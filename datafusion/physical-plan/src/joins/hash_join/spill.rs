// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Disk-spilling fallback for [`super::HashJoinExec`] (grace hash join).
//!
//! The fast path is untouched: the operator buffers the build side in memory
//! exactly as before. Only when the build-side [`MemoryReservation`] first
//! fails does the build switch into *scatter mode*: already-buffered batches
//! and the rest of the build input are hash-partitioned into `K` disk
//! partitions (releasing their memory), the probe side is scattered with the
//! same hash, and the join then runs partition-pair by partition-pair, each
//! pair building an ordinary in-memory hash table within the memory budget.
//!
//! Memory ownership rules (every reservation has exactly one owner):
//! - `HashJoinInput[p]` (the existing build reservation): grows per buffered
//!   batch on the fast path, shrinks batch-by-batch as buffers are scattered,
//!   reaches zero once scatter mode is fully engaged.
//! - `HashJoinSpillHeadroom[p]`: a fixed reservation covering scatter scratch
//!   and per-partition write buffers; held until the spill join completes.
//! - `HashJoinSpillPartition[p.k]`: per partition-pair; covers the loaded
//!   build batches plus the hash table built from them (moved into the
//!   pair's [`JoinLeftData`]); dropped when the pair finishes.
//!
//! The scatter hash uses seeds distinct from both `RepartitionExec`'s
//! `(0,0,0,0)` routing seeds and the join hash map's `HASH_JOIN_SEED`
//! (`'J','O','I','N'`), so disk partitions do not correlate with either the
//! upstream partitioning or the per-pair hash tables.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::task::{Context, Poll};

use crate::hash_utils::create_hashes;
use crate::joins::PartitionMode;
use crate::joins::SharedBitmapBuilder;
use crate::joins::hash_join::exec::{
    BuildPhaseOutput, build_left_data, hash_table_estimate,
};
use crate::joins::hash_join::shared_bounds::PartitionBounds;
use crate::joins::hash_join::stream::{
    BuildSide, BuildSideInitialState, HashJoinStream, HashJoinStreamState,
};
use crate::joins::utils::{
    BuildProbeJoinMetrics, ColumnIndex, JoinFilter, OnceFut, need_produce_result_in_final,
};
use crate::metrics::{Count, ExecutionPlanMetricsSet};
use crate::spill::get_record_batch_memory_size;
use crate::spill::in_progress_spill_file::InProgressSpillFile;
use crate::spill::spill_manager::SpillManager;
use crate::stream::RecordBatchStreamAdapter;
use crate::{RecordBatchStream, SendableRecordBatchStream};

use arrow::array::{Array, BooleanBufferBuilder, StringViewArray, UInt32Array};
use arrow::compute::{concat_batches, take_record_batch};
use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use datafusion_common::config::ConfigOptions;
use datafusion_common::{DataFusionError, JoinType, NullEquality, Result, internal_err};
use datafusion_execution::disk_manager::RefCountedTempFile;
use datafusion_execution::memory_pool::{
    MemoryConsumer, MemoryLimit, MemoryPool, MemoryReservation,
};
use datafusion_physical_expr::PhysicalExprRef;
use datafusion_physical_expr_common::utils::evaluate_expressions_to_arrays;

use ahash::RandomState;
use futures::future::BoxFuture;
use futures::{Stream, StreamExt, ready};
use parking_lot::Mutex;

/// Per-partition write-buffer flush threshold: coalesce scattered slices into
/// IPC batches of roughly this size before writing.
const WRITE_BUFFER_FLUSH_BYTES: usize = 1024 * 1024;

/// Fallback per-partition target size when the memory pool cannot report a
/// finite limit.
const DEFAULT_PARTITION_TARGET_BYTES: usize = 128 * 1024 * 1024;

/// SplitMix64 finalizer: a strong 64-bit mixer for seed derivation.
fn splitmix64(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Hash seeds for scattering rows into disk partitions.
///
/// Must stay independent from `REPARTITION_RANDOM_STATE` (0,0,0,0) and
/// `HASH_JOIN_SEED` ('J','O','I','N'): rows arrive pre-bucketed by the former
/// and are hashed by the latter inside each pair's hash table.
///
/// All four seeds are derived from the level through a strong mixer: ahash
/// folds some seed words in ADDITIVELY, so states differing only by a small
/// constant in one word produce hashes shifted by a constant — under `% K`
/// that maps an entire parent partition into a single child, making
/// recursive repartitioning useless. (Caught by
/// `recursive_scatter_levels_are_decorrelated`.)
pub(super) fn scatter_random_state(level: usize) -> RandomState {
    let base = 0x53_50_4C_00_C0_FF_EE_00u64 ^ (level as u64);
    RandomState::with_seeds(
        splitmix64(base),
        splitmix64(base ^ 0xA5A5_A5A5_A5A5_A5A5),
        splitmix64(base.rotate_left(17)),
        splitmix64(base.rotate_left(43) ^ 0x5A5A_5A5A_5A5A_5A5A),
    )
}

/// Choose the number of disk partitions when spilling engages.
///
/// `buffered_bytes` is what fit in memory before the reservation failed; the
/// 4x factor assumes overflow happened early in the stream. Recursion
/// corrects underestimates, so precision is not important.
/// `concurrent_consumers` is how many partition tables can be resident at
/// once: 1 for Partitioned mode (one pair at a time), N for a shared
/// CollectLeft build whose N probe partitions walk the k's out of sync —
/// the per-partition target shrinks accordingly.
pub(super) fn compute_partition_count(
    buffered_bytes: usize,
    pool_limit: &MemoryLimit,
    override_count: usize,
    concurrent_consumers: usize,
) -> usize {
    if override_count > 0 {
        return override_count;
    }
    let base_target = match pool_limit {
        MemoryLimit::Finite(limit) => {
            (*limit / 16).clamp(32 * 1024 * 1024, 256 * 1024 * 1024)
        }
        _ => DEFAULT_PARTITION_TARGET_BYTES,
    };
    let target = base_target / concurrent_consumers.max(1);
    buffered_bytes
        .saturating_mul(4)
        .div_ceil(target.max(1))
        .clamp(16, 64)
}

/// Everything a hash join partition needs to engage and drive spilling.
/// Built once per output partition in `HashJoinExec::execute` when the
/// feature is enabled and applicable; shared by the build phase and the
/// spill-join driver.
pub(super) struct HashJoinSpillContext {
    pub(super) pool: Arc<dyn MemoryPool>,
    /// Spill manager for build-side partitions (build schema).
    pub(super) build_spill: Arc<SpillManager>,
    /// Spill manager for probe-side partitions (probe schema).
    pub(super) probe_spill: Arc<SpillManager>,
    /// Incremented once when a build side switches into scatter mode.
    pub(super) spill_engaged: Count,
    /// Incremented per recursive repartition pass on an oversized pair.
    pub(super) repartition_passes: Count,
    /// Incremented per chunk executed by the chunked build fallback.
    pub(super) fallback_chunks: Count,
    pub(super) partition_count_override: usize,
    pub(super) headroom_bytes: usize,
    pub(super) max_recursion_depth: usize,
    pub(super) max_spill_file_size: usize,
    /// Output partition index (consumer naming / diagnostics).
    pub(super) partition: usize,
    /// Build-side join key expressions (needed to rebuild per-pair tables).
    pub(super) on_left: Vec<PhysicalExprRef>,
    pub(super) config: Arc<ConfigOptions>,
}

/// One side's scattered partition: the rotated spill files plus totals.
pub(super) struct SpilledSide {
    pub(super) files: Vec<RefCountedTempFile>,
    pub(super) bytes: usize,
    pub(super) rows: usize,
}

/// The build phase's output when it overflowed and scattered to disk.
pub(super) struct SpilledBuild {
    pub(super) partitions: Vec<SpilledSide>,
    pub(super) partition_count: usize,
    /// Recursion level the scatter used (level 0 at first engage).
    pub(super) level: usize,
    /// Dynamic-filter bounds computed before/during the scatter (bounds
    /// accumulators keep running, so these stay exact under spill).
    pub(super) bounds: Option<PartitionBounds>,
    /// Headroom reservation kept alive for the probe scatter and the
    /// partition-pair loop; dropped when the spill join completes.
    pub(super) headroom: MemoryReservation,
}

/// A spilled build shared by all `CollectLeft` probe partitions.
///
/// The single build was scattered once into `K` disk partitions; each of the
/// `probe_threads` output partitions scatters ITS OWN probe stream with the
/// same seed and then walks k = 0..K, loading its OWN copy of k's hash
/// table (memory: at most one table per probe partition at a time, which is
/// what the auto partition-count sizing assumes). What IS shared per k is
/// the visited bitmap and the probe-completion counter injected into each
/// copy's [`JoinLeftData`]: matches from every probe partition mark one
/// bitmap, and the existing last-finisher logic emits unmatched build rows
/// exactly once. Row order is deterministic across copies (same files, same
/// order), so bitmap indices agree.
///
/// [`JoinLeftData`]: super::exec::JoinLeftData
pub(super) struct SharedSpilledBuild {
    pub(super) partitions: Vec<SharedSpilledSide>,
    pub(super) partition_count: usize,
    pub(super) level: usize,
    /// Bounds for dynamic-filter reporting (exact — accumulators keep
    /// running during the scatter).
    pub(super) bounds: Option<PartitionBounds>,
    /// Number of probe partitions that will walk the shared build.
    pub(super) probe_threads: usize,
    /// Per-k shared visited bitmap + probe-completion counter.
    shared_state: Vec<(Arc<SharedBitmapBuilder>, Arc<AtomicUsize>)>,
    /// Scatter headroom, held until the whole shared build is dropped.
    _headroom: MemoryReservation,
}

/// One shared build partition: file handles are `Arc`d so every output
/// partition can read them.
pub(super) struct SharedSpilledSide {
    pub(super) files: Arc<Vec<RefCountedTempFile>>,
}

impl SharedSpilledBuild {
    pub(super) fn new(
        partitions: Vec<SpilledSide>,
        level: usize,
        bounds: Option<PartitionBounds>,
        probe_threads: usize,
        headroom: MemoryReservation,
    ) -> Self {
        let partition_count = partitions.len();
        let partitions = partitions
            .into_iter()
            .map(|side| SharedSpilledSide {
                files: Arc::new(side.files),
            })
            .collect();
        Self {
            partitions,
            partition_count,
            level,
            bounds,
            probe_threads,
            shared_state: (0..partition_count)
                .map(|_| {
                    (
                        Arc::new(Mutex::new(BooleanBufferBuilder::new(0))),
                        Arc::new(AtomicUsize::new(probe_threads)),
                    )
                })
                .collect(),
            _headroom: headroom,
        }
    }
}

/// A single partition's spill writer: buffers scattered slices and appends
/// them as coalesced IPC batches, rotating files at `max_file_size` bytes.
struct PartitionSpillWriter {
    spill_manager: Arc<SpillManager>,
    description: String,
    max_file_size: usize,
    current: Option<InProgressSpillFile>,
    current_bytes: usize,
    files: Vec<RefCountedTempFile>,
    buffered: Vec<RecordBatch>,
    buffered_bytes: usize,
    bytes: usize,
    rows: usize,
}

impl PartitionSpillWriter {
    fn new(
        spill_manager: Arc<SpillManager>,
        description: String,
        max_file_size: usize,
    ) -> Self {
        Self {
            spill_manager,
            description,
            max_file_size,
            current: None,
            current_bytes: 0,
            files: Vec::new(),
            buffered: Vec::new(),
            buffered_bytes: 0,
            bytes: 0,
            rows: 0,
        }
    }

    fn append(&mut self, batch: RecordBatch) -> Result<()> {
        if batch.num_rows() == 0 {
            return Ok(());
        }
        self.rows += batch.num_rows();
        self.buffered_bytes += get_record_batch_memory_size(&batch);
        self.buffered.push(batch);
        if self.buffered_bytes >= WRITE_BUFFER_FLUSH_BYTES {
            self.flush()?;
        }
        Ok(())
    }

    /// Write the buffered slices out as one coalesced IPC batch.
    fn flush(&mut self) -> Result<()> {
        if self.buffered.is_empty() {
            return Ok(());
        }
        let schema = Arc::clone(self.spill_manager.schema());
        let batch = concat_batches(&schema, self.buffered.iter())?;
        // Compact StringView columns: `take`d views reference the source
        // batch's full data buffers, which the IPC writer would otherwise
        // serialize wholesale into every partition.
        let batch = compact_string_views(batch)?;
        self.buffered.clear();
        self.buffered_bytes = 0;

        if self.current.is_none() {
            self.current = Some(
                self.spill_manager
                    .create_in_progress_file(&self.description)?,
            );
            self.current_bytes = 0;
        }
        if let Some(writer) = self.current.as_mut() {
            writer.append_batch(&batch)?;
        }
        let written = get_record_batch_memory_size(&batch);
        self.current_bytes += written;
        self.bytes += written;

        if self.current_bytes >= self.max_file_size {
            self.rotate_file()?;
        }
        Ok(())
    }

    /// Close the in-progress file (if any) and add it to the finished set.
    fn rotate_file(&mut self) -> Result<()> {
        if let Some(mut writer) = self.current.take()
            && let Some(file) = writer.finish()?
        {
            self.files.push(file);
        }
        Ok(())
    }

    fn finish(mut self) -> Result<SpilledSide> {
        self.flush()?;
        self.rotate_file()?;
        Ok(SpilledSide {
            files: self.files,
            bytes: self.bytes,
            rows: self.rows,
        })
    }
}

/// Rebuild any StringView columns so they own only the data they reference.
fn compact_string_views(batch: RecordBatch) -> Result<RecordBatch> {
    let mut mutated = false;
    let mut columns = Vec::with_capacity(batch.num_columns());
    for array in batch.columns() {
        if let Some(string_view) = array.as_any().downcast_ref::<StringViewArray>() {
            columns.push(Arc::new(string_view.gc()) as _);
            mutated = true;
        } else {
            columns.push(Arc::clone(array));
        }
    }
    if mutated {
        Ok(RecordBatch::try_new(batch.schema(), columns)?)
    } else {
        Ok(batch)
    }
}

/// Scatters one side's batches into `K` disk partitions by join-key hash.
pub(super) struct SideScatter {
    on_exprs: Vec<PhysicalExprRef>,
    random_state: RandomState,
    writers: Vec<PartitionSpillWriter>,
    /// Flush all write buffers when their sum exceeds this.
    buffer_cap: usize,
    hashes_buffer: Vec<u64>,
}

impl SideScatter {
    pub(super) fn new(
        partition_count: usize,
        on_exprs: Vec<PhysicalExprRef>,
        random_state: RandomState,
        spill_manager: &Arc<SpillManager>,
        max_file_size: usize,
        buffer_cap: usize,
        description: &str,
    ) -> Self {
        let writers = (0..partition_count)
            .map(|i| {
                PartitionSpillWriter::new(
                    Arc::clone(spill_manager),
                    format!("{description}[{i}]"),
                    max_file_size,
                )
            })
            .collect();
        Self {
            on_exprs,
            random_state,
            writers,
            buffer_cap,
            hashes_buffer: Vec::new(),
        }
    }

    pub(super) fn scatter_batch(&mut self, batch: &RecordBatch) -> Result<()> {
        let num_rows = batch.num_rows();
        if num_rows == 0 {
            return Ok(());
        }
        let keys = evaluate_expressions_to_arrays(&self.on_exprs, batch)?;
        self.hashes_buffer.clear();
        self.hashes_buffer.resize(num_rows, 0);
        create_hashes(&keys, &self.random_state, &mut self.hashes_buffer)?;

        let partition_count = self.writers.len() as u64;
        let mut indices: Vec<Vec<u32>> = vec![Vec::new(); self.writers.len()];
        for (row, hash) in self.hashes_buffer.iter().enumerate() {
            indices[(hash % partition_count) as usize].push(row as u32);
        }

        for (partition, rows) in indices.into_iter().enumerate() {
            if rows.is_empty() {
                continue;
            }
            let take_indices = UInt32Array::from(rows);
            let slice = take_record_batch(batch, &take_indices)?;
            self.writers[partition].append(slice)?;
        }

        let total_buffered: usize = self.writers.iter().map(|w| w.buffered_bytes).sum();
        if total_buffered > self.buffer_cap {
            for writer in &mut self.writers {
                writer.flush()?;
            }
        }
        Ok(())
    }

    pub(super) fn finish(self) -> Result<Vec<SpilledSide>> {
        self.writers.into_iter().map(|w| w.finish()).collect()
    }
}

/// Lazily chains a partition's spill files into one record batch stream.
struct ChainedSpillReader {
    schema: SchemaRef,
    files: VecDeque<RefCountedTempFile>,
    spill_manager: Arc<SpillManager>,
    current: Option<SendableRecordBatchStream>,
}

impl Stream for ChainedSpillReader {
    type Item = Result<RecordBatch>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        loop {
            if let Some(current) = &mut self.current {
                match ready!(current.poll_next_unpin(cx)) {
                    Some(item) => return Poll::Ready(Some(item)),
                    None => self.current = None,
                }
            }
            match self.files.pop_front() {
                Some(file) => {
                    let this = &mut *self;
                    match this.spill_manager.read_spill_as_stream(file, None) {
                        Ok(stream) => this.current = Some(stream),
                        Err(e) => return Poll::Ready(Some(Err(e))),
                    }
                }
                None => return Poll::Ready(None),
            }
        }
    }
}

impl RecordBatchStream for ChainedSpillReader {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}

fn spilled_side_stream(
    files: Vec<RefCountedTempFile>,
    spill_manager: Arc<SpillManager>,
) -> SendableRecordBatchStream {
    Box::pin(ChainedSpillReader {
        schema: Arc::clone(spill_manager.schema()),
        files: files.into(),
        spill_manager,
        current: None,
    })
}

/// An immediately-empty record batch stream (used to take ownership of the
/// outer probe stream when handing it to the spill driver).
pub(super) fn empty_record_batch_stream(schema: SchemaRef) -> SendableRecordBatchStream {
    Box::pin(RecordBatchStreamAdapter::new(
        schema,
        futures::stream::empty(),
    ))
}

/// Parameters copied from the outer [`HashJoinStream`] needed to construct
/// the per-partition-pair inner join streams.
pub(super) struct InnerJoinSpec {
    pub(super) partition: usize,
    pub(super) schema: SchemaRef,
    pub(super) on_right: Vec<PhysicalExprRef>,
    pub(super) filter: Option<JoinFilter>,
    pub(super) join_type: JoinType,
    /// The join hash map's random state (`HASH_JOIN_SEED`).
    pub(super) random_state: RandomState,
    pub(super) column_indices: Vec<ColumnIndex>,
    pub(super) null_equality: NullEquality,
    pub(super) batch_size: usize,
}

/// Fan-out when an oversized pair is recursively repartitioned.
const CHILD_FANOUT: usize = 8;

/// One partition pair waiting to be joined.
struct PairWork {
    build: SpilledSide,
    probe: SpilledSide,
    /// Recursion level this pair's data was scattered at.
    level: usize,
    /// False once a repartition pass failed to shrink the data
    /// (duplicate-key mass): further passes cannot help, go straight to the
    /// chunked fallback.
    allow_recursion: bool,
}

/// Per-pair state carried through the chunked fallback: the remaining build
/// stream and the probe files that get re-read once per chunk.
struct ChunkedState {
    pair_id: usize,
    level: usize,
    /// None once the build partition is fully consumed.
    build_stream: Option<SendableRecordBatchStream>,
    /// A batch that did not fit in the previous chunk; leads the next one.
    pending: Option<RecordBatch>,
    probe_files: Vec<RefCountedTempFile>,
    chunk_index: usize,
}

impl ChunkedState {
    fn exhausted(&self) -> bool {
        self.build_stream.is_none() && self.pending.is_none()
    }
}

enum DriverPhase {
    /// Scatter the entire probe stream into `K` disk partitions.
    ScatterProbe {
        probe: SendableRecordBatchStream,
        scatter: SideScatter,
    },
    /// Pick the next partition pair to join.
    NextPair,
    /// Stream the pair's build partition into memory under its reservation.
    LoadBuild {
        pair_id: usize,
        level: usize,
        allow_recursion: bool,
        probe: SpilledSide,
        stream: SendableRecordBatchStream,
        batches: Vec<RecordBatch>,
        num_rows: usize,
        reservation: MemoryReservation,
        /// Bytes of the hash-table estimate reserved so far (grows with the
        /// row count so table overflow surfaces during the load).
        table_reserved: usize,
        parent_build_bytes: usize,
    },
    /// Run the pair's in-memory join and forward its output.
    RunPair {
        inner: SendableRecordBatchStream,
    },
    /// The pair's build side did not fit: re-scatter the remaining build
    /// rows (already-loaded ones went in first) at the next seed level.
    RepartitionBuild {
        level: usize,
        probe: SpilledSide,
        stream: SendableRecordBatchStream,
        scatter: SideScatter,
        parent_build_bytes: usize,
    },
    /// Re-scatter the oversized pair's probe partition with the same child
    /// seed, then enqueue the child pairs.
    RepartitionProbe {
        children_build: Vec<SpilledSide>,
        child_level: usize,
        allow_recursion: bool,
        stream: SendableRecordBatchStream,
        scatter: SideScatter,
    },
    /// Chunked fallback: load the next slice of build rows that fits.
    ChunkedLoad {
        state: ChunkedState,
        batches: Vec<RecordBatch>,
        num_rows: usize,
        reservation: MemoryReservation,
        table_reserved: usize,
    },
    /// Chunked fallback: run one chunk's join over a full probe re-read.
    ChunkedRun {
        inner: SendableRecordBatchStream,
        state: ChunkedState,
    },
    Done,
}

/// Drives the spilled join: probe scatter, then a sequential loop over
/// partition pairs, each joined by an ordinary in-memory [`HashJoinStream`].
///
/// A pair whose build side exceeds the budget is recursively repartitioned
/// with a fresh seed (up to `hash_join_spill_max_recursion_depth` passes);
/// when repartitioning cannot shrink it (duplicate-key mass), build-side
/// emission join types degrade to a chunked build over full probe re-reads,
/// and the rest surface a descriptive resources-exhausted error.
pub(super) struct SpillJoinDriver {
    ctx: Arc<HashJoinSpillContext>,
    spec: InnerJoinSpec,
    phase: DriverPhase,
    /// Build partitions from the initial scatter, consumed when the probe
    /// scatter finishes and the pair queue is built.
    initial_build: Vec<SpilledSide>,
    base_level: usize,
    queue: VecDeque<PairWork>,
    /// Monotonic pair counter for reservation naming across recursion.
    pair_seq: usize,
    /// Outer metrics: probe input rows/batches are recorded during scatter.
    outer_metrics: BuildProbeJoinMetrics,
    /// Private metrics for inner per-pair streams (not exposed; prevents
    /// double counting on the operator's metrics set).
    inner_metrics_set: ExecutionPlanMetricsSet,
    /// Held for the driver's lifetime; released on drop.
    _headroom: MemoryReservation,
}

impl SpillJoinDriver {
    pub(super) fn new(
        spilled: SpilledBuild,
        probe: SendableRecordBatchStream,
        ctx: Arc<HashJoinSpillContext>,
        spec: InnerJoinSpec,
        outer_metrics: BuildProbeJoinMetrics,
    ) -> Self {
        let SpilledBuild {
            partitions,
            partition_count,
            level,
            bounds: _,
            headroom,
        } = spilled;

        let scatter = SideScatter::new(
            partition_count,
            spec.on_right.clone(),
            scatter_random_state(level),
            &ctx.probe_spill,
            ctx.max_spill_file_size,
            ctx.headroom_bytes / 2,
            "hash_join_probe_spill",
        );

        Self {
            spec,
            phase: DriverPhase::ScatterProbe { probe, scatter },
            initial_build: partitions,
            base_level: level,
            queue: VecDeque::new(),
            pair_seq: 0,
            outer_metrics,
            inner_metrics_set: ExecutionPlanMetricsSet::new(),
            _headroom: headroom,
            ctx,
        }
    }

    /// Whether an empty side lets the pair be skipped entirely, i.e. the
    /// join type provably produces no rows for it. Pairs that emit unmatched
    /// rows from a non-empty side must still run: the inner stream handles
    /// an empty probe (unmatched build emission) and an empty build
    /// (probe-side null padding) natively.
    fn can_skip_pair(&self, build_rows: usize, probe_rows: usize) -> bool {
        use JoinType::*;
        match self.spec.join_type {
            // Emit only matches: either side empty means no output.
            Inner | LeftSemi | RightSemi => build_rows == 0 || probe_rows == 0,
            // Emit (un)matched build rows: only an empty build side is silent.
            Left | LeftAnti | LeftMark => build_rows == 0,
            // Emit (un)matched probe rows: only an empty probe side is silent.
            Right | RightAnti | RightMark => probe_rows == 0,
            // Emits unmatched rows from both sides.
            Full => build_rows == 0 && probe_rows == 0,
        }
    }

    /// Chunking the build side is correct only for join types whose output
    /// is decided per build row (each build row lives in exactly one chunk).
    /// Types that emit unmatched PROBE rows would need cross-chunk match
    /// tracking, which chunked execution cannot provide.
    fn chunked_fallback_supported(&self) -> bool {
        matches!(
            self.spec.join_type,
            JoinType::Inner
                | JoinType::Left
                | JoinType::LeftSemi
                | JoinType::LeftAnti
                | JoinType::LeftMark
        )
    }

    pub(super) fn poll_next(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<RecordBatch>>> {
        loop {
            match &self.phase {
                DriverPhase::ScatterProbe { .. } => {
                    ready!(self.poll_scatter_probe(cx))?;
                }
                DriverPhase::NextPair => self.select_next_pair()?,
                DriverPhase::LoadBuild { .. } => {
                    ready!(self.poll_load_build(cx))?;
                }
                DriverPhase::RunPair { .. } => match ready!(self.poll_run_pair(cx)) {
                    Some(item) => return Poll::Ready(Some(item)),
                    None => self.phase = DriverPhase::NextPair,
                },
                DriverPhase::RepartitionBuild { .. } => {
                    ready!(self.poll_repartition_build(cx))?;
                }
                DriverPhase::RepartitionProbe { .. } => {
                    ready!(self.poll_repartition_probe(cx))?;
                }
                DriverPhase::ChunkedLoad { .. } => {
                    ready!(self.poll_chunked_load(cx))?;
                }
                DriverPhase::ChunkedRun { .. } => {
                    match ready!(self.poll_chunked_run(cx)) {
                        Some(item) => return Poll::Ready(Some(item)),
                        None => self.finish_chunk()?,
                    }
                }
                DriverPhase::Done => return Poll::Ready(None),
            }
        }
    }

    /// One step of probe scatter. Returns `Ready(Ok(()))` after a state
    /// change; the caller loops.
    fn poll_scatter_probe(&mut self, cx: &mut Context<'_>) -> Poll<Result<()>> {
        let polled = {
            let DriverPhase::ScatterProbe { probe, .. } = &mut self.phase else {
                return Poll::Ready(internal_err!("expected ScatterProbe phase"));
            };
            ready!(probe.poll_next_unpin(cx))
        };
        match polled {
            Some(batch) => {
                let batch = batch?;
                self.outer_metrics.input_batches.add(1);
                self.outer_metrics.input_rows.add(batch.num_rows());
                let DriverPhase::ScatterProbe { scatter, .. } = &mut self.phase else {
                    return Poll::Ready(internal_err!("expected ScatterProbe phase"));
                };
                scatter.scatter_batch(&batch)?;
            }
            None => {
                let DriverPhase::ScatterProbe { scatter, .. } =
                    std::mem::replace(&mut self.phase, DriverPhase::NextPair)
                else {
                    return Poll::Ready(internal_err!("expected ScatterProbe phase"));
                };
                let probe_parts = scatter.finish()?;
                let level = self.base_level;
                self.queue = std::mem::take(&mut self.initial_build)
                    .into_iter()
                    .zip(probe_parts)
                    .map(|(build, probe)| PairWork {
                        build,
                        probe,
                        level,
                        allow_recursion: true,
                    })
                    .collect();
            }
        }
        Poll::Ready(Ok(()))
    }

    /// Pop the next pair; skip pairs this join type provably emits nothing
    /// for.
    fn select_next_pair(&mut self) -> Result<()> {
        loop {
            let Some(work) = self.queue.pop_front() else {
                self.phase = DriverPhase::Done;
                return Ok(());
            };

            if self.can_skip_pair(work.build.rows, work.probe.rows) {
                // Dropping the sides deletes their spill files.
                continue;
            }

            let pair_id = self.pair_seq;
            self.pair_seq += 1;

            let reservation = MemoryConsumer::new(format!(
                "HashJoinSpillPartition[{}.{}]",
                self.ctx.partition, pair_id
            ))
            .with_can_spill(true)
            .register(&self.ctx.pool);

            let parent_build_bytes = work.build.bytes;
            let stream =
                spilled_side_stream(work.build.files, Arc::clone(&self.ctx.build_spill));
            self.phase = DriverPhase::LoadBuild {
                pair_id,
                level: work.level,
                allow_recursion: work.allow_recursion,
                probe: work.probe,
                stream,
                batches: Vec::new(),
                num_rows: 0,
                reservation,
                table_reserved: 0,
                parent_build_bytes,
            };
            return Ok(());
        }
    }

    /// One step of loading the current pair's build partition.
    fn poll_load_build(&mut self, cx: &mut Context<'_>) -> Poll<Result<()>> {
        let polled = {
            let DriverPhase::LoadBuild { stream, .. } = &mut self.phase else {
                return Poll::Ready(internal_err!("expected LoadBuild phase"));
            };
            ready!(stream.poll_next_unpin(cx))
        };
        match polled {
            Some(batch) => {
                let batch = batch?;
                let size = get_record_batch_memory_size(&batch);
                let grow_result = {
                    let DriverPhase::LoadBuild {
                        reservation,
                        num_rows,
                        table_reserved,
                        ..
                    } = &mut self.phase
                    else {
                        return Poll::Ready(internal_err!("expected LoadBuild phase"));
                    };
                    // Also reserve the hash table's estimated growth so the
                    // pair overflows here (where recursion/fallback applies)
                    // rather than at table construction.
                    let table_delta = hash_table_estimate(*num_rows + batch.num_rows())?
                        .saturating_sub(*table_reserved);
                    reservation
                        .try_grow(size + table_delta)
                        .map(|()| table_delta)
                };
                match grow_result {
                    Ok(table_delta) => {
                        let DriverPhase::LoadBuild {
                            batches,
                            num_rows,
                            table_reserved,
                            ..
                        } = &mut self.phase
                        else {
                            return Poll::Ready(internal_err!(
                                "expected LoadBuild phase"
                            ));
                        };
                        *num_rows += batch.num_rows();
                        *table_reserved += table_delta;
                        batches.push(batch);
                    }
                    Err(e) => {
                        return Poll::Ready(self.handle_oversized_build(batch, e));
                    }
                }
            }
            None => {
                let DriverPhase::LoadBuild {
                    pair_id,
                    probe,
                    batches,
                    num_rows,
                    reservation,
                    table_reserved,
                    ..
                } = std::mem::replace(&mut self.phase, DriverPhase::NextPair)
                else {
                    return Poll::Ready(internal_err!("expected LoadBuild phase"));
                };
                let inner = self.build_inner_join(
                    pair_id,
                    batches,
                    num_rows,
                    reservation,
                    table_reserved,
                    probe.files,
                )?;
                self.phase = DriverPhase::RunPair { inner };
            }
        }
        Poll::Ready(Ok(()))
    }

    /// The current pair's build partition exceeded the budget mid-load.
    /// Choose: recursive repartition, chunked fallback, or a clean error.
    fn handle_oversized_build(
        &mut self,
        failed_batch: RecordBatch,
        source: DataFusionError,
    ) -> Result<()> {
        let (level, allow_recursion) = {
            let DriverPhase::LoadBuild {
                level,
                allow_recursion,
                ..
            } = &self.phase
            else {
                return internal_err!("expected LoadBuild phase");
            };
            (*level, *allow_recursion)
        };

        if allow_recursion && level < self.ctx.max_recursion_depth {
            self.begin_repartition(failed_batch)
        } else if self.chunked_fallback_supported() {
            self.begin_chunked(failed_batch)
        } else {
            let join_type = self.spec.join_type;
            Err(source.context(format!(
                "spilled hash join partition remains oversized after {level} \
                 repartition pass(es) (extreme key skew), and join type \
                 {join_type:?} decides probe-side rows from matches against \
                 the WHOLE build side, which the chunked build fallback \
                 cannot track across chunks; increase the memory budget to \
                 run this join"
            )))
        }
    }

    /// Re-scatter the oversized pair's build side at the next seed level:
    /// already-loaded batches first (releasing their memory), then the rest
    /// of the pair's build stream.
    fn begin_repartition(&mut self, failed_batch: RecordBatch) -> Result<()> {
        let DriverPhase::LoadBuild {
            level,
            probe,
            stream,
            batches,
            reservation,
            parent_build_bytes,
            ..
        } = std::mem::replace(&mut self.phase, DriverPhase::NextPair)
        else {
            return internal_err!("expected LoadBuild phase");
        };

        self.ctx.repartition_passes.add(1);
        let child_level = level + 1;
        let mut scatter = SideScatter::new(
            CHILD_FANOUT,
            self.ctx.on_left.clone(),
            scatter_random_state(child_level),
            &self.ctx.build_spill,
            self.ctx.max_spill_file_size,
            self.ctx.headroom_bytes / 2,
            "hash_join_build_respill",
        );

        for batch in batches {
            scatter.scatter_batch(&batch)?;
        }
        // The freed bytes exactly cover the scattered batches — this
        // reservation held nothing else.
        drop(reservation);
        scatter.scatter_batch(&failed_batch)?;
        drop(failed_batch);

        self.phase = DriverPhase::RepartitionBuild {
            level,
            probe,
            stream,
            scatter,
            parent_build_bytes,
        };
        Ok(())
    }

    fn poll_repartition_build(&mut self, cx: &mut Context<'_>) -> Poll<Result<()>> {
        let polled = {
            let DriverPhase::RepartitionBuild { stream, .. } = &mut self.phase else {
                return Poll::Ready(internal_err!("expected RepartitionBuild phase"));
            };
            ready!(stream.poll_next_unpin(cx))
        };
        match polled {
            Some(batch) => {
                let batch = batch?;
                let DriverPhase::RepartitionBuild { scatter, .. } = &mut self.phase
                else {
                    return Poll::Ready(internal_err!("expected RepartitionBuild phase"));
                };
                scatter.scatter_batch(&batch)?;
            }
            None => {
                let DriverPhase::RepartitionBuild {
                    level,
                    probe,
                    scatter,
                    parent_build_bytes,
                    ..
                } = std::mem::replace(&mut self.phase, DriverPhase::NextPair)
                else {
                    return Poll::Ready(internal_err!("expected RepartitionBuild phase"));
                };
                let children_build = scatter.finish()?;

                // No-shrink shortcut: when the largest child still holds
                // most of the parent, the mass sits on duplicate keys and
                // more passes cannot split it — the children go straight to
                // the fallback if they do not fit.
                let max_child = children_build.iter().map(|c| c.bytes).max();
                let allow_recursion = max_child
                    .is_none_or(|m| m.saturating_mul(4) <= parent_build_bytes * 3);

                let child_level = level + 1;
                let probe_scatter = SideScatter::new(
                    CHILD_FANOUT,
                    self.spec.on_right.clone(),
                    scatter_random_state(child_level),
                    &self.ctx.probe_spill,
                    self.ctx.max_spill_file_size,
                    self.ctx.headroom_bytes / 2,
                    "hash_join_probe_respill",
                );
                let probe_stream =
                    spilled_side_stream(probe.files, Arc::clone(&self.ctx.probe_spill));
                self.phase = DriverPhase::RepartitionProbe {
                    children_build,
                    child_level,
                    allow_recursion,
                    stream: probe_stream,
                    scatter: probe_scatter,
                };
            }
        }
        Poll::Ready(Ok(()))
    }

    fn poll_repartition_probe(&mut self, cx: &mut Context<'_>) -> Poll<Result<()>> {
        let polled = {
            let DriverPhase::RepartitionProbe { stream, .. } = &mut self.phase else {
                return Poll::Ready(internal_err!("expected RepartitionProbe phase"));
            };
            ready!(stream.poll_next_unpin(cx))
        };
        match polled {
            Some(batch) => {
                let batch = batch?;
                let DriverPhase::RepartitionProbe { scatter, .. } = &mut self.phase
                else {
                    return Poll::Ready(internal_err!("expected RepartitionProbe phase"));
                };
                scatter.scatter_batch(&batch)?;
            }
            None => {
                let DriverPhase::RepartitionProbe {
                    children_build,
                    child_level,
                    allow_recursion,
                    scatter,
                    ..
                } = std::mem::replace(&mut self.phase, DriverPhase::NextPair)
                else {
                    return Poll::Ready(internal_err!("expected RepartitionProbe phase"));
                };
                let children_probe = scatter.finish()?;
                // Children go to the FRONT so their disk space is reclaimed
                // before other top-level pairs run.
                for (build, probe) in children_build.into_iter().zip(children_probe).rev()
                {
                    self.queue.push_front(PairWork {
                        build,
                        probe,
                        level: child_level,
                        allow_recursion,
                    });
                }
            }
        }
        Poll::Ready(Ok(()))
    }

    /// Enter the chunked fallback: the already-loaded batches (which fit)
    /// become chunk 0; the failed batch leads chunk 1.
    fn begin_chunked(&mut self, failed_batch: RecordBatch) -> Result<()> {
        let DriverPhase::LoadBuild {
            pair_id,
            level,
            probe,
            stream,
            batches,
            num_rows,
            reservation,
            table_reserved,
            ..
        } = std::mem::replace(&mut self.phase, DriverPhase::NextPair)
        else {
            return internal_err!("expected LoadBuild phase");
        };

        if batches.is_empty() {
            // Not even one batch fits: nothing to chunk.
            return internal_err!(
                "spilled hash join partition {pair_id}: a single build batch \
                 exceeds the memory budget"
            );
        }

        let state = ChunkedState {
            pair_id,
            level,
            build_stream: Some(stream),
            pending: Some(failed_batch),
            probe_files: probe.files,
            chunk_index: 0,
        };
        let inner = self.start_chunk_join(
            &state,
            batches,
            num_rows,
            reservation,
            table_reserved,
        )?;
        self.phase = DriverPhase::ChunkedRun { inner, state };
        Ok(())
    }

    /// One step of loading the next chunk of build rows.
    fn poll_chunked_load(&mut self, cx: &mut Context<'_>) -> Poll<Result<()>> {
        // A pending batch from the previous chunk leads this one.
        let pending_step = {
            let DriverPhase::ChunkedLoad {
                state,
                batches,
                num_rows,
                reservation,
                table_reserved,
            } = &mut self.phase
            else {
                return Poll::Ready(internal_err!("expected ChunkedLoad phase"));
            };
            if let Some(batch) = state.pending.take() {
                let size = get_record_batch_memory_size(&batch);
                let table_delta = hash_table_estimate(*num_rows + batch.num_rows())?
                    .saturating_sub(*table_reserved);
                match reservation.try_grow(size + table_delta) {
                    Ok(()) => {
                        *num_rows += batch.num_rows();
                        *table_reserved += table_delta;
                        batches.push(batch);
                        true
                    }
                    Err(e) => {
                        if batches.is_empty() {
                            let pair_id = state.pair_id;
                            return Poll::Ready(Err(e.context(format!(
                                "spilled hash join partition {pair_id}: a single \
                                 build batch exceeds the memory budget"
                            ))));
                        }
                        // Chunk full before the pending batch: run with what
                        // we have, keep the batch pending.
                        state.pending = Some(batch);
                        return Poll::Ready(self.start_chunk_run());
                    }
                }
            } else {
                false
            }
        };
        if pending_step {
            return Poll::Ready(Ok(()));
        }

        // Poll the build stream (it may already be exhausted).
        let polled = {
            let DriverPhase::ChunkedLoad { state, .. } = &mut self.phase else {
                return Poll::Ready(internal_err!("expected ChunkedLoad phase"));
            };
            match &mut state.build_stream {
                Some(stream) => ready!(stream.poll_next_unpin(cx)),
                None => None,
            }
        };
        match polled {
            Some(batch) => {
                let batch = batch?;
                let size = get_record_batch_memory_size(&batch);
                let DriverPhase::ChunkedLoad {
                    state,
                    batches,
                    num_rows,
                    reservation,
                    table_reserved,
                } = &mut self.phase
                else {
                    return Poll::Ready(internal_err!("expected ChunkedLoad phase"));
                };
                let table_delta = hash_table_estimate(*num_rows + batch.num_rows())?
                    .saturating_sub(*table_reserved);
                match reservation.try_grow(size + table_delta) {
                    Ok(()) => {
                        *num_rows += batch.num_rows();
                        *table_reserved += table_delta;
                        batches.push(batch);
                    }
                    Err(e) => {
                        if batches.is_empty() {
                            let pair_id = state.pair_id;
                            return Poll::Ready(Err(e.context(format!(
                                "spilled hash join partition {pair_id}: a single \
                                 build batch exceeds the memory budget"
                            ))));
                        }
                        state.pending = Some(batch);
                        return Poll::Ready(self.start_chunk_run());
                    }
                }
            }
            None => {
                {
                    let DriverPhase::ChunkedLoad { state, .. } = &mut self.phase else {
                        return Poll::Ready(internal_err!("expected ChunkedLoad phase"));
                    };
                    state.build_stream = None;
                }
                return Poll::Ready(self.start_chunk_run());
            }
        }
        Poll::Ready(Ok(()))
    }

    /// Transition ChunkedLoad -> ChunkedRun with the accumulated chunk.
    fn start_chunk_run(&mut self) -> Result<()> {
        let DriverPhase::ChunkedLoad {
            state,
            batches,
            num_rows,
            reservation,
            table_reserved,
        } = std::mem::replace(&mut self.phase, DriverPhase::NextPair)
        else {
            return internal_err!("expected ChunkedLoad phase");
        };
        if batches.is_empty() {
            // Final poll found no more rows: the pair is done.
            debug_assert!(state.exhausted());
            self.phase = DriverPhase::NextPair;
            return Ok(());
        }
        let inner = self.start_chunk_join(
            &state,
            batches,
            num_rows,
            reservation,
            table_reserved,
        )?;
        self.phase = DriverPhase::ChunkedRun { inner, state };
        Ok(())
    }

    fn poll_run_pair(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<RecordBatch>>> {
        let DriverPhase::RunPair { inner } = &mut self.phase else {
            return Poll::Ready(Some(internal_err!("expected RunPair phase")));
        };
        inner.poll_next_unpin(cx)
    }

    fn poll_chunked_run(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<RecordBatch>>> {
        let DriverPhase::ChunkedRun { inner, .. } = &mut self.phase else {
            return Poll::Ready(Some(internal_err!("expected ChunkedRun phase")));
        };
        inner.poll_next_unpin(cx)
    }

    /// A chunk's join finished: move to the next chunk or the next pair.
    fn finish_chunk(&mut self) -> Result<()> {
        let DriverPhase::ChunkedRun { state, .. } =
            std::mem::replace(&mut self.phase, DriverPhase::NextPair)
        else {
            return internal_err!("expected ChunkedRun phase");
        };
        if state.exhausted() {
            // Dropping the state deletes the probe partition's files.
            return Ok(());
        }
        let mut state = state;
        state.chunk_index += 1;
        let reservation = MemoryConsumer::new(format!(
            "HashJoinSpillChunk[{}.{}.{}]",
            self.ctx.partition, state.pair_id, state.chunk_index
        ))
        .with_can_spill(true)
        .register(&self.ctx.pool);
        self.phase = DriverPhase::ChunkedLoad {
            state,
            batches: Vec::new(),
            num_rows: 0,
            reservation,
            table_reserved: 0,
        };
        Ok(())
    }

    /// Build one chunk's `JoinLeftData` and join it against a full re-read
    /// of the pair's probe partition.
    fn start_chunk_join(
        &mut self,
        state: &ChunkedState,
        batches: Vec<RecordBatch>,
        num_rows: usize,
        reservation: MemoryReservation,
        table_reserved: usize,
    ) -> Result<SendableRecordBatchStream> {
        self.ctx.fallback_chunks.add(1);
        let _ = state.level; // recorded for diagnostics via consumer names
        self.build_inner_join(
            state.pair_id,
            batches,
            num_rows,
            reservation,
            table_reserved,
            state.probe_files.clone(),
        )
    }

    fn build_inner_join(
        &mut self,
        pair_id: usize,
        batches: Vec<RecordBatch>,
        num_rows: usize,
        reservation: MemoryReservation,
        table_reserved: usize,
        probe_files: Vec<RefCountedTempFile>,
    ) -> Result<SendableRecordBatchStream> {
        let inner_metrics = BuildProbeJoinMetrics::new(pair_id, &self.inner_metrics_set);
        let array_map_count = Count::new();
        let need_bitmap = need_produce_result_in_final(self.spec.join_type);

        let left_data = build_left_data(
            &batches,
            num_rows,
            self.ctx.build_spill.schema(),
            &self.ctx.on_left,
            &self.spec.random_state,
            reservation,
            &inner_metrics,
            need_bitmap,
            1,
            None,
            false,
            &self.ctx.config,
            self.spec.null_equality,
            &array_map_count,
            table_reserved,
            None,
        )?;
        drop(batches);

        let probe_stream =
            spilled_side_stream(probe_files, Arc::clone(&self.ctx.probe_spill));

        let build_output = BuildPhaseOutput::InMemory(Arc::new(left_data));
        let left_fut = OnceFut::new(std::future::ready(Ok(build_output)));

        let inner = HashJoinStream::new(
            self.spec.partition,
            Arc::clone(&self.spec.schema),
            self.spec.on_right.clone(),
            self.spec.filter.clone(),
            self.spec.join_type,
            probe_stream,
            self.spec.random_state.clone(),
            inner_metrics,
            self.spec.column_indices.clone(),
            self.spec.null_equality,
            HashJoinStreamState::WaitBuildSide,
            BuildSide::Initial(BuildSideInitialState { left_fut }),
            self.spec.batch_size,
            vec![],
            false,
            None,
            PartitionMode::Partitioned,
            false,
            None,
            None,
        );

        Ok(Box::pin(inner))
    }
}

enum SharedDriverPhase {
    /// Scatter this output partition's OWN probe stream into `K` partitions.
    ScatterProbe {
        probe: SendableRecordBatchStream,
        scatter: SideScatter,
    },
    /// Move to the next shared build partition.
    NextK,
    /// Run k's join: shared build (lazily built once across partitions)
    /// against this partition's probe files for k.
    RunK {
        inner: SendableRecordBatchStream,
    },
    Done,
}

/// Drives one output partition's share of a spilled `CollectLeft` join.
///
/// Unlike the Partitioned-mode driver, every k MUST run even when this
/// partition's probe side is empty: the shared `JoinLeftData` counts probe
/// completions, and skipping would leave the counter high so unmatched
/// build rows would never be emitted. Oversized shared partitions surface a
/// clean error (no recursion/chunking for the shared build in v1 — the
/// shared visited bitmap cannot be split across chunks).
pub(super) struct SharedSpillJoinDriver {
    ctx: Arc<HashJoinSpillContext>,
    spec: InnerJoinSpec,
    shared: Arc<SharedSpilledBuild>,
    phase: SharedDriverPhase,
    probe_parts: Vec<Option<SpilledSide>>,
    next_k: usize,
    /// Outer metrics: probe input rows/batches are recorded during scatter.
    outer_metrics: BuildProbeJoinMetrics,
    /// Private metrics for inner per-k streams.
    inner_metrics_set: ExecutionPlanMetricsSet,
}

impl SharedSpillJoinDriver {
    pub(super) fn new(
        shared: Arc<SharedSpilledBuild>,
        probe: SendableRecordBatchStream,
        ctx: Arc<HashJoinSpillContext>,
        spec: InnerJoinSpec,
        outer_metrics: BuildProbeJoinMetrics,
    ) -> Self {
        let scatter = SideScatter::new(
            shared.partition_count,
            spec.on_right.clone(),
            scatter_random_state(shared.level),
            &ctx.probe_spill,
            ctx.max_spill_file_size,
            ctx.headroom_bytes / 2,
            "hash_join_probe_spill",
        );
        Self {
            spec,
            phase: SharedDriverPhase::ScatterProbe { probe, scatter },
            probe_parts: Vec::new(),
            next_k: 0,
            outer_metrics,
            inner_metrics_set: ExecutionPlanMetricsSet::new(),
            ctx,
            shared,
        }
    }

    pub(super) fn poll_next(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<RecordBatch>>> {
        loop {
            match &self.phase {
                SharedDriverPhase::ScatterProbe { .. } => {
                    ready!(self.poll_scatter_probe(cx))?;
                }
                SharedDriverPhase::NextK => self.start_next_k()?,
                SharedDriverPhase::RunK { .. } => {
                    let polled = {
                        let SharedDriverPhase::RunK { inner } = &mut self.phase else {
                            return Poll::Ready(Some(internal_err!(
                                "expected RunK phase"
                            )));
                        };
                        ready!(inner.poll_next_unpin(cx))
                    };
                    match polled {
                        Some(item) => return Poll::Ready(Some(item)),
                        None => {
                            self.phase = SharedDriverPhase::NextK;
                        }
                    }
                }
                SharedDriverPhase::Done => return Poll::Ready(None),
            }
        }
    }

    fn poll_scatter_probe(&mut self, cx: &mut Context<'_>) -> Poll<Result<()>> {
        let polled = {
            let SharedDriverPhase::ScatterProbe { probe, .. } = &mut self.phase else {
                return Poll::Ready(internal_err!("expected ScatterProbe phase"));
            };
            ready!(probe.poll_next_unpin(cx))
        };
        match polled {
            Some(batch) => {
                let batch = batch?;
                self.outer_metrics.input_batches.add(1);
                self.outer_metrics.input_rows.add(batch.num_rows());
                let SharedDriverPhase::ScatterProbe { scatter, .. } = &mut self.phase
                else {
                    return Poll::Ready(internal_err!("expected ScatterProbe phase"));
                };
                scatter.scatter_batch(&batch)?;
            }
            None => {
                let SharedDriverPhase::ScatterProbe { scatter, .. } =
                    std::mem::replace(&mut self.phase, SharedDriverPhase::NextK)
                else {
                    return Poll::Ready(internal_err!("expected ScatterProbe phase"));
                };
                self.probe_parts = scatter.finish()?.into_iter().map(Some).collect();
            }
        }
        Poll::Ready(Ok(()))
    }

    fn start_next_k(&mut self) -> Result<()> {
        let k = self.next_k;
        if k >= self.shared.partition_count {
            self.phase = SharedDriverPhase::Done;
            return Ok(());
        }
        self.next_k += 1;

        let probe = self.probe_parts[k].take().ok_or_else(|| {
            DataFusionError::Internal(format!(
                "shared spill probe partition {k} already consumed"
            ))
        })?;

        let (bitmap, counter) = &self.shared.shared_state[k];
        let left_fut = OnceFut::new(shared_build_loader(
            Arc::clone(&self.ctx),
            Arc::clone(&self.shared.partitions[k].files),
            k,
            self.spec.random_state.clone(),
            self.spec.join_type,
            self.spec.null_equality,
            self.shared.probe_threads,
            (Arc::clone(bitmap), Arc::clone(counter)),
        ));

        let inner_metrics = BuildProbeJoinMetrics::new(k, &self.inner_metrics_set);
        let probe_stream =
            spilled_side_stream(probe.files, Arc::clone(&self.ctx.probe_spill));
        let inner = HashJoinStream::new(
            self.spec.partition,
            Arc::clone(&self.spec.schema),
            self.spec.on_right.clone(),
            self.spec.filter.clone(),
            self.spec.join_type,
            probe_stream,
            self.spec.random_state.clone(),
            inner_metrics,
            self.spec.column_indices.clone(),
            self.spec.null_equality,
            HashJoinStreamState::WaitBuildSide,
            BuildSide::Initial(BuildSideInitialState { left_fut }),
            self.spec.batch_size,
            vec![],
            false,
            None,
            PartitionMode::CollectLeft,
            false,
            None,
            None,
        );
        self.phase = SharedDriverPhase::RunK {
            inner: Box::pin(inner),
        };
        Ok(())
    }
}

/// Load shared build partition `k` into memory and construct its
/// `JoinLeftData` with `probe_threads_count = N` so the existing
/// last-finisher machinery handles shared unmatched-row emission.
#[expect(clippy::too_many_arguments)]
fn shared_build_loader(
    ctx: Arc<HashJoinSpillContext>,
    files: Arc<Vec<RefCountedTempFile>>,
    k: usize,
    random_state: RandomState,
    join_type: JoinType,
    null_equality: NullEquality,
    probe_threads: usize,
    shared_state: (Arc<SharedBitmapBuilder>, Arc<AtomicUsize>),
) -> BoxFuture<'static, Result<BuildPhaseOutput>> {
    Box::pin(async move {
        let reservation =
            MemoryConsumer::new(format!("HashJoinSpillShared[{}.{k}]", ctx.partition))
                .with_can_spill(true)
                .register(&ctx.pool);

        let mut stream =
            spilled_side_stream(files.as_ref().clone(), Arc::clone(&ctx.build_spill));
        let mut batches = Vec::new();
        let mut num_rows = 0usize;
        let mut table_reserved = 0usize;
        while let Some(batch) = stream.next().await {
            let batch = batch?;
            let size = get_record_batch_memory_size(&batch);
            let table_delta = hash_table_estimate(num_rows + batch.num_rows())?
                .saturating_sub(table_reserved);
            reservation.try_grow(size + table_delta).map_err(|e| {
                e.context(format!(
                    "shared spilled hash join build partition {k} does not fit \
                     in the memory budget (CollectLeft spill does not \
                     repartition recursively); increase the memory budget"
                ))
            })?;
            table_reserved += table_delta;
            num_rows += batch.num_rows();
            batches.push(batch);
        }

        let metrics_set = ExecutionPlanMetricsSet::new();
        let inner_metrics = BuildProbeJoinMetrics::new(k, &metrics_set);
        let array_map_count = Count::new();
        let need_bitmap = need_produce_result_in_final(join_type);
        let data = build_left_data(
            &batches,
            num_rows,
            ctx.build_spill.schema(),
            &ctx.on_left,
            &random_state,
            reservation,
            &inner_metrics,
            need_bitmap,
            probe_threads,
            None,
            false,
            &ctx.config,
            null_equality,
            &array_map_count,
            table_reserved,
            Some(shared_state),
        )?;
        Ok(BuildPhaseOutput::InMemory(Arc::new(data)))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::joins::HashJoinExec;
    use crate::joins::PartitionMode;
    use crate::metrics::SpillMetrics;
    use crate::repartition::REPARTITION_RANDOM_STATE;
    use crate::test::TestMemoryExec;
    use crate::{ExecutionPlan, common};

    use arrow::array::{Int32Array, RecordBatch};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::util::pretty::pretty_format_batches;
    use datafusion_common::JoinType;
    use datafusion_common::NullEquality;
    use datafusion_execution::TaskContext;
    use datafusion_execution::config::SessionConfig;
    use datafusion_execution::disk_manager::{DiskManagerBuilder, DiskManagerMode};
    use datafusion_execution::runtime_env::{RuntimeEnv, RuntimeEnvBuilder};
    use datafusion_physical_expr::expressions::Column;

    fn test_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int32, false),
            Field::new("v", DataType::Int32, false),
        ]))
    }

    fn make_batch(start: i32, len: usize) -> RecordBatch {
        let keys = Int32Array::from_iter_values(start..start + len as i32);
        let vals = Int32Array::from_iter_values((0..len as i32).map(|v| v * 10));
        RecordBatch::try_new(test_schema(), vec![Arc::new(keys), Arc::new(vals)]).unwrap()
    }

    fn key_exprs() -> Vec<PhysicalExprRef> {
        vec![Arc::new(Column::new("k", 0)) as _]
    }

    fn test_spill_manager(env: Arc<RuntimeEnv>) -> Arc<SpillManager> {
        Arc::new(SpillManager::new(
            env,
            SpillMetrics::new(&ExecutionPlanMetricsSet::new(), 0),
            test_schema(),
        ))
    }

    async fn read_all(
        side: SpilledSide,
        manager: &Arc<SpillManager>,
    ) -> Vec<RecordBatch> {
        let stream = spilled_side_stream(side.files, Arc::clone(manager));
        common::collect(stream).await.unwrap()
    }

    fn sorted_rows(batches: &[RecordBatch]) -> Vec<String> {
        let non_empty: Vec<RecordBatch> = batches
            .iter()
            .filter(|b| b.num_rows() > 0)
            .cloned()
            .collect();
        if non_empty.is_empty() {
            return vec![];
        }
        let formatted = pretty_format_batches(&non_empty).unwrap().to_string();
        let mut rows: Vec<String> = formatted
            .lines()
            .filter(|l| l.starts_with('|') && !l.contains(" k "))
            .map(|s| s.to_string())
            .collect();
        rows.sort();
        rows
    }

    /// Cheap multiset fingerprint of a result set: (row count, wrapping sum
    /// of row hashes). Equal row multisets yield equal fingerprints; the
    /// converse holds with overwhelming probability, which is plenty for
    /// comparing a spilled run against its in-memory reference.
    fn result_fingerprint(batches: &[RecordBatch]) -> (usize, u64) {
        let state = RandomState::with_seeds(1, 2, 3, 4);
        let mut count = 0usize;
        let mut sum = 0u64;
        for batch in batches.iter().filter(|b| b.num_rows() > 0) {
            let mut hashes = vec![0u64; batch.num_rows()];
            create_hashes(batch.columns(), &state, &mut hashes).unwrap();
            count += batch.num_rows();
            sum = hashes.iter().fold(sum, |acc, h| acc.wrapping_add(*h));
        }
        (count, sum)
    }

    #[tokio::test]
    async fn scatter_round_trip_preserves_all_rows() {
        let env = Arc::new(RuntimeEnv::default());
        let manager = test_spill_manager(Arc::clone(&env));
        let mut scatter = SideScatter::new(
            8,
            key_exprs(),
            scatter_random_state(0),
            &manager,
            128 * 1024 * 1024,
            1024 * 1024,
            "test_scatter",
        );

        let inputs: Vec<RecordBatch> =
            (0..10).map(|i| make_batch(i * 1000, 1000)).collect();
        for batch in &inputs {
            scatter.scatter_batch(batch).unwrap();
        }
        let sides = scatter.finish().unwrap();
        assert_eq!(sides.len(), 8);
        let total_rows: usize = sides.iter().map(|s| s.rows).sum();
        assert_eq!(total_rows, 10_000);

        let mut all = Vec::new();
        for side in sides {
            all.extend(read_all(side, &manager).await);
        }
        assert_eq!(sorted_rows(&all), sorted_rows(&inputs));
    }

    /// Rows that all land in ONE bucket of the repartition hash (the shape
    /// Partitioned-mode inputs actually have) must still spread across the
    /// scatter's disk partitions — i.e. the seeds are independent.
    #[tokio::test]
    async fn scatter_spreads_rows_prebucketed_by_repartition_hash() {
        const K: u64 = 8;

        // Select keys whose REPARTITION hash lands in bucket 0.
        let candidates = Int32Array::from_iter_values(0..200_000);
        let mut hashes = vec![0u64; candidates.len()];
        let candidate_arrays: Vec<arrow::array::ArrayRef> =
            vec![Arc::new(candidates.clone())];
        create_hashes(
            &candidate_arrays,
            REPARTITION_RANDOM_STATE.random_state(),
            &mut hashes,
        )
        .unwrap();
        let selected: Vec<i32> = hashes
            .iter()
            .enumerate()
            .filter(|(_, h)| *h % K == 0)
            .map(|(i, _)| candidates.value(i))
            .collect();
        assert!(selected.len() > 10_000, "want a meaningful sample");

        let keys = Int32Array::from(selected.clone());
        let vals = Int32Array::from(vec![0; selected.len()]);
        let batch =
            RecordBatch::try_new(test_schema(), vec![Arc::new(keys), Arc::new(vals)])
                .unwrap();

        let env = Arc::new(RuntimeEnv::default());
        let manager = test_spill_manager(Arc::clone(&env));
        let mut scatter = SideScatter::new(
            K as usize,
            key_exprs(),
            scatter_random_state(0),
            &manager,
            128 * 1024 * 1024,
            1024 * 1024,
            "test_seed_independence",
        );
        scatter.scatter_batch(&batch).unwrap();
        let sides = scatter.finish().unwrap();

        let non_empty = sides.iter().filter(|s| s.rows > 0).count();
        assert_eq!(non_empty, K as usize, "scatter must use every partition");
        let max = sides.iter().map(|s| s.rows).max().unwrap();
        let min = sides.iter().map(|s| s.rows).min().unwrap();
        assert!(
            max < min * 3,
            "scatter should be roughly uniform, got min={min} max={max}"
        );
    }

    #[tokio::test]
    async fn partition_writer_rotates_and_cleans_up_files() {
        let env = Arc::new(RuntimeEnv::default());
        let manager = test_spill_manager(Arc::clone(&env));
        // Tiny rotation threshold: every flushed buffer closes its file.
        let mut writer = PartitionSpillWriter::new(
            Arc::clone(&manager),
            "test_rotation".to_string(),
            1,
        );
        for i in 0..4 {
            writer.append(make_batch(i * 100_000, 100_000)).unwrap();
        }
        let side = writer.finish().unwrap();
        assert!(
            side.files.len() >= 2,
            "expected file rotation, got {} file(s)",
            side.files.len()
        );
        assert_eq!(side.rows, 400_000);

        let paths: Vec<std::path::PathBuf> =
            side.files.iter().map(|f| f.path().to_path_buf()).collect();
        for p in &paths {
            assert!(p.exists());
        }
        drop(side);
        for p in &paths {
            assert!(!p.exists(), "spill file must be deleted on drop: {p:?}");
        }
    }

    // ---- end-to-end HashJoinExec tests ----

    fn partitioned_join(
        build_rows: usize,
        join_type: JoinType,
    ) -> (Arc<dyn ExecutionPlan>, RecordBatch, RecordBatch) {
        // Feed the build side in ~100k-row batches so the overflow happens
        // mid-stream (buffered batches get re-scattered, the rest stream).
        let left_batches: Vec<RecordBatch> = (0..build_rows.div_ceil(100_000))
            .map(|i| {
                make_batch(i as i32 * 100_000, 100_000.min(build_rows - i * 100_000))
            })
            .collect();
        let left_batch = left_batches[0].clone();
        let right_batch = make_batch(0, 4096);
        let left =
            TestMemoryExec::try_new_exec(&[left_batches], left_batch.schema(), None)
                .unwrap();
        let right = TestMemoryExec::try_new_exec(
            &[vec![right_batch.clone()]],
            right_batch.schema(),
            None,
        )
        .unwrap();
        let on = vec![(
            Arc::new(Column::new_with_schema("k", &left_batch.schema()).unwrap()) as _,
            Arc::new(Column::new_with_schema("k", &right_batch.schema()).unwrap()) as _,
        )];
        let join = HashJoinExec::try_new(
            left,
            right,
            on,
            None,
            &join_type,
            None,
            PartitionMode::Partitioned,
            NullEquality::NullEqualsNothing,
            false,
        )
        .unwrap();
        (Arc::new(join), left_batch, right_batch)
    }

    fn spill_task_ctx(memory_limit: usize) -> Arc<TaskContext> {
        let runtime = RuntimeEnvBuilder::new()
            .with_memory_limit(memory_limit, 1.0)
            .build_arc()
            .unwrap();
        let mut session_config = SessionConfig::default().with_batch_size(4096);
        {
            let exec = &mut session_config.options_mut().execution;
            exec.enable_hash_join_spill = true;
            exec.hash_join_spill_headroom_bytes = 256 * 1024;
            exec.hash_join_spill_partition_count = 16;
        }
        Arc::new(
            TaskContext::default()
                .with_session_config(session_config)
                .with_runtime(runtime),
        )
    }

    #[tokio::test]
    async fn forced_spill_inner_join_matches_in_memory() {
        // ~2M rows * 8B ≈ 16MB build side vs an 8MB pool: must spill. With
        // K=16, each pair holds ~1MB of data plus its hash table — well
        // within the budget.
        let (join, _, _) = partitioned_join(2_000_000, JoinType::Inner);

        let spill_ctx = spill_task_ctx(8 * 1024 * 1024);
        let stream = join.execute(0, Arc::clone(&spill_ctx)).unwrap();
        let spilled_result = common::collect(stream).await.unwrap();

        let metrics = join.metrics().unwrap();
        assert_eq!(
            metrics
                .sum_by_name("join_spill_engaged")
                .map(|m| m.as_usize()),
            Some(1),
            "join must have engaged spilling"
        );
        assert!(
            metrics.spill_count().unwrap_or(0) > 0,
            "spill files must have been written"
        );

        // Reference: same join, no limit, feature off.
        let (reference_join, _, _) = partitioned_join(2_000_000, JoinType::Inner);
        let reference_ctx = Arc::new(TaskContext::default());
        let reference =
            common::collect(reference_join.execute(0, reference_ctx).unwrap())
                .await
                .unwrap();

        let spilled_rows: usize = spilled_result.iter().map(|b| b.num_rows()).sum();
        let reference_rows: usize = reference.iter().map(|b| b.num_rows()).sum();
        assert_eq!(spilled_rows, reference_rows);
        assert_eq!(sorted_rows(&spilled_result), sorted_rows(&reference));

        // All join memory must be released.
        assert_eq!(spill_ctx.memory_pool().reserved(), 0);
    }

    #[tokio::test]
    async fn fast_path_untouched_when_memory_sufficient() {
        let (join, _, _) = partitioned_join(100_000, JoinType::Inner);
        // Plenty of memory: flag on, but no spill may occur.
        let ctx = spill_task_ctx(512 * 1024 * 1024);
        let result = common::collect(join.execute(0, ctx).unwrap())
            .await
            .unwrap();

        let metrics = join.metrics().unwrap();
        assert_eq!(
            metrics
                .sum_by_name("join_spill_engaged")
                .map(|m| m.as_usize()),
            Some(0)
        );
        assert_eq!(metrics.spill_count(), Some(0));
        let rows: usize = result.iter().map(|b| b.num_rows()).sum();
        assert_eq!(rows, 4096);
    }

    #[tokio::test]
    async fn pool_released_when_stream_dropped_mid_spill_join() {
        let (join, _, _) = partitioned_join(2_000_000, JoinType::Inner);
        let ctx = spill_task_ctx(8 * 1024 * 1024);
        let mut stream = join.execute(0, Arc::clone(&ctx)).unwrap();

        // Pull a single batch, then drop the stream mid-flight.
        let first = stream.next().await;
        assert!(matches!(first, Some(Ok(_))), "expected at least one batch");
        drop(stream);

        assert_eq!(
            ctx.memory_pool().reserved(),
            0,
            "dropping the stream must release all join memory"
        );
    }

    #[tokio::test]
    async fn disk_cap_exhaustion_is_a_clean_error() {
        let (join, _, _) = partitioned_join(500_000, JoinType::Inner);

        let runtime = RuntimeEnvBuilder::new()
            .with_memory_limit(2 * 1024 * 1024, 1.0)
            .with_disk_manager_builder(
                DiskManagerBuilder::default()
                    .with_mode(DiskManagerMode::OsTmpDirectory)
                    .with_max_temp_directory_size(64 * 1024),
            )
            .build_arc()
            .unwrap();
        let mut session_config = SessionConfig::default().with_batch_size(4096);
        {
            let exec = &mut session_config.options_mut().execution;
            exec.enable_hash_join_spill = true;
            exec.hash_join_spill_headroom_bytes = 256 * 1024;
            exec.hash_join_spill_partition_count = 16;
        }
        let ctx = Arc::new(
            TaskContext::default()
                .with_session_config(session_config)
                .with_runtime(runtime),
        );

        let err = common::collect(join.execute(0, Arc::clone(&ctx)).unwrap())
            .await
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("disk") || err.contains("Resources exhausted"),
            "expected a clean disk/resources error, got: {err}"
        );
        assert_eq!(ctx.memory_pool().reserved(), 0);
    }

    #[tokio::test]
    async fn spill_disabled_keeps_todays_clean_error() {
        let (join, _, _) = partitioned_join(500_000, JoinType::Inner);
        let runtime = RuntimeEnvBuilder::new()
            .with_memory_limit(2 * 1024 * 1024, 1.0)
            .build_arc()
            .unwrap();
        let ctx = Arc::new(
            TaskContext::default()
                .with_session_config(SessionConfig::default().with_batch_size(4096))
                .with_runtime(runtime),
        );
        let err = common::collect(join.execute(0, ctx).unwrap())
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("Resources exhausted"), "got: {err}");
    }
    /// Build a Partitioned-mode join from explicit batch lists.
    fn join_from_batches(
        left_batches: Vec<RecordBatch>,
        right_batches: Vec<RecordBatch>,
        join_type: JoinType,
        null_equality: NullEquality,
    ) -> Arc<dyn ExecutionPlan> {
        let left_schema = left_batches[0].schema();
        let right_schema = right_batches[0].schema();
        let left =
            TestMemoryExec::try_new_exec(&[left_batches], Arc::clone(&left_schema), None)
                .unwrap();
        let right = TestMemoryExec::try_new_exec(
            &[right_batches],
            Arc::clone(&right_schema),
            None,
        )
        .unwrap();
        let on = vec![(
            Arc::new(Column::new_with_schema("k", &left_schema).unwrap()) as _,
            Arc::new(Column::new_with_schema("k", &right_schema).unwrap()) as _,
        )];
        Arc::new(
            HashJoinExec::try_new(
                left,
                right,
                on,
                None,
                &join_type,
                None,
                PartitionMode::Partitioned,
                null_equality,
                false,
            )
            .unwrap(),
        )
    }

    fn chunked_build_batches(rows: usize) -> Vec<RecordBatch> {
        (0..rows.div_ceil(100_000))
            .map(|i| make_batch(i as i32 * 100_000, 100_000.min(rows - i * 100_000)))
            .collect()
    }

    /// Every join type must produce identical results spilled vs in-memory.
    /// The probe range half-overlaps the build range so matches, unmatched
    /// build rows, and unmatched probe rows all occur.
    #[tokio::test]
    async fn forced_spill_matches_in_memory_for_all_join_types() {
        use JoinType::*;
        let join_types = [
            Inner, Left, Right, Full, LeftSemi, LeftAnti, RightSemi, RightAnti, LeftMark,
            RightMark,
        ];
        for join_type in join_types {
            let build = chunked_build_batches(1_000_000);
            // 8192 probe rows: first half matches build keys, second half
            // (>= 1M) has no build match.
            let probe = vec![make_batch(996_000, 8192)];

            let spilled_join = join_from_batches(
                build.clone(),
                probe.clone(),
                join_type,
                NullEquality::NullEqualsNothing,
            );
            let ctx = spill_task_ctx(4 * 1024 * 1024);
            let spilled =
                common::collect(spilled_join.execute(0, Arc::clone(&ctx)).unwrap())
                    .await
                    .unwrap_or_else(|e| panic!("{join_type:?} spilled run failed: {e}"));
            let metrics = spilled_join.metrics().unwrap();
            assert_eq!(
                metrics
                    .sum_by_name("join_spill_engaged")
                    .map(|m| m.as_usize()),
                Some(1),
                "{join_type:?} must have engaged spilling"
            );

            let reference_join = join_from_batches(
                build,
                probe,
                join_type,
                NullEquality::NullEqualsNothing,
            );
            let reference = common::collect(
                reference_join
                    .execute(0, Arc::new(TaskContext::default()))
                    .unwrap(),
            )
            .await
            .unwrap();

            assert_eq!(
                result_fingerprint(&spilled),
                result_fingerprint(&reference),
                "{join_type:?} results differ between spilled and in-memory"
            );
            assert_eq!(ctx.memory_pool().reserved(), 0, "{join_type:?} leaked");
        }
    }

    fn nullable_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int32, true),
            Field::new("v", DataType::Int32, false),
        ]))
    }

    /// Like `make_batch` but every 100th key is NULL. All NULL keys hash to
    /// one disk partition, and under NullEqualsNull they cross-join — the
    /// density keeps that product (and the partition skew, which is the
    /// chunked fallback's job) small.
    fn make_nullable_batch(start: i32, len: usize) -> RecordBatch {
        let keys = Int32Array::from_iter(
            (start..start + len as i32).map(|i| (i % 100 != 0).then_some(i)),
        );
        let vals = Int32Array::from_iter_values((0..len as i32).map(|v| v * 10));
        RecordBatch::try_new(nullable_schema(), vec![Arc::new(keys), Arc::new(vals)])
            .unwrap()
    }

    /// NULL join keys must behave identically spilled vs in-memory under
    /// both null-equality semantics (all NULLs scatter to one partition, so
    /// NullEqualsNull matching stays complete).
    #[tokio::test]
    async fn forced_spill_null_equality_matrix() {
        use JoinType::*;
        for join_type in [Inner, Left, Full] {
            for null_equality in [
                NullEquality::NullEqualsNothing,
                NullEquality::NullEqualsNull,
            ] {
                let build: Vec<RecordBatch> = (0..10)
                    .map(|i| make_nullable_batch(i * 100_000, 100_000))
                    .collect();
                // Half the probe keys match build keys, half exceed them.
                let probe = vec![make_nullable_batch(999_500, 1024)];

                let spilled_join = join_from_batches(
                    build.clone(),
                    probe.clone(),
                    join_type,
                    null_equality,
                );
                let ctx = spill_task_ctx(4 * 1024 * 1024);
                let spilled =
                    common::collect(spilled_join.execute(0, Arc::clone(&ctx)).unwrap())
                        .await
                        .unwrap_or_else(|e| {
                            panic!(
                                "{join_type:?}/{null_equality:?} spilled run failed: {e}"
                            )
                        });
                let metrics = spilled_join.metrics().unwrap();
                assert_eq!(
                    metrics
                        .sum_by_name("join_spill_engaged")
                        .map(|m| m.as_usize()),
                    Some(1),
                    "{join_type:?}/{null_equality:?} must have engaged spilling"
                );

                let reference_join =
                    join_from_batches(build, probe, join_type, null_equality);
                let reference = common::collect(
                    reference_join
                        .execute(0, Arc::new(TaskContext::default()))
                        .unwrap(),
                )
                .await
                .unwrap();

                assert_eq!(
                    result_fingerprint(&spilled),
                    result_fingerprint(&reference),
                    "{join_type:?}/{null_equality:?} spilled vs in-memory mismatch"
                );
            }
        }
    }
    /// Extreme duplicate-key skew: every build row has the same key, so
    /// repartitioning cannot shrink the partition and the chunked fallback
    /// must carry the join. Inner join: matches only.
    #[tokio::test]
    async fn extreme_skew_falls_back_to_chunked_inner() {
        // 600k rows, all key=42 (~4.8MB) vs a 4MB pool.
        let key = Int32Array::from(vec![42; 100_000]);
        let build: Vec<RecordBatch> = (0..6)
            .map(|i| {
                let vals = Int32Array::from_iter_values(i * 100_000..(i + 1) * 100_000);
                RecordBatch::try_new(
                    test_schema(),
                    vec![Arc::new(key.clone()), Arc::new(vals)],
                )
                .unwrap()
            })
            .collect();
        // 4 probe rows match key 42; 4 miss.
        let probe = vec![
            RecordBatch::try_new(
                test_schema(),
                vec![
                    Arc::new(Int32Array::from(vec![42, 42, 42, 42, 1, 2, 3, 4])),
                    Arc::new(Int32Array::from(vec![0, 1, 2, 3, 4, 5, 6, 7])),
                ],
            )
            .unwrap(),
        ];

        let spilled_join = join_from_batches(
            build.clone(),
            probe.clone(),
            JoinType::Inner,
            NullEquality::NullEqualsNothing,
        );
        let ctx = spill_task_ctx(4 * 1024 * 1024);
        let spilled = common::collect(spilled_join.execute(0, Arc::clone(&ctx)).unwrap())
            .await
            .unwrap();

        let metrics = spilled_join.metrics().unwrap();
        assert!(
            metrics
                .sum_by_name("join_spill_repartition_passes")
                .map(|m| m.as_usize())
                .unwrap_or(0)
                >= 1,
            "skewed pair should attempt at least one repartition pass"
        );
        assert!(
            metrics
                .sum_by_name("join_spill_fallback_chunks")
                .map(|m| m.as_usize())
                .unwrap_or(0)
                >= 2,
            "skewed pair must run multiple fallback chunks"
        );

        let reference_join = join_from_batches(
            build,
            probe,
            JoinType::Inner,
            NullEquality::NullEqualsNothing,
        );
        let reference = common::collect(
            reference_join
                .execute(0, Arc::new(TaskContext::default()))
                .unwrap(),
        )
        .await
        .unwrap();
        assert_eq!(result_fingerprint(&spilled), result_fingerprint(&reference));
        assert_eq!(ctx.memory_pool().reserved(), 0);
    }

    /// Chunked fallback with a build-side-emission type: LeftAnti emits
    /// every build row exactly once across all chunks.
    #[tokio::test]
    async fn extreme_skew_falls_back_to_chunked_left_anti() {
        let key = Int32Array::from(vec![42; 100_000]);
        let build: Vec<RecordBatch> = (0..6)
            .map(|i| {
                let vals = Int32Array::from_iter_values(i * 100_000..(i + 1) * 100_000);
                RecordBatch::try_new(
                    test_schema(),
                    vec![Arc::new(key.clone()), Arc::new(vals)],
                )
                .unwrap()
            })
            .collect();
        // No probe row matches key 42: all 600k build rows are anti-matches.
        let probe = vec![make_batch(100, 64)];

        let spilled_join = join_from_batches(
            build.clone(),
            probe.clone(),
            JoinType::LeftAnti,
            NullEquality::NullEqualsNothing,
        );
        let ctx = spill_task_ctx(4 * 1024 * 1024);
        let spilled = common::collect(spilled_join.execute(0, Arc::clone(&ctx)).unwrap())
            .await
            .unwrap();
        let spilled_rows: usize = spilled.iter().map(|b| b.num_rows()).sum();
        assert_eq!(spilled_rows, 600_000);

        let reference_join = join_from_batches(
            build,
            probe,
            JoinType::LeftAnti,
            NullEquality::NullEqualsNothing,
        );
        let reference = common::collect(
            reference_join
                .execute(0, Arc::new(TaskContext::default()))
                .unwrap(),
        )
        .await
        .unwrap();
        assert_eq!(result_fingerprint(&spilled), result_fingerprint(&reference));
    }

    /// Probe-side-emission types cannot use the chunked fallback: extreme
    /// skew must surface a descriptive clean error, not wrong results.
    #[tokio::test]
    async fn extreme_skew_unsupported_type_is_a_clean_error() {
        let key = Int32Array::from(vec![42; 100_000]);
        let build: Vec<RecordBatch> = (0..6)
            .map(|i| {
                let vals = Int32Array::from_iter_values(i * 100_000..(i + 1) * 100_000);
                RecordBatch::try_new(
                    test_schema(),
                    vec![Arc::new(key.clone()), Arc::new(vals)],
                )
                .unwrap()
            })
            .collect();
        let probe = vec![make_batch(0, 1024)];

        let join = join_from_batches(
            build,
            probe,
            JoinType::RightSemi,
            NullEquality::NullEqualsNothing,
        );
        let ctx = spill_task_ctx(4 * 1024 * 1024);
        let err = common::collect(join.execute(0, Arc::clone(&ctx)).unwrap())
            .await
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("extreme key skew") && err.contains("RightSemi"),
            "expected a descriptive skew error, got: {err}"
        );
        assert_eq!(ctx.memory_pool().reserved(), 0);
    }

    /// Moderate skew (too few initial partitions) is fixed by one recursion
    /// pass — no fallback chunks needed.
    #[tokio::test]
    async fn recursion_splits_oversized_uniform_partitions() {
        let build = chunked_build_batches(1_000_000);
        let probe = vec![make_batch(996_000, 8192)];

        let spilled_join = join_from_batches(
            build.clone(),
            probe.clone(),
            JoinType::Inner,
            NullEquality::NullEqualsNothing,
        );
        // Force only 2 initial partitions: each ~4MB against a 4MB pool, so
        // both need one repartition pass into 8 children each.
        let runtime = RuntimeEnvBuilder::new()
            .with_memory_limit(4 * 1024 * 1024, 1.0)
            .build_arc()
            .unwrap();
        let mut session_config = SessionConfig::default().with_batch_size(4096);
        {
            let exec = &mut session_config.options_mut().execution;
            exec.enable_hash_join_spill = true;
            exec.hash_join_spill_headroom_bytes = 256 * 1024;
            exec.hash_join_spill_partition_count = 2;
        }
        let ctx = Arc::new(
            TaskContext::default()
                .with_session_config(session_config)
                .with_runtime(runtime),
        );
        let spilled = common::collect(spilled_join.execute(0, Arc::clone(&ctx)).unwrap())
            .await
            .unwrap();

        let metrics = spilled_join.metrics().unwrap();
        assert!(
            metrics
                .sum_by_name("join_spill_repartition_passes")
                .map(|m| m.as_usize())
                .unwrap_or(0)
                >= 1,
            "expected at least one repartition pass"
        );
        assert_eq!(
            metrics
                .sum_by_name("join_spill_fallback_chunks")
                .map(|m| m.as_usize()),
            Some(0),
            "uniform keys must not need the chunked fallback"
        );

        let reference_join = join_from_batches(
            build,
            probe,
            JoinType::Inner,
            NullEquality::NullEqualsNothing,
        );
        let reference = common::collect(
            reference_join
                .execute(0, Arc::new(TaskContext::default()))
                .unwrap(),
        )
        .await
        .unwrap();
        assert_eq!(result_fingerprint(&spilled), result_fingerprint(&reference));
        assert_eq!(ctx.memory_pool().reserved(), 0);
    }
    /// Re-scattering one level's partition at the next level must spread it
    /// across all children. (Catches additive seed correlation: with weakly
    /// derived per-level seeds, `hash_l1 = hash_l0 + const`, so a parent
    /// partition maps into a single child under `% K` and recursion never
    /// shrinks anything.)
    #[tokio::test]
    async fn recursive_scatter_levels_are_decorrelated() {
        const K: u64 = 8;
        let candidates = Int32Array::from_iter_values(0..200_000);
        let candidate_arrays: Vec<arrow::array::ArrayRef> =
            vec![Arc::new(candidates.clone())];
        for parent_level in 0..3usize {
            let mut hashes = vec![0u64; candidates.len()];
            create_hashes(
                &candidate_arrays,
                &scatter_random_state(parent_level),
                &mut hashes,
            )
            .unwrap();
            // Keys in the parent scatter's bucket 0.
            let selected: Vec<i32> = hashes
                .iter()
                .enumerate()
                .filter(|(_, h)| *h % K == 0)
                .map(|(i, _)| candidates.value(i))
                .collect();
            let keys = Int32Array::from(selected.clone());
            let vals = Int32Array::from(vec![0; selected.len()]);
            let batch =
                RecordBatch::try_new(test_schema(), vec![Arc::new(keys), Arc::new(vals)])
                    .unwrap();

            let env = Arc::new(RuntimeEnv::default());
            let manager = test_spill_manager(Arc::clone(&env));
            let mut scatter = SideScatter::new(
                K as usize,
                key_exprs(),
                scatter_random_state(parent_level + 1),
                &manager,
                128 * 1024 * 1024,
                1024 * 1024,
                "test_level_decorrelation",
            );
            scatter.scatter_batch(&batch).unwrap();
            let sides = scatter.finish().unwrap();
            let non_empty = sides.iter().filter(|s| s.rows > 0).count();
            assert_eq!(
                non_empty,
                K as usize,
                "level {parent_level}->{}: child scatter must use every \
                 partition",
                parent_level + 1
            );
            let max = sides.iter().map(|s| s.rows).max().unwrap();
            let min = sides.iter().map(|s| s.rows).min().unwrap();
            assert!(
                max < min * 3,
                "level {parent_level}->{}: got min={min} max={max}",
                parent_level + 1
            );
        }
    }
    /// Build a CollectLeft-mode join: one shared build, two probe partitions.
    fn collect_left_join(
        left_batches: Vec<RecordBatch>,
        right_parts: Vec<Vec<RecordBatch>>,
        join_type: JoinType,
    ) -> Arc<dyn ExecutionPlan> {
        let left_schema = left_batches[0].schema();
        let right_schema = right_parts[0][0].schema();
        let left =
            TestMemoryExec::try_new_exec(&[left_batches], Arc::clone(&left_schema), None)
                .unwrap();
        let right =
            TestMemoryExec::try_new_exec(&right_parts, Arc::clone(&right_schema), None)
                .unwrap();
        let on = vec![(
            Arc::new(Column::new_with_schema("k", &left_schema).unwrap()) as _,
            Arc::new(Column::new_with_schema("k", &right_schema).unwrap()) as _,
        )];
        Arc::new(
            HashJoinExec::try_new(
                left,
                right,
                on,
                None,
                &join_type,
                None,
                PartitionMode::CollectLeft,
                NullEquality::NullEqualsNothing,
                false,
            )
            .unwrap(),
        )
    }

    /// CollectLeft spill: the shared scattered build must produce identical
    /// results to the in-memory join across all probe partitions — including
    /// exactly-once unmatched build emission (Left/Full/LeftAnti), which
    /// exercises the shared per-k bitmap and last-finisher accounting.
    #[tokio::test]
    async fn collect_left_forced_spill_matches_in_memory() {
        use JoinType::*;
        for join_type in [Inner, Left, LeftAnti, Full] {
            let build = chunked_build_batches(2_000_000);
            // Partition 0 half-overlaps the build key range; partition 1 is
            // fully matched.
            let right_parts =
                vec![vec![make_batch(1_996_000, 8192)], vec![make_batch(0, 4096)]];

            let spilled_join =
                collect_left_join(build.clone(), right_parts.clone(), join_type);
            // Two probe partitions walk the shared k's out of sync, so the
            // pool must admit two resident per-k tables at once: K=32 keeps
            // each table ~2.7MB against the 8MB pool.
            let runtime = RuntimeEnvBuilder::new()
                .with_memory_limit(8 * 1024 * 1024, 1.0)
                .build_arc()
                .unwrap();
            let mut session_config = SessionConfig::default().with_batch_size(4096);
            {
                let exec = &mut session_config.options_mut().execution;
                exec.enable_hash_join_spill = true;
                exec.hash_join_spill_headroom_bytes = 256 * 1024;
                exec.hash_join_spill_partition_count = 32;
            }
            let ctx = Arc::new(
                TaskContext::default()
                    .with_session_config(session_config)
                    .with_runtime(runtime),
            );
            // Poll both output partitions concurrently (as real plans do):
            // out-of-sync consumption is what the shared per-k accounting
            // must survive.
            let s0 = spilled_join.execute(0, Arc::clone(&ctx)).unwrap();
            let s1 = spilled_join.execute(1, Arc::clone(&ctx)).unwrap();
            let (r0, r1) = futures::join!(common::collect(s0), common::collect(s1));
            let mut spilled =
                r0.unwrap_or_else(|e| panic!("{join_type:?} partition 0 failed: {e}"));
            spilled.extend(
                r1.unwrap_or_else(|e| panic!("{join_type:?} partition 1 failed: {e}")),
            );

            let metrics = spilled_join.metrics().unwrap();
            assert_eq!(
                metrics
                    .sum_by_name("join_spill_engaged")
                    .map(|m| m.as_usize()),
                Some(1),
                "{join_type:?} must have engaged spilling"
            );

            let reference_join = collect_left_join(build, right_parts, join_type);
            let ref_ctx = Arc::new(TaskContext::default());
            let (f0, f1) = futures::join!(
                common::collect(reference_join.execute(0, Arc::clone(&ref_ctx)).unwrap()),
                common::collect(reference_join.execute(1, ref_ctx).unwrap())
            );
            let mut reference = f0.unwrap();
            reference.extend(f1.unwrap());

            assert_eq!(
                result_fingerprint(&spilled),
                result_fingerprint(&reference),
                "{join_type:?} CollectLeft spilled vs in-memory mismatch"
            );
            // The exec node's left_fut cache retains the shared build (and
            // its scatter headroom) until the plan drops — the same lifetime
            // the in-memory CollectLeft build has today.
            drop(spilled_join);
            assert_eq!(
                ctx.memory_pool().reserved(),
                0,
                "{join_type:?} leaked memory"
            );
        }
    }
}
