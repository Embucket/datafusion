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
use std::task::{Context, Poll};

use crate::hash_utils::create_hashes;
use crate::joins::PartitionMode;
use crate::joins::hash_join::exec::{BuildPhaseOutput, build_left_data};
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

use arrow::array::{Array, StringViewArray, UInt32Array};
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
use futures::{Stream, StreamExt, ready};

/// Per-partition write-buffer flush threshold: coalesce scattered slices into
/// IPC batches of roughly this size before writing.
const WRITE_BUFFER_FLUSH_BYTES: usize = 1024 * 1024;

/// Fallback per-partition target size when the memory pool cannot report a
/// finite limit.
const DEFAULT_PARTITION_TARGET_BYTES: usize = 128 * 1024 * 1024;

/// Hash seeds for scattering rows into disk partitions.
///
/// Must stay independent from `REPARTITION_RANDOM_STATE` (0,0,0,0) and
/// `HASH_JOIN_SEED` ('J','O','I','N'): rows arrive pre-bucketed by the former
/// and are hashed by the latter inside each pair's hash table. The seed
/// varies per recursion level so a repartition pass re-randomizes bucket
/// assignment.
pub(super) fn scatter_random_state(level: usize) -> RandomState {
    RandomState::with_seeds('S' as u64, 'P' as u64, 'L' as u64, 0xC0FFEE + level as u64)
}

/// Choose the number of disk partitions when spilling engages.
///
/// `buffered_bytes` is what fit in memory before the reservation failed; the
/// 4x factor assumes overflow happened early in the stream. Recursion
/// corrects underestimates, so precision is not important.
pub(super) fn compute_partition_count(
    buffered_bytes: usize,
    pool_limit: &MemoryLimit,
    override_count: usize,
) -> usize {
    if override_count > 0 {
        return override_count;
    }
    let target = match pool_limit {
        MemoryLimit::Finite(limit) => {
            (*limit / 16).clamp(32 * 1024 * 1024, 256 * 1024 * 1024)
        }
        _ => DEFAULT_PARTITION_TARGET_BYTES,
    };
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
    pub(super) partition_count_override: usize,
    pub(super) headroom_bytes: usize,
    #[expect(dead_code)] // used by recursive repartitioning (M4)
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
    #[expect(dead_code)] // used by recursion sizing decisions (M4)
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
        pair: usize,
        stream: SendableRecordBatchStream,
        batches: Vec<RecordBatch>,
        num_rows: usize,
        reservation: MemoryReservation,
    },
    /// Run the pair's in-memory join and forward its output.
    RunPair {
        inner: SendableRecordBatchStream,
    },
    Done,
}

/// Drives the spilled join: probe scatter, then a sequential loop over
/// partition pairs, each joined by an ordinary in-memory [`HashJoinStream`].
pub(super) struct SpillJoinDriver {
    ctx: Arc<HashJoinSpillContext>,
    spec: InnerJoinSpec,
    phase: DriverPhase,
    build_parts: Vec<Option<SpilledSide>>,
    probe_parts: Vec<Option<SpilledSide>>,
    next_pair: usize,
    level: usize,
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
            build_parts: partitions.into_iter().map(Some).collect(),
            probe_parts: Vec::new(),
            next_pair: 0,
            level,
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
                self.probe_parts = scatter.finish()?.into_iter().map(Some).collect();
            }
        }
        Poll::Ready(Ok(()))
    }

    /// Pick the next pair; skip pairs an inner join cannot produce rows for.
    fn select_next_pair(&mut self) -> Result<()> {
        loop {
            let pair = self.next_pair;
            if pair >= self.build_parts.len() {
                self.phase = DriverPhase::Done;
                return Ok(());
            }
            self.next_pair += 1;

            let (Some(build), Some(probe)) =
                (self.build_parts[pair].take(), self.probe_parts[pair].take())
            else {
                return internal_err!(
                    "spilled hash join partition {pair} already consumed"
                );
            };

            if self.can_skip_pair(build.rows, probe.rows) {
                // Dropping the sides deletes their spill files.
                continue;
            }

            let reservation = MemoryConsumer::new(format!(
                "HashJoinSpillPartition[{}.{}]",
                self.ctx.partition, pair
            ))
            .with_can_spill(true)
            .register(&self.ctx.pool);

            let stream =
                spilled_side_stream(build.files, Arc::clone(&self.ctx.build_spill));
            // Stash the probe side back; consumed when the pair's join starts.
            self.probe_parts[pair] = Some(probe);
            self.phase = DriverPhase::LoadBuild {
                pair,
                stream,
                batches: Vec::new(),
                num_rows: 0,
                reservation,
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
                let DriverPhase::LoadBuild {
                    pair,
                    batches,
                    num_rows,
                    reservation,
                    ..
                } = &mut self.phase
                else {
                    return Poll::Ready(internal_err!("expected LoadBuild phase"));
                };
                if let Err(e) = reservation.try_grow(size) {
                    // Recursive repartitioning arrives in M4; for now an
                    // oversized partition surfaces the clean
                    // resources-exhausted error.
                    return Poll::Ready(Err(oversize_partition_err(
                        e, *pair, self.level,
                    )));
                }
                *num_rows += batch.num_rows();
                batches.push(batch);
            }
            None => {
                let DriverPhase::LoadBuild {
                    pair,
                    batches,
                    num_rows,
                    reservation,
                    ..
                } = std::mem::replace(&mut self.phase, DriverPhase::NextPair)
                else {
                    return Poll::Ready(internal_err!("expected LoadBuild phase"));
                };
                let inner = self.start_pair_join(pair, batches, num_rows, reservation)?;
                self.phase = DriverPhase::RunPair { inner };
            }
        }
        Poll::Ready(Ok(()))
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

    /// Build the pair's `JoinLeftData` from the loaded batches and construct
    /// the inner in-memory join stream over the pair's probe partition.
    fn start_pair_join(
        &mut self,
        pair: usize,
        batches: Vec<RecordBatch>,
        num_rows: usize,
        reservation: MemoryReservation,
    ) -> Result<SendableRecordBatchStream> {
        let inner_metrics = BuildProbeJoinMetrics::new(pair, &self.inner_metrics_set);
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
        )?;
        drop(batches);

        let probe = self.probe_parts[pair].take().ok_or_else(|| {
            DataFusionError::Internal(format!(
                "probe partition {pair} missing for spilled hash join"
            ))
        })?;
        let probe_stream =
            spilled_side_stream(probe.files, Arc::clone(&self.ctx.probe_spill));

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

fn oversize_partition_err(
    source: DataFusionError,
    pair: usize,
    level: usize,
) -> DataFusionError {
    source.context(format!(
        "spilled hash join partition {pair} (repartition level {level}) does not fit \
         in the memory budget"
    ))
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
}
