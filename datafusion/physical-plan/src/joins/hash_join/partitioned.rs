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

//! Partitioned Hash Join implementation
//!
//! This module implements a partitioned hash join that can handle large datasets
//! by partitioning both build and probe sides into multiple partitions and
//! processing them sequentially. This approach is similar to sort-merge join
//! but uses hash-based partitioning instead of sorting.
//!
//! # State Machine Overview
//!
//! The partitioned hash join follows this state machine pattern:
//!
//! ```text
//! PartitionBuildSide → ProcessPartitions(i) → Done
//! ```
//!
//! ## PartitionBuildSide State
//! - Partitions build-side data into multiple partitions based on hash values
//! - Keeps one partition resident in memory (partition 0)
//! - Spills other partitions to disk when memory pressure occurs
//! - Uses consistent hashing to ensure same keys go to same partition
//!
//! ## ProcessPartitions State
//! - Processes each partition sequentially
//! - Loads build-side hash map for current partition (from memory or disk)
//! - Probes all probe batches for this partition against the hash map
//! - Generates join results and handles unmatched rows for outer joins
//! - Tracks matched rows for proper outer join semantics

use std::collections::VecDeque;
use std::mem::{self, size_of};
use std::sync::Arc;
use std::task::{Context, Poll};

#[cfg(feature = "hybrid_hash_join_scheduler")]
use super::scheduler::{HybridTaskScheduler, SchedulerConfig};
use crate::joins::hash_join::exec::JoinLeftData;
use crate::joins::join_hash_map::{JoinHashMapType, JoinHashMapU32, JoinHashMapU64};
use crate::joins::utils::{
    adjust_indices_by_join_type, apply_join_filter_to_indices, build_batch_from_indices,
    equal_rows_arr, get_final_indices_from_bit_map, need_produce_result_in_final,
    uint32_to_uint64_indices, BuildProbeJoinMetrics, ColumnIndex, JoinFilter, OnceFut,
    StatefulStreamResult,
};
use crate::metrics::SpillMetrics;
use crate::spill::in_progress_spill_file::InProgressSpillFile;
use crate::spill::spill_manager::SpillManager;
use crate::{RecordBatchStream, SendableRecordBatchStream};

use arrow::array::{Array, ArrayRef, BooleanBufferBuilder, UInt32Array, UInt64Array};
use arrow::compute::{concat_batches, take};
use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use datafusion_common::utils::memory::estimate_memory_size;
use datafusion_common::{
    hash_utils::create_hashes, internal_datafusion_err, internal_err, DataFusionError,
    JoinSide, JoinType, NullEquality, Result,
};
use datafusion_execution::disk_manager::RefCountedTempFile;
use datafusion_execution::memory_pool::{MemoryConsumer, MemoryReservation};
use datafusion_execution::runtime_env::RuntimeEnv;
use datafusion_physical_expr::PhysicalExprRef;

use ahash::RandomState;
use futures::{executor::block_on, ready, Stream, StreamExt};

const HYBRID_HASH_MAX_REPARTITION_DEPTH: usize = 6;
const HYBRID_HASH_MIN_FANOUT: usize = 2;
const HYBRID_HASH_MIN_PARTITION_BYTES: usize = 8 * 1024 * 1024;
const HYBRID_HASH_ROWS_PER_PARTITION_TARGET_MULTIPLIER: usize = 8;
const HYBRID_HASH_ROWS_PER_PARTITION_MIN: usize = 32 * 1024;

fn highest_power_of_two_leq(n: usize) -> usize {
    if n <= 1 {
        1
    } else {
        let mut power = 1usize;
        while (power << 1) <= n {
            power <<= 1;
        }
        power
    }
}

/// State of the partitioned hash join stream
#[derive(Debug, Clone)]
pub(super) enum PartitionedHashJoinState {
    /// Initial state - partitioning build side
    PartitionBuildSide,
    /// Processing a specific partition
    ProcessPartition(ProcessPartitionState),
    /// All partitions processed, handling unmatched rows for outer joins
    HandleUnmatchedRows,
    /// Join completed
    Completed,
}

/// State for processing a specific partition
#[derive(Debug, Clone)]
pub(super) struct ProcessPartitionState {
    /// Descriptor for the partition currently being processed
    descriptor: PartitionDescriptor,
}

/// Represents a partition of build-side data
pub(super) enum BuildPartition {
    /// Partition data in memory
    InMemory {
        /// Hash map for this partition
        hash_map: Box<dyn JoinHashMapType>,
        /// Build-side batch data
        batch: RecordBatch,
        /// Join key values
        values: Vec<ArrayRef>,
        /// Memory reservation for this partition
        reservation: MemoryReservation,
    },
    /// Partition data spilled to disk
    Spilled {
        /// Spill file containing the partition data (taken on load)
        spill_file: Option<RefCountedTempFile>,
        /// Memory reservation (released when spilled)
        reservation: MemoryReservation,
        /// Total bytes written for this spill partition
        spilled_bytes: usize,
        /// Total rows written for this spill partition
        spilled_rows: usize,
    },
    /// Partition resources released and not available
    Released {
        /// Placeholder reservation
        reservation: MemoryReservation,
    },
    /// Empty partition (no rows)
    Empty,
}

/// Represents a partition of probe-side data
#[derive(Debug)]
pub(super) struct ProbePartition {
    /// Batches in this partition
    pub batches: Vec<RecordBatch>,
    /// Join key values for each batch
    pub values: Vec<Vec<ArrayRef>>,
    /// Hash values for each batch
    pub hashes: Vec<Vec<u64>>,
}

impl ProbePartition {
    fn new() -> Self {
        Self {
            batches: Vec::new(),
            values: Vec::new(),
            hashes: Vec::new(),
        }
    }
}

enum PartitionBuildStatus {
    Ready(StatefulStreamResult<Option<RecordBatch>>),
    NeedMorePartitions { next_count: usize },
}

struct PartitionAccumulator {
    buffered_batches: Vec<RecordBatch>,
    buffered_bytes: usize,
    total_rows: usize,
    spill_writer: Option<InProgressSpillFile>,
    spilled_bytes: usize,
}

impl PartitionAccumulator {
    fn new() -> Self {
        Self {
            buffered_batches: Vec::new(),
            buffered_bytes: 0,
            total_rows: 0,
            spill_writer: None,
            spilled_bytes: 0,
        }
    }
}

impl Default for PartitionAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub(super) struct PartitionDescriptor {
    /// Index into build/probe storage vectors
    build_index: usize,
    /// Index of the original (generation 0) partition
    root_index: usize,
    /// Number of refinement passes applied so far
    generation: usize,
    /// Total number of radix bits used to identify this partition
    radix_bits: usize,
    /// Hash prefix (lower `radix_bits`) identifying this partition
    hash_prefix: u64,
    /// Latest spilled byte estimate for this partition
    spilled_bytes: usize,
    /// Latest spilled row estimate for this partition
    spilled_rows: usize,
}

// Use RefCountedTempFile from datafusion_execution::disk_manager

/// Partitioned Hash Join stream that can handle large datasets by partitioning
/// both build and probe sides and processing them sequentially.
pub(super) struct PartitionedHashJoinStream {
    // ========================================================================
    // PROPERTIES:
    // These fields are initialized at the start and remain constant throughout
    // the execution.
    // ========================================================================
    /// Partition identifier for debugging and determinism
    pub partition: usize,
    /// Output schema
    pub schema: SchemaRef,
    /// Join key columns from the right (probe side)
    pub on_right: Vec<PhysicalExprRef>,
    /// Join key columns from the left (build side)
    pub on_left: Vec<PhysicalExprRef>,
    /// Optional join filter
    pub filter: Option<JoinFilter>,
    /// Type of the join (left, right, semi, etc)
    pub join_type: JoinType,
    /// Right (probe) input stream
    pub right: SendableRecordBatchStream,
    /// Future that yields the collected build-side data
    pub left_fut: OnceFut<JoinLeftData>,
    /// Random state used for hashing initialization
    pub random_state: RandomState,
    /// Metrics
    pub join_metrics: BuildProbeJoinMetrics,
    /// Information of index and left / right placement of columns
    pub column_indices: Vec<ColumnIndex>,
    /// Defines the null equality for the join
    pub null_equality: NullEquality,
    /// Maximum output batch size
    pub batch_size: usize,
    /// Number of partitions to use
    pub num_partitions: usize,
    /// Maximum partition fanout allowed when recursively repartitioning
    pub max_partition_count: usize,
    /// Memory threshold for spilling (in bytes)
    pub memory_threshold: usize,

    // ========================================================================
    // STATE:
    // These fields track the execution state and are updated during execution.
    // ========================================================================
    /// Current state of the stream
    pub state: PartitionedHashJoinState,
    /// Build-side partitions
    pub build_partitions: Vec<BuildPartition>,
    /// Probe-side partitions
    pub probe_partitions: Vec<ProbePartition>,
    /// Current partition being processed
    pub current_partition: Option<usize>,
    /// Queue of pending partitions to process (supports recursive fan-out)
    pub pending_partitions: VecDeque<PartitionDescriptor>,
    /// Spill manager for probe-side (right) batches
    pub probe_spill_manager: SpillManager,
    /// Spill manager for build-side (left) batches
    pub build_spill_manager: SpillManager,
    /// Memory reservation for the entire operation
    pub memory_reservation: MemoryReservation,
    /// Tracks how many repartition passes have been attempted
    pub partition_pass: usize,
    /// Runtime environment
    pub runtime_env: Arc<RuntimeEnv>,
    /// Scratch space for computing hashes
    pub hashes_buffer: Vec<u64>,
    /// Whether the right side has an ordering to potentially preserve
    pub right_side_ordered: bool,
    /// Whether this stream has emitted a placeholder batch for downstream scheduling
    pub placeholder_emitted: bool,
    /// Running alignment start for right indices across probe batches (for semi/anti/mark)
    pub right_alignment_start: usize,
    /// Shared bounds accumulator for coordinating dynamic filter updates (optional)
    pub bounds_accumulator:
        Option<Arc<crate::joins::hash_join::shared_bounds::SharedBoundsAccumulator>>,
    /// Future used to synchronize dynamic filter updates across partitions
    pub bounds_waiter: Option<OnceFut<()>>,
    /// Cached build-side schema
    pub build_schema: SchemaRef,
    /// Cached probe-side schema
    pub probe_schema: SchemaRef,
    /// Current probe batch (filtered to the active partition), if any
    pub current_probe_batch: Option<RecordBatch>,
    /// Current probe values for ON expressions
    pub current_probe_values: Vec<ArrayRef>,
    /// Current probe hashes (filtered to the active partition)
    pub current_probe_hashes: Vec<u64>,
    /// Current lookup offset within the join hash map
    pub current_offset: crate::joins::join_hash_map::JoinHashMapOffset,
    /// Max joined probe-side index from current batch (for Right/Semi/Anti alignment)
    pub joined_probe_idx: Option<usize>,
    /// Bitmaps to track matched build-side rows for outer joins (one per partition)
    pub matched_build_rows_per_partition: Vec<BooleanBufferBuilder>,
    /// Current partition being processed for unmatched rows
    pub unmatched_partition: usize,
    /// Cached unmatched build/probe indices for current partition (chunked emission)
    pub unmatched_left_indices_cache: Option<UInt64Array>,
    pub unmatched_right_indices_cache: Option<UInt32Array>,
    pub unmatched_offset: usize,
    /// Whether the probe stream has reached EOF
    pub probe_stream_finished: bool,
    /// Current read position per partition within buffered probe batches
    pub probe_batch_positions: Vec<usize>,
    /// Metrics: total probe rows buffered per partition (RAM)
    pub probe_buffered_rows_per_part: Vec<usize>,
    /// Metrics: total probe rows spilled per partition (disk)
    pub probe_spilled_rows_per_part: Vec<usize>,
    /// Metrics: total probe rows consumed during probing per partition
    pub probe_consumed_rows_per_part: Vec<usize>,
    /// Metrics: total matches after equality per partition
    pub matched_rows_per_part: Vec<usize>,
    /// Metrics: total rows emitted per partition
    pub emitted_rows_per_part: Vec<usize>,
    /// Metrics: total candidate pairs before equality per partition
    pub candidate_pairs_per_part: Vec<usize>,
    /// One-time flag to run shadow verification per partition
    pub verify_once_per_part: Vec<bool>,
    /// One-time flag for filter debug logging per partition
    pub filter_debug_once_per_part: Vec<bool>,
    /// Pending async spill reload stream for build partitions
    pub pending_reload_stream: Option<SendableRecordBatchStream>,
    /// Accumulated batches for pending reload
    pub pending_reload_batches: Vec<RecordBatch>,
    /// Target partition id for pending reload
    pub pending_reload_partition: Option<usize>,
    /// In-progress probe spill writers, one per partition (used when corresponding build is spilled)
    pub probe_spill_in_progress: Vec<Option<InProgressSpillFile>>,
    /// Finalized probe spill files per partition (queue of ready-to-read files)
    pub probe_spill_files: Vec<VecDeque<RefCountedTempFile>>,
    /// Pending probe stream for the current partition's probe spill file
    pub pending_probe_stream: Option<SendableRecordBatchStream>,
    /// Target partition id for pending probe stream
    pub pending_probe_partition: Option<usize>,
    /// Whether a partition is currently queued for processing
    pub partition_pending: Vec<bool>,
    /// Latest descriptor metadata per partition
    pub partition_descriptors: Vec<Option<PartitionDescriptor>>,
}

impl PartitionedHashJoinStream {
    /// Compute partition id for a given hash using radix mask when possible
    #[inline]
    fn partition_for_hash(&self, hash: u64) -> usize {
        if self.num_partitions.is_power_of_two() {
            (hash as usize) & (self.num_partitions - 1)
        } else {
            // Fallback when num_partitions is not a power of two
            (hash as usize) % self.num_partitions
        }
    }

    fn resize_partition_vectors(&mut self) {
        let n = self.num_partitions;
        self.probe_spill_in_progress = (0..n).map(|_| None).collect();
        self.probe_spill_files = (0..n).map(|_| VecDeque::new()).collect();
        self.probe_batch_positions = vec![0; n];
        self.probe_buffered_rows_per_part = vec![0; n];
        self.probe_spilled_rows_per_part = vec![0; n];
        self.probe_consumed_rows_per_part = vec![0; n];
        self.matched_rows_per_part = vec![0; n];
        self.emitted_rows_per_part = vec![0; n];
        self.candidate_pairs_per_part = vec![0; n];
        self.verify_once_per_part = vec![false; n];
        self.filter_debug_once_per_part = vec![false; n];
        self.partition_pending = vec![false; n];
        self.partition_descriptors = (0..n).map(|_| None).collect();
    }

    fn allocate_partition_slot(&mut self) -> usize {
        let idx = self.build_partitions.len();
        self.build_partitions.push(BuildPartition::Empty);
        self.matched_build_rows_per_partition
            .push(BooleanBufferBuilder::new(0));
        self.probe_partitions.push(ProbePartition::new());
        self.probe_batch_positions.push(0);
        self.probe_spill_in_progress.push(None);
        self.probe_spill_files.push(VecDeque::new());
        self.probe_buffered_rows_per_part.push(0);
        self.probe_spilled_rows_per_part.push(0);
        self.probe_consumed_rows_per_part.push(0);
        self.matched_rows_per_part.push(0);
        self.emitted_rows_per_part.push(0);
        self.candidate_pairs_per_part.push(0);
        self.verify_once_per_part.push(false);
        self.filter_debug_once_per_part.push(false);
        self.partition_pending.push(false);
        self.partition_descriptors.push(None);
        idx
    }

    fn schedule_partition(&mut self, part_id: usize) -> Result<()> {
        if part_id >= self.partition_pending.len() {
            let new_len = part_id + 1;
            self.partition_pending.resize(new_len, false);
            self.partition_descriptors.resize_with(new_len, || None);
        }

        if self.current_partition == Some(part_id) {
            return Ok(());
        }

        if self.partition_pending[part_id] {
            return Ok(());
        }

        if let Some(desc) = self
            .partition_descriptors
            .get(part_id)
            .and_then(|d| d.clone())
        {
            self.pending_partitions.push_back(desc);
            self.partition_pending[part_id] = true;
        }

        Ok(())
    }

    fn flush_probe_writer(
        &mut self,
        part_id: usize,
    ) -> Result<Option<RefCountedTempFile>> {
        if part_id >= self.probe_spill_in_progress.len() {
            return Ok(None);
        }
        if let Some(mut writer) = self.probe_spill_in_progress[part_id].take() {
            let file = writer.finish()?;
            return Ok(file);
        }
        Ok(None)
    }

    fn finalize_spilled_partition(&mut self, part_id: usize) -> Result<bool> {
        if part_id >= self.probe_spill_in_progress.len() {
            return Ok(false);
        }
        if let Some(file) = self.flush_probe_writer(part_id)? {
            if part_id >= self.probe_spill_files.len() {
                self.probe_spill_files
                    .resize_with(part_id + 1, VecDeque::new);
            }
            self.probe_spill_files[part_id].push_back(file);
            self.schedule_partition(part_id)?;
            return Ok(true);
        }
        Ok(false)
    }

    fn compute_recursive_fanout(
        &self,
        descriptor: &PartitionDescriptor,
    ) -> Option<(usize, usize)> {
        if descriptor.generation >= HYBRID_HASH_MAX_REPARTITION_DEPTH {
            return None;
        }
        if self.max_partition_count == 0 {
            return None;
        }
        let current_total = self.build_partitions.len();
        if current_total == 0 {
            return None;
        }

        let max_fanout_allowed = self
            .max_partition_count
            .saturating_sub(current_total.saturating_sub(1));
        if max_fanout_allowed < HYBRID_HASH_MIN_FANOUT {
            return None;
        }

        let mut per_partition_budget = self
            .memory_threshold
            .checked_div(self.max_partition_count.max(1))
            .unwrap_or(self.memory_threshold);
        if per_partition_budget == 0 {
            per_partition_budget = HYBRID_HASH_MIN_PARTITION_BYTES;
        }
        per_partition_budget = per_partition_budget.max(HYBRID_HASH_MIN_PARTITION_BYTES);

        let rows_budget = self
            .batch_size
            .saturating_mul(HYBRID_HASH_ROWS_PER_PARTITION_TARGET_MULTIPLIER)
            .max(HYBRID_HASH_ROWS_PER_PARTITION_MIN);

        let should_repartition_bytes = descriptor.spilled_bytes > per_partition_budget;
        let should_repartition_rows = descriptor.spilled_rows > rows_budget;

        if !should_repartition_bytes && !should_repartition_rows {
            return None;
        }

        let mut required = HYBRID_HASH_MIN_FANOUT;

        if should_repartition_bytes {
            let budget = per_partition_budget.max(1);
            let needed = descriptor.spilled_bytes.saturating_add(budget - 1) / budget;
            required = required.max(needed);
        }

        if should_repartition_rows {
            let budget = rows_budget.max(1);
            let needed = descriptor.spilled_rows.saturating_add(budget - 1) / budget;
            required = required.max(needed);
        }

        let mut fanout = required.next_power_of_two();
        if fanout == 0 {
            fanout = HYBRID_HASH_MIN_FANOUT;
        }
        if fanout > max_fanout_allowed {
            fanout = highest_power_of_two_leq(max_fanout_allowed);
        }
        if fanout < HYBRID_HASH_MIN_FANOUT {
            return None;
        }

        let additional_bits = fanout.trailing_zeros() as usize;
        if additional_bits == 0 {
            return None;
        }
        Some((additional_bits, fanout))
    }

    fn repartition_spilled_partition(
        &mut self,
        descriptor: &PartitionDescriptor,
        additional_bits: usize,
        fanout: usize,
    ) -> Result<Vec<PartitionDescriptor>> {
        let build_index = descriptor.build_index;
        if build_index >= self.build_partitions.len() {
            return Ok(vec![]);
        }

        let placeholder_reservation =
            MemoryConsumer::new("partition_repartition_placeholder")
                .with_can_spill(true)
                .register(&self.runtime_env.memory_pool);

        let old_partition = mem::replace(
            &mut self.build_partitions[build_index],
            BuildPartition::Released {
                reservation: placeholder_reservation,
            },
        );

        let (spill_file, _spilled_bytes, _spilled_rows) = match old_partition {
            BuildPartition::Spilled {
                spill_file,
                spilled_bytes,
                spilled_rows,
                ..
            } => (
                spill_file.ok_or_else(|| {
                    internal_datafusion_err!(
                        "spill file already consumed for partition {}",
                        build_index
                    )
                })?,
                spilled_bytes,
                spilled_rows,
            ),
            other => {
                self.build_partitions[build_index] = other;
                return Ok(vec![]);
            }
        };

        // Collect spilled build batches
        let mut build_batches = block_on(async {
            let mut stream = self.build_spill_manager.read_spill_as_stream(spill_file)?;
            let mut batches = Vec::new();
            while let Some(batch) = stream.next().await {
                batches.push(batch?);
            }
            Result::<Vec<RecordBatch>>::Ok(batches)
        })?;

        if build_batches.is_empty() {
            // Nothing to repartition; keep placeholder as empty partition
            let mut new_descriptor = descriptor.clone();
            new_descriptor.spilled_bytes = 0;
            new_descriptor.spilled_rows = 0;
            self.matched_build_rows_per_partition[build_index] =
                BooleanBufferBuilder::new(0);
            self.build_partitions[build_index] = BuildPartition::Empty;
            return Ok(vec![new_descriptor]);
        }

        let shift_bits = descriptor.radix_bits;
        let mask = (fanout - 1) as u64;
        let mut sub_accumulators = (0..fanout)
            .map(|_| PartitionAccumulator::new())
            .collect::<Vec<_>>();

        self.join_metrics.recursive_repartition_events.add(1);
        self.join_metrics.recursive_partitions_created.add(fanout);
        self.join_metrics
            .recursive_partition_depth
            .set_max(descriptor.generation.saturating_add(1));
        self.join_metrics
            .recursive_repartition_fanout
            .set_max(fanout);

        for batch in build_batches.drain(..) {
            let mut keys_values: Vec<ArrayRef> = Vec::with_capacity(self.on_left.len());
            for expr in &self.on_left {
                keys_values.push(expr.evaluate(&batch)?.into_array(batch.num_rows())?);
            }
            let mut hashes = vec![0u64; batch.num_rows()];
            create_hashes(&keys_values, &self.random_state, &mut hashes)?;

            let mut indices_per_part: Vec<Vec<u32>> = vec![Vec::new(); fanout];
            for (row_idx, hash) in hashes.iter().enumerate() {
                let sub_idx = (((*hash >> shift_bits) as usize) & mask as usize) % fanout;
                indices_per_part[sub_idx].push(row_idx as u32);
            }

            for (sub_idx, indices) in indices_per_part.into_iter().enumerate() {
                if indices.is_empty() {
                    continue;
                }
                let idx_array = UInt32Array::from(indices);
                let mut filtered_columns: Vec<ArrayRef> =
                    Vec::with_capacity(batch.num_columns());
                for col in batch.columns() {
                    filtered_columns.push(
                        take(col, &idx_array, None).map_err(DataFusionError::from)?,
                    );
                }
                let filtered_batch =
                    RecordBatch::try_new(batch.schema(), filtered_columns)
                        .map_err(DataFusionError::from)?;
                let batch_size = filtered_batch.get_array_memory_size();

                let accum = &mut sub_accumulators[sub_idx];
                accum.total_rows += filtered_batch.num_rows();

                match self.memory_reservation.try_grow(batch_size) {
                    Ok(_) => {
                        accum.buffered_bytes += batch_size;
                        accum.buffered_batches.push(filtered_batch);
                        self.join_metrics
                            .build_mem_used
                            .set_max(self.memory_reservation.size());
                        if self.memory_reservation.size() > self.memory_threshold {
                            self.spill_partition(sub_idx, accum)?;
                        }
                    }
                    Err(_) => {
                        self.spill_partition(sub_idx, accum)?;
                        self.append_spilled_batch(accum, filtered_batch)?;
                    }
                }
            }
        }

        // Finalize sub partitions
        let new_radix_bits = descriptor.radix_bits + additional_bits;
        let mut new_descriptors = Vec::with_capacity(fanout);
        let mut partition_indices = Vec::with_capacity(fanout);

        for sub_idx in 0..fanout {
            let accum = &mut sub_accumulators[sub_idx];
            let mut matched_bitmap = BooleanBufferBuilder::new(accum.total_rows);
            matched_bitmap.append_n(accum.total_rows, false);

            let new_index = if sub_idx == 0 {
                build_index
            } else {
                self.allocate_partition_slot()
            };
            partition_indices.push(new_index);

            self.matched_build_rows_per_partition[new_index] = matched_bitmap;

            if accum.spill_writer.is_some() || !accum.buffered_batches.is_empty() {
                if accum.spill_writer.is_some() {
                    if !accum.buffered_batches.is_empty() {
                        self.spill_partition(sub_idx, accum)?;
                    }
                    let mut writer = accum.spill_writer.take().ok_or_else(|| {
                        internal_datafusion_err!("missing spill writer")
                    })?;
                    let spill_file = writer.finish()?.ok_or_else(|| {
                        internal_datafusion_err!("expected spill file after repartition")
                    })?;
                    let reservation = MemoryConsumer::new("partition_spilled")
                        .with_can_spill(true)
                        .register(&self.runtime_env.memory_pool);
                    self.build_partitions[new_index] = BuildPartition::Spilled {
                        spill_file: Some(spill_file),
                        reservation,
                        spilled_bytes: accum.spilled_bytes,
                        spilled_rows: accum.total_rows,
                    };
                } else {
                    let mut buffered_batches = mem::take(&mut accum.buffered_batches);
                    let partition_batch = if buffered_batches.len() == 1 {
                        buffered_batches.pop().unwrap()
                    } else {
                        let batch_refs: Vec<_> = buffered_batches.iter().collect();
                        concat_batches(&self.build_schema, batch_refs)?
                    };
                    let num_rows = partition_batch.num_rows();
                    let partition_values = self
                        .on_left
                        .iter()
                        .map(|expr| expr.evaluate(&partition_batch)?.into_array(num_rows))
                        .collect::<Result<Vec<_>>>()?;

                    let fixed_size_u32 = size_of::<JoinHashMapU32>();
                    let fixed_size_u64 = size_of::<JoinHashMapU64>();
                    let mut hash_map: Box<dyn JoinHashMapType> = if num_rows
                        > u32::MAX as usize
                    {
                        let estimated_hashtable_size =
                            estimate_memory_size::<(u64, u64)>(num_rows, fixed_size_u64)?;
                        self.memory_reservation.try_grow(estimated_hashtable_size)?;
                        self.join_metrics
                            .build_mem_used
                            .set_max(self.memory_reservation.size());
                        Box::new(JoinHashMapU64::with_capacity(num_rows))
                    } else {
                        let estimated_hashtable_size =
                            estimate_memory_size::<(u32, u64)>(num_rows, fixed_size_u32)?;
                        self.memory_reservation.try_grow(estimated_hashtable_size)?;
                        self.join_metrics
                            .build_mem_used
                            .set_max(self.memory_reservation.size());
                        Box::new(JoinHashMapU32::with_capacity(num_rows))
                    };

                    self.hashes_buffer.clear();
                    self.hashes_buffer.resize(num_rows, 0);
                    create_hashes(
                        &partition_values,
                        &self.random_state,
                        &mut self.hashes_buffer,
                    )?;
                    hash_map.extend_zero(num_rows);
                    let iter = self
                        .hashes_buffer
                        .iter()
                        .enumerate()
                        .map(|(idx, hash)| (idx, hash));
                    hash_map.update_from_iter(Box::new(iter), 0);

                    let reservation = MemoryConsumer::new("partition_memory")
                        .with_can_spill(true)
                        .register(&self.runtime_env.memory_pool);

                    self.build_partitions[new_index] = BuildPartition::InMemory {
                        hash_map,
                        batch: partition_batch,
                        values: partition_values,
                        reservation,
                    };
                    accum.spilled_bytes = 0;
                }
            } else {
                self.build_partitions[new_index] = BuildPartition::Empty;
            }

            let hash_prefix =
                (descriptor.hash_prefix << additional_bits) | (sub_idx as u64);
            new_descriptors.push(PartitionDescriptor {
                build_index: new_index,
                root_index: descriptor.root_index,
                generation: descriptor.generation + 1,
                radix_bits: new_radix_bits,
                hash_prefix,
                spilled_bytes: accum.spilled_bytes,
                spilled_rows: accum.total_rows,
            });
        }

        self.repartition_probe_partition(descriptor, fanout, &partition_indices)?;

        Ok(new_descriptors)
    }

    fn repartition_probe_partition(
        &mut self,
        descriptor: &PartitionDescriptor,
        fanout: usize,
        partition_indices: &[usize],
    ) -> Result<()> {
        let parent_index = descriptor.build_index;
        if parent_index >= self.probe_partitions.len() {
            return Ok(());
        }

        // Reset parent metrics
        self.probe_buffered_rows_per_part
            .get_mut(parent_index)
            .map(|v| *v = 0);
        self.probe_spilled_rows_per_part
            .get_mut(parent_index)
            .map(|v| *v = 0);
        self.probe_consumed_rows_per_part
            .get_mut(parent_index)
            .map(|v| *v = 0);
        if parent_index < self.probe_batch_positions.len() {
            self.probe_batch_positions[parent_index] = 0;
        }
        if parent_index < self.probe_spill_in_progress.len() {
            self.probe_spill_in_progress[parent_index] = None;
        }

        let shift_bits = descriptor.radix_bits;
        let mask = (fanout - 1) as u64;

        if let Some(file) = self
            .probe_spill_files
            .get_mut(parent_index)
            .and_then(|queue| queue.pop_front())
        {
            let mut writers = Vec::with_capacity(fanout);
            for _ in 0..fanout {
                let writer = self
                    .probe_spill_manager
                    .create_in_progress_file("hash_join_probe_repartition")?;
                writers.push(writer);
            }

            let mut file_opt = Some(file);
            block_on(async {
                let mut stream = self
                    .probe_spill_manager
                    .read_spill_as_stream(file_opt.take().unwrap())?;
                while let Some(batch) = stream.next().await {
                    let batch = batch?;
                    let mut key_arrays: Vec<ArrayRef> =
                        Vec::with_capacity(self.on_right.len());
                    for expr in &self.on_right {
                        key_arrays
                            .push(expr.evaluate(&batch)?.into_array(batch.num_rows())?);
                    }
                    let mut hashes = vec![0u64; batch.num_rows()];
                    create_hashes(&key_arrays, &self.random_state, &mut hashes)?;

                    let mut indices_per_part: Vec<Vec<u32>> = vec![Vec::new(); fanout];
                    for (row_idx, hash) in hashes.iter().enumerate() {
                        let sub_idx =
                            (((*hash >> shift_bits) as usize) & mask as usize) % fanout;
                        indices_per_part[sub_idx].push(row_idx as u32);
                    }

                    for (sub_idx, indices) in indices_per_part.into_iter().enumerate() {
                        if indices.is_empty() {
                            continue;
                        }
                        let indices_arr = UInt32Array::from(indices);
                        let mut filtered_columns: Vec<ArrayRef> =
                            Vec::with_capacity(batch.num_columns());
                        for col in batch.columns() {
                            filtered_columns.push(
                                take(col, &indices_arr, None)
                                    .map_err(DataFusionError::from)?,
                            );
                        }
                        let filtered_batch =
                            RecordBatch::try_new(batch.schema(), filtered_columns)
                                .map_err(DataFusionError::from)?;
                        let writer = writers
                            .get_mut(sub_idx)
                            .ok_or_else(|| internal_datafusion_err!("missing writer"))?;
                        writer.append_batch(&filtered_batch)?;
                        self.join_metrics
                            .probe_spilled_rows
                            .add(filtered_batch.num_rows());
                        self.join_metrics
                            .probe_spilled_bytes
                            .add(filtered_batch.get_array_memory_size());
                    }
                }
                Result::<()>::Ok(())
            })?;

            for (sub_idx, mut writer) in writers.into_iter().enumerate() {
                let file = writer.finish()?.ok_or_else(|| {
                    internal_datafusion_err!("expected probe spill file")
                })?;
                let partitions_idx = partition_indices[sub_idx];
                self.probe_spill_files[partitions_idx].push_back(file);
                self.probe_spilled_rows_per_part[partitions_idx] = 0;
                self.probe_buffered_rows_per_part[partitions_idx] = 0;
                self.probe_consumed_rows_per_part[partitions_idx] = 0;
            }
            return Ok(());
        }

        // In-memory probe data
        let parent_partition = mem::replace(
            &mut self.probe_partitions[parent_index],
            ProbePartition::new(),
        );
        for idx in 0..parent_partition.batches.len() {
            let batch = &parent_partition.batches[idx];
            let values = &parent_partition.values[idx];
            let hashes = &parent_partition.hashes[idx];
            let mut indices_per_part: Vec<Vec<u32>> = vec![Vec::new(); fanout];
            for (row_idx, hash) in hashes.iter().enumerate() {
                let sub_idx = (((*hash >> shift_bits) as usize) & mask as usize) % fanout;
                indices_per_part[sub_idx].push(row_idx as u32);
            }

            for (sub_idx, indices) in indices_per_part.into_iter().enumerate() {
                if indices.is_empty() {
                    continue;
                }
                let indices_arr = UInt32Array::from(indices);
                let mut filtered_columns: Vec<ArrayRef> =
                    Vec::with_capacity(batch.num_columns());
                for col in batch.columns() {
                    filtered_columns.push(
                        take(col, &indices_arr, None).map_err(DataFusionError::from)?,
                    );
                }
                let filtered_batch =
                    RecordBatch::try_new(batch.schema(), filtered_columns)
                        .map_err(DataFusionError::from)?;

                let mut filtered_values: Vec<ArrayRef> = Vec::with_capacity(values.len());
                for arr in values.iter() {
                    filtered_values.push(
                        take(arr, &indices_arr, None).map_err(DataFusionError::from)?,
                    );
                }

                let mut filtered_hashes: Vec<u64> = Vec::with_capacity(indices_arr.len());
                for i in indices_arr.values().iter() {
                    filtered_hashes.push(hashes[*i as usize]);
                }

                let idx = partition_indices[sub_idx];
                let part = self
                    .probe_partitions
                    .get_mut(idx)
                    .ok_or_else(|| internal_datafusion_err!("missing probe partition"))?;
                part.batches.push(filtered_batch);
                part.values.push(filtered_values);
                part.hashes.push(filtered_hashes);
                let buffered = part
                    .batches
                    .last()
                    .map(|b| b.num_rows())
                    .unwrap_or_default();
                self.probe_buffered_rows_per_part[idx] =
                    self.probe_buffered_rows_per_part[idx].saturating_add(buffered);
            }
        }

        Ok(())
    }

    fn buffer_probe_side(&mut self, cx: &mut Context<'_>) -> Poll<Result<()>> {
        if self.probe_spill_in_progress.len() != self.num_partitions
            || self.probe_spill_files.len() != self.num_partitions
        {
            self.resize_partition_vectors();
        }
        if self.probe_partitions.len() != self.num_partitions {
            self.probe_partitions.clear();
        }
        if self.probe_partitions.is_empty() {
            self.probe_partitions = (0..self.num_partitions)
                .map(|_| ProbePartition::new())
                .collect();
        }
        loop {
            match self.right.poll_next_unpin(cx) {
                Poll::Ready(Some(Ok(batch))) => {
                    let mut keys_values: Vec<ArrayRef> =
                        Vec::with_capacity(self.on_right.len());
                    for c in &self.on_right {
                        let v = c.evaluate(&batch)?.into_array(batch.num_rows())?;
                        keys_values.push(v);
                    }
                    let mut hashes = vec![0u64; batch.num_rows()];
                    create_hashes(&keys_values, &self.random_state, &mut hashes)?;

                    let mut indices_per_part: Vec<Vec<u32>> =
                        vec![Vec::new(); self.num_partitions];
                    for (row_idx, &hash) in hashes.iter().enumerate() {
                        let pid = self.partition_for_hash(hash) as usize;
                        indices_per_part[pid].push(row_idx as u32);
                    }

                    for part_id in 0..self.num_partitions {
                        let part_indices = &indices_per_part[part_id];
                        if part_indices.is_empty() {
                            continue;
                        }

                        let indices_arr: UInt32Array = part_indices.clone().into();

                        let mut filtered_columns: Vec<ArrayRef> =
                            Vec::with_capacity(batch.num_columns());
                        for col in batch.columns() {
                            filtered_columns.push(
                                take(col, &indices_arr, None)
                                    .map_err(DataFusionError::from)?,
                            );
                        }
                        let filtered_batch =
                            RecordBatch::try_new(batch.schema(), filtered_columns)
                                .map_err(DataFusionError::from)?;

                        let mut filtered_on_values: Vec<ArrayRef> =
                            Vec::with_capacity(self.on_right.len());
                        for arr in &keys_values {
                            filtered_on_values.push(
                                take(arr, &indices_arr, None)
                                    .map_err(DataFusionError::from)?,
                            );
                        }

                        let mut filtered_hashes: Vec<u64> =
                            Vec::with_capacity(part_indices.len());
                        for &i in part_indices.iter() {
                            filtered_hashes.push(hashes[i as usize]);
                        }

                        match self.build_partitions.get_mut(part_id) {
                            Some(BuildPartition::Spilled { .. }) => {
                                if self.probe_spill_in_progress[part_id].is_none() {
                                    let ipf = self
                                        .probe_spill_manager
                                        .create_in_progress_file(
                                            "hash_join_probe_partition",
                                        )?;
                                    self.probe_spill_in_progress[part_id] = Some(ipf);
                                    self.join_metrics.probe_spill_count.add(1);
                                }
                                if let Some(ref mut ipf) =
                                    self.probe_spill_in_progress[part_id]
                                {
                                    ipf.append_batch(&filtered_batch)?;
                                    self.join_metrics
                                        .probe_spilled_rows
                                        .add(filtered_batch.num_rows());
                                    self.join_metrics
                                        .probe_spilled_bytes
                                        .add(filtered_batch.get_array_memory_size());
                                }
                                self.probe_spilled_rows_per_part[part_id] +=
                                    filtered_batch.num_rows();
                                let queue_ready = self
                                    .probe_spill_files
                                    .get(part_id)
                                    .map(|q| !q.is_empty())
                                    .unwrap_or(false);
                                let stream_active = self
                                    .pending_probe_partition
                                    .is_some_and(|p| p == part_id);
                                if !queue_ready && !stream_active {
                                    self.finalize_spilled_partition(part_id)?;
                                }
                            }
                            _ => {
                                self.probe_partitions[part_id]
                                    .batches
                                    .push(filtered_batch);
                                self.probe_partitions[part_id]
                                    .values
                                    .push(filtered_on_values);
                                self.probe_partitions[part_id]
                                    .hashes
                                    .push(filtered_hashes);
                                let last = self.probe_partitions[part_id]
                                    .batches
                                    .last()
                                    .unwrap();
                                self.probe_buffered_rows_per_part[part_id] +=
                                    last.num_rows();
                            }
                        }
                    }

                    return Poll::Ready(Ok(()));
                }
                Poll::Ready(Some(Err(e))) => return Poll::Ready(Err(e)),
                Poll::Ready(None) => {
                    self.probe_stream_finished = true;
                    self.probe_batch_positions = vec![0; self.num_partitions];
                    for part_id in 0..self.num_partitions {
                        self.finalize_spilled_partition(part_id)?;
                    }
                    return Poll::Ready(Ok(()));
                }
                Poll::Pending => {
                    return Poll::Pending;
                }
            }
        }
    }

    fn maybe_recursive_repartition(
        &mut self,
        descriptor: &PartitionDescriptor,
    ) -> Result<bool> {
        if descriptor.build_index >= self.build_partitions.len() {
            return Ok(false);
        }
        match self.build_partitions.get(descriptor.build_index) {
            Some(BuildPartition::Spilled { .. }) => {}
            _ => return Ok(false),
        }
        let Some((additional_bits, fanout)) = self.compute_recursive_fanout(descriptor)
        else {
            return Ok(false);
        };
        let new_descriptors =
            self.repartition_spilled_partition(descriptor, additional_bits, fanout)?;
        if new_descriptors.is_empty() {
            return Ok(false);
        }
        // Enqueue new descriptors in order
        for desc in new_descriptors.into_iter().rev() {
            self.pending_partitions.push_front(desc);
        }
        Ok(true)
    }

    fn ensure_build_spill_writer<'a>(
        &self,
        accum: &'a mut PartitionAccumulator,
    ) -> Result<&'a mut InProgressSpillFile> {
        if accum.spill_writer.is_none() {
            accum.spill_writer = Some(
                self.build_spill_manager
                    .create_in_progress_file("hash_join_build_partition")?,
            );
        }
        Ok(accum.spill_writer.as_mut().unwrap())
    }

    fn spill_partition(
        &mut self,
        _build_index: usize,
        accum: &mut PartitionAccumulator,
    ) -> Result<()> {
        let buffered_batches = mem::take(&mut accum.buffered_batches);
        if buffered_batches.is_empty() {
            return Ok(());
        }

        let created_writer = accum.spill_writer.is_none();
        let mut total_spilled_bytes = 0usize;
        {
            let writer = self.ensure_build_spill_writer(accum)?;
            if created_writer {
                self.join_metrics.build_spill_count.add(1);
            }
            for batch in buffered_batches {
                let batch_size = batch.get_array_memory_size();
                total_spilled_bytes = total_spilled_bytes.saturating_add(batch_size);
                self.join_metrics.build_spilled_rows.add(batch.num_rows());
                self.join_metrics.build_spilled_bytes.add(batch_size);
                writer.append_batch(&batch)?;
            }
        }
        accum.spilled_bytes = accum.spilled_bytes.saturating_add(total_spilled_bytes);
        if accum.buffered_bytes > 0 {
            let _ = self.memory_reservation.try_shrink(accum.buffered_bytes);
            accum.buffered_bytes = 0;
        }
        Ok(())
    }

    fn append_spilled_batch(
        &self,
        accum: &mut PartitionAccumulator,
        batch: RecordBatch,
    ) -> Result<()> {
        let batch_size = batch.get_array_memory_size();
        self.join_metrics.build_spilled_rows.add(batch.num_rows());
        self.join_metrics.build_spilled_bytes.add(batch_size);
        {
            let writer = self.ensure_build_spill_writer(accum)?;
            writer.append_batch(&batch)?;
        }
        accum.spilled_bytes = accum.spilled_bytes.saturating_add(batch_size);
        Ok(())
    }

    fn reset_partition_state(&mut self) {
        for writer in self.probe_spill_in_progress.iter_mut() {
            if let Some(mut writer) = writer.take() {
                let _ = writer.finish();
            }
        }

        self.build_partitions.clear();
        self.matched_build_rows_per_partition.clear();
        self.current_partition = None;
        self.pending_partitions.clear();
        self.current_probe_batch = None;
        self.current_probe_values.clear();
        self.current_probe_hashes.clear();
        self.current_offset = (0, None);
        self.joined_probe_idx = None;
        self.placeholder_emitted = false;
        self.right_alignment_start = 0;
        self.unmatched_partition = 0;
        self.unmatched_left_indices_cache = None;
        self.unmatched_right_indices_cache = None;
        self.unmatched_offset = 0;
        self.probe_partitions.clear();
        self.probe_batch_positions.clear();
        self.probe_stream_finished = false;
        self.pending_reload_stream = None;
        self.pending_reload_batches.clear();
        self.pending_reload_partition = None;
        self.pending_probe_stream = None;
        self.pending_probe_partition = None;
        for queue in self.probe_spill_files.iter_mut() {
            queue.clear();
        }
        self.partition_pending.clear();
        self.partition_descriptors.clear();
        self.bounds_waiter = None;

        self.resize_partition_vectors();

        let reserved = self.memory_reservation.size();
        if reserved > 0 {
            let _ = self.memory_reservation.try_shrink(reserved);
        }

        self.state = PartitionedHashJoinState::PartitionBuildSide;
    }

    fn next_partition_count(&self) -> Option<usize> {
        if self.num_partitions >= self.max_partition_count {
            return None;
        }

        let mut next = self.num_partitions.saturating_mul(2);
        if next <= self.num_partitions {
            next = self.num_partitions.saturating_add(1);
        }
        if next > self.max_partition_count {
            next = self.max_partition_count;
        }
        if next > self.num_partitions {
            Some(next)
        } else {
            None
        }
    }

    fn prepare_partition_queue(&mut self) {
        self.pending_partitions.clear();
        let radix_bits =
            self.num_partitions.next_power_of_two().trailing_zeros() as usize;
        for part_id in 0..self.build_partitions.len() {
            let (spilled_bytes, spilled_rows) = match &self.build_partitions[part_id] {
                BuildPartition::Spilled {
                    spilled_bytes,
                    spilled_rows,
                    ..
                } => (*spilled_bytes, *spilled_rows),
                _ => (0, 0),
            };
            if self.partition_descriptors.len() <= part_id {
                self.partition_descriptors.resize_with(part_id + 1, || None);
            }
            if self.partition_pending.len() <= part_id {
                self.partition_pending.resize(part_id + 1, false);
            }
            self.pending_partitions.push_back(PartitionDescriptor {
                build_index: part_id,
                root_index: part_id,
                generation: self.partition_pass,
                radix_bits,
                hash_prefix: part_id as u64,
                spilled_bytes,
                spilled_rows,
            });
            if let Some(desc) = self.pending_partitions.back() {
                self.partition_descriptors[part_id] = Some(desc.clone());
                self.partition_pending[part_id] = true;
            }
        }
    }

    fn transition_to_next_partition(&mut self) {
        if let Some(descriptor) = self.pending_partitions.pop_front() {
            let build_index = descriptor.build_index;
            if self.partition_descriptors.len() <= build_index {
                self.partition_descriptors
                    .resize_with(build_index + 1, || None);
            }
            if self.partition_pending.len() <= build_index {
                self.partition_pending.resize(build_index + 1, false);
            }
            self.partition_descriptors[build_index] = Some(descriptor.clone());
            self.partition_pending[build_index] = false;
            self.current_partition = Some(build_index);
            self.state =
                PartitionedHashJoinState::ProcessPartition(ProcessPartitionState {
                    descriptor,
                });
        } else {
            self.current_partition = None;
            self.state = PartitionedHashJoinState::HandleUnmatchedRows;
        }
    }

    fn advance_to_next_partition(&mut self) {
        self.current_partition = None;
        self.transition_to_next_partition();
    }

    /// Report build-side bounds to the shared accumulator when dynamic filtering is enabled
    fn poll_bounds_update(
        &mut self,
        cx: &mut Context<'_>,
        build_data: &Arc<JoinLeftData>,
    ) -> Poll<Result<()>> {
        if let Some(ref accumulator) = self.bounds_accumulator {
            if self.bounds_waiter.is_none() {
                // println!(
                //     "[spill-join] partition={} reporting build bounds (rows={})",
                //     self.partition,
                //     build_data.batch().num_rows()
                // );
                let accumulator = Arc::clone(accumulator);
                let partition = self.partition;
                let bounds = build_data.bounds.clone();
                self.bounds_waiter = Some(OnceFut::new(async move {
                    accumulator.report_partition_bounds(partition, bounds).await
                }));
            }

            if let Some(waiter) = self.bounds_waiter.as_mut() {
                match waiter.get(cx) {
                    Poll::Ready(Ok(_)) => {
                        // println!(
                        //     "[spill-join] partition={} build bounds reported",
                        //     self.partition
                        // );
                        self.bounds_waiter = None;
                    }
                    Poll::Ready(Err(e)) => return Poll::Ready(Err(e)),
                    Poll::Pending => {
                        // println!(
                        //     "[spill-join] partition={} waiting on shared bounds barrier",
                        //     self.partition
                        // );
                        return Poll::Pending;
                    }
                }
            }
        }

        Poll::Ready(Ok(()))
    }

    /// Ensure the build partition is loaded in-memory (reload if spilled)
    fn ensure_build_partition_loaded(
        &mut self,
        cx: &mut Context<'_>,
        part_id: usize,
    ) -> Poll<Result<()>> {
        let needs_reload = matches!(
            self.build_partitions.get(part_id),
            Some(BuildPartition::Spilled { .. })
        );
        if !needs_reload {
            return Poll::Ready(Ok(()));
        }

        // Kick off reload if needed
        if self.pending_reload_partition.is_none() {
            if let Some(BuildPartition::Spilled { spill_file, .. }) =
                self.build_partitions.get_mut(part_id)
            {
                let spill_file = spill_file.take().ok_or_else(|| {
                    internal_datafusion_err!(
                        "spill file already consumed for this partition"
                    )
                })?;
                let stream = self.build_spill_manager.read_spill_as_stream(spill_file)?;
                self.pending_reload_stream = Some(stream);
                self.pending_reload_batches.clear();
                self.pending_reload_partition = Some(part_id);
            }
        }

        // Drive stream forward
        if self.pending_reload_partition == Some(part_id) {
            if let Some(stream) = self.pending_reload_stream.as_mut() {
                loop {
                    match stream.poll_next_unpin(cx) {
                        Poll::Ready(Some(Ok(batch))) => {
                            self.pending_reload_batches.push(batch);
                            // Continue draining ready batches without yielding.
                            continue;
                        }
                        Poll::Ready(Some(Err(e))) => return Poll::Ready(Err(e)),
                        Poll::Ready(None) => {
                            // Concatenate
                            let first_schema = self
                                .pending_reload_batches
                                .get(0)
                                .ok_or_else(|| {
                                    internal_datafusion_err!("empty spilled partition")
                                })?
                                .schema();
                            let concatenated = concat_batches(
                                &first_schema,
                                self.pending_reload_batches.as_slice(),
                            )
                            .map_err(DataFusionError::from)?;

                            // println!(
                            //     "Reloaded spilled build partition {} for probing (rows={})",
                            //     part_id,
                            //     concatenated.num_rows()
                            // );

                            // Grow global reservation conservatively by concatenated batch size
                            let concat_size = concatenated.get_array_memory_size();
                            let _ = self.memory_reservation.try_grow(concat_size);

                            // Recompute values and hashmap
                            let mut values: Vec<ArrayRef> =
                                Vec::with_capacity(self.on_left.len());
                            for c in &self.on_left {
                                values.push(
                                    c.evaluate(&concatenated)?
                                        .into_array(concatenated.num_rows())?,
                                );
                            }

                            let mut hash_map: Box<dyn JoinHashMapType> = Box::new(
                                JoinHashMapU32::with_capacity(concatenated.num_rows()),
                            );
                            self.hashes_buffer.clear();
                            self.hashes_buffer.resize(concatenated.num_rows(), 0);
                            // Build HT for reloaded partition from precomputed key arrays (no re-eval)
                            create_hashes(
                                &values,
                                &self.random_state,
                                &mut self.hashes_buffer,
                            )?;
                            hash_map.extend_zero(concatenated.num_rows());
                            let iter = self
                                .hashes_buffer
                                .iter()
                                .enumerate()
                                .map(|(i, h)| (i, h));
                            hash_map.update_from_iter(Box::new(iter), 0);

                            let new_reservation = MemoryConsumer::new("partition_reload")
                                .with_can_spill(true)
                                .register(&self.runtime_env.memory_pool);

                            self.build_partitions[part_id] = BuildPartition::InMemory {
                                hash_map,
                                batch: concatenated,
                                values,
                                reservation: new_reservation,
                            };

                            self.pending_reload_stream = None;
                            self.pending_reload_batches.clear();
                            self.pending_reload_partition = None;
                            // Shrink global reservation now that partition is resident with per-partition reservation
                            let _ = self.memory_reservation.try_shrink(concat_size);
                            return Poll::Ready(Ok(()));
                        }
                        Poll::Pending => {
                            cx.waker().wake_by_ref();
                            return Poll::Pending;
                        }
                    }
                }
            }
        }

        Poll::Pending
    }
    /// Create a new partitioned hash join stream
    pub fn new(
        partition: usize,
        schema: SchemaRef,
        on_left: Vec<PhysicalExprRef>,
        on_right: Vec<PhysicalExprRef>,
        filter: Option<JoinFilter>,
        join_type: JoinType,
        right: SendableRecordBatchStream,
        left_fut: OnceFut<JoinLeftData>,
        random_state: RandomState,
        join_metrics: BuildProbeJoinMetrics,
        probe_spill_metrics: SpillMetrics,
        build_spill_metrics: SpillMetrics,
        column_indices: Vec<ColumnIndex>,
        null_equality: NullEquality,
        batch_size: usize,
        num_partitions: usize,
        max_partition_count: usize,
        memory_threshold: usize,
        memory_reservation: MemoryReservation,
        runtime_env: Arc<RuntimeEnv>,
        build_schema: SchemaRef,
        probe_schema: SchemaRef,
        right_side_ordered: bool,
        bounds_accumulator: Option<
            Arc<crate::joins::hash_join::shared_bounds::SharedBoundsAccumulator>,
        >,
    ) -> Result<Self> {
        let probe_spill_manager = SpillManager::new(
            runtime_env.clone(),
            probe_spill_metrics,
            Arc::clone(&probe_schema),
        );

        let build_spill_manager = SpillManager::new(
            runtime_env.clone(),
            build_spill_metrics,
            Arc::clone(&build_schema),
        );

        Ok(Self {
            partition,
            schema,
            on_left,
            on_right,
            filter,
            join_type,
            right,
            left_fut,
            random_state,
            join_metrics,
            column_indices,
            null_equality,
            batch_size,
            num_partitions,
            max_partition_count,
            memory_threshold,
            state: PartitionedHashJoinState::PartitionBuildSide,
            build_partitions: Vec::new(),
            probe_partitions: Vec::new(),
            current_partition: None,
            pending_partitions: VecDeque::new(),
            probe_spill_manager,
            build_spill_manager,
            memory_reservation,
            partition_pass: 0,
            runtime_env,
            hashes_buffer: Vec::new(),
            right_side_ordered,
            placeholder_emitted: false,
            right_alignment_start: 0,
            bounds_accumulator,
            bounds_waiter: None,
            build_schema,
            probe_schema,
            current_probe_batch: None,
            current_probe_values: vec![],
            current_probe_hashes: vec![],
            current_offset: (0, None),
            joined_probe_idx: None,
            matched_build_rows_per_partition: Vec::new(),
            unmatched_partition: 0,
            unmatched_left_indices_cache: None,
            unmatched_right_indices_cache: None,
            unmatched_offset: 0,
            probe_stream_finished: false,
            probe_batch_positions: vec![],
            pending_reload_stream: None,
            pending_reload_batches: Vec::new(),
            pending_reload_partition: None,
            probe_spill_in_progress: (0..num_partitions).map(|_| None).collect(),
            probe_spill_files: (0..num_partitions).map(|_| VecDeque::new()).collect(),
            pending_probe_stream: None,
            pending_probe_partition: None,
            probe_buffered_rows_per_part: vec![0; num_partitions],
            probe_spilled_rows_per_part: vec![0; num_partitions],
            probe_consumed_rows_per_part: vec![0; num_partitions],
            matched_rows_per_part: vec![0; num_partitions],
            emitted_rows_per_part: vec![0; num_partitions],
            candidate_pairs_per_part: vec![0; num_partitions],
            verify_once_per_part: vec![false; num_partitions],
            filter_debug_once_per_part: vec![false; num_partitions],
            partition_pending: vec![false; num_partitions],
            partition_descriptors: (0..num_partitions).map(|_| None).collect(),
        })
    }

    /// Partition build-side data into multiple partitions
    #[cfg(feature = "hybrid_hash_join_scheduler")]
    fn partition_build_side(
        &mut self,
        build_data: Arc<JoinLeftData>,
    ) -> Result<StatefulStreamResult<Option<RecordBatch>>> {
        let config = SchedulerConfig::from_stream(self);
        HybridTaskScheduler::with_build_task(config, build_data)
            .run_until_build_finished(self)
    }

    /// Partition build-side data into multiple partitions (legacy serial path)
    #[cfg(not(feature = "hybrid_hash_join_scheduler"))]
    fn partition_build_side(
        &mut self,
        build_data: Arc<JoinLeftData>,
    ) -> Result<StatefulStreamResult<Option<RecordBatch>>> {
        self.partition_build_side_serial(build_data)
    }

    /// Legacy build partitioning logic shared with the experimental scheduler.
    pub(super) fn partition_build_side_serial(
        &mut self,
        build_data: Arc<JoinLeftData>,
    ) -> Result<StatefulStreamResult<Option<RecordBatch>>> {
        if self.partition_pass == 0 {
            self.join_metrics.build_input_batches.add(1);
            let total_rows: usize = build_data
                .original_batches()
                .iter()
                .map(|b| b.num_rows())
                .sum();
            self.join_metrics.build_input_rows.add(total_rows);
        }

        let build_total_size: usize = build_data
            .original_batches()
            .iter()
            .map(|batch| batch.get_array_memory_size())
            .sum();
        if build_total_size <= self.memory_threshold {
            self.num_partitions = 1;
            self.max_partition_count = 1;
        }

        let mut allow_repartition = true;
        loop {
            self.reset_partition_state();

            match self.try_partition_build_side(&build_data, allow_repartition)? {
                PartitionBuildStatus::Ready(result) => return Ok(result),
                PartitionBuildStatus::NeedMorePartitions { next_count } => {
                    if next_count <= self.num_partitions
                        || next_count == 0
                        || next_count > self.max_partition_count
                    {
                        allow_repartition = false;
                        continue;
                    }

                    self.num_partitions = next_count;
                    self.partition_pass += 1;
                    allow_repartition = true;
                }
            }
        }
    }

    fn try_partition_build_side(
        &mut self,
        build_data: &Arc<JoinLeftData>,
        allow_repartition: bool,
    ) -> Result<PartitionBuildStatus> {
        self.build_partitions = Vec::with_capacity(self.num_partitions);
        self.matched_build_rows_per_partition = Vec::with_capacity(self.num_partitions);

        let mut partition_accumulators = (0..self.num_partitions)
            .map(|_| PartitionAccumulator::new())
            .collect::<Vec<_>>();
        let mut repartition_request: Option<usize> = None;
        let mut max_spilled_bytes: usize = 0;
        let mut any_spilled = false;

        for batch in build_data.original_batches() {
            let mut keys_values: Vec<ArrayRef> = Vec::with_capacity(self.on_left.len());
            for expr in &self.on_left {
                keys_values.push(expr.evaluate(batch)?.into_array(batch.num_rows())?);
            }
            let mut hashes = vec![0u64; batch.num_rows()];
            create_hashes(&keys_values, &self.random_state, &mut hashes)?;

            let mut indices_per_part: Vec<Vec<u32>> =
                vec![Vec::new(); self.num_partitions];
            for (row_idx, hash) in hashes.iter().enumerate() {
                let build_index = self.partition_for_hash(*hash);
                indices_per_part[build_index].push(row_idx as u32);
            }

            for (build_index, indices) in indices_per_part.into_iter().enumerate() {
                if indices.is_empty() {
                    continue;
                }

                let idx_array = UInt32Array::from(indices);
                let mut filtered_columns: Vec<ArrayRef> =
                    Vec::with_capacity(batch.num_columns());
                for col in batch.columns() {
                    filtered_columns.push(
                        take(col, &idx_array, None).map_err(DataFusionError::from)?,
                    );
                }
                let filtered_batch =
                    RecordBatch::try_new(batch.schema(), filtered_columns)
                        .map_err(DataFusionError::from)?;
                let batch_size = filtered_batch.get_array_memory_size();
                let accum = &mut partition_accumulators[build_index];
                accum.total_rows += filtered_batch.num_rows();

                if accum.spill_writer.is_some() {
                    self.append_spilled_batch(accum, filtered_batch)?;
                    continue;
                }

                match self.memory_reservation.try_grow(batch_size) {
                    Ok(_) => {
                        accum.buffered_bytes += batch_size;
                        accum.buffered_batches.push(filtered_batch);
                        self.join_metrics
                            .build_mem_used
                            .set_max(self.memory_reservation.size());
                        if self.memory_reservation.size() > self.memory_threshold {
                            if allow_repartition {
                                if let Some(next_count) = self.next_partition_count() {
                                    repartition_request = Some(next_count);
                                    break;
                                }
                            }
                            if !self.runtime_env.disk_manager.tmp_files_enabled() {
                                return Err(internal_datafusion_err!(
                                    "Insufficient memory for build partitioning and spilling is disabled"
                                ));
                            }
                            self.spill_partition(build_index, accum)?;
                        }
                    }
                    Err(_) => {
                        if allow_repartition {
                            if let Some(next_count) = self.next_partition_count() {
                                repartition_request = Some(next_count);
                                break;
                            }
                        }
                        if !self.runtime_env.disk_manager.tmp_files_enabled() {
                            return Err(internal_datafusion_err!(
                                "Unable to allocate memory for build partition"
                            ));
                        }
                        self.spill_partition(build_index, accum)?;
                        self.append_spilled_batch(accum, filtered_batch)?;
                    }
                }

                if repartition_request.is_some() {
                    break;
                }
            }

            if repartition_request.is_some() {
                break;
            }
        }

        if let Some(next_count) = repartition_request {
            return Ok(PartitionBuildStatus::NeedMorePartitions { next_count });
        }

        self.build_partitions.reserve(self.num_partitions);
        self.matched_build_rows_per_partition
            .reserve(self.num_partitions);

        for part_id in 0..self.num_partitions {
            let mut accum = mem::take(&mut partition_accumulators[part_id]);
            max_spilled_bytes = max_spilled_bytes.max(accum.spilled_bytes);
            if accum.spill_writer.is_some() {
                if !accum.buffered_batches.is_empty() {
                    self.spill_partition(part_id, &mut accum)?;
                }
                if let Some(mut writer) = accum.spill_writer.take() {
                    let spill_file = writer
                        .finish()?
                        .ok_or_else(|| internal_datafusion_err!("expected spill file"))?;
                    let mut matched_bitmap = BooleanBufferBuilder::new(accum.total_rows);
                    matched_bitmap.append_n(accum.total_rows, false);
                    self.matched_build_rows_per_partition.push(matched_bitmap);
                    let reservation = MemoryConsumer::new("partition_spilled")
                        .with_can_spill(true)
                        .register(&self.runtime_env.memory_pool);
                    any_spilled = true;
                    self.build_partitions.push(BuildPartition::Spilled {
                        spill_file: Some(spill_file),
                        reservation,
                        spilled_bytes: accum.spilled_bytes,
                        spilled_rows: accum.total_rows,
                    });
                }
                continue;
            }

            if accum.buffered_batches.is_empty() {
                self.matched_build_rows_per_partition
                    .push(BooleanBufferBuilder::new(0));
                self.build_partitions.push(BuildPartition::Empty);
                continue;
            }

            let mut buffered_batches = accum.buffered_batches;
            let partition_batch = if buffered_batches.len() == 1 {
                buffered_batches.pop().unwrap()
            } else {
                let batch_refs: Vec<_> = buffered_batches.iter().collect();
                concat_batches(&self.build_schema, batch_refs)?
            };
            let num_rows = partition_batch.num_rows();
            let partition_values = self
                .on_left
                .iter()
                .map(|expr| expr.evaluate(&partition_batch)?.into_array(num_rows))
                .collect::<Result<Vec<_>>>()?;
            let fixed_size_u32 = size_of::<JoinHashMapU32>();
            let fixed_size_u64 = size_of::<JoinHashMapU64>();
            let mut hash_map: Box<dyn JoinHashMapType> = if num_rows > u32::MAX as usize {
                let estimated_hashtable_size =
                    estimate_memory_size::<(u64, u64)>(num_rows, fixed_size_u64)?;
                self.memory_reservation.try_grow(estimated_hashtable_size)?;
                self.join_metrics
                    .build_mem_used
                    .set_max(self.memory_reservation.size());
                Box::new(JoinHashMapU64::with_capacity(num_rows))
            } else {
                let estimated_hashtable_size =
                    estimate_memory_size::<(u32, u64)>(num_rows, fixed_size_u32)?;
                self.memory_reservation.try_grow(estimated_hashtable_size)?;
                self.join_metrics
                    .build_mem_used
                    .set_max(self.memory_reservation.size());
                Box::new(JoinHashMapU32::with_capacity(num_rows))
            };

            self.hashes_buffer.clear();
            self.hashes_buffer.resize(num_rows, 0);
            create_hashes(
                &partition_values,
                &self.random_state,
                &mut self.hashes_buffer,
            )?;
            hash_map.extend_zero(num_rows);
            let iter = self
                .hashes_buffer
                .iter()
                .enumerate()
                .map(|(idx, hash)| (idx, hash));
            hash_map.update_from_iter(Box::new(iter), 0);

            let mut matched_bitmap = BooleanBufferBuilder::new(num_rows);
            matched_bitmap.append_n(num_rows, false);
            self.matched_build_rows_per_partition.push(matched_bitmap);

            let reservation = MemoryConsumer::new("partition_memory")
                .with_can_spill(true)
                .register(&self.runtime_env.memory_pool);

            let approx_partition_size = partition_batch.get_array_memory_size()
                + partition_values
                    .iter()
                    .map(|arr| arr.get_array_memory_size())
                    .sum::<usize>();
            self.join_metrics.build_mem_used.set_max(
                self.memory_reservation
                    .size()
                    .saturating_add(approx_partition_size),
            );

            self.build_partitions.push(BuildPartition::InMemory {
                hash_map,
                batch: partition_batch,
                values: partition_values,
                reservation,
            });
        }

        if (max_spilled_bytes > self.memory_threshold || any_spilled) && allow_repartition
        {
            if let Some(next_count) = self.next_partition_count() {
                return Ok(PartitionBuildStatus::NeedMorePartitions { next_count });
            }
        }

        self.prepare_partition_queue();
        self.transition_to_next_partition();

        Ok(PartitionBuildStatus::Ready(StatefulStreamResult::Continue))
    }
    /// Release resources associated with a finished partition when safe to do so.
    /// Only releases memory eagerly when we don't need unmatched rows in the final phase.
    fn release_partition_resources(&mut self, build_index: usize) {
        if need_produce_result_in_final(self.join_type) {
            return;
        }

        if build_index >= self.build_partitions.len() {
            return;
        }

        // Take ownership of the old partition to drop heavy resources
        let placeholder_reservation =
            MemoryConsumer::new("partition_released_placeholder")
                .with_can_spill(true)
                .register(&self.runtime_env.memory_pool);
        let old_partition = mem::replace(
            &mut self.build_partitions[build_index],
            BuildPartition::Released {
                reservation: placeholder_reservation,
            },
        );

        match old_partition {
            BuildPartition::InMemory {
                batch,
                values,
                reservation,
                ..
            } => {
                // Estimate memory held by this partition and shrink global reservation
                let mut estimated_size = batch.get_array_memory_size();
                estimated_size += values
                    .iter()
                    .map(|a| a.get_array_memory_size())
                    .sum::<usize>();
                let _ = self.memory_reservation.try_shrink(estimated_size);

                // Replace with an empty in-memory partition to keep indexing stable
                let empty_batch = RecordBatch::new_empty(batch.schema());
                let empty_values: Vec<ArrayRef> = self
                    .on_left
                    .iter()
                    .filter_map(|expr| expr.evaluate(&empty_batch).ok())
                    .filter_map(|v| v.into_array(empty_batch.num_rows()).ok())
                    .collect();
                let empty_hash_map: Box<dyn JoinHashMapType> =
                    Box::new(JoinHashMapU32::with_capacity(0));

                self.build_partitions[build_index] = BuildPartition::InMemory {
                    hash_map: empty_hash_map,
                    batch: empty_batch,
                    values: empty_values,
                    reservation,
                };
            }
            BuildPartition::Spilled { reservation, .. } => {
                // Transition to Released; no files remain
                self.build_partitions[build_index] =
                    BuildPartition::Released { reservation };
            }
            BuildPartition::Released { reservation } => {
                self.build_partitions[build_index] =
                    BuildPartition::Released { reservation };
            }
            BuildPartition::Empty => {
                // no-op
            }
        }
    }

    fn partition_has_pending_probe(&self, part_id: usize) -> bool {
        if part_id < self.probe_partitions.len()
            && part_id < self.probe_batch_positions.len()
            && self.probe_batch_positions[part_id]
                < self.probe_partitions[part_id].batches.len()
        {
            return true;
        }

        if self.current_partition.is_some_and(|idx| idx == part_id)
            && self.current_probe_batch.is_some()
        {
            return true;
        }

        if part_id < self.probe_spill_files.len()
            && !self.probe_spill_files[part_id].is_empty()
        {
            return true;
        }

        if self
            .pending_probe_partition
            .is_some_and(|idx| idx == part_id)
        {
            return true;
        }

        if part_id < self.probe_spill_in_progress.len()
            && self.probe_spill_in_progress[part_id].is_some()
        {
            return true;
        }

        false
    }

    /// Process a specific partition
    fn process_partition(
        &mut self,
        cx: &mut Context<'_>,
        partition_state: &ProcessPartitionState,
    ) -> Poll<Result<StatefulStreamResult<Option<RecordBatch>>>> {
        let build_index = partition_state.descriptor.build_index;

        // Guard against invalid partition ids (off-by-one protection)
        if build_index >= self.build_partitions.len() {
            self.state = PartitionedHashJoinState::HandleUnmatchedRows;
            return Poll::Ready(Ok(StatefulStreamResult::Continue));
        }

        if self.maybe_recursive_repartition(&partition_state.descriptor)? {
            self.current_partition = None;
            self.transition_to_next_partition();
            return Poll::Ready(Ok(StatefulStreamResult::Continue));
        }

        if self.current_partition != Some(build_index) {
            self.current_partition = Some(build_index);
        }

        // Do not buffer probe side here; selection happens below depending on num_partitions

        // (Spill reload handled by ensure_build_partition_loaded earlier if needed)

        // (Build partition will be immutably borrowed later within a narrower scope)

        // Ensure the build partition is ready (reload if spilled) BEFORE any immutable borrows
        match self.ensure_build_partition_loaded(cx, build_index) {
            Poll::Ready(Ok(())) => {}
            Poll::Ready(Err(e)) => return Poll::Ready(Err(e)),
            Poll::Pending => return Poll::Pending,
        }

        // Ensure probe side is fully buffered into per-partition containers
        if !self.probe_stream_finished {
            match self.buffer_probe_side(cx) {
                Poll::Ready(Ok(())) => {}
                Poll::Ready(Err(e)) => return Poll::Ready(Err(e)),
                Poll::Pending => {
                    let no_current_data = !self.partition_has_pending_probe(build_index);
                    let no_other_pending = self.pending_partitions.is_empty();
                    if no_current_data && no_other_pending {
                        return Poll::Pending;
                    }
                }
            }
        }

        // Select next probe batch for current partition
        if self.current_probe_batch.is_none() {
            // Decide probe source based on whether we spilled probe for this partition
            let has_spilled_probe = self
                .probe_spill_in_progress
                .get(build_index)
                .and_then(|o| o.as_ref())
                .is_some()
                || self
                    .probe_spill_files
                    .get(build_index)
                    .map(|queue| !queue.is_empty())
                    .unwrap_or(false)
                || self
                    .pending_probe_partition
                    .is_some_and(|p| p == build_index);
            let has_buffered_probe = self
                .probe_partitions
                .get(build_index)
                .map(|p| !p.batches.is_empty())
                .unwrap_or(false);

            // Prefer buffered probe batches first; when exhausted, consume spilled probe stream
            let pos = self.probe_batch_positions[build_index];
            let buffered_len = self
                .probe_partitions
                .get(build_index)
                .map(|p| p.batches.len())
                .unwrap_or(0);
            if has_buffered_probe && pos < buffered_len {
                let part = &self.probe_partitions[build_index];
                // Take buffered batch/values/hashes
                let batch = part.batches[pos].clone();
                let values = part.values[pos].clone();
                let hashes = part.hashes[pos].clone();
                self.probe_batch_positions[build_index] = pos + 1;

                self.current_probe_batch = Some(batch);
                self.current_probe_values = values;
                self.current_probe_hashes = hashes;
                self.current_offset = (0, None);
                if let Some(b) = &self.current_probe_batch {
                    self.probe_consumed_rows_per_part[build_index] = self
                        .probe_consumed_rows_per_part[build_index]
                        .saturating_add(b.num_rows());
                }
            } else if has_spilled_probe {
                loop {
                    if self.pending_probe_partition != Some(build_index) {
                        let mut next_file = self
                            .probe_spill_files
                            .get_mut(build_index)
                            .and_then(|queue| queue.pop_front());
                        if next_file.is_none()
                            && self.finalize_spilled_partition(build_index)?
                        {
                            next_file = self
                                .probe_spill_files
                                .get_mut(build_index)
                                .and_then(|queue| queue.pop_front());
                        }
                        if let Some(file) = next_file {
                            let stream =
                                self.probe_spill_manager.read_spill_as_stream(file)?;
                            self.pending_probe_stream = Some(stream);
                            self.pending_probe_partition = Some(build_index);
                        } else {
                            let writer_open = self
                                .probe_spill_in_progress
                                .get(build_index)
                                .and_then(|o| o.as_ref())
                                .is_some();
                            if self.probe_stream_finished && !writer_open {
                                self.pending_probe_stream = None;
                                self.pending_probe_partition = None;
                                self.release_partition_resources(build_index);
                                self.advance_to_next_partition();
                                return Poll::Ready(Ok(StatefulStreamResult::Continue));
                            } else {
                                return Poll::Pending;
                            }
                        }
                    }

                    if let Some(stream) = self.pending_probe_stream.as_mut() {
                        match stream.poll_next_unpin(cx) {
                            Poll::Ready(Some(Ok(batch))) => {
                                let mut keys_values: Vec<ArrayRef> =
                                    Vec::with_capacity(self.on_right.len());
                                for c in &self.on_right {
                                    let v = c
                                        .evaluate(&batch)?
                                        .into_array(batch.num_rows())?;
                                    keys_values.push(v);
                                }
                                let mut hashes = vec![0u64; batch.num_rows()];
                                create_hashes(
                                    &keys_values,
                                    &self.random_state,
                                    &mut hashes,
                                )?;

                                self.current_probe_batch = Some(batch);
                                self.current_probe_values = keys_values;
                                self.current_probe_hashes = hashes;
                                self.current_offset = (0, None);
                                if let Some(b) = &self.current_probe_batch {
                                    self.probe_consumed_rows_per_part[build_index] = self
                                        .probe_consumed_rows_per_part[build_index]
                                        .saturating_add(b.num_rows());
                                }
                                break;
                            }
                            Poll::Ready(Some(Err(e))) => return Poll::Ready(Err(e)),
                            Poll::Ready(None) => {
                                self.pending_probe_stream = None;
                                self.pending_probe_partition = None;
                                continue;
                            }
                            Poll::Pending => return Poll::Pending,
                        }
                    } else {
                        return Poll::Pending;
                    }
                }
            } else {
                // Neither spilled nor buffered probe for this partition: advance
                // println!(
                //     "[spill-join][summary] part={} buffered={} spilled={} consumed={} candidates={} matched={} emitted={}",
                //     build_index,
                //     self.probe_buffered_rows_per_part[build_index],
                //     self.probe_spilled_rows_per_part[build_index],
                //     self.probe_consumed_rows_per_part[build_index],
                //     self.candidate_pairs_per_part[build_index],
                //     self.matched_rows_per_part[build_index],
                //     self.emitted_rows_per_part[build_index]
                // );
                self.release_partition_resources(build_index);
                self.advance_to_next_partition();
                return Poll::Ready(Ok(StatefulStreamResult::Continue));
            }
        }

        // If no probe batch selected, advance to next partition (no probe rows here)
        if self.current_probe_batch.is_none() {
            // println!(
            //     "[spill-join][summary] part={} buffered={} spilled={} consumed={} candidates={} matched={} emitted={}",
            //     build_index,
            //     self.probe_buffered_rows_per_part[build_index],
            //     self.probe_spilled_rows_per_part[build_index],
            //     self.probe_consumed_rows_per_part[build_index],
            //     self.candidate_pairs_per_part[build_index],
            //     self.matched_rows_per_part[build_index],
            //     self.emitted_rows_per_part[build_index]
            // );
            self.release_partition_resources(build_index);
            self.advance_to_next_partition();
            return Poll::Ready(Ok(StatefulStreamResult::Continue));
        }

        // At this point we have a current probe batch for this partition
        let (result, build_ids_to_mark, next_offset) = {
            let probe_batch = self
                .current_probe_batch
                .as_ref()
                .ok_or_else(|| internal_datafusion_err!("expected probe batch"))?;

            let (build_hashmap, build_batch, build_values) =
                match self.build_partitions.get(build_index) {
                    Some(BuildPartition::InMemory {
                        hash_map,
                        batch,
                        values,
                        ..
                    }) => (&**hash_map, batch, values as &Vec<ArrayRef>),
                    Some(BuildPartition::Spilled { .. }) => {
                        return Poll::Ready(Ok(StatefulStreamResult::Continue));
                    }
                    Some(BuildPartition::Released { .. })
                    | Some(BuildPartition::Empty)
                    | None => {
                        return Poll::Ready(internal_err!(
                            "Missing or invalid build partition"
                        ));
                    }
                };
            // Debug: log ON expressions and output mapping once we have both sides
            /* let on_left_desc = self
                .on_left
                .iter()
                .map(|e| format!("{}", e))
                .collect::<Vec<_>>()
                .join(", ");
            let on_right_desc = self
                .on_right
                .iter()
                .map(|e| format!("{}", e))
                .collect::<Vec<_>>()
                .join(", ");
            let mapping_desc = self
                .column_indices
                .iter()
                .map(|ci| {
                    let side = match ci.side {
                        JoinSide::Left => "L",
                        JoinSide::Right => "R",
                        JoinSide::None => "M",
                    };
                    format!("{}@{}", side, ci.index)
                })
                .collect::<Vec<_>>()
                .join(", ");*/
            // println!(
            //     "[spill-join] ON build=[{}] | probe=[{}] | out=[{}]",
            //     on_left_desc, on_right_desc, mapping_desc
            // );

            // Log resolved output column names for the current mapping
            /*let out_names = self
                .column_indices
                .iter()
                .map(|ci| match ci.side {
                    JoinSide::Left => {
                        format!("L:{}", build_batch.schema().field(ci.index).name())
                    }
                    JoinSide::Right => {
                        format!("R:{}", probe_batch.schema().field(ci.index).name())
                    }
                    JoinSide::None => "M:mark".to_string(),
                })
                .collect::<Vec<_>>()
                .join(", ");
            // println!("[spill-join] OUT columns: {}", out_names);

            // println!(
            //     "[spill-join] Partition {} build hashmap empty? {}",
            //     build_index,
            //     build_hashmap.is_empty()
            // );*/

            // Lookup against hash map with limit
            let (probe_indices, build_indices, next_offset) = build_hashmap
                .get_matched_indices_with_limit_offset(
                    &self.current_probe_hashes,
                    self.batch_size,
                    self.current_offset,
                );

            let build_indices: UInt64Array = build_indices.into();
            let probe_indices: UInt32Array = probe_indices.into();

            // Track candidate pairs before equality
            self.candidate_pairs_per_part[build_index] = self.candidate_pairs_per_part
                [build_index]
                .saturating_add(build_indices.len());
            // println!(
            //     "[spill-join] Candidates before equality: build_ids={}, probe_ids={}, build_rows={}, probe_rows={}",
            //     build_indices.len(),
            //     probe_indices.len(),
            //     build_batch.num_rows(),
            //     probe_batch.num_rows()
            // );

            // Resolve hash collisions
            let (build_indices, probe_indices) = equal_rows_arr(
                &build_indices,
                &probe_indices,
                build_values,
                &self.current_probe_values,
                self.null_equality,
            )?;

            // Shadow verify on INNER join with single Int64 key (first 50k rows)
            /*if matches!(self.join_type, JoinType::Inner)
                && build_values.len() == 1
                && self.current_probe_values.len() == 1
                && build_values[0].data_type() == &arrow::datatypes::DataType::Int64
                && self.current_probe_values[0].data_type()
                    == &arrow::datatypes::DataType::Int64
                && !self.verify_once_per_part[build_index]
            {
                use arrow::array::Int64Array;
                use std::collections::HashMap;
                let bcol = build_values[0]
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap();
                let pcol = self.current_probe_values[0]
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap();
                let mut map: HashMap<i64, usize> = HashMap::new();
                let max_b = bcol.len().min(50_000);
                for i in 0..max_b {
                    if bcol.is_null(i) {
                        continue;
                    }
                    let k = bcol.value(i);
                    *map.entry(k).or_insert(0) += 1;
                }
                /*let mut expect = 0usize;
                let max_p = pcol.len().min(50_000);
                for i in 0..max_p {
                    if pcol.is_null(i) {
                        continue;
                    }
                    let k = pcol.value(i);
                    if let Some(&c) = map.get(&k) {
                        expect += c;
                    }
                }
                // println!(
                //     "[spill-join][verify] part={} expect_pairs~{} vs actual_after_eq={}",
                //     build_index,
                //     expect,
                //     build_indices.len()
                // );*/
                self.verify_once_per_part[build_index] = true;
            }*/

            // Debug: log key data types and sample matched pairs
            /*if !build_indices.is_empty() {
                /*let build_types = build_values
                    .iter()
                    .map(|a| format!("{:?}", a.data_type()))
                    .collect::<Vec<_>>()
                    .join(", ");
                let probe_types = self
                    .current_probe_values
                    .iter()
                    .map(|a| format!("{:?}", a.data_type()))
                    .collect::<Vec<_>>()
                    .join(", ");*/
                // println!(
                //     "[spill-join] Key types: build=[{}], probe=[{}], null_equality={:?}",
                //     build_types, probe_types, self.null_equality
                // );
                let sample = build_indices.len().min(5);
                let mut pairs = Vec::new();
                for i in 0..sample {
                    let b = build_indices.value(i) as usize;
                    let p = probe_indices.value(i) as usize;
                    // Include actual first-key values for sanity checks
                    let bk = &build_values[0];
                    let pk = &self.current_probe_values[0];
                    let bv = arrow::util::display::array_value_to_string(bk.as_ref(), b)
                        .unwrap_or_else(|_| "<err>".to_string());
                    let pv = arrow::util::display::array_value_to_string(pk.as_ref(), p)
                        .unwrap_or_else(|_| "<err>".to_string());
                    pairs.push(format!("({},{})", bv, pv));
                }
                // println!(
                //     "[spill-join] Sample key pairs {} -> {}: {}",
                //     sample,
                //     build_indices.len(),
                //     pairs.join(", ")
                // );
            }*/

            // Apply residual join filter if present
            let mut build_indices = build_indices;
            let mut probe_indices = probe_indices;
            if let Some(filter) = &self.filter {
                let before_len = build_indices.len();
                // let before_build_indices = build_indices.clone();
                //let before_probe_indices = probe_indices.clone();

                let (filtered_build_indices, filtered_probe_indices) =
                    apply_join_filter_to_indices(
                        build_batch,
                        probe_batch,
                        build_indices,
                        probe_indices,
                        filter,
                        JoinSide::Left,
                        None,
                    )?;

                if !self.filter_debug_once_per_part[build_index] {
                    /*
                    // println!(
                    //     "[spill-join][filter-debug] part={} filter_before={} filter_after={}",
                    //     build_index,
                    //     before_len,
                    //     filtered_build_indices.len()
                    // );

                    let sample = filtered_build_indices.len().min(5);
                    for i in 0..sample {
                        let build_row = filtered_build_indices.value(i) as usize;
                        let probe_row = filtered_probe_indices.value(i) as usize;

                        let build_schema = build_batch.schema();
                        let build_vals = (0..build_batch.num_columns())
                            .map(|col| {
                                let name = build_schema.field(col).name();
                                let value = arrow::util::display::array_value_to_string(
                                    build_batch.column(col).as_ref(),
                                    build_row,
                                )
                                .unwrap_or_else(|_| "<err>".to_string());
                                format!("{}={}", name, value)
                            })
                            .collect::<Vec<_>>()
                            .join(", ");

                        let probe_schema = probe_batch.schema();
                        let probe_vals = (0..probe_batch.num_columns())
                            .map(|col| {
                                let name = probe_schema.field(col).name();
                                let value = arrow::util::display::array_value_to_string(
                                    probe_batch.column(col).as_ref(),
                                    probe_row,
                                )
                                .unwrap_or_else(|_| "<err>".to_string());
                                format!("{}={}", name, value)
                            })
                            .collect::<Vec<_>>()
                            .join(", ");

                        // println!(
                        //     "[spill-join][filter-debug] sample {} build {{{}}} probe {{{}}}",
                        //     i, build_vals, probe_vals
                        // );
                    }

                    if filtered_build_indices.len() == 0 {
                        let sample_removed = before_build_indices.len().min(5);
                        for i in 0..sample_removed {
                            let build_row = before_build_indices.value(i) as usize;
                            let probe_row = before_probe_indices.value(i) as usize;

                            let build_schema = build_batch.schema();
                            let build_vals = (0..build_batch.num_columns())
                                .map(|col| {
                                    let name = build_schema.field(col).name();
                                    let value =
                                        arrow::util::display::array_value_to_string(
                                            build_batch.column(col).as_ref(),
                                            build_row,
                                        )
                                        .unwrap_or_else(|_| "<err>".to_string());
                                    format!("{}={}", name, value)
                                })
                                .collect::<Vec<_>>()
                                .join(", ");

                            let probe_schema = probe_batch.schema();
                            /*let probe_vals = (0..probe_batch.num_columns())
                                .map(|col| {
                                    let name = probe_schema.field(col).name();
                                    let value =
                                        arrow::util::display::array_value_to_string(
                                            probe_batch.column(col).as_ref(),
                                            probe_row,
                                        )
                                        .unwrap_or_else(|_| "<err>".to_string());
                                    format!("{}={}", name, value)
                                })
                                .collect::<Vec<_>>()
                                .join(", ");*/

                            // println!(
                            //     "[spill-join][filter-debug] removed sample {} build {{{}}} probe {{{}}}",
                            //     i, build_vals, probe_vals
                            // );
                        }
                    }*/

                    self.filter_debug_once_per_part[build_index] = true;
                }

                if before_len != filtered_build_indices.len() {
                    // println!(
                    //     "[spill-join][filter-debug] part={} filter removed {} rows",
                    //     build_index,
                    //     before_len - filtered_build_indices.len()
                    // );
                }

                build_indices = filtered_build_indices;
                probe_indices = filtered_probe_indices;
            }

            // Capture matched build indices prior to alignment so we can mark bitmaps even if
            // the join type drops them (e.g. LeftAnti emits matches only in the final phase).
            let build_indices_for_marking =
                if need_produce_result_in_final(self.join_type) {
                    Some(build_indices.clone())
                } else {
                    None
                };

            // Log sample matches even if no residual filter remains, to debug equality behavior
            /*if !self.filter_debug_once_per_part[build_index]
                || build_indices.len() != probe_indices.len()
            {
                let sample = build_indices.len().min(5);
                for i in 0..sample {
                    let build_row = build_indices.value(i) as usize;
                    let probe_row = probe_indices.value(i) as usize;

                    let build_schema = build_batch.schema();
                    let build_vals = (0..build_batch.num_columns())
                        .map(|col| {
                            let name = build_schema.field(col).name();
                            let value = arrow::util::display::array_value_to_string(
                                build_batch.column(col).as_ref(),
                                build_row,
                            )
                            .unwrap_or_else(|_| "<err>".to_string());
                            format!("{}={}", name, value)
                        })
                        .collect::<Vec<_>>()
                        .join(", ");

                    let probe_schema = probe_batch.schema();
                   /* let probe_vals = (0..probe_batch.num_columns())
                        .map(|col| {
                            let name = probe_schema.field(col).name();
                            let value = arrow::util::display::array_value_to_string(
                                probe_batch.column(col).as_ref(),
                                probe_row,
                            )
                            .unwrap_or_else(|_| "<err>".to_string());
                            format!("{}={}", name, value)
                        })
                        .collect::<Vec<_>>()
                        .join(", ");*/

                    // println!(
                    //     "[spill-join][match-debug] part={} pair {} build {{{}}} probe {{{}}}",
                    //     build_index,
                    //     i,
                    //     build_vals,
                    //     probe_vals
                    // );
                }

                if build_indices.len() != probe_indices.len() {
                    // println!(
                    //     "[spill-join][match-debug] part={} MISMATCH len build={} probe={}",
                    //     build_index,
                    //     build_indices.len(),
                    //     probe_indices.len()
                    // );
                }

                self.filter_debug_once_per_part[build_index] = true;
            }*/

            // Debug counter: post-equality (before any alignment)
            // println!(
            //     "[spill-join] After equality{} (pre-align): {}",
            //     if self.filter.is_some() { "+filter" } else { "" },
            //     build_indices.len()
            // );
            // Shadow verify for two-key joins (stringified) to catch type coercion issues
            /*if matches!(self.join_type, JoinType::Inner)
                && build_values.len() == 2
                && self.current_probe_values.len() == 2
                && !self.verify_once_per_part[build_index]
            {
                use std::collections::HashMap;
                let mut map: HashMap<String, usize> = HashMap::new();
                let max_b = build_batch.num_rows().min(50_000);
                for i in 0..max_b {
                    let k0 = arrow::util::display::array_value_to_string(
                        build_values[0].as_ref(),
                        i,
                    )
                    .unwrap_or_else(|_| "<err>".to_string());
                    let k1 = arrow::util::display::array_value_to_string(
                        build_values[1].as_ref(),
                        i,
                    )
                    .unwrap_or_else(|_| "<err>".to_string());
                    let key = format!("{}|{}", k0, k1);
                    *map.entry(key).or_insert(0) += 1;
                }
                let max_p = probe_batch.num_rows().min(50_000);
                for i in 0..max_p {
                    let k0 = arrow::util::display::array_value_to_string(
                        self.current_probe_values[0].as_ref(),
                        i,
                    )
                    .unwrap_or_else(|_| "<err>".to_string());
                    let k1 = arrow::util::display::array_value_to_string(
                        self.current_probe_values[1].as_ref(),
                        i,
                    )
                    .unwrap_or_else(|_| "<err>".to_string());
                    let key = format!("{}|{}", k0, k1);
                    if let Some(&c) = map.get(&key) {
                        expect += c;
                    }
                }
                // println!(
                //     "[spill-join][verify2] part={} expect_pairs~{} vs actual_after_eq={}",
                //     build_index,
                //     expect,
                //     build_indices.len()
                // );
                self.verify_once_per_part[build_index] = true;
            }*/
            // Accumulate matched rows per partition
            self.matched_rows_per_part[build_index] = self.matched_rows_per_part
                [build_index]
                .saturating_add(build_indices.len());

            // Compute alignment window (used by adjust_indices for all join types)
            let last_joined_right_idx = match probe_indices.len() {
                0 => None,
                n => Some(probe_indices.value(n - 1) as usize),
            };
            let probe_num_rows = probe_batch.num_rows();
            let mut index_alignment_range_start =
                self.joined_probe_idx.map_or(0, |v| v + 1);
            let mut index_alignment_range_end = if next_offset.is_none() {
                probe_num_rows
            } else {
                last_joined_right_idx.map_or(index_alignment_range_start, |v| v + 1)
            };

            if index_alignment_range_start > probe_num_rows {
                index_alignment_range_start = probe_num_rows;
            }
            if index_alignment_range_end > probe_num_rows {
                index_alignment_range_end = probe_num_rows;
            }
            if index_alignment_range_end < index_alignment_range_start {
                index_alignment_range_end = index_alignment_range_start;
            }

            let (build_indices, probe_indices) = adjust_indices_by_join_type(
                build_indices,
                probe_indices,
                index_alignment_range_start..index_alignment_range_end,
                self.join_type,
                self.right_side_ordered,
            )?;

            // Only right-oriented joins need to preserve alignment state across batches
            let needs_alignment = matches!(
                self.join_type,
                JoinType::RightSemi | JoinType::RightAnti | JoinType::RightMark
            );

            // Debug counter: after alignment (or effective no-op for other join types)
            // println!("[spill-join] After alignment: {}", build_indices.len());

            // Prepare ids for marking after we release borrows. Prefer the pre-alignment
            // matches (for join types like LeftAnti) so bitmap tracking remains accurate.
            let build_ids_to_mark: Vec<u64> =
                if let Some(indices) = build_indices_for_marking {
                    indices.values().to_vec()
                } else {
                    build_indices.values().to_vec()
                };
            // Track last joined probe row only for right-oriented joins; otherwise clear it
            self.joined_probe_idx = if needs_alignment && next_offset.is_some() {
                last_joined_right_idx
            } else {
                None
            };

            // Build output batch depending on join side semantics
            let result = if matches!(
                self.join_type,
                JoinType::RightMark | JoinType::RightSemi | JoinType::RightAnti
            ) {
                if matches!(self.join_type, JoinType::RightMark) {
                    // println!("[spill-join] Building output with JoinSide::Right (RightMark)");
                } else {
                    // println!(
                    //     "[spill-join] Building output with JoinSide::Right ({:?})",
                    //     self.join_type
                    // );
                }
                let right_indices_u64 = uint32_to_uint64_indices(&probe_indices);
                build_batch_from_indices(
                    &self.schema,
                    probe_batch,
                    build_batch,
                    &right_indices_u64,
                    &probe_indices,
                    &self.column_indices,
                    JoinSide::Right,
                )?
            } else {
                build_batch_from_indices(
                    &self.schema,
                    build_batch,
                    probe_batch,
                    &build_indices,
                    &probe_indices,
                    &self.column_indices,
                    JoinSide::Left,
                )?
            };

            let emitted_rows = result.num_rows();
            self.emitted_rows_per_part[build_index] =
                self.emitted_rows_per_part[build_index].saturating_add(emitted_rows);
            (result, build_ids_to_mark, next_offset)
        };

        // Mark matched build-side rows for outer joins (use current partition's bitmap)
        if let Some(bitmap) = self.matched_build_rows_per_partition.get_mut(build_index) {
            for build_idx in build_ids_to_mark {
                bitmap.set_bit(build_idx as usize, true);
            }
        }

        // Update offset or fetch a new probe batch
        if let Some(offset) = next_offset {
            self.current_offset = offset;
        } else {
            // Finished this probe batch
            self.current_probe_batch = None;
            self.current_probe_values.clear();
            self.current_probe_hashes.clear();
            self.current_offset = (0, None);
            self.joined_probe_idx = None;
            // Alignment is batch-local for semi/anti/mark in spillable path; do not carry across batches
        }

        if result.num_rows() == 0 {
            // println!(
            //     "[spill-join] Skipping empty batch emission (partition={})",
            //     build_index
            // );
            return Poll::Ready(Ok(StatefulStreamResult::Continue));
        }
        self.join_metrics.output_batches.add(1);
        self.join_metrics.baseline.record_output(result.num_rows());
        // println!(
        //     "[spill-join] Emitting batch: rows={} (partition={})",
        //     result.num_rows(),
        //     build_index
        // );
        Poll::Ready(Ok(StatefulStreamResult::Ready(Some(result))))
    }

    /// Handle unmatched rows for outer joins (poll-based, non-blocking spill reload)
    fn handle_unmatched_rows(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<Result<StatefulStreamResult<Option<RecordBatch>>>> {
        if !need_produce_result_in_final(self.join_type) {
            self.state = PartitionedHashJoinState::Completed;
            return Poll::Ready(Ok(StatefulStreamResult::Ready(None)));
        }

        // If we have cached unmatched indices for current partition, emit them chunk-by-chunk
        if let (Some(left_all), Some(right_all)) = (
            self.unmatched_left_indices_cache.as_ref(),
            self.unmatched_right_indices_cache.as_ref(),
        ) {
            let total = left_all.len();
            if self.unmatched_offset < total {
                let remaining = total - self.unmatched_offset;
                let to_emit = remaining.min(self.batch_size);

                let left_chunk_ref = left_all.slice(self.unmatched_offset, to_emit);
                let right_chunk_ref = right_all.slice(self.unmatched_offset, to_emit);
                let left_chunk = left_chunk_ref
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .ok_or_else(|| {
                        internal_datafusion_err!("failed to downcast left indices chunk")
                    })?;
                let right_chunk = right_chunk_ref
                    .as_any()
                    .downcast_ref::<UInt32Array>()
                    .ok_or_else(|| {
                        internal_datafusion_err!("failed to downcast right indices chunk")
                    })?;

                // Use current partition's build batch
                let partition = self
                    .build_partitions
                    .get(self.unmatched_partition)
                    .ok_or_else(|| {
                        internal_datafusion_err!(
                            "missing build partition during unmatched cached emission"
                        )
                    })?;
                let build_batch = match partition {
                    BuildPartition::InMemory { batch, .. } => batch,
                    BuildPartition::Spilled { .. } => {
                        // Should not happen because we only cache after loading InMemory indices
                        return Poll::Ready(Ok(StatefulStreamResult::Continue));
                    }
                    BuildPartition::Released { .. } => {
                        return Poll::Ready(Ok(StatefulStreamResult::Continue))
                    }
                    BuildPartition::Empty => {
                        return Poll::Ready(Ok(StatefulStreamResult::Continue))
                    }
                };

                let empty_right_batch =
                    RecordBatch::new_empty(Arc::clone(&self.probe_schema));
                // println!(
                //     "Emitting unmatched rows chunk: partition={}, offset={}, size={} (total={})",
                //     self.unmatched_partition,
                //     self.unmatched_offset,
                //     to_emit,
                //     total
                // );

                let result = build_batch_from_indices(
                    &self.schema,
                    build_batch,
                    &empty_right_batch,
                    left_chunk,
                    right_chunk,
                    &self.column_indices,
                    JoinSide::Left,
                )?;

                self.unmatched_offset += to_emit;
                if self.unmatched_offset >= total {
                    // finished this partition's unmatched rows
                    self.unmatched_left_indices_cache = None;
                    self.unmatched_right_indices_cache = None;
                    self.unmatched_offset = 0;
                    // println!(
                    //     "Finished emitting unmatched rows for partition {}",
                    //     self.unmatched_partition
                    // );
                    self.unmatched_partition += 1;
                }

                return Poll::Ready(Ok(StatefulStreamResult::Ready(Some(result))));
            } else {
                // Safety: should not reach here; reset caches
                self.unmatched_left_indices_cache = None;
                self.unmatched_right_indices_cache = None;
                self.unmatched_offset = 0;
            }
        }

        // Process unmatched rows for the current partition
        if self.unmatched_partition < self.build_partitions.len() {
            let partition = self
                .build_partitions
                .get_mut(self.unmatched_partition)
                .ok_or_else(|| {
                    internal_datafusion_err!(
                        "missing build partition during unmatched processing"
                    )
                })?;

            match partition {
                BuildPartition::InMemory { batch: _batch, .. } => {
                    // Get unmatched indices for this partition using its bitmap
                    let (left_indices, right_indices) = if let Some(bitmap) = self
                        .matched_build_rows_per_partition
                        .get(self.unmatched_partition)
                    {
                        get_final_indices_from_bit_map(bitmap, self.join_type)
                    } else {
                        // If no bitmap, skip this partition
                        self.unmatched_partition += 1;
                        return Poll::Ready(Ok(StatefulStreamResult::Continue));
                    };

                    // println!(
                    //     "Unmatched calculation for partition {} -> {} rows",
                    //     self.unmatched_partition,
                    //     left_indices.len()
                    // );

                    if left_indices.len() > 0 {
                        // Cache the full indices and emit first chunk via cached path next call
                        self.unmatched_left_indices_cache = Some(left_indices.clone());
                        self.unmatched_right_indices_cache = Some(right_indices.clone());
                        self.unmatched_offset = 0;
                        // Fall-through into cached emission on next invocation
                        return Poll::Ready(Ok(StatefulStreamResult::Continue));
                    } else {
                        // No unmatched rows in this partition, move to next
                        self.unmatched_partition += 1;
                        return Poll::Ready(Ok(StatefulStreamResult::Continue));
                    }
                }
                BuildPartition::Spilled { spill_file, .. } => {
                    // Non-blocking reload of spilled partition for unmatched rows
                    if self.pending_reload_partition.is_none() {
                        let taken = spill_file.take().ok_or_else(|| {
                            internal_datafusion_err!(
                                "spill file already consumed for unmatched"
                            )
                        })?;
                        let stream =
                            self.build_spill_manager.read_spill_as_stream(taken)?;
                        self.pending_reload_stream = Some(stream);
                        self.pending_reload_batches.clear();
                        self.pending_reload_partition = Some(self.unmatched_partition);
                    }

                    if self.pending_reload_partition == Some(self.unmatched_partition) {
                        if let Some(stream) = self.pending_reload_stream.as_mut() {
                            match stream.poll_next_unpin(cx) {
                                Poll::Ready(Some(Ok(batch))) => {
                                    // println!(
                                    //     "Reload stream yielded batch for build partition {} (rows={})",
                                    //     self.unmatched_partition,
                                    //     batch.num_rows()
                                    // );
                                    self.pending_reload_batches.push(batch);
                                    return Poll::Pending;
                                }
                                Poll::Ready(Some(Err(e))) => return Poll::Ready(Err(e)),
                                Poll::Ready(None) => {
                                    let first_schema = self
                                        .pending_reload_batches
                                        .get(0)
                                        .ok_or_else(|| {
                                            internal_datafusion_err!(
                                                "empty spilled partition for unmatched"
                                            )
                                        })?
                                        .schema();
                                    let concatenated = concat_batches(
                                        &first_schema,
                                        self.pending_reload_batches.as_slice(),
                                    )
                                    .map_err(DataFusionError::from)?;

                                    // println!(
                                    //     "Reloaded spilled build partition {} for unmatched rows (rows={})",
                                    //     self.unmatched_partition,
                                    //     concatenated.num_rows()
                                    // );

                                    let new_reservation =
                                        MemoryConsumer::new("partition_reload_unmatched")
                                            .with_can_spill(true)
                                            .register(&self.runtime_env.memory_pool);
                                    let mut values: Vec<ArrayRef> =
                                        Vec::with_capacity(self.on_left.len());
                                    for c in &self.on_left {
                                        values.push(
                                            c.evaluate(&concatenated)?
                                                .into_array(concatenated.num_rows())?,
                                        );
                                    }
                                    let hash_map: Box<dyn JoinHashMapType> =
                                        Box::new(JoinHashMapU32::with_capacity(
                                            concatenated.num_rows(),
                                        ));
                                    self.build_partitions[self.unmatched_partition] =
                                        BuildPartition::InMemory {
                                            hash_map,
                                            batch: concatenated,
                                            values,
                                            reservation: new_reservation,
                                        };
                                    // println!(
                                    //     "Prepared spilled partition {} as InMemory for unmatched emission",
                                    //     self.unmatched_partition
                                    // );

                                    // Clear pending
                                    self.pending_reload_stream = None;
                                    self.pending_reload_batches.clear();
                                    self.pending_reload_partition = None;

                                    // Continue; next iteration will handle InMemory branch
                                    return Poll::Ready(Ok(
                                        StatefulStreamResult::Continue,
                                    ));
                                }
                                Poll::Pending => {
                                    // Yield until more data is available from reload stream
                                    // println!(
                                    //     "Reload stream pending for build partition {} (accumulated_batches={})",
                                    //     self.unmatched_partition,
                                    //     self.pending_reload_batches.len()
                                    // );
                                    return Poll::Pending;
                                }
                            }
                        }
                    }
                    Poll::Pending
                }
                BuildPartition::Released { .. } => {
                    // Nothing to emit; advance
                    self.unmatched_partition += 1;
                    return Poll::Ready(Ok(StatefulStreamResult::Continue));
                }
                BuildPartition::Empty => {
                    self.unmatched_partition += 1;
                    return Poll::Ready(Ok(StatefulStreamResult::Continue));
                }
            }
        } else {
            // All partitions processed
            self.state = PartitionedHashJoinState::Completed;
            return Poll::Ready(Ok(StatefulStreamResult::Ready(None)));
        }
    }
}

impl RecordBatchStream for PartitionedHashJoinStream {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}

impl Stream for PartitionedHashJoinStream {
    type Item = Result<RecordBatch>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        loop {
            match self.state.clone() {
                PartitionedHashJoinState::PartitionBuildSide => {
                    // Collect build side and partition it
                    let left_data = {
                        let fut = &mut self.left_fut;
                        ready!(fut.get_shared(cx))?
                    };
                    match self.poll_bounds_update(cx, &left_data) {
                        Poll::Ready(Ok(())) => {}
                        Poll::Ready(Err(e)) => return Poll::Ready(Some(Err(e))),
                        Poll::Pending => return Poll::Pending,
                    }
                    match self.partition_build_side(left_data) {
                        Ok(StatefulStreamResult::Continue) => continue,
                        Ok(StatefulStreamResult::Ready(Some(batch))) => {
                            // println!(
                            //     "[spill-join] poll_next yielding initial batch: rows={}",
                            //     batch.num_rows()
                            // );
                            return Poll::Ready(Some(Ok(batch)));
                        }
                        Ok(StatefulStreamResult::Ready(None)) => {
                            return Poll::Ready(None)
                        }
                        Err(e) => return Poll::Ready(Some(Err(e))),
                    }
                }
                PartitionedHashJoinState::ProcessPartition(partition_state) => {
                    // Emit a zero-row placeholder once in multi-output mode to satisfy downstream schedulers
                    if self.num_partitions > 1 && !self.placeholder_emitted {
                        self.placeholder_emitted = true;
                        let empty = RecordBatch::new_empty(self.schema.clone());
                        // println!(
                        //     "[spill-join] Emitting placeholder empty batch for partition {}",
                        //     build_index
                        // );
                        return Poll::Ready(Some(Ok(empty)));
                    }
                    match self.process_partition(cx, &partition_state) {
                        Poll::Ready(Ok(StatefulStreamResult::Ready(Some(batch)))) => {
                            // println!(
                            //     "[spill-join] poll_next yielding process batch: rows={} (state partition={})",
                            //     batch.num_rows(), build_index
                            // );
                            return Poll::Ready(Some(Ok(batch)));
                        }
                        Poll::Ready(Ok(StatefulStreamResult::Ready(None))) => {
                            return Poll::Ready(None);
                        }
                        Poll::Ready(Ok(StatefulStreamResult::Continue)) => {
                            continue;
                        }
                        Poll::Ready(Err(e)) => return Poll::Ready(Some(Err(e))),
                        Poll::Pending => return Poll::Pending,
                    }
                }
                PartitionedHashJoinState::HandleUnmatchedRows => {
                    match self.handle_unmatched_rows(cx) {
                        Poll::Ready(Ok(StatefulStreamResult::Ready(Some(batch)))) => {
                            // println!(
                            //     "[spill-join] poll_next yielding unmatched batch: rows={}",
                            //     batch.num_rows()
                            // );
                            return Poll::Ready(Some(Ok(batch)));
                        }
                        Poll::Ready(Ok(StatefulStreamResult::Ready(None))) => {
                            return Poll::Ready(None);
                        }
                        Poll::Ready(Ok(StatefulStreamResult::Continue)) => {
                            continue;
                        }
                        Poll::Ready(Err(e)) => return Poll::Ready(Some(Err(e))),
                        Poll::Pending => return Poll::Pending,
                    }
                }
                PartitionedHashJoinState::Completed => return Poll::Ready(None),
            }
        }
    }
}
