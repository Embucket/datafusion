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

use std::sync::Arc;
use std::task::{Context, Poll};
use std::mem;

use crate::joins::hash_join::exec::JoinLeftData;
use crate::joins::join_hash_map::JoinHashMapType;
use crate::joins::utils::{
    build_batch_from_indices, equal_rows_arr, get_final_indices_from_bit_map,
    need_produce_result_in_final, BuildProbeJoinMetrics, ColumnIndex, JoinFilter,
    OnceFut, StatefulStreamResult,
};
use crate::metrics::{SpillMetrics};
use crate::spill::spill_manager::SpillManager;
use crate::{RecordBatchStream, SendableRecordBatchStream};

use arrow::array::{Array, ArrayRef, BooleanBufferBuilder, UInt32Array, UInt64Array};
use arrow::compute::{take, concat_batches};
use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use datafusion_common::{
    hash_utils::create_hashes, internal_datafusion_err, internal_err, DataFusionError,
    JoinSide, JoinType, NullEquality, Result,
};
use datafusion_execution::memory_pool::{MemoryConsumer, MemoryReservation};
use datafusion_execution::runtime_env::RuntimeEnv;
use datafusion_execution::disk_manager::RefCountedTempFile;
use datafusion_physical_expr::PhysicalExprRef;

use ahash::RandomState;
use futures::{ready, Stream, StreamExt};
 

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
    /// Current partition being processed
    pub partition_id: usize,
    /// Total number of partitions
    pub total_partitions: usize,
    /// Whether we're processing the last partition
    pub is_last_partition: bool,
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
    },
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
    /// Manages the process of spilling and reading back intermediate data
    pub spill_manager: SpillManager,
    /// Memory reservation for the entire operation
    pub memory_reservation: MemoryReservation,
    /// Runtime environment
    pub runtime_env: Arc<RuntimeEnv>,
    /// Scratch space for computing hashes
    pub hashes_buffer: Vec<u64>,
    /// Whether the right side has an ordering to potentially preserve
    pub right_side_ordered: bool,
    /// Shared bounds accumulator for coordinating dynamic filter updates (optional)
    pub bounds_accumulator: Option<Arc<crate::joins::hash_join::shared_bounds::SharedBoundsAccumulator>>,
    /// Current probe batch (filtered to the active partition), if any
    pub current_probe_batch: Option<RecordBatch>,
    /// Current probe values for ON expressions
    pub current_probe_values: Vec<ArrayRef>,
    /// Current probe hashes (filtered to the active partition)
    pub current_probe_hashes: Vec<u64>,
    /// Current lookup offset within the join hash map
    pub current_offset: crate::joins::join_hash_map::JoinHashMapOffset,
    /// Bitmaps to track matched build-side rows for outer joins (one per partition)
    pub matched_build_rows_per_partition: Vec<BooleanBufferBuilder>,
    /// Current partition being processed for unmatched rows
    pub unmatched_partition: usize,
    /// Cached unmatched build/probe indices for current partition (chunked emission)
    pub unmatched_left_indices_cache: Option<UInt64Array>,
    pub unmatched_right_indices_cache: Option<UInt32Array>,
    pub unmatched_offset: usize,
    /// Whether we've buffered the entire probe side into per-partition batches
    pub probes_buffered: bool,
    /// Current read position per partition within buffered probe batches
    pub probe_batch_positions: Vec<usize>,
    /// Pending async spill reload stream for build partitions
    pub pending_reload_stream: Option<SendableRecordBatchStream>,
    /// Accumulated batches for pending reload
    pub pending_reload_batches: Vec<RecordBatch>,
    /// Target partition id for pending reload
    pub pending_reload_partition: Option<usize>,
}

impl PartitionedHashJoinStream {
    /// Ensure the build partition is loaded in-memory (reload if spilled)
    fn ensure_build_partition_loaded(&mut self, cx: &mut Context<'_>, part_id: usize) -> Poll<Result<()>> {
        let needs_reload = matches!(
            self.build_partitions.get(part_id),
            Some(BuildPartition::Spilled { .. })
        );
        if !needs_reload {
            return Poll::Ready(Ok(()));
        }

        // Kick off reload if needed
        if self.pending_reload_partition.is_none() {
            if let Some(BuildPartition::Spilled { spill_file, .. }) = self.build_partitions.get_mut(part_id) {
                let spill_file = spill_file.take().ok_or_else(|| internal_datafusion_err!("spill file already consumed for this partition"))?;
                let stream = self.spill_manager.read_spill_as_stream(spill_file)?;
                self.pending_reload_stream = Some(stream);
                self.pending_reload_batches.clear();
                self.pending_reload_partition = Some(part_id);
            }
        }

        // Drive stream forward
        if self.pending_reload_partition == Some(part_id) {
            if let Some(stream) = self.pending_reload_stream.as_mut() {
                match stream.poll_next_unpin(cx) {
                    Poll::Ready(Some(Ok(batch))) => {
                        self.pending_reload_batches.push(batch);
                        return Poll::Pending;
                    }
                    Poll::Ready(Some(Err(e))) => return Poll::Ready(Err(e)),
                    Poll::Ready(None) => {
                        // Concatenate
                        let first_schema = self.pending_reload_batches.get(0)
                            .ok_or_else(|| internal_datafusion_err!("empty spilled partition"))?
                            .schema();
                        let concatenated = concat_batches(&first_schema, self.pending_reload_batches.as_slice())
                            .map_err(DataFusionError::from)?;

                        println!("Reloaded spilled build partition {} for probing (rows={})", part_id, concatenated.num_rows());

                        // Recompute values and hashmap
                        let mut values: Vec<ArrayRef> = Vec::with_capacity(self.on_left.len());
                        for c in &self.on_left {
                            values.push(c.evaluate(&concatenated)?.into_array(concatenated.num_rows())?);
                        }

                        let mut hash_map: Box<dyn JoinHashMapType> = Box::new(
                            crate::joins::join_hash_map::JoinHashMapU32::with_capacity(concatenated.num_rows()),
                        );
                        self.hashes_buffer.clear();
                        self.hashes_buffer.resize(concatenated.num_rows(), 0);
                        crate::joins::utils::update_hash(
                            &self.on_left,
                            &concatenated,
                            &mut *hash_map,
                            0,
                            &self.random_state,
                            &mut self.hashes_buffer,
                            0,
                            true,
                        )?;

                        let new_reservation = MemoryConsumer::new("partition_reload").with_can_spill(true).register(&self.runtime_env.memory_pool);

                        self.build_partitions[part_id] = BuildPartition::InMemory {
                            hash_map,
                            batch: concatenated,
                            values,
                            reservation: new_reservation,
                        };

                        self.pending_reload_stream = None;
                        self.pending_reload_batches.clear();
                        self.pending_reload_partition = None;
                        return Poll::Ready(Ok(()));
                    }
                    Poll::Pending => return Poll::Pending,
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
        spill_metrics: SpillMetrics,
        column_indices: Vec<ColumnIndex>,
        null_equality: NullEquality,
        batch_size: usize,
        num_partitions: usize,
        memory_threshold: usize,
        memory_reservation: MemoryReservation,
        runtime_env: Arc<RuntimeEnv>,
    ) -> Result<Self> {
        let spill_manager = SpillManager::new(
            runtime_env.clone(),
            spill_metrics,
            schema.clone(),
        );

        println!(
            "PartitionedHashJoinStream created: partition={}, num_partitions={}, memory_threshold={} bytes",
            partition, num_partitions, memory_threshold
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
            memory_threshold,
            state: PartitionedHashJoinState::PartitionBuildSide,
            build_partitions: Vec::new(),
            probe_partitions: Vec::new(),
            current_partition: None,
            spill_manager,
            memory_reservation,
            runtime_env,
            hashes_buffer: Vec::new(),
            right_side_ordered: false,
            bounds_accumulator: None,
            current_probe_batch: None,
            current_probe_values: vec![],
            current_probe_hashes: vec![],
            current_offset: (0, None),
            matched_build_rows_per_partition: Vec::new(),
            unmatched_partition: 0,
            unmatched_left_indices_cache: None,
            unmatched_right_indices_cache: None,
            unmatched_offset: 0,
            probes_buffered: false,
            probe_batch_positions: vec![],
            pending_reload_stream: None,
            pending_reload_batches: Vec::new(),
            pending_reload_partition: None,
        })
    }

    /// Buffer the entire probe side stream into per-partition batches.
    /// Returns Pending until the right stream is fully consumed.
    fn buffer_probe_side(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<Result<()>> {
        if self.probe_partitions.is_empty() {
            self.probe_partitions = (0..self.num_partitions)
                .map(|_| ProbePartition {
                    batches: Vec::new(),
                    values: Vec::new(),
                    hashes: Vec::new(),
                })
                .collect();
        }
        loop {
            match self.right.poll_next_unpin(cx) {
                Poll::Ready(Some(Ok(batch))) => {
                    // Compute ON values for the full batch
                    let mut keys_values: Vec<ArrayRef> = Vec::with_capacity(self.on_right.len());
                    for c in &self.on_right {
                        let v = c.evaluate(&batch)?.into_array(batch.num_rows())?;
                        keys_values.push(v);
                    }
                    let mut hashes = vec![0u64; batch.num_rows()];
                    create_hashes(&keys_values, &self.random_state, &mut hashes)?;

                    // For each partition, select rows and push filtered batch
                    for part_id in 0..self.num_partitions {
                        let indices: Vec<u32> = hashes
                            .iter()
                            .enumerate()
                            .filter_map(|(i, &h)| ((h as usize) % self.num_partitions == part_id).then_some(i as u32))
                            .collect();
                        if indices.is_empty() {
                            continue;
                        }
                        let indices_arr: UInt32Array = indices.clone().into();
                        let mut filtered_columns: Vec<ArrayRef> = Vec::with_capacity(batch.num_columns());
                        for col in batch.columns() {
                            filtered_columns.push(take(col, &indices_arr, None).map_err(DataFusionError::from)?);
                        }
                        let filtered_batch = RecordBatch::try_new(batch.schema(), filtered_columns)
                            .map_err(DataFusionError::from)?;

                        // Filtered ON values for this partition's batch
                        let mut filtered_on_values: Vec<ArrayRef> = Vec::with_capacity(self.on_right.len());
                        for c in &self.on_right {
                            let v = c.evaluate(&filtered_batch)?.into_array(filtered_batch.num_rows())?;
                            filtered_on_values.push(v);
                        }
                        let filtered_hashes: Vec<u64> = indices
                            .iter()
                            .map(|&i| hashes[i as usize])
                            .collect();

                        self.probe_partitions[part_id].batches.push(filtered_batch);
                        self.probe_partitions[part_id].values.push(filtered_on_values);
                        self.probe_partitions[part_id].hashes.push(filtered_hashes);
                    }
                }
                Poll::Ready(Some(Err(e))) => return Poll::Ready(Err(e)),
                Poll::Ready(None) => {
                    // Finished buffering
                    self.probes_buffered = true;
                    self.probe_batch_positions = vec![0; self.num_partitions];
                    println!(
                        "Buffered probe side: per-partition batch counts = {:?}",
                        self.probe_partitions.iter().map(|p| p.batches.len()).collect::<Vec<_>>()
                    );
                    return Poll::Ready(Ok(()));
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }

    /// Partition build-side data into multiple partitions
    fn partition_build_side(
        &mut self,
        build_data: Arc<JoinLeftData>,
    ) -> Result<StatefulStreamResult<Option<RecordBatch>>> {
        println!("Partitioning build side data into {} partitions", self.num_partitions);
        // Initialize partitions
        self.build_partitions = Vec::with_capacity(self.num_partitions);
        // Initialize per-partition matched rows bitmaps
        self.matched_build_rows_per_partition = Vec::with_capacity(self.num_partitions);
        
        // Extract build-side data
        let batch = build_data.batch();
        let values = build_data.values();
        
        // Compute hash values for all rows in the build-side batch
        let mut hashes = vec![0u64; batch.num_rows()];
        create_hashes(values, &self.random_state, &mut hashes)?;
        
        // Partition the data based on hash values
        let mut partition_batches: Vec<Vec<usize>> = vec![Vec::new(); self.num_partitions];
        
        for (row_idx, &hash) in hashes.iter().enumerate() {
            let partition_id = (hash as usize) % self.num_partitions;
            partition_batches[partition_id].push(row_idx);
        }
        
        // Create partitions; spill when memory_threshold is exceeded
        for partition_id in 0..self.num_partitions {
            let row_indices = &partition_batches[partition_id];
            if row_indices.is_empty() {
                // Empty partition - create empty hash map
                let empty_hash_map: Box<dyn JoinHashMapType> = 
                    Box::new(crate::joins::join_hash_map::JoinHashMapU32::with_capacity(0));
                let empty_batch = batch.slice(0, 0);
                let empty_values: Vec<ArrayRef> = values.iter().map(|arr| arr.slice(0, 0)).collect();
                
                // Initialize empty matched rows bitmap for this partition
                let matched_bitmap = BooleanBufferBuilder::new(0);
                self.matched_build_rows_per_partition.push(matched_bitmap);
                
                self.build_partitions.push(BuildPartition::InMemory {
                    hash_map: empty_hash_map,
                    batch: empty_batch,
                    values: empty_values,
                    reservation: MemoryConsumer::new("empty_partition").with_can_spill(true).register(&self.runtime_env.memory_pool),
                });
                continue;
            }
            
            // Create batch slice for this partition
            let partition_batch = self.take_rows(batch, row_indices)?;
            let partition_values: Vec<ArrayRef> = values.iter()
                .map(|arr| self.take_rows_from_array(arr, row_indices))
                .collect::<Result<Vec<_>>>()?;

            // Estimate memory for this partition
            let estimated_size = partition_batch.get_array_memory_size()
                + partition_values
                    .iter()
                    .map(|a| a.get_array_memory_size())
                    .sum::<usize>();

            // Decide spilling using global reservation (per DF best practice)
            let mut will_spill = false;
            match self.memory_reservation.try_grow(estimated_size) {
                Ok(_) => {
                    if self.memory_reservation.size() > self.memory_threshold {
                        // Exceeds threshold: roll back and spill
                        let _ = self.memory_reservation.try_shrink(estimated_size);
                        will_spill = true;
                    }
                }
                Err(_) => {
                    will_spill = true;
                }
            }

            if will_spill && self.runtime_env.disk_manager.tmp_files_enabled() {
                println!(
                    "Spilling build partition {} (rows={}) due to memory threshold (threshold={} bytes, current={})",
                    partition_id,
                    row_indices.len(),
                    self.memory_threshold,
                    self.memory_reservation.size()
                );
                // Spill this partition to disk and do not keep it in memory
                let spill_file = self
                    .spill_manager
                    .spill_record_batch_and_finish(&[partition_batch.clone()], "hash_join_build_partition")?
                    .ok_or_else(|| internal_datafusion_err!("expected spill file"))?;

                // Initialize matched rows bitmap for this partition
                let mut matched_bitmap = BooleanBufferBuilder::new(row_indices.len());
                matched_bitmap.append_n(row_indices.len(), false);
                self.matched_build_rows_per_partition.push(matched_bitmap);

                // Per-partition reservation kept as zero-sized placeholder
                let reservation = MemoryConsumer::new("partition_spilled").with_can_spill(true).register(&self.runtime_env.memory_pool);

                self.build_partitions.push(BuildPartition::Spilled {
                    spill_file: Some(spill_file),
                    reservation,
                });
                continue;
            }
            
            // Create hash map for this partition
            let partition_hash_map: Box<dyn JoinHashMapType> = 
                Box::new(crate::joins::join_hash_map::JoinHashMapU32::with_capacity(row_indices.len()));
            
            // Build the hash map for this partition using existing utilities
            let mut partition_hash_map = partition_hash_map;
            self.hashes_buffer.clear();
            self.hashes_buffer.resize(partition_batch.num_rows(), 0);
            crate::joins::utils::update_hash(
                &self.on_left,
                &partition_batch,
                &mut *partition_hash_map,
                0,
                &self.random_state,
                &mut self.hashes_buffer,
                0,
                true,
            )?;

            println!(
                "Built in-memory hash map for partition {} (rows={})",
                partition_id,
                row_indices.len()
            );
            
            // Initialize matched rows bitmap for this partition
            let mut matched_bitmap = BooleanBufferBuilder::new(row_indices.len());
            matched_bitmap.append_n(row_indices.len(), false);
            self.matched_build_rows_per_partition.push(matched_bitmap);
            
            // Per-partition reservation: zero-sized placeholder; global reservation tracks memory
            let reservation = MemoryConsumer::new("partition_memory").with_can_spill(true).register(&self.runtime_env.memory_pool);

            self.build_partitions.push(BuildPartition::InMemory {
                hash_map: partition_hash_map,
                batch: partition_batch,
                values: partition_values,
                reservation,
            });
        }
        
        // Start processing the first partition
        println!(
            "Partitioning complete. Created {} partitions. Starting to process partition 0",
            self.build_partitions.len()
        );
        
        self.state = PartitionedHashJoinState::ProcessPartition(ProcessPartitionState {
            partition_id: 0,
            total_partitions: self.num_partitions,
            is_last_partition: self.num_partitions == 1,
        });
        
        Ok(StatefulStreamResult::Continue)
    }
    
    /// Take specific rows from a RecordBatch
    fn take_rows(&self, batch: &RecordBatch, indices: &[usize]) -> Result<RecordBatch> {
        use arrow::compute::take;
        use arrow::array::UInt32Array;
        
        let indices_array = UInt32Array::from(
            indices.iter().map(|&i| i as u32).collect::<Vec<_>>()
        );
        
        let columns: Result<Vec<_>, DataFusionError> = batch.columns().iter()
            .map(|col| take(col, &indices_array, None).map_err(|e| e.into()))
            .collect();
        
        Ok(RecordBatch::try_new(batch.schema(), columns?)?)
    }
    
    /// Take specific rows from an ArrayRef
    fn take_rows_from_array(&self, array: &ArrayRef, indices: &[usize]) -> Result<ArrayRef> {
        use arrow::compute::take;
        use arrow::array::UInt32Array;
        
        let indices_array = UInt32Array::from(
            indices.iter().map(|&i| i as u32).collect::<Vec<_>>()
        );
        
        Ok(take(array, &indices_array, None).map_err(DataFusionError::from)?)
    }

    /// Release resources associated with a finished partition when safe to do so.
    /// Only releases memory eagerly when we don't need unmatched rows in the final phase.
    fn release_partition_resources(&mut self, partition_id: usize) {
        if need_produce_result_in_final(self.join_type) {
            return;
        }

        if partition_id >= self.build_partitions.len() {
            return;
        }

        // Take ownership of the old partition to drop heavy resources
        let placeholder_reservation = MemoryConsumer::new("partition_released_placeholder")
            .with_can_spill(true)
            .register(&self.runtime_env.memory_pool);
        let old_partition = mem::replace(
            &mut self.build_partitions[partition_id],
            BuildPartition::Spilled {
                spill_file: None,
                reservation: placeholder_reservation,
            },
        );

        match old_partition {
            BuildPartition::InMemory { batch, values, reservation, .. } => {
                // Estimate memory held by this partition and shrink global reservation
                let mut estimated_size = batch.get_array_memory_size();
                estimated_size += values.iter().map(|a| a.get_array_memory_size()).sum::<usize>();
                let _ = self.memory_reservation.try_shrink(estimated_size);

                // Replace with an empty in-memory partition to keep indexing stable
                let empty_batch = RecordBatch::new_empty(batch.schema());
                let empty_values: Vec<ArrayRef> = self
                    .on_left
                    .iter()
                    .filter_map(|expr| expr.evaluate(&empty_batch).ok())
                    .filter_map(|v| v.into_array(empty_batch.num_rows()).ok())
                    .collect();
                let empty_hash_map: Box<dyn JoinHashMapType> = Box::new(
                    crate::joins::join_hash_map::JoinHashMapU32::with_capacity(0),
                );

                self.build_partitions[partition_id] = BuildPartition::InMemory {
                    hash_map: empty_hash_map,
                    batch: empty_batch,
                    values: empty_values,
                    reservation,
                };
            }
            BuildPartition::Spilled { reservation, .. } => {
                // Keep as empty spilled (no further action needed)
                self.build_partitions[partition_id] = BuildPartition::Spilled {
                    spill_file: None,
                    reservation,
                };
            }
        }
    }

    /// Process a specific partition
    fn process_partition(
        &mut self,
        cx: &mut Context<'_>,
        partition_state: &ProcessPartitionState,
    ) -> Poll<Result<StatefulStreamResult<Option<RecordBatch>>>> {
        // Guard against invalid partition ids (off-by-one protection)
        if partition_state.partition_id >= partition_state.total_partitions {
            self.state = PartitionedHashJoinState::HandleUnmatchedRows;
            return Poll::Ready(Ok(StatefulStreamResult::Continue));
        }
        println!(
            "Processing partition {} (total_partitions={}), build_partitions.len()={}",
            partition_state.partition_id,
            partition_state.total_partitions,
            self.build_partitions.len()
        );
        
        // Do not buffer probe side here; selection happens below depending on num_partitions

        // (Spill reload handled by ensure_build_partition_loaded earlier if needed)

        // (Build partition will be immutably borrowed later within a narrower scope)

        // Ensure the build partition is ready (reload if spilled) BEFORE any immutable borrows
        match self.ensure_build_partition_loaded(cx, partition_state.partition_id) {
            Poll::Ready(Ok(())) => {}
            Poll::Ready(Err(e)) => return Poll::Ready(Err(e)),
            Poll::Pending => return Poll::Pending,
        }

        // If only 1 partition, stream the probe side directly (simpler and correct across executor partitions)
        if self.num_partitions == 1 {
            if self.current_probe_batch.is_none() {
                match ready!(self.right.poll_next_unpin(cx)) {
                    Some(Ok(batch)) => {
                        // Compute hashes for the full batch
                        let mut keys_values: Vec<ArrayRef> = Vec::with_capacity(self.on_right.len());
                        for c in &self.on_right {
                            let v = c.evaluate(&batch)?.into_array(batch.num_rows())?;
                            keys_values.push(v);
                        }
                        let mut hashes = vec![0u64; batch.num_rows()];
                        create_hashes(&keys_values, &self.random_state, &mut hashes)?;

                        // No filtering needed when only one partition
                        self.current_probe_hashes = hashes;
                        self.current_probe_values = keys_values;
                        self.current_probe_batch = Some(batch);
                        self.current_offset = (0, None);

                        if let Some(pb) = self.current_probe_batch.as_ref() {
                            println!(
                                "[spill-join] Direct probe batch rows={} (partitions=1)",
                                pb.num_rows()
                            );
                            self.join_metrics.input_batches.add(1);
                            self.join_metrics.input_rows.add(pb.num_rows());
                        }
                    }
                    Some(Err(e)) => return Poll::Ready(Err(e)),
                    None => {
                        // No more probe data for this partition, release and advance
                        self.release_partition_resources(partition_state.partition_id);
                        if partition_state.is_last_partition {
                            self.state = PartitionedHashJoinState::HandleUnmatchedRows;
                        } else {
                            self.state = PartitionedHashJoinState::ProcessPartition(
                                ProcessPartitionState {
                                    partition_id: partition_state.partition_id + 1,
                                    total_partitions: partition_state.total_partitions,
                                    is_last_partition: partition_state.partition_id + 1
                                        == partition_state.total_partitions,
                                },
                            );
                        }
                        return Poll::Ready(Ok(StatefulStreamResult::Continue));
                    }
                }
            }
        } else {
            // For multiple inner partitions, buffer the probe side once and consume per partition
            if !self.probes_buffered {
                ready!(self.buffer_probe_side(cx))?;
            }
            if self.current_probe_batch.is_none() {
                let part_id = partition_state.partition_id;
                let pos = *self.probe_batch_positions.get(part_id).unwrap_or(&0);
                if let Some(probe_part) = self.probe_partitions.get(part_id) {
                    if pos < probe_part.batches.len() {
                        let filtered_batch = probe_part.batches[pos].clone();
                        let filtered_on_values = probe_part.values[pos].clone();
                        let filtered_hashes = probe_part.hashes[pos].clone();

                        self.current_probe_hashes = filtered_hashes;
                        self.current_probe_values = filtered_on_values;
                        self.current_probe_batch = Some(filtered_batch);
                        self.current_offset = (0, None);
                        self.probe_batch_positions[part_id] = pos + 1;
                    } else {
                        // No more probe data for this partition, release and advance
                        self.release_partition_resources(partition_state.partition_id);
                        if partition_state.is_last_partition {
                            self.state = PartitionedHashJoinState::HandleUnmatchedRows;
                        } else {
                            self.state = PartitionedHashJoinState::ProcessPartition(
                                ProcessPartitionState {
                                    partition_id: partition_state.partition_id + 1,
                                    total_partitions: partition_state.total_partitions,
                                    is_last_partition: partition_state.partition_id + 1
                                        == partition_state.total_partitions,
                                },
                            );
                        }
                        return Poll::Ready(Ok(StatefulStreamResult::Continue));
                    }
                }
            }
        }

        // At this point we have a current probe batch for this partition
        let (result, build_ids_to_mark, next_offset) = {
            let probe_batch = self
                .current_probe_batch
                .as_ref()
                .ok_or_else(|| internal_datafusion_err!("expected probe batch"))?;

            let (build_hashmap, build_batch, build_values) = match self
                .build_partitions
                .get(partition_state.partition_id)
            {
                Some(BuildPartition::InMemory {
                    hash_map,
                    batch,
                    values,
                    ..
                }) => (&**hash_map, batch, values as &Vec<ArrayRef>),
                _ => return Poll::Ready(internal_err!("Missing or invalid build partition")),
            };

            // Lookup against hash map with limit
            let (probe_indices, build_indices, next_offset) = build_hashmap
                .get_matched_indices_with_limit_offset(
                    &self.current_probe_hashes,
                    self.batch_size,
                    self.current_offset,
                );

            let build_indices: UInt64Array = build_indices.into();
            let probe_indices: UInt32Array = probe_indices.into();

            println!(
                "[spill-join] Candidates before equality: build_ids={}, probe_ids={}, build_rows={}, probe_rows={}",
                build_indices.len(),
                probe_indices.len(),
                build_batch.num_rows(),
                probe_batch.num_rows()
            );

            // Resolve hash collisions
            let (build_indices, probe_indices) = equal_rows_arr(
                &build_indices,
                &probe_indices,
                build_values,
                &self.current_probe_values,
                self.null_equality,
            )?;

            println!(
                "[spill-join] Matched after equality: {}",
                build_indices.len()
            );

            // Prepare ids for marking after we release borrows
            let build_ids_to_mark: Vec<u64> = build_indices.values().to_vec();

            // Build output batch (Left side is build)
            let result = build_batch_from_indices(
                &self.schema,
                build_batch,
                probe_batch,
                &build_indices,
                &probe_indices,
                &self.column_indices,
                JoinSide::Left,
            )?;

            (result, build_ids_to_mark, next_offset)
        };

        // Mark matched build-side rows for outer joins (use current partition's bitmap)
        if let Some(bitmap) = self.matched_build_rows_per_partition.get_mut(partition_state.partition_id) {
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
        }

        if result.num_rows() == 0 {
            println!(
                "[spill-join] Skipping empty batch emission (partition={})",
                partition_state.partition_id
            );
            return Poll::Ready(Ok(StatefulStreamResult::Continue));
        }
        self.join_metrics.output_batches.add(1);
        self.join_metrics.baseline.record_output(result.num_rows());
        println!(
            "[spill-join] Emitting batch: rows={} (partition={})",
            result.num_rows(),
            partition_state.partition_id
        );
        Poll::Ready(Ok(StatefulStreamResult::Ready(Some(result))))
    }

    /// Handle unmatched rows for outer joins (poll-based, non-blocking spill reload)
    fn handle_unmatched_rows(&mut self, cx: &mut Context<'_>) -> Poll<Result<StatefulStreamResult<Option<RecordBatch>>>> {
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
                    .ok_or_else(|| internal_datafusion_err!("failed to downcast left indices chunk"))?;
                let right_chunk = right_chunk_ref
                    .as_any()
                    .downcast_ref::<UInt32Array>()
                    .ok_or_else(|| internal_datafusion_err!("failed to downcast right indices chunk"))?;

                // Use current partition's build batch
                let partition = self
                    .build_partitions
                    .get(self.unmatched_partition)
                    .ok_or_else(|| internal_datafusion_err!("missing build partition during unmatched cached emission"))?;
                let build_batch = match partition {
                    BuildPartition::InMemory { batch, .. } => batch,
                    BuildPartition::Spilled { .. } => {
                        // Should not happen because we only cache after loading InMemory indices
                        return Poll::Ready(Ok(StatefulStreamResult::Continue));
                    }
                };

                let empty_right_batch = RecordBatch::new_empty(self.right.schema());
                println!(
                    "Emitting unmatched rows chunk: partition={}, offset={}, size={} (total={})",
                    self.unmatched_partition,
                    self.unmatched_offset,
                    to_emit,
                    total
                );

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
                    println!(
                        "Finished emitting unmatched rows for partition {}",
                        self.unmatched_partition
                    );
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
            let partition = self.build_partitions.get_mut(self.unmatched_partition)
                .ok_or_else(|| internal_datafusion_err!("missing build partition during unmatched processing"))?;

            match partition {
                BuildPartition::InMemory { batch: _batch, .. } => {
                    // Get unmatched indices for this partition using its bitmap
                    let (left_indices, right_indices) = if let Some(bitmap) = self.matched_build_rows_per_partition.get(self.unmatched_partition) {
                        get_final_indices_from_bit_map(
                            bitmap,
                            self.join_type,
                        )
                    } else {
                        // If no bitmap, skip this partition
                        self.unmatched_partition += 1;
                        return Poll::Ready(Ok(StatefulStreamResult::Continue));
                    };
                    
                    println!(
                        "Unmatched calculation for partition {} -> {} rows",
                        self.unmatched_partition,
                        left_indices.len()
                    );

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
                        let taken = spill_file.take().ok_or_else(|| internal_datafusion_err!("spill file already consumed for unmatched"))?;
                        let stream = self.spill_manager.read_spill_as_stream(taken)?;
                        self.pending_reload_stream = Some(stream);
                        self.pending_reload_batches.clear();
                        self.pending_reload_partition = Some(self.unmatched_partition);
                    }

                    if self.pending_reload_partition == Some(self.unmatched_partition) {
                        if let Some(stream) = self.pending_reload_stream.as_mut() {
                            match stream.poll_next_unpin(cx) {
                                Poll::Ready(Some(Ok(batch))) => {
                                    self.pending_reload_batches.push(batch);
                                    return Poll::Pending;
                                }
                                Poll::Ready(Some(Err(e))) => return Poll::Ready(Err(e)),
                                Poll::Ready(None) => {
                                    let first_schema = self.pending_reload_batches.get(0)
                                        .ok_or_else(|| internal_datafusion_err!("empty spilled partition for unmatched"))?
                                        .schema();
                                    let concatenated = concat_batches(&first_schema, self.pending_reload_batches.as_slice())
                                        .map_err(DataFusionError::from)?;

                                    println!(
                                        "Reloaded spilled build partition {} for unmatched rows (rows={})",
                                        self.unmatched_partition,
                                        concatenated.num_rows()
                                    );

                                    let new_reservation = MemoryConsumer::new("partition_reload_unmatched")
                                        .with_can_spill(true)
                                        .register(&self.runtime_env.memory_pool);
                                    let mut values: Vec<ArrayRef> = Vec::with_capacity(self.on_left.len());
                                    for c in &self.on_left {
                                        values.push(c.evaluate(&concatenated)?.into_array(concatenated.num_rows())?);
                                    }
                                    let hash_map: Box<dyn JoinHashMapType> = Box::new(
                                        crate::joins::join_hash_map::JoinHashMapU32::with_capacity(concatenated.num_rows()),
                                    );
                                    self.build_partitions[self.unmatched_partition] = BuildPartition::InMemory {
                                        hash_map,
                                        batch: concatenated,
                                        values,
                                        reservation: new_reservation,
                                    };
                                    println!(
                                        "Prepared spilled partition {} as InMemory for unmatched emission",
                                        self.unmatched_partition
                                    );

                                    // Clear pending
                                    self.pending_reload_stream = None;
                                    self.pending_reload_batches.clear();
                                    self.pending_reload_partition = None;

                                    // Continue; next iteration will handle InMemory branch
                                    return Poll::Ready(Ok(StatefulStreamResult::Continue));
                                }
                                Poll::Pending => return Poll::Pending,
                            }
                        }
                    }
                    Poll::Pending
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
                    match self.partition_build_side(left_data) {
                        Ok(StatefulStreamResult::Continue) => continue,
                        Ok(StatefulStreamResult::Ready(Some(batch))) => {
                            println!(
                                "[spill-join] poll_next yielding initial batch: rows={}",
                                batch.num_rows()
                            );
                            return Poll::Ready(Some(Ok(batch)));
                        }
                        Ok(StatefulStreamResult::Ready(None)) => return Poll::Ready(None),
                        Err(e) => return Poll::Ready(Some(Err(e))),
                    }
                }
                PartitionedHashJoinState::ProcessPartition(partition_state) => {
                    match self.process_partition(cx, &partition_state) {
                        Poll::Ready(Ok(StatefulStreamResult::Ready(Some(batch)))) => {
                            println!(
                                "[spill-join] poll_next yielding process batch: rows={} (state partition={})",
                                batch.num_rows(), partition_state.partition_id
                            );
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
                            println!(
                                "[spill-join] poll_next yielding unmatched batch: rows={}",
                                batch.num_rows()
                            );
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
