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

//! Stream implementation for Hash Join
//!
//! This module implements [`HashJoinStream`], the streaming engine for
//! [`super::HashJoinExec`]. See comments in [`HashJoinStream`] for more details.

use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Instant;

use crate::joins::utils::OnceFut;
use crate::{
    joins::utils::{BuildProbeJoinMetrics, ColumnIndex, JoinFilter},
    ExecutionPlan, RecordBatchStream, SendableRecordBatchStream, SpillManager,
};

use crate::empty::EmptyExec;
use crate::joins::grace_hash_join::exec::{
    partition_and_spill, PartitionIndex, SpillChunk,
};
use crate::joins::{HashJoinExec, PartitionMode};
use crate::spill::get_record_batch_memory_size;
use crate::test::TestMemoryExec;
use ahash::RandomState;
use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use datafusion_common::{JoinType, NullEquality, Result};
use datafusion_execution::memory_pool::{
    human_readable_size, MemoryConsumer, MemoryReservation,
};
use datafusion_execution::TaskContext;
use datafusion_physical_expr::PhysicalExprRef;
use futures::stream::FuturesUnordered;
use futures::{ready, Stream, StreamExt};
use log::{debug, info};
use parking_lot::Mutex;
#[cfg(target_os = "linux")]
use std::sync::OnceLock;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

/// Maximum number of partitions we allow after recursive repartitioning to
/// prevent explosive fan-out.
const MAX_REPARTITION_PARTITIONS: usize = 256;
/// Upper bound for concurrent spill chunk read-ahead when loading partitions.
/// Keep this modest to avoid competing with the main partition budget.
const SPILL_READAHEAD_BYTES: usize = 64 * 1024 * 1024;
/// Below this size we avoid further repartitioning to keep file counts under control.
const MIN_REPARTITION_BYTES: usize = 16 * 1024 * 1024;

fn global_join_semaphore() -> &'static Arc<Semaphore> {
    static SEM: OnceLock<Arc<Semaphore>> = OnceLock::new();
    SEM.get_or_init(|| Arc::new(Semaphore::new(1)))
}

/// Prefetch is disabled to avoid overlapping memory use with the active partition.
fn prefetch_cap_bytes(_current_limit: usize) -> usize {
    0
}

#[cfg(target_os = "linux")]
fn current_rss_bytes() -> Option<u64> {
    let statm = std::fs::read_to_string("/proc/self/statm").ok()?;
    let mut parts = statm.split_whitespace();
    let _size_pages = parts.next()?;
    let resident_pages = parts.next()?.parse::<u64>().ok()?;
    let page_size = unsafe { libc_sysconf_page_size() };
    if page_size <= 0 {
        return None;
    }
    Some(resident_pages.saturating_mul(page_size as u64))
}

#[cfg(not(target_os = "linux"))]
fn current_rss_bytes() -> Option<u64> {
    None
}

#[cfg(target_os = "linux")]
unsafe fn libc_sysconf_page_size() -> i64 {
    #[link(name = "c")]
    extern "C" {
        fn sysconf(name: i32) -> i64;
    }
    const _SC_PAGESIZE: i32 = 30; // POSIX _SC_PAGESIZE
    sysconf(_SC_PAGESIZE)
}

enum GraceJoinState {
    /// Waiting for the partitioning phase (Phase 1) to finish
    WaitPartitioning,

    /// Currently joining partition(s)
    JoinPartition {
        work_queue: VecDeque<PartitionWorkItem>,
        current_work: Option<PartitionWorkItem>,
        current_stream: Option<SendableRecordBatchStream>,
        left_fut: Option<OnceFut<LoadedPartitionBatches>>,
        right_fut: Option<OnceFut<LoadedPartitionBatches>>,
        base_reservation: Arc<Mutex<MemoryReservation>>,
        prefetch_reservation: Arc<Mutex<MemoryReservation>>,
        /// Bytes reserved in the memory pool for the current partition's
        /// loaded left batches
        left_bytes: Arc<Mutex<usize>>,
        /// Bytes reserved in the memory pool for the current partition's
        /// loaded right batches
        right_bytes: Arc<Mutex<usize>>,
        /// Reservation used to track memory for the current partition's loaded batches
        reservation: Arc<Mutex<MemoryReservation>>,
        /// Permit to limit concurrent in-memory joins across tasks
        join_permit: Option<Arc<OwnedSemaphorePermit>>,
        join_permit_fut: Option<OnceFut<OwnedSemaphorePermit>>,
        current_join_start: Option<Instant>,
        repartition_fut: Option<OnceFut<Vec<PartitionWorkItem>>>,
        /// Prefetch for the next partition (at most one in-flight)
        prefetch: Option<PrefetchState>,
        /// Last partition we logged a prefetch skip for, to avoid log spam
        last_prefetch_skip: Option<(usize, usize)>,
    },

    Done,
}

#[derive(Clone)]
struct PartitionWorkItem {
    partition_id: usize,
    pass: usize,
    partition_count: usize,
    left: PartitionIndex,
    right: PartitionIndex,
}

impl PartitionWorkItem {
    fn total_bytes(&self) -> usize {
        self.left.total_bytes() + self.right.total_bytes()
    }
}

#[derive(Clone, Default)]
struct PartitionStatsSummary {
    count: usize,
    min_bytes: usize,
    median_bytes: usize,
    max_bytes: usize,
}

impl PartitionStatsSummary {
    fn from_samples(samples: &[usize]) -> Self {
        if samples.is_empty() {
            return Self::default();
        }
        let mut sorted = samples.to_vec();
        sorted.sort_unstable();
        let median = sorted[sorted.len() / 2];
        let min_bytes = *sorted.first().unwrap();
        let max_bytes = *sorted.last().unwrap();
        Self {
            count: samples.len(),
            min_bytes,
            median_bytes: median,
            max_bytes,
        }
    }

    fn is_empty(&self) -> bool {
        self.count == 0
    }

    fn partitions(&self) -> usize {
        self.count
    }

    fn min_bytes(&self) -> usize {
        self.min_bytes
    }

    fn median_bytes(&self) -> usize {
        self.median_bytes
    }

    fn max_bytes(&self) -> usize {
        self.max_bytes
    }

    fn max_partition_bytes(&self) -> Option<usize> {
        if self.is_empty() {
            None
        } else {
            Some(self.max_bytes)
        }
    }
}

#[derive(Debug)]
struct AdaptivePartitionBudget {
    base_budget: usize,
    preferred_cap: usize,
    absolute_cap: usize,
    observed_max: usize,
    current_limit: usize,
    active_partitions: usize,
}

impl AdaptivePartitionBudget {
    fn new(
        base_budget: usize,
        preferred_cap: usize,
        absolute_cap: usize,
        active_partitions: usize,
    ) -> Self {
        let base_budget = base_budget.max(1);
        let preferred_cap = preferred_cap.max(base_budget);
        let absolute_cap = absolute_cap.max(preferred_cap);
        let mut budget = Self {
            base_budget,
            preferred_cap,
            absolute_cap,
            observed_max: base_budget,
            current_limit: 1,
            active_partitions: active_partitions.max(1),
        };
        budget.recompute_limit();
        budget
    }

    fn observe(&mut self, bytes: usize) {
        if bytes > self.observed_max {
            self.observed_max = bytes;
        }
    }

    fn update_active_partitions(&mut self, active_partitions: usize) {
        let active_partitions = active_partitions.max(1);
        if self.active_partitions != active_partitions {
            self.active_partitions = active_partitions;
            self.recompute_limit();
        }
    }

    fn current_limit(&self) -> usize {
        self.current_limit
    }

    fn current_concurrency(&self) -> usize {
        self.active_partitions
    }

    fn prime_with_stats(&mut self, stats: &PartitionStatsSummary) {
        if let Some(max_bytes) = stats.max_partition_bytes() {
            self.observe(max_bytes);
            self.recompute_limit();
        }
    }

    fn base_for_current_concurrency(&self) -> usize {
        let concurrent_share = self.preferred_cap / self.active_partitions;
        self.base_budget.max(concurrent_share).max(1)
    }

    fn recompute_limit(&mut self) -> usize {
        if self.active_partitions <= 1 {
            // When processing sequentially (one active partition), allow using the full absolute budget.
            // This prevents artificial constraints from "preferred" caps or previous observations
            // when we are in a fallback/recovery mode.
            self.current_limit = self.absolute_cap.max(1);
        } else {
            let needed = self.base_for_current_concurrency().max(self.observed_max);
            let limit = needed.min(self.preferred_cap);
            self.current_limit = limit.max(1);
        }
        self.current_limit
    }

    fn ensure_fits(&mut self, bytes: usize) -> AdaptiveBudgetOutcome {
        let previous = self.current_limit;
        self.observe(bytes);
        let limit = self.recompute_limit();
        if bytes <= limit {
            if limit > previous {
                AdaptiveBudgetOutcome::Raised {
                    previous,
                    new_limit: limit,
                }
            } else {
                AdaptiveBudgetOutcome::Fits
            }
        } else {
            AdaptiveBudgetOutcome::CannotFit { limit }
        }
    }

    /// Try to lower concurrency to a single in-flight partition so we can
    /// safely increase the per-partition budget. Returns the previous
    /// concurrency and limits when serialization helps, otherwise leaves the
    /// budget unchanged.
    fn serialize_if_helpful(&mut self, bytes: usize) -> Option<(usize, usize, usize)> {
        if self.active_partitions <= 1 {
            return None;
        }
        let previous_concurrency = self.active_partitions;
        let previous_limit = self.current_limit;
        self.active_partitions = 1;
        let new_limit = self.recompute_limit();
        if bytes <= new_limit {
            Some((previous_concurrency, previous_limit, new_limit))
        } else {
            // Restore previous settings if serialization didn't help
            self.active_partitions = previous_concurrency;
            self.recompute_limit();
            None
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum AdaptiveBudgetOutcome {
    Fits,
    Raised { previous: usize, new_limit: usize },
    CannotFit { limit: usize },
}

fn compute_repartition_count(
    partition_count: usize,
    input_size: usize,
    target_size: usize,
    max_partition_count: usize,
    allow_slack: bool,
    compute_soft_cap_bytes: usize,
) -> usize {
    let target_size = target_size.max(1);
    let effective_target = target_size.min(compute_soft_cap_bytes.max(1));
    // If the current partition is already within the target, keep the same fan-out
    // to avoid pointless recursive repartitioning.
    if allow_slack && input_size <= effective_target {
        return partition_count.min(max_partition_count);
    }
    // Allow some slack: if we're within 2x of the target, keep the fan-out to avoid
    // a costly extra pass when the current size will still fit in memory.
    if allow_slack && input_size <= effective_target.saturating_mul(2) {
        return partition_count.min(max_partition_count);
    }

    let fan_out = (input_size + effective_target - 1) / effective_target;
    let fan_out = fan_out.max(2);

    let base = partition_count.min(max_partition_count);
    let desired = base.saturating_mul(fan_out).max(base + 1);
    desired.min(max_partition_count)
}

fn build_repartition_future(
    work: PartitionWorkItem,
    random_state: RandomState,
    on_pairs: Vec<(PhysicalExprRef, PhysicalExprRef)>,
    spill_left: Arc<SpillManager>,
    spill_right: Arc<SpillManager>,
    join_metrics: Arc<BuildProbeJoinMetrics>,
    context: Arc<TaskContext>,
    left_schema: SchemaRef,
    right_schema: SchemaRef,
    partition_batch_size: usize,
    target_size: usize,
    partition_write_buffer_bytes: usize,
    max_partition_count: usize,
    compute_soft_cap_bytes: usize,
) -> OnceFut<Vec<PartitionWorkItem>> {
    OnceFut::new(async move {
        let PartitionWorkItem {
            partition_id,
            pass,
            partition_count,
            left,
            right,
        } = work;

        let input_size = left.total_bytes() + right.total_bytes();
        let target_size = target_size.max(1);
        let new_partition_count = compute_repartition_count(
            partition_count,
            input_size,
            target_size,
            max_partition_count,
            true,
            compute_soft_cap_bytes,
        );
        if new_partition_count == max_partition_count
            && partition_count < max_partition_count
        {
            debug!(
                "Grace hash join partition {} capped repartition fan-out at {} (pass {}, input {}, target {})",
                partition_id,
                max_partition_count,
                pass,
                human_readable_size(input_size),
                human_readable_size(target_size),
            );
        }

        let left_stream: SendableRecordBatchStream =
            Box::pin(SpilledPartitionStream::new(
                Arc::clone(&left_schema),
                Arc::clone(&spill_left),
                left.chunks,
            ));
        let right_stream: SendableRecordBatchStream =
            Box::pin(SpilledPartitionStream::new(
                Arc::clone(&right_schema),
                Arc::clone(&spill_right),
                right.chunks,
            ));

        let mut left_reservation = MemoryConsumer::new(format!(
            "GraceHashJoinRepartitionLeft[{partition_id}-{pass}]"
        ))
        .with_can_spill(true)
        .register(context.memory_pool());

        let mut right_reservation = MemoryConsumer::new(format!(
            "GraceHashJoinRepartitionRight[{partition_id}-{pass}]"
        ))
        .with_can_spill(true)
        .register(context.memory_pool());

        let (new_left, new_right) = partition_and_spill(
            random_state,
            on_pairs,
            left_stream,
            right_stream,
            join_metrics,
            false,
            new_partition_count,
            Arc::clone(&spill_left),
            Arc::clone(&spill_right),
            partition_id,
            &mut left_reservation,
            &mut right_reservation,
            partition_batch_size,
            partition_write_buffer_bytes,
        )
        .await?;

        let items = new_left
            .into_iter()
            .zip(new_right.into_iter())
            .map(|(left_idx, right_idx)| PartitionWorkItem {
                partition_id,
                pass: pass + 1,
                partition_count: new_partition_count,
                left: left_idx,
                right: right_idx,
            })
            .collect();

        Ok(items)
    })
}

pub struct GraceHashJoinStream {
    schema: SchemaRef,
    left_input_schema: SchemaRef,
    right_input_schema: SchemaRef,
    spill_fut: OnceFut<SpillFut>,
    partition: usize,
    spill_left: Arc<SpillManager>,
    spill_right: Arc<SpillManager>,
    on_left: Vec<PhysicalExprRef>,
    on_right: Vec<PhysicalExprRef>,
    projection: Option<Vec<usize>>,
    filter: Option<JoinFilter>,
    join_type: JoinType,
    column_indices: Vec<ColumnIndex>,
    join_metrics: Arc<BuildProbeJoinMetrics>,
    context: Arc<TaskContext>,
    /// Lazily registered reservation tracking in-memory buffers used by the join stream
    reservation: OnceLock<Arc<Mutex<MemoryReservation>>>,
    /// Lazily registered reservation dedicated to prefetching the next partition
    prefetch_reservation: OnceLock<Arc<Mutex<MemoryReservation>>>,
    random_state: RandomState,
    partition_batch_size: usize,
    adaptive_budget: AdaptivePartitionBudget,
    max_partition_passes: usize,
    compute_soft_cap_bytes: usize,
    repartition_enabled: bool,
    partition_stats: PartitionStatsSummary,
    state: GraceJoinState,
}

#[derive(Debug, Clone)]
pub struct SpillFut {
    left: Vec<PartitionIndex>,
    right: Vec<PartitionIndex>,
}
impl SpillFut {
    pub(crate) fn new(
        _partition: usize,
        left: Vec<PartitionIndex>,
        right: Vec<PartitionIndex>,
    ) -> Self {
        SpillFut { left, right }
    }
}

#[derive(Debug, Clone)]
struct LoadedPartitionBatches {
    batches: Vec<RecordBatch>,
    total_bytes: usize,
}

#[derive(Clone)]
struct PrefetchState {
    work: PartitionWorkItem,
    left_fut: OnceFut<LoadedPartitionBatches>,
    right_fut: OnceFut<LoadedPartitionBatches>,
    left_bytes: Arc<Mutex<usize>>,
    right_bytes: Arc<Mutex<usize>>,
    reservation: Arc<Mutex<MemoryReservation>>,
}

impl RecordBatchStream for GraceHashJoinStream {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}

impl GraceHashJoinStream {
    /// Ensure the main join reservation is registered only when the join phase starts.
    fn main_reservation(&self) -> Arc<Mutex<MemoryReservation>> {
        Arc::clone(self.reservation.get_or_init(|| {
            let reservation =
                MemoryConsumer::new(format!("GraceHashJoinStream[{}]", self.partition))
                    .with_can_spill(true)
                    .register(self.context.memory_pool());
            Arc::new(Mutex::new(reservation))
        }))
    }

    /// Lazily register a reservation for prefetching; avoids bloating spillable consumer count while partitioning.
    fn ensure_prefetch_reservation(&self) -> Arc<Mutex<MemoryReservation>> {
        Arc::clone(self.prefetch_reservation.get_or_init(|| {
            let reservation =
                MemoryConsumer::new(format!("GraceHashJoinPrefetch[{}]", self.partition))
                    .with_can_spill(true)
                    .register(self.context.memory_pool());
            Arc::new(Mutex::new(reservation))
        }))
    }

    pub fn new(
        schema: SchemaRef,
        left_input_schema: SchemaRef,
        right_input_schema: SchemaRef,
        spill_fut: OnceFut<SpillFut>,
        partition: usize,
        spill_left: Arc<SpillManager>,
        spill_right: Arc<SpillManager>,
        on_left: Vec<PhysicalExprRef>,
        on_right: Vec<PhysicalExprRef>,
        projection: Option<Vec<usize>>,
        filter: Option<JoinFilter>,
        join_type: JoinType,
        column_indices: Vec<ColumnIndex>,
        join_metrics: Arc<BuildProbeJoinMetrics>,
        context: Arc<TaskContext>,
        random_state: RandomState,
        partition_batch_size: usize,
        base_partition_budget_bytes: usize,
        preferred_partition_budget_bytes: usize,
        absolute_partition_cap_bytes: usize,
        max_partition_passes: usize,
        compute_soft_cap_bytes: usize,
    ) -> Self {
        let adaptive_budget = AdaptivePartitionBudget::new(
            base_partition_budget_bytes,
            preferred_partition_budget_bytes,
            absolute_partition_cap_bytes,
            1,
        );
        Self {
            schema,
            left_input_schema,
            right_input_schema,
            spill_fut,
            partition,
            spill_left,
            spill_right,
            on_left,
            on_right,
            projection,
            filter,
            join_type,
            column_indices,
            join_metrics,
            context,
            reservation: OnceLock::new(),
            prefetch_reservation: OnceLock::new(),
            random_state,
            partition_batch_size,
            adaptive_budget,
            max_partition_passes,
            compute_soft_cap_bytes,
            repartition_enabled: true,
            partition_stats: PartitionStatsSummary::default(),
            state: GraceJoinState::WaitPartitioning,
        }
    }

    /// Core state machine logic (poll implementation)
    fn poll_next_impl(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<RecordBatch>>> {
        let join_time_metric = self.join_metrics.join_time.clone();
        loop {
            match &mut self.state {
                GraceJoinState::WaitPartitioning => {
                    let shared = ready!(self.spill_fut.get_shared(cx))?;
                    let partition_count = shared.left.len();
                    let mut partition_samples = Vec::with_capacity(partition_count);
                    let mut work_queue = VecDeque::with_capacity(partition_count);
                    for (partition_id, (left, right)) in shared
                        .left
                        .iter()
                        .cloned()
                        .zip(shared.right.iter().cloned())
                        .enumerate()
                    {
                        partition_samples.push(left.total_bytes() + right.total_bytes());
                        work_queue.push_back(PartitionWorkItem {
                            partition_id,
                            pass: 0,
                            partition_count,
                            left,
                            right,
                        });
                    }
                    let stats = PartitionStatsSummary::from_samples(&partition_samples);
                    if !stats.is_empty() {
                        debug!(
                            "Grace hash join partition stats: count={}, min={}, median={}, max={}",
                            stats.partitions(),
                            human_readable_size(stats.min_bytes()),
                            human_readable_size(stats.median_bytes()),
                            human_readable_size(stats.max_bytes())
                        );
                    }
                    self.partition_stats = stats.clone();
                    self.adaptive_budget.prime_with_stats(&stats);
                    self.adaptive_budget.update_active_partitions(1);
                    // If every partition already fits in the current budget, disable further repartitioning.
                    if let Some(max_bytes) = stats.max_partition_bytes() {
                        if max_bytes <= self.adaptive_budget.current_limit() {
                            self.repartition_enabled = false;
                            debug!(
                                "Grace hash join repartition disabled: max partition {} fits current limit {}",
                                human_readable_size(max_bytes),
                                human_readable_size(self.adaptive_budget.current_limit())
                            );
                        }
                    }
                    let left_bytes = Arc::new(Mutex::new(0usize));
                    let right_bytes = Arc::new(Mutex::new(0usize));
                    let base_reservation = self.main_reservation();
                    let prefetch_reservation = self.ensure_prefetch_reservation();
                    self.state = GraceJoinState::JoinPartition {
                        work_queue,
                        current_work: None,
                        current_stream: None,
                        left_fut: None,
                        right_fut: None,
                        base_reservation: Arc::clone(&base_reservation),
                        prefetch_reservation: Arc::clone(&prefetch_reservation),
                        left_bytes,
                        right_bytes,
                        reservation: base_reservation,
                        join_permit: None,
                        join_permit_fut: None,
                        current_join_start: None,
                        repartition_fut: None,
                        prefetch: None,
                        last_prefetch_skip: None,
                    };
                    continue;
                }
                GraceJoinState::JoinPartition {
                    work_queue,
                    current_work,
                    current_stream,
                    left_fut,
                    right_fut,
                    base_reservation,
                    prefetch_reservation,
                    left_bytes,
                    right_bytes,
                    reservation,
                    join_permit,
                    join_permit_fut,
                    current_join_start,
                    repartition_fut,
                    prefetch,
                    last_prefetch_skip,
                } => {
                    if current_work.is_none() {
                        match work_queue.pop_front() {
                            Some(work) => {
                                *current_work = Some(work);
                                *left_bytes.lock() = 0;
                                *right_bytes.lock() = 0;
                                *reservation = Arc::clone(base_reservation);
                                *join_permit = None;
                                *join_permit_fut = None;
                                self.adaptive_budget.update_active_partitions(1);
                            }
                            None => {
                                self.state = GraceJoinState::Done;
                                continue;
                            }
                        }
                    }

                    // Acquire global join permit before loading/joining to cap concurrent joins.
                    if join_permit.is_none() {
                        if join_permit_fut.is_none() {
                            let sem = Arc::clone(global_join_semaphore());
                            *join_permit_fut = Some(OnceFut::new(async move {
                                let permit = sem.acquire_owned().await.unwrap();
                                Ok(permit)
                            }));
                        }
                        if let Some(fut) = join_permit_fut.as_mut() {
                            match fut.get_shared(cx) {
                                Poll::Ready(Ok(permit)) => {
                                    *join_permit = Some(permit);
                                    if let Some(rss) = current_rss_bytes() {
                                        info!(
                                                "Grace hash join acquired join permit for partition {} (pass {}), rss={}",
                                                current_work.as_ref().map(|w| w.partition_id).unwrap_or(usize::MAX),
                                                current_work.as_ref().map(|w| w.pass).unwrap_or(usize::MAX),
                                                human_readable_size(rss as usize)
                                            );
                                    }
                                    *join_permit_fut = None;
                                }
                                Poll::Ready(Err(e)) => {
                                    return Poll::Ready(Some(Err(e)));
                                }
                                Poll::Pending => {
                                    return Poll::Pending;
                                }
                            }
                        }
                        continue;
                    }

                    if current_stream.is_none() {
                        let work = current_work.as_ref().expect("work must exist");

                        // If we prefetched this work item, reuse its futures/bytes
                        if let Some(pref) = prefetch.take() {
                            if pref.work.partition_id == work.partition_id
                                && pref.work.pass == work.pass
                            {
                                *left_fut = Some(pref.left_fut);
                                *right_fut = Some(pref.right_fut);
                                *left_bytes = pref.left_bytes;
                                *right_bytes = pref.right_bytes;
                                *reservation = pref.reservation;
                            } else {
                                // Not the same work, keep prefetch for later
                                *prefetch = Some(pref);
                            }
                        }
                        // Expansion factor to account for hash table overhead
                        let estimated_size = (work.total_bytes() as f64 * 1.5) as usize;
                        let mut effective_limit = self.adaptive_budget.current_limit();
                        let mut preload_budget_change_logged = false;
                        let force_compute_repartition = estimated_size
                            > self.compute_soft_cap_bytes
                            && work.pass < self.max_partition_passes;

                        // If it definitely won't fit, try serializing first to raise the limit.
                        if estimated_size > effective_limit
                            && work.pass < self.max_partition_passes
                            && self.adaptive_budget.current_concurrency() > 1
                            && !force_compute_repartition
                        {
                            if let Some((
                                previous_concurrency,
                                previous_limit,
                                new_limit,
                            )) =
                                self.adaptive_budget.serialize_if_helpful(estimated_size)
                            {
                                effective_limit = new_limit;
                                let msg = format!(
                                    "Grace hash join partition {} serializing before load to raise budget from {} to {} ({} -> 1 in-flight partitions, pass {}, estimated {})",
                                    work.partition_id,
                                    human_readable_size(previous_limit),
                                    human_readable_size(new_limit),
                                    previous_concurrency,
                                    work.pass,
                                    human_readable_size(estimated_size),
                                );
                                if new_limit > self.adaptive_budget.preferred_cap {
                                    info!("{msg}");
                                } else {
                                    debug!("{msg}");
                                }
                                preload_budget_change_logged = true;
                            }
                        }

                        let skip_load = (estimated_size > effective_limit
                            && work.pass < self.max_partition_passes)
                            || force_compute_repartition;

                        if left_fut.is_none()
                            && right_fut.is_none()
                            && !skip_load
                            && !force_compute_repartition
                        {
                            *left_fut = Some(load_partition_async(
                                Arc::clone(&self.spill_left),
                                work.left.clone(),
                                Arc::clone(reservation),
                                Arc::clone(left_bytes),
                            ));
                            *right_fut = Some(load_partition_async(
                                Arc::clone(&self.spill_right),
                                work.right.clone(),
                                Arc::clone(reservation),
                                Arc::clone(right_bytes),
                            ));
                        } else if skip_load || force_compute_repartition {
                            debug!(
                                "Grace hash join partition {} estimated size {} exceeds {} (limit {}, pass {}), repartitioning without loading",
                                work.partition_id,
                                human_readable_size(estimated_size),
                                if force_compute_repartition {
                                    human_readable_size(self.compute_soft_cap_bytes)
                                } else {
                                    human_readable_size(effective_limit)
                                },
                                effective_limit,
                                work.pass
                            );
                        }

                        let (left_batches, right_batches) = if left_fut.is_some() {
                            let left =
                                (*ready!(left_fut.as_mut().unwrap().get_shared(cx))?)
                                    .clone();
                            let right =
                                (*ready!(right_fut.as_mut().unwrap().get_shared(cx))?)
                                    .clone();
                            (left, right)
                        } else {
                            // Should only happen when we force repartition without loading.
                            (
                                LoadedPartitionBatches {
                                    batches: vec![],
                                    total_bytes: 0,
                                },
                                LoadedPartitionBatches {
                                    batches: vec![],
                                    total_bytes: 0,
                                },
                            )
                        };

                        let work = current_work.as_ref().expect("work must exist");

                        // If we skipped loading, use estimated size. Otherwise use actual loaded size.
                        let total_loaded_bytes = if left_fut.is_some() {
                            left_batches
                                .total_bytes
                                .saturating_add(right_batches.total_bytes)
                        } else {
                            (work.total_bytes() as f64 * 1.5) as usize
                        };

                        let mut outcome =
                            self.adaptive_budget.ensure_fits(total_loaded_bytes);
                        let mut budget_change_logged = preload_budget_change_logged;

                        if matches!(outcome, AdaptiveBudgetOutcome::CannotFit { .. })
                            && self.adaptive_budget.current_concurrency() > 1
                            && !force_compute_repartition
                        {
                            if let Some((
                                previous_concurrency,
                                previous_limit,
                                new_limit,
                            )) = self
                                .adaptive_budget
                                .serialize_if_helpful(total_loaded_bytes)
                            {
                                let msg = format!(
                                    "Grace hash join partition {} switching to serial processing to raise budget from {} to {} ({} -> 1 in-flight partitions, pass {}, loaded {})",
                                    work.partition_id,
                                    human_readable_size(previous_limit),
                                    human_readable_size(new_limit),
                                    previous_concurrency,
                                    work.pass,
                                    human_readable_size(total_loaded_bytes),
                                );
                                if new_limit > self.adaptive_budget.preferred_cap {
                                    info!("{msg}");
                                } else {
                                    debug!("{msg}");
                                }
                                budget_change_logged = true;
                                outcome = if new_limit > previous_limit {
                                    AdaptiveBudgetOutcome::Raised {
                                        previous: previous_limit,
                                        new_limit,
                                    }
                                } else {
                                    AdaptiveBudgetOutcome::Fits
                                };
                            }
                        }

                        // Track whether we must repartition this partition
                        let mut need_repartition = false;
                        match outcome {
                            AdaptiveBudgetOutcome::Fits => {}
                            AdaptiveBudgetOutcome::Raised {
                                previous,
                                new_limit,
                            } => {
                                if !budget_change_logged {
                                    let msg = format!(
                                        "Grace hash join partition {} raised budget from {} to {} (pass {}, loaded {})",
                                        work.partition_id,
                                        human_readable_size(previous),
                                        human_readable_size(new_limit),
                                        work.pass,
                                        human_readable_size(total_loaded_bytes),
                                    );
                                    if new_limit > self.adaptive_budget.preferred_cap {
                                        info!("{msg}");
                                    } else {
                                        debug!("{msg}");
                                    }
                                }
                            }
                            AdaptiveBudgetOutcome::CannotFit { limit, .. } => {
                                if work.pass >= self.max_partition_passes {
                                    let msg = format!(
                                        "Grace hash join partition {} requires {} but maximum budget is {} after {} passes",
                                        work.partition_id,
                                        human_readable_size(total_loaded_bytes),
                                        human_readable_size(limit),
                                        self.max_partition_passes
                                    );
                                    use log::error;
                                    error!("{}", msg);
                                    return Poll::Ready(Some(Err(
                                        datafusion_common::DataFusionError::ResourcesExhausted(msg),
                                    )));
                                }
                                need_repartition = true;
                            }
                        }

                        if force_compute_repartition {
                            need_repartition = true;
                        }

                        // Even if it fits the memory budget, avoid building extremely large partitions
                        // by capping compute-friendly size.
                        if !need_repartition
                            && work.pass < self.max_partition_passes
                            && work.partition_count < MAX_REPARTITION_PARTITIONS
                            && total_loaded_bytes > self.compute_soft_cap_bytes
                        {
                            debug!(
                                "Grace hash join partition {} size {} exceeds compute soft cap {}, repartitioning to reduce per-partition work",
                                work.partition_id,
                                human_readable_size(total_loaded_bytes),
                                human_readable_size(self.compute_soft_cap_bytes)
                            );
                            need_repartition = true;
                        }

                        if need_repartition && !self.repartition_enabled {
                            debug!(
                                "Grace hash join repartition suppressed: current limit {} covers loaded {}",
                                human_readable_size(self.adaptive_budget.current_limit()),
                                human_readable_size(total_loaded_bytes)
                            );
                            need_repartition = false;
                        }

                        if need_repartition {
                            // Do not split already-small partitions; instead keep them as-is.
                            if work.total_bytes() <= MIN_REPARTITION_BYTES {
                                debug!(
                                    "Grace hash join partition {} size {} below minimum repartition threshold {}, joining without further split",
                                    work.partition_id,
                                    human_readable_size(work.total_bytes()),
                                    human_readable_size(MIN_REPARTITION_BYTES)
                                );
                                need_repartition = false;
                            } else if !force_compute_repartition {
                                // If the loaded size now fits the current limit, join directly.
                                let current_limit = self.adaptive_budget.current_limit();
                                if total_loaded_bytes <= current_limit {
                                    debug!(
                                        "Grace hash join partition {} now fits current limit {} (loaded {}), skipping repartition",
                                        work.partition_id,
                                        human_readable_size(current_limit),
                                        human_readable_size(total_loaded_bytes)
                                    );
                                    need_repartition = false;
                                }
                            }
                        }

                        if need_repartition {
                            // If repartitioning would not increase fan-out, skip it to avoid a useless extra pass.
                            let prospective = compute_repartition_count(
                                work.partition_count,
                                work.total_bytes(),
                                if force_compute_repartition {
                                    self.compute_soft_cap_bytes
                                } else {
                                    self.adaptive_budget.current_limit()
                                },
                                MAX_REPARTITION_PARTITIONS,
                                !force_compute_repartition,
                                self.compute_soft_cap_bytes,
                            );
                            if prospective <= work.partition_count
                                && force_compute_repartition
                            {
                                debug!(
                                    "Grace hash join partition {} compute-split planned fan-out {} -> {} (no increase), loading and joining instead",
                                    work.partition_id,
                                    work.partition_count,
                                    prospective
                                );
                            } else if prospective <= work.partition_count {
                                debug!(
                                    "Grace hash join partition {} already at fan-out {}, skipping repartition (prospective {})",
                                    work.partition_id, work.partition_count, prospective
                                );
                                need_repartition = false;
                            } else if !force_compute_repartition {
                                // If we are only mildly above the budget (<= 2x), prefer joining with serialization instead of splitting.
                                let current_limit = self.adaptive_budget.current_limit();
                                if total_loaded_bytes <= current_limit.saturating_mul(2) {
                                    debug!(
                                        "Grace hash join partition {} within 2x budget (limit {}, loaded {}), skipping repartition",
                                        work.partition_id,
                                        human_readable_size(current_limit),
                                        human_readable_size(total_loaded_bytes)
                                    );
                                    need_repartition = false;
                                }
                            }
                        }

                        if need_repartition {
                            // Free loaded bytes before repartitioning
                            let bytes_to_free = {
                                let mut l = left_bytes.lock();
                                let mut r = right_bytes.lock();
                                let total = *l + *r;
                                *l = 0;
                                *r = 0;
                                total
                            };
                            if bytes_to_free > 0 {
                                let mut res = reservation.lock();
                                if res.try_shrink(bytes_to_free).is_err() {
                                    let freed = res.free();
                                    debug!(
                                        "Grace hash join reservation underflow: freed {} after shrink failure (requested {})",
                                        human_readable_size(freed),
                                        human_readable_size(bytes_to_free)
                                    );
                                }
                            }
                            *join_permit = None;
                            *left_fut = None;
                            *right_fut = None;

                            if repartition_fut.is_none() {
                                let to_split = current_work.take().unwrap();
                                let planned_fanout = compute_repartition_count(
                                    to_split.partition_count,
                                    to_split.total_bytes(),
                                    if force_compute_repartition {
                                        self.compute_soft_cap_bytes
                                    } else {
                                        self.adaptive_budget.current_limit()
                                    },
                                    MAX_REPARTITION_PARTITIONS,
                                    !force_compute_repartition,
                                    self.compute_soft_cap_bytes,
                                );
                                if planned_fanout <= to_split.partition_count
                                    && force_compute_repartition
                                {
                                    debug!(
                                        "Grace hash join partition {} compute-split planned fan-out {} -> {} (no increase), loading and joining instead",
                                        to_split.partition_id,
                                        to_split.partition_count,
                                        planned_fanout
                                    );
                                    *current_work = Some(to_split);
                                    continue;
                                } else if planned_fanout <= to_split.partition_count {
                                    debug!(
                                        "Grace hash join partition {} would not increase fan-out ({} -> {}), skipping repartition",
                                        to_split.partition_id,
                                        to_split.partition_count,
                                        planned_fanout
                                    );
                                    *current_work = Some(to_split);
                                    continue;
                                } else {
                                    debug!(
                                        "Grace hash join repartitioning partition {} (pass {}) fan-out {} -> {} (size {}, target {})",
                                        to_split.partition_id,
                                        to_split.pass,
                                        to_split.partition_count,
                                        planned_fanout,
                                        human_readable_size(to_split.total_bytes()),
                                        if force_compute_repartition {
                                            human_readable_size(self.compute_soft_cap_bytes)
                                        } else {
                                            human_readable_size(self.adaptive_budget.current_limit())
                                        }
                                    );
                                }
                                let future = build_repartition_future(
                                    to_split,
                                    self.random_state.clone(),
                                    self.on_left
                                        .iter()
                                        .cloned()
                                        .zip(self.on_right.iter().cloned())
                                        .collect(),
                                    Arc::clone(&self.spill_left),
                                    Arc::clone(&self.spill_right),
                                    Arc::clone(&self.join_metrics),
                                    Arc::clone(&self.context),
                                    Arc::clone(&self.left_input_schema),
                                    Arc::clone(&self.right_input_schema),
                                    self.partition_batch_size,
                                    if force_compute_repartition {
                                        self.compute_soft_cap_bytes
                                    } else {
                                        self.adaptive_budget.current_limit()
                                    },
                                    self.adaptive_budget.current_limit(),
                                    MAX_REPARTITION_PARTITIONS,
                                    self.compute_soft_cap_bytes,
                                );
                                *repartition_fut = Some(future);
                            }
                            if let Some(fut) = repartition_fut.as_mut() {
                                let new_parts = (*ready!(fut.get_shared(cx))?).clone();
                                *repartition_fut = None;
                                for part in &new_parts {
                                    self.adaptive_budget.observe(part.total_bytes());
                                }
                                work_queue.extend(new_parts.into_iter());
                                self.adaptive_budget.update_active_partitions(1);
                            }
                            continue;
                        }

                        // If we decided not to repartition but skipped loading earlier,
                        // load now and poll again to avoid joining empty batches.
                        if left_fut.is_none() {
                            *left_fut = Some(load_partition_async(
                                Arc::clone(&self.spill_left),
                                work.left.clone(),
                                Arc::clone(reservation),
                                Arc::clone(left_bytes),
                            ));
                            *right_fut = Some(load_partition_async(
                                Arc::clone(&self.spill_right),
                                work.right.clone(),
                                Arc::clone(reservation),
                                Arc::clone(right_bytes),
                            ));
                            continue;
                        }

                        let stream = build_in_memory_join_stream(
                            Arc::clone(&self.schema),
                            Arc::clone(&self.left_input_schema),
                            Arc::clone(&self.right_input_schema),
                            left_batches.batches,
                            right_batches.batches,
                            &self.on_left,
                            &self.on_right,
                            self.projection.clone(),
                            self.filter.clone(),
                            self.join_type,
                            &self.column_indices,
                            &self.join_metrics,
                            &self.context,
                        )?;

                        *current_stream = Some(stream);
                        info!(
                            "Grace hash join starting partition {} (pass {}, loaded {} left / {} right, limit {})",
                            work.partition_id,
                            work.pass,
                            human_readable_size(left_batches.total_bytes),
                            human_readable_size(right_batches.total_bytes),
                            human_readable_size(self.adaptive_budget.current_limit())
                        );
                        *current_join_start = Some(Instant::now());
                        *left_fut = None;
                        *right_fut = None;
                    }

                    // Trigger prefetch of the next partition (if any) while joining this one
                    if current_stream.is_some() && prefetch.is_none() {
                        if let Some(next_work) = work_queue.front() {
                            let estimated_size =
                                (next_work.total_bytes() as f64 * 1.1) as usize;
                            let cap =
                                prefetch_cap_bytes(self.adaptive_budget.current_limit());
                            if cap > 0 && estimated_size <= cap {
                                let left_bytes_pf = Arc::new(Mutex::new(0usize));
                                let right_bytes_pf = Arc::new(Mutex::new(0usize));
                                let left_fut_pf = load_partition_async(
                                    Arc::clone(&self.spill_left),
                                    next_work.left.clone(),
                                    Arc::clone(prefetch_reservation),
                                    Arc::clone(&left_bytes_pf),
                                );
                                let right_fut_pf = load_partition_async(
                                    Arc::clone(&self.spill_right),
                                    next_work.right.clone(),
                                    Arc::clone(prefetch_reservation),
                                    Arc::clone(&right_bytes_pf),
                                );
                                debug!(
                                    "Prefetching next partition {} (est {} <= cap {})",
                                    next_work.partition_id,
                                    human_readable_size(estimated_size),
                                    human_readable_size(cap)
                                );
                                *prefetch = Some(PrefetchState {
                                    work: next_work.clone(),
                                    left_fut: left_fut_pf,
                                    right_fut: right_fut_pf,
                                    left_bytes: left_bytes_pf,
                                    right_bytes: right_bytes_pf,
                                    reservation: Arc::clone(prefetch_reservation),
                                });
                            } else {
                                let key = (next_work.partition_id, next_work.pass);
                                if last_prefetch_skip.as_ref() != Some(&key) {
                                    debug!(
                                        "Skipping prefetch for partition {} (est {} > cap {})",
                                        next_work.partition_id,
                                        human_readable_size(estimated_size),
                                        human_readable_size(cap)
                                    );
                                    *last_prefetch_skip = Some(key);
                                }
                            }
                        }
                    }

                    if let Some(stream) = current_stream {
                        match ready!(stream.poll_next_unpin(cx)) {
                            Some(Ok(batch)) => return Poll::Ready(Some(Ok(batch))),
                            Some(Err(e)) => {
                                if let Some(start) = current_join_start.take() {
                                    join_time_metric.add_elapsed(start);
                                    if let Some(work) = current_work.as_ref() {
                                        info!(
                                            "Grace hash join failed partition {} (pass {}) after {:?}",
                                            work.partition_id,
                                            work.pass,
                                            start.elapsed()
                                        );
                                    }
                                }
                                *join_permit = None;
                                return Poll::Ready(Some(Err(e)));
                            }
                            None => {
                                let bytes_to_free = {
                                    let mut l = left_bytes.lock();
                                    let mut r = right_bytes.lock();
                                    let total = *l + *r;
                                    *l = 0;
                                    *r = 0;
                                    total
                                };
                                if bytes_to_free > 0 {
                                    let mut res = reservation.lock();
                                    if res.try_shrink(bytes_to_free).is_err() {
                                        let freed = res.free();
                                        debug!(
                                        "Grace hash join stream completion freed {} after shrink failure (requested {})",
                                            human_readable_size(freed),
                                            human_readable_size(bytes_to_free)
                                        );
                                    }
                                }
                                if let Some(start) = current_join_start.take() {
                                    join_time_metric.add_elapsed(start);
                                    if let Some(work) = current_work.as_ref() {
                                        info!(
                                            "Grace hash join finished partition {} (pass {}) in {:?}",
                                            work.partition_id,
                                            work.pass,
                                            start.elapsed()
                                        );
                                    }
                                }
                                *current_stream = None;
                                *current_work = None;
                                *last_prefetch_skip = None;
                                *reservation = Arc::clone(base_reservation);
                                *join_permit = None;
                                self.adaptive_budget.update_active_partitions(1);
                                continue;
                            }
                        }
                    }
                }
                GraceJoinState::Done => return Poll::Ready(None),
            }
        }
    }
}

fn load_partition_async(
    spill_manager: Arc<SpillManager>,
    partition: PartitionIndex,
    reservation: Arc<Mutex<MemoryReservation>>,
    bytes_counter: Arc<Mutex<usize>>,
) -> OnceFut<LoadedPartitionBatches> {
    OnceFut::new(async move {
        // Load spill chunks with bounded parallelism to overlap IO.
        let mut tasks = FuturesUnordered::new();
        let mut in_flight_bytes = 0usize;
        let mut all_batches = Vec::new();
        let mut total_bytes = 0usize;

        for chunk in partition.chunks {
            let estimated = chunk.size;
            // backpressure: wait for at least one task to finish if we'd exceed cap
            while in_flight_bytes > 0
                && in_flight_bytes + estimated > SPILL_READAHEAD_BYTES
            {
                if let Some(res) = tasks.next().await {
                    let (batches, bytes) = res?;
                    in_flight_bytes = in_flight_bytes.saturating_sub(bytes);
                    total_bytes = total_bytes.saturating_add(bytes);
                    all_batches.extend(batches);
                }
            }

            let sm = Arc::clone(&spill_manager);
            let resv = Arc::clone(&reservation);
            let counter = Arc::clone(&bytes_counter);
            tasks.push(async move { load_spill_chunk(sm, chunk, resv, counter).await });
            in_flight_bytes = in_flight_bytes.saturating_add(estimated);
        }

        while let Some(res) = tasks.next().await {
            let (batches, bytes) = res?;
            in_flight_bytes = in_flight_bytes.saturating_sub(bytes);
            total_bytes = total_bytes.saturating_add(bytes);
            all_batches.extend(batches);
        }

        Ok(LoadedPartitionBatches {
            batches: all_batches,
            total_bytes,
        })
    })
}

async fn load_spill_chunk(
    spill_manager: Arc<SpillManager>,
    chunk: SpillChunk,
    reservation: Arc<Mutex<MemoryReservation>>,
    bytes_counter: Arc<Mutex<usize>>,
) -> Result<(Vec<RecordBatch>, usize)> {
    let mut reader = spill_manager.load_spilled_batch(&chunk.location)?;
    let mut batches = Vec::new();
    let mut total_bytes = 0usize;
    while let Some(batch_result) = reader.next().await {
        let batch = batch_result?;
        // Use de-duplicated record batch memory size to avoid
        // drastically overestimating memory when arrays share buffers.
        let batch_size = get_record_batch_memory_size(&batch);
        {
            let mut res = reservation.lock();
            res.try_grow(batch_size)?;
            let mut b = bytes_counter.lock();
            *b += batch_size;
        }
        total_bytes = total_bytes.saturating_add(batch_size);
        batches.push(batch);
    }
    Ok((batches, total_bytes))
}

/// Build an in-memory HashJoinExec for one pair of spilled partitions
fn build_in_memory_join_stream(
    output_schema: SchemaRef,
    left_input_schema: SchemaRef,
    right_input_schema: SchemaRef,
    left_batches: Vec<RecordBatch>,
    right_batches: Vec<RecordBatch>,
    on_left: &[PhysicalExprRef],
    on_right: &[PhysicalExprRef],
    projection: Option<Vec<usize>>,
    filter: Option<JoinFilter>,
    join_type: JoinType,
    _column_indices: &[ColumnIndex],
    _join_metrics: &BuildProbeJoinMetrics,
    context: &Arc<TaskContext>,
) -> Result<SendableRecordBatchStream> {
    if left_batches.is_empty() && right_batches.is_empty() {
        return EmptyExec::new(output_schema).execute(0, Arc::clone(context));
    }

    // Build memory execution nodes for each side
    let left_plan: Arc<dyn ExecutionPlan> = Arc::new(TestMemoryExec::try_new(
        &[left_batches],
        left_input_schema,
        None,
    )?);
    let right_plan: Arc<dyn ExecutionPlan> = Arc::new(TestMemoryExec::try_new(
        &[right_batches],
        right_input_schema,
        None,
    )?);

    // Combine join expressions into pairs
    let on: Vec<(PhysicalExprRef, PhysicalExprRef)> = on_left
        .iter()
        .cloned()
        .zip(on_right.iter().cloned())
        .collect();

    // For one partition pair: always CollectLeft (build left, stream right)
    let join_exec = HashJoinExec::try_new(
        left_plan,
        right_plan,
        on,
        filter,
        &join_type,
        projection,
        PartitionMode::CollectLeft,
        NullEquality::NullEqualsNothing,
    )?;

    // Each join executes locally with the same context
    join_exec.execute(0, Arc::clone(context))
}

struct SpilledPartitionStream {
    schema: SchemaRef,
    spill_manager: Arc<SpillManager>,
    chunks: VecDeque<SpillChunk>,
    current_stream: Option<SendableRecordBatchStream>,
}

impl SpilledPartitionStream {
    fn new(
        schema: SchemaRef,
        spill_manager: Arc<SpillManager>,
        chunks: Vec<SpillChunk>,
    ) -> Self {
        Self {
            schema,
            spill_manager,
            chunks: VecDeque::from(chunks),
            current_stream: None,
        }
    }
}

impl Stream for SpilledPartitionStream {
    type Item = Result<RecordBatch>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        loop {
            if let Some(stream) = self.current_stream.as_mut() {
                match stream.poll_next_unpin(cx) {
                    Poll::Ready(Some(batch)) => return Poll::Ready(Some(batch)),
                    Poll::Ready(None) => {
                        self.current_stream = None;
                        continue;
                    }
                    Poll::Pending => return Poll::Pending,
                }
            }

            match self.chunks.pop_front() {
                Some(chunk) => {
                    match self.spill_manager.load_spilled_batch(&chunk.location) {
                        Ok(stream) => {
                            self.current_stream = Some(stream);
                            continue;
                        }
                        Err(e) => return Poll::Ready(Some(Err(e))),
                    }
                }
                None => return Poll::Ready(None),
            }
        }
    }
}

impl RecordBatchStream for SpilledPartitionStream {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}

impl Stream for GraceHashJoinStream {
    type Item = Result<RecordBatch>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        self.poll_next_impl(cx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adaptive_budget_scales_up_to_preferred_cap() {
        let mut budget = AdaptivePartitionBudget::new(
            512 * 1024 * 1024,
            2 * 1024 * 1024 * 1024,
            4 * 1024 * 1024 * 1024,
            4,
        );
        match budget.ensure_fits(768 * 1024 * 1024) {
            AdaptiveBudgetOutcome::Raised {
                previous,
                new_limit,
            } => {
                assert_eq!(previous, 512 * 1024 * 1024);
                assert_eq!(new_limit, 768 * 1024 * 1024);
            }
            other => panic!("unexpected outcome: {other:?}"),
        }
        assert!(matches!(
            budget.ensure_fits(512 * 1024 * 1024),
            AdaptiveBudgetOutcome::Fits
        ));
    }

    #[test]
    fn adaptive_budget_borrows_up_to_absolute_cap() {
        let mut budget = AdaptivePartitionBudget::new(512, 1024, 2048, 1);
        // With aggressive budgeting, a single active partition gets the absolute cap immediately.
        assert_eq!(budget.current_limit(), 2048);

        // It fits immediately
        match budget.ensure_fits(1500) {
            AdaptiveBudgetOutcome::Fits => {}
            other => panic!("unexpected outcome {other:?}"),
        }

        // Still bounded by absolute cap
        match budget.ensure_fits(3000) {
            AdaptiveBudgetOutcome::CannotFit { limit } => {
                assert_eq!(limit, 2048);
            }
            other => panic!("expected absolute cap exhaustion, got {other:?}"),
        }
    }

    #[test]
    fn adaptive_budget_serializes_to_raise_limit() {
        let mut budget = AdaptivePartitionBudget::new(256, 512, 1024, 4);
        match budget.ensure_fits(800) {
            AdaptiveBudgetOutcome::CannotFit { limit } => {
                assert_eq!(limit, 512);
            }
            other => panic!("expected initial budget exhaustion, got {other:?}"),
        }

        match budget.serialize_if_helpful(800) {
            Some((previous_concurrency, previous_limit, new_limit)) => {
                assert_eq!(previous_concurrency, 4);
                assert_eq!(previous_limit, 512);
                // Should jump to absolute cap (1024)
                assert_eq!(new_limit, 1024);
                assert_eq!(budget.current_concurrency(), 1);
                assert_eq!(budget.current_limit(), 1024);
            }
            None => panic!("expected serialization to help"),
        }
    }

    #[test]
    fn repartition_count_is_capped() {
        let count = compute_repartition_count(
            16,
            1 * 1024 * 1024 * 1024,
            64 * 1024 * 1024,
            64,
            true,
            512 * 1024 * 1024,
        );
        assert_eq!(count, 64);
    }

    #[test]
    fn repartition_count_grows_by_fan_out() {
        let count = compute_repartition_count(
            8,
            64 * 1024 * 1024,
            64 * 1024 * 1024,
            256,
            true,
            512 * 1024 * 1024,
        );
        // fan_out=2 (min), base=8 -> 16
        assert_eq!(count, 16);
    }
}
