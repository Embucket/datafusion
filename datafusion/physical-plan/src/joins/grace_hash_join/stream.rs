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
use futures::{ready, Stream, StreamExt};
use parking_lot::Mutex;

enum GraceJoinState {
    /// Waiting for the partitioning phase (Phase 1) to finish
    WaitPartitioning,

    /// Currently joining partition(s)
    JoinPartition {
        work_queue: VecDeque<PartitionWorkItem>,
        current_work: Option<PartitionWorkItem>,
        current_stream: Option<SendableRecordBatchStream>,
        left_fut: Option<OnceFut<Vec<RecordBatch>>>,
        right_fut: Option<OnceFut<Vec<RecordBatch>>>,
        /// Bytes reserved in the memory pool for the current partition's
        /// loaded left batches
        left_bytes: Arc<Mutex<usize>>,
        /// Bytes reserved in the memory pool for the current partition's
        /// loaded right batches
        right_bytes: Arc<Mutex<usize>>,
        repartition_fut: Option<OnceFut<Vec<PartitionWorkItem>>>,
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
) -> OnceFut<Vec<PartitionWorkItem>> {
    OnceFut::new(async move {
        let PartitionWorkItem {
            partition_id,
            pass,
            partition_count,
            left,
            right,
        } = work;

        let new_partition_count =
            partition_count.saturating_mul(2).max(partition_count + 1);

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
    /// Memory reservation tracking in-memory buffers used by the join stream
    reservation: Arc<Mutex<MemoryReservation>>,
    random_state: RandomState,
    partition_batch_size: usize,
    partition_budget_bytes: usize,
    max_partition_passes: usize,
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

impl RecordBatchStream for GraceHashJoinStream {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}

impl GraceHashJoinStream {
    pub fn new(
        schema: SchemaRef,
        left_input_schema: SchemaRef,
        right_input_schema: SchemaRef,
        spill_fut: OnceFut<SpillFut>,
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
        reservation: MemoryReservation,
        random_state: RandomState,
        partition_batch_size: usize,
        partition_budget_bytes: usize,
        max_partition_passes: usize,
    ) -> Self {
        Self {
            schema,
            left_input_schema,
            right_input_schema,
            spill_fut,
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
            reservation: Arc::new(Mutex::new(reservation)),
            random_state,
            partition_batch_size,
            partition_budget_bytes,
            max_partition_passes,
            state: GraceJoinState::WaitPartitioning,
        }
    }

    /// Core state machine logic (poll implementation)
    fn poll_next_impl(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<RecordBatch>>> {
        loop {
            match &mut self.state {
                GraceJoinState::WaitPartitioning => {
                    let shared = ready!(self.spill_fut.get_shared(cx))?;
                    let partition_count = shared.left.len();
                    let work_queue = shared
                        .left
                        .iter()
                        .cloned()
                        .zip(shared.right.iter().cloned())
                        .enumerate()
                        .map(|(partition_id, (left, right))| PartitionWorkItem {
                            partition_id,
                            pass: 0,
                            partition_count,
                            left,
                            right,
                        })
                        .collect::<VecDeque<_>>();
                    let left_bytes = Arc::new(Mutex::new(0usize));
                    let right_bytes = Arc::new(Mutex::new(0usize));
                    self.state = GraceJoinState::JoinPartition {
                        work_queue,
                        current_work: None,
                        current_stream: None,
                        left_fut: None,
                        right_fut: None,
                        left_bytes,
                        right_bytes,
                        repartition_fut: None,
                    };
                    continue;
                }
                GraceJoinState::JoinPartition {
                    work_queue,
                    current_work,
                    current_stream,
                    left_fut,
                    right_fut,
                    left_bytes,
                    right_bytes,
                    repartition_fut,
                } => {
                    if current_work.is_none() {
                        match work_queue.pop_front() {
                            Some(work) => {
                                *current_work = Some(work);
                                *left_bytes.lock() = 0;
                                *right_bytes.lock() = 0;
                            }
                            None => {
                                self.state = GraceJoinState::Done;
                                continue;
                            }
                        }
                    }

                    if current_stream.is_none() {
                        if let Some(work) = current_work.as_ref() {
                            if work.total_bytes() > self.partition_budget_bytes {
                                if work.pass >= self.max_partition_passes {
                                    let msg = format!(
                                        "Grace hash join partition {} requires {} but budget is {} after {} passes",
                                        work.partition_id,
                                        human_readable_size(work.total_bytes()),
                                        human_readable_size(self.partition_budget_bytes),
                                        self.max_partition_passes
                                    );
                                    return Poll::Ready(Some(Err(
                                        datafusion_common::DataFusionError::ResourcesExhausted(
                                            msg,
                                        ),
                                    )));
                                }
                                if repartition_fut.is_none() {
                                    let to_split = current_work.take().unwrap();
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
                                    );
                                    *repartition_fut = Some(future);
                                }
                                if let Some(fut) = repartition_fut.as_mut() {
                                    let new_parts =
                                        (*ready!(fut.get_shared(cx))?).clone();
                                    *repartition_fut = None;
                                    work_queue.extend(new_parts.into_iter());
                                }
                                continue;
                            }
                        }

                        if left_fut.is_none() && right_fut.is_none() {
                            if let Some(work) = current_work.as_ref() {
                                *left_fut = Some(load_partition_async(
                                    Arc::clone(&self.spill_left),
                                    work.left.clone(),
                                    Arc::clone(&self.reservation),
                                    Arc::clone(left_bytes),
                                ));
                                *right_fut = Some(load_partition_async(
                                    Arc::clone(&self.spill_right),
                                    work.right.clone(),
                                    Arc::clone(&self.reservation),
                                    Arc::clone(right_bytes),
                                ));
                            }
                        }

                        let left_batches =
                            (*ready!(left_fut.as_mut().unwrap().get_shared(cx))?).clone();
                        let right_batches =
                            (*ready!(right_fut.as_mut().unwrap().get_shared(cx))?)
                                .clone();

                        let stream = build_in_memory_join_stream(
                            Arc::clone(&self.schema),
                            Arc::clone(&self.left_input_schema),
                            Arc::clone(&self.right_input_schema),
                            left_batches,
                            right_batches,
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
                        *left_fut = None;
                        *right_fut = None;
                    }

                    if let Some(stream) = current_stream {
                        match ready!(stream.poll_next_unpin(cx)) {
                            Some(Ok(batch)) => return Poll::Ready(Some(Ok(batch))),
                            Some(Err(e)) => return Poll::Ready(Some(Err(e))),
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
                                    let mut res = self.reservation.lock();
                                    res.shrink(bytes_to_free);
                                }
                                *current_stream = None;
                                *current_work = None;
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
) -> OnceFut<Vec<RecordBatch>> {
    OnceFut::new(async move {
        let mut all_batches = Vec::new();

        for chunk in partition.chunks {
            let mut reader = spill_manager.load_spilled_batch(&chunk.location)?;
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
                all_batches.push(batch);
            }
        }
        Ok(all_batches)
    })
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
