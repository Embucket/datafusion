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

//! Experimental Hybrid Hash Join scheduler abstractions.

#![cfg(feature = "hybrid_hash_join_scheduler")]

use std::collections::VecDeque;
use std::sync::Arc;
use std::task::Context;

use arrow::{array::ArrayRef, record_batch::RecordBatch};

use crate::joins::hash_join::exec::JoinLeftData;
use crate::joins::hash_join::partitioned::{
    PartitionDescriptor, PartitionedHashJoinStream, ProbePartition,
};
use crate::joins::utils::StatefulStreamResult;
use crate::SendableRecordBatchStream;

use datafusion_common::{internal_datafusion_err, Result};
use datafusion_execution::disk_manager::RefCountedTempFile;

use crate::joins::join_hash_map::JoinHashMapOffset;
use crate::spill::in_progress_spill_file::InProgressSpillFile;

/// Configuration shared across scheduler components.
#[derive(Clone, Debug)]
pub(super) struct SchedulerConfig {
    pub memory_threshold: usize,
    pub batch_size: usize,
    pub max_partition_count: usize,
    pub max_probe_streams: usize,
}

impl SchedulerConfig {
    pub fn from_stream(stream: &PartitionedHashJoinStream) -> Self {
        Self {
            memory_threshold: stream.memory_threshold,
            batch_size: stream.batch_size,
            max_partition_count: stream.max_partition_count,
            max_probe_streams: std::cmp::max(1, std::cmp::min(4, stream.num_partitions)),
        }
    }
}

/// Minimal scheduler capable of running build / probe / finalize tasks.
pub(super) struct HybridTaskScheduler {
    config: SchedulerConfig,
    ready_queue: VecDeque<SchedulerTask>,
}

impl HybridTaskScheduler {
    pub fn new(config: SchedulerConfig) -> Self {
        Self {
            config,
            ready_queue: VecDeque::new(),
        }
    }

    pub fn push_task(&mut self, task: SchedulerTask) {
        self.ready_queue.push_back(task);
    }

    pub fn pop_task(&mut self) -> Option<SchedulerTask> {
        self.ready_queue.pop_front()
    }

    pub fn len(&self) -> usize {
        self.ready_queue.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ready_queue.is_empty()
    }

    pub fn with_build_task(
        config: SchedulerConfig,
        build_data: Arc<JoinLeftData>,
    ) -> Self {
        let mut scheduler = Self::new(config.clone());
        scheduler
            .ready_queue
            .push_back(SchedulerTask::Build(BuildStageTask::new(
                config, build_data,
            )));
        scheduler
    }

    pub fn enqueue_probe_task(&mut self, descriptor: PartitionDescriptor) {
        self.ready_queue
            .push_back(SchedulerTask::Probe(ProbeStageTask::new(
                self.config.clone(),
                descriptor,
            )));
    }

    pub fn enqueue_finalize_task(&mut self, descriptor: PartitionDescriptor) {
        self.ready_queue
            .push_back(SchedulerTask::Finalize(FinalizeStageTask::new(
                self.config.clone(),
                descriptor,
            )));
    }

    pub fn run_until_build_finished(
        &mut self,
        stream: &mut PartitionedHashJoinStream,
    ) -> Result<StatefulStreamResult<Option<RecordBatch>>> {
        while let Some(task) = self.ready_queue.pop_front() {
            match task.poll(stream, None)? {
                TaskPoll::Ready(_) => continue,
                TaskPoll::ProbeReady(_) => continue,
                TaskPoll::Pending(task) => self.ready_queue.push_back(task),
                TaskPoll::BuildFinished(result) => return Ok(result),
                TaskPoll::YieldProbe { task, .. } => self.ready_queue.push_back(task),
                TaskPoll::YieldFinalize(task) => self.ready_queue.push_back(task),
                TaskPoll::ProbeFinished(_) | TaskPoll::FinalizeFinished => continue,
            }
        }
        Err(internal_datafusion_err!(
            "scheduler queue exhausted without producing build output"
        ))
    }
}

pub(super) enum SchedulerTask {
    Build(BuildStageTask),
    Probe(ProbeStageTask),
    Finalize(FinalizeStageTask),
}

pub(super) enum TaskPoll {
    Ready(Option<RecordBatch>),
    ProbeReady(PartitionDescriptor),
    Pending(SchedulerTask),
    BuildFinished(StatefulStreamResult<Option<RecordBatch>>),
    /// Probe task yielded without producing output (e.g. waiting on IO).
    YieldProbe {
        task: SchedulerTask,
        descriptor: PartitionDescriptor,
    },
    /// Finalize task yielded without producing output.
    YieldFinalize(SchedulerTask),
    ProbeFinished(PartitionDescriptor),
    FinalizeFinished,
}

impl SchedulerTask {
    pub(super) fn poll(
        self,
        stream: &mut PartitionedHashJoinStream,
        cx: Option<&mut Context<'_>>,
    ) -> Result<TaskPoll> {
        match self {
            SchedulerTask::Build(task) => match task.poll(stream)? {
                BuildTaskEvent::Pending(next_state) => {
                    Ok(TaskPoll::Pending(SchedulerTask::Build(next_state)))
                }
                BuildTaskEvent::Finished(result) => Ok(TaskPoll::BuildFinished(result)),
            },
            SchedulerTask::Probe(task) => {
                let cx = cx.expect("probe task requires runtime context");
                let descriptor = task.descriptor().clone();
                match task.poll(stream, cx)? {
                    ProbeTaskEvent::Pending(next_task) => {
                        Ok(TaskPoll::Pending(SchedulerTask::Probe(next_task)))
                    }
                    ProbeTaskEvent::Ready => Ok(TaskPoll::ProbeReady(descriptor)),
                    ProbeTaskEvent::NeedStream(next_task) => {
                        let wait_descriptor = next_task.descriptor().clone();
                        Ok(TaskPoll::YieldProbe {
                            task: SchedulerTask::Probe(next_task),
                            descriptor: wait_descriptor,
                        })
                    }
                    ProbeTaskEvent::Finished => Ok(TaskPoll::ProbeFinished(descriptor)),
                }
            }
            SchedulerTask::Finalize(task) => match task.poll(stream)? {
                FinalizeTaskEvent::Pending(next_task) => {
                    Ok(TaskPoll::YieldFinalize(SchedulerTask::Finalize(next_task)))
                }
                FinalizeTaskEvent::Finished => Ok(TaskPoll::FinalizeFinished),
            },
        }
    }
}

/// Build stage broken into multiple cooperative steps so the scheduler can interleave it.
struct BuildStageTask {
    config: SchedulerConfig,
    build_data: Option<Arc<JoinLeftData>>,
    step: BuildTaskStep,
    warmup_remaining: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BuildTaskStep {
    Init,
    Partitioning,
    Finished,
}

impl BuildStageTask {
    fn new(config: SchedulerConfig, build_data: Arc<JoinLeftData>) -> Self {
        Self {
            config,
            build_data: Some(build_data),
            step: BuildTaskStep::Init,
            warmup_remaining: 2, // allow a couple of yields before heavy work
        }
    }

    fn poll(mut self, stream: &mut PartitionedHashJoinStream) -> Result<BuildTaskEvent> {
        match self.step {
            BuildTaskStep::Init => {
                if self.warmup_remaining > 0 {
                    self.warmup_remaining -= 1;
                    return Ok(BuildTaskEvent::Pending(self));
                }
                self.step = BuildTaskStep::Partitioning;
                Ok(BuildTaskEvent::Pending(self))
            }
            BuildTaskStep::Partitioning => {
                let build_data = self.build_data.take().ok_or_else(|| {
                    internal_datafusion_err!("build task missing input data")
                })?;
                let result = stream.partition_build_side_serial(build_data)?;
                self.step = BuildTaskStep::Finished;
                Ok(BuildTaskEvent::Finished(result))
            }
            BuildTaskStep::Finished => {
                Err(internal_datafusion_err!("build task already finished"))
            }
        }
    }
}

enum BuildTaskEvent {
    Pending(BuildStageTask),
    Finished(StatefulStreamResult<Option<RecordBatch>>),
}

pub(super) struct ProbePartitionState {
    pub buffered: ProbePartition,
    pub batch_position: usize,
    pub buffered_rows: usize,
    pub buffered_bytes: usize,
    pub spilled_rows: usize,
    pub consumed_rows: usize,
    pub spill_in_progress: Option<InProgressSpillFile>,
    pub spill_files: VecDeque<RefCountedTempFile>,
    pub pending_stream: Option<SendableRecordBatchStream>,
    pub active_batch: Option<RecordBatch>,
    pub active_values: Vec<ArrayRef>,
    pub active_hashes: Vec<u64>,
    pub active_offset: JoinHashMapOffset,
    pub joined_probe_idx: Option<usize>,
}

impl ProbePartitionState {
    pub fn new() -> Self {
        Self {
            buffered: ProbePartition::new(),
            batch_position: 0,
            buffered_rows: 0,
            buffered_bytes: 0,
            spilled_rows: 0,
            consumed_rows: 0,
            spill_in_progress: None,
            spill_files: VecDeque::new(),
            pending_stream: None,
            active_batch: None,
            active_values: Vec::new(),
            active_hashes: Vec::new(),
            active_offset: (0, None),
            joined_probe_idx: None,
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ProbeDataPoll {
    Ready,
    Pending,
    NeedStream,
    Finished,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProbeTaskState {
    Init,
    Ready,
    Finished,
}

pub(super) struct ProbeStageTask {
    _config: SchedulerConfig,
    descriptor: PartitionDescriptor,
    state: ProbeTaskState,
}

impl ProbeStageTask {
    pub fn new(config: SchedulerConfig, descriptor: PartitionDescriptor) -> Self {
        Self {
            _config: config,
            descriptor,
            state: ProbeTaskState::Init,
        }
    }

    pub fn descriptor(&self) -> &PartitionDescriptor {
        &self.descriptor
    }

    fn poll(
        mut self,
        stream: &mut PartitionedHashJoinStream,
        cx: &mut Context<'_>,
    ) -> Result<ProbeTaskEvent> {
        match self.state {
            ProbeTaskState::Init => {
                self.state = ProbeTaskState::Ready;
                Ok(ProbeTaskEvent::Pending(self))
            }
            ProbeTaskState::Ready => {
                match stream
                    .poll_probe_data_for_partition(self.descriptor.build_index, cx)?
                {
                    ProbeDataPoll::Ready => Ok(ProbeTaskEvent::Ready),
                    ProbeDataPoll::Pending => Ok(ProbeTaskEvent::Pending(self)),
                    ProbeDataPoll::NeedStream => Ok(ProbeTaskEvent::NeedStream(self)),
                    ProbeDataPoll::Finished => {
                        self.state = ProbeTaskState::Finished;
                        Ok(ProbeTaskEvent::Finished)
                    }
                }
            }
            ProbeTaskState::Finished => Ok(ProbeTaskEvent::Finished),
        }
    }
}

enum ProbeTaskEvent {
    Pending(ProbeStageTask),
    Ready,
    NeedStream(ProbeStageTask),
    Finished,
}

struct FinalizeStageTask {
    config: SchedulerConfig,
    descriptor: PartitionDescriptor,
    yielded_once: bool,
}

impl FinalizeStageTask {
    fn new(config: SchedulerConfig, descriptor: PartitionDescriptor) -> Self {
        Self {
            config,
            descriptor,
            yielded_once: false,
        }
    }

    fn poll(self, _stream: &mut PartitionedHashJoinStream) -> Result<FinalizeTaskEvent> {
        if self.yielded_once {
            Ok(FinalizeTaskEvent::Finished)
        } else {
            Ok(FinalizeTaskEvent::Pending(Self {
                yielded_once: true,
                ..self
            }))
        }
    }
}

enum FinalizeTaskEvent {
    Pending(FinalizeStageTask),
    Finished,
}
