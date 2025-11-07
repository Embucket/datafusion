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

use arrow::record_batch::RecordBatch;

use crate::joins::hash_join::exec::JoinLeftData;
use crate::joins::hash_join::partitioned::{
    PartitionDescriptor, PartitionedHashJoinStream,
};
use crate::joins::utils::StatefulStreamResult;

use datafusion_common::{internal_datafusion_err, Result};

/// Configuration shared across scheduler components.
#[derive(Clone, Debug)]
pub(super) struct SchedulerConfig {
    pub memory_threshold: usize,
    pub batch_size: usize,
    pub max_partition_count: usize,
}

impl SchedulerConfig {
    pub fn from_stream(stream: &PartitionedHashJoinStream) -> Self {
        Self {
            memory_threshold: stream.memory_threshold,
            batch_size: stream.batch_size,
            max_partition_count: stream.max_partition_count,
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
            match task.poll(stream)? {
                TaskPoll::Pending(task) => self.ready_queue.push_back(task),
                TaskPoll::BuildFinished(result) => return Ok(result),
                TaskPoll::YieldProbe(task) => self.ready_queue.push_back(task),
                TaskPoll::YieldFinalize(task) => self.ready_queue.push_back(task),
                TaskPoll::ProbeFinished | TaskPoll::FinalizeFinished => continue,
            }
        }
        Err(internal_datafusion_err!(
            "scheduler queue exhausted without producing build output"
        ))
    }
}

enum SchedulerTask {
    Build(BuildStageTask),
    Probe(ProbeStageTask),
    Finalize(FinalizeStageTask),
}

enum TaskPoll {
    Pending(SchedulerTask),
    BuildFinished(StatefulStreamResult<Option<RecordBatch>>),
    /// Probe task yielded without producing output (to be expanded later).
    YieldProbe(SchedulerTask),
    /// Finalize task yielded without producing output.
    YieldFinalize(SchedulerTask),
    ProbeFinished,
    FinalizeFinished,
}

impl SchedulerTask {
    fn poll(self, stream: &mut PartitionedHashJoinStream) -> Result<TaskPoll> {
        match self {
            SchedulerTask::Build(task) => match task.poll(stream)? {
                BuildTaskEvent::Pending(next_state) => {
                    Ok(TaskPoll::Pending(SchedulerTask::Build(next_state)))
                }
                BuildTaskEvent::Finished(result) => Ok(TaskPoll::BuildFinished(result)),
            },
            SchedulerTask::Probe(task) => match task.poll(stream)? {
                ProbeTaskEvent::Pending(next_task) => {
                    Ok(TaskPoll::YieldProbe(SchedulerTask::Probe(next_task)))
                }
                ProbeTaskEvent::Finished => Ok(TaskPoll::ProbeFinished),
            },
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

struct ProbeStageTask {
    config: SchedulerConfig,
    descriptor: PartitionDescriptor,
    yielded_once: bool,
}

impl ProbeStageTask {
    fn new(config: SchedulerConfig, descriptor: PartitionDescriptor) -> Self {
        Self {
            config,
            descriptor,
            yielded_once: false,
        }
    }

    fn poll(self, _stream: &mut PartitionedHashJoinStream) -> Result<ProbeTaskEvent> {
        if self.yielded_once {
            Ok(ProbeTaskEvent::Finished)
        } else {
            Ok(ProbeTaskEvent::Pending(Self {
                yielded_once: true,
                ..self
            }))
        }
    }
}

enum ProbeTaskEvent {
    Pending(ProbeStageTask),
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
