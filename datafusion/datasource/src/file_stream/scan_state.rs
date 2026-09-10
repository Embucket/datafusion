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

use datafusion_common::internal_datafusion_err;
use std::collections::VecDeque;
use std::task::{Context, Poll};

use crate::morsel::{Morsel, MorselPlanner, Morselizer, PendingMorselPlanner};
use arrow::record_batch::RecordBatch;
use datafusion_common::{DataFusionError, Result};
use datafusion_physical_plan::metrics::ScopedTimerGuard;
use futures::stream::BoxStream;
use futures::{FutureExt as _, StreamExt as _};

use super::work_source::WorkSource;
use super::{FileStreamMetrics, OnError};

/// State [`FileStreamState::Scan`].
///
/// There is one `ScanState` per `FileStream`, and thus per output partition.
///
/// It groups together the lifecycle of scanning that partition's files:
/// unopened files, CPU-ready planners, pending planner I/O, ready morsels,
/// the active reader, and the metrics associated with processing that work.
///
/// # I/O
///
/// To avoid challenges controlling buffering, the ScanState only ever has a
/// single I/O outstanding at any time.
///
/// # State Transitions
///
/// ```text
/// work_source
///    |
///    v
/// morselizer.plan_file(file)
///    |
///    v
/// ready_planners ---> plan() ---> ready_morsels ---> into_stream() ---> reader ---> RecordBatches
///       ^               |
///       |               v
///       |          pending_planner
///       |               |
///       |               v
///       +-------- poll until ready
/// ```
///
/// [`FileStreamState::Scan`]: super::FileStreamState::Scan
pub(super) struct ScanState {
    /// Unopened files that still need to be planned for this stream.
    work_source: WorkSource,
    /// Remaining row limit, if any.
    remain: Option<usize>,
    /// The morselizer used to plan files.
    morselizer: Box<dyn Morselizer>,
    /// Behavior if opening or scanning a file fails.
    on_error: OnError,
    /// CPU-ready planners for the current file.
    ready_planners: VecDeque<Box<dyn MorselPlanner>>,
    /// Ready morsels for the current file.
    ready_morsels: VecDeque<Box<dyn Morsel>>,
    /// The active reader, if any.
    reader: Option<BoxStream<'static, Result<RecordBatch>>>,
    /// The single planner currently blocked on I/O, if any.
    ///
    /// Once the I/O completes, yields the next planner and is pushed back
    /// onto `ready_planners`.
    pending_planner: Option<PendingMorselPlanner>,
    /// Whether to start opening the next file while the active reader is
    /// still producing batches. See [`Self::advance_open_ahead`].
    open_ahead: bool,
    /// Metrics for the active scan queues.
    metrics: FileStreamMetrics,
}

impl ScanState {
    pub(super) fn new(
        work_source: WorkSource,
        remain: Option<usize>,
        morselizer: Box<dyn Morselizer>,
        on_error: OnError,
        open_ahead: bool,
        metrics: FileStreamMetrics,
    ) -> Self {
        Self {
            work_source,
            remain,
            morselizer,
            on_error,
            ready_planners: Default::default(),
            ready_morsels: Default::default(),
            reader: None,
            pending_planner: None,
            open_ahead,
            metrics,
        }
    }

    /// Updates how scan errors are handled while the stream is still active.
    pub(super) fn set_on_error(&mut self, on_error: OnError) {
        self.on_error = on_error;
    }

    /// Drives one iteration of the active scan state.
    ///
    /// Work is attempted in this order:
    /// 1. resolve any pending planner I/O
    /// 2. with `open_ahead`, start planning the next file behind the active
    ///    reader (see [`Self::advance_open_ahead`])
    /// 3. poll the active reader
    /// 4. turn a ready morsel into the active reader
    /// 5. run CPU planning on a ready planner
    /// 6. morselize the next unopened file
    ///
    /// The return [`ScanAndReturn`] tells `poll_inner` how to update the
    /// outer `FileStreamState`.
    pub(super) fn poll_scan(&mut self, cx: &mut Context<'_>) -> ScanAndReturn {
        // Guard a clone of the shared timer so the borrow does not pin
        // `self.metrics` for the whole poll (the look-ahead needs `&mut self`).
        let time_processing = self.metrics.time_processing.clone();
        let _processing_timer: ScopedTimerGuard<'_> = time_processing.timer();

        // Try and resolve outstanding IO first. If it is still pending, check
        // the current reader or ready morsels before yielding. New planning
        // work must still wait for this I/O to resolve.
        if let Some(mut pending_planner) = self.pending_planner.take() {
            match pending_planner.poll_unpin(cx) {
                // IO is still pending
                Poll::Pending => {
                    self.pending_planner = Some(pending_planner);
                }
                // IO resolved, and the planner is ready for CPU work
                Poll::Ready(Ok(planner)) => {
                    self.ready_planners.push_back(planner);
                }
                // IO Error
                Poll::Ready(Err(err)) => {
                    self.metrics.file_open_errors.add(1);
                    self.metrics.time_opening.stop();
                    return match self.on_error {
                        OnError::Skip => {
                            self.metrics.files_processed.add(1);
                            ScanAndReturn::Continue
                        }
                        OnError::Fail => ScanAndReturn::Error(err),
                    };
                }
            }
        }

        // With open-ahead enabled, use the time the active reader is busy to
        // get the next file's planning (and its single outstanding I/O)
        // under way, so per-file open latency overlaps with data reads.
        if self.open_ahead
            && self.reader.is_some()
            && let Some(ret) = self.advance_open_ahead()
        {
            return ret;
        }

        // Next try and get the next batch from the active reader, if any.
        if let Some(reader) = self.reader.as_mut() {
            match reader.poll_next_unpin(cx) {
                // Morsels should ideally only expose ready-to-decode streams,
                // but tolerate pending readers here.
                Poll::Pending => return ScanAndReturn::Return(Poll::Pending),
                Poll::Ready(Some(Ok(batch))) => {
                    self.metrics.time_scanning_until_data.stop();
                    self.metrics.time_scanning_total.stop();
                    // Apply any remaining row limit.
                    let (batch, finished) = match &mut self.remain {
                        Some(remain) => {
                            if *remain > batch.num_rows() {
                                *remain -= batch.num_rows();
                                self.metrics.time_scanning_total.start();
                                (batch, false)
                            } else {
                                let batch = batch.slice(0, *remain);
                                let done = 1 + self.work_source.skipped_on_limit();
                                self.metrics.files_processed.add(done);
                                *remain = 0;
                                (batch, true)
                            }
                        }
                        None => {
                            self.metrics.time_scanning_total.start();
                            (batch, false)
                        }
                    };
                    return if finished {
                        ScanAndReturn::Done(Some(Ok(batch)))
                    } else {
                        ScanAndReturn::Return(Poll::Ready(Some(Ok(batch))))
                    };
                }
                Poll::Ready(Some(Err(err))) => {
                    self.reader = None;
                    self.metrics.file_scan_errors.add(1);
                    self.metrics.time_scanning_until_data.stop();
                    self.metrics.time_scanning_total.stop();
                    return match self.on_error {
                        OnError::Skip => {
                            self.metrics.files_processed.add(1);
                            ScanAndReturn::Continue
                        }
                        OnError::Fail => ScanAndReturn::Error(err),
                    };
                }
                Poll::Ready(None) => {
                    self.reader = None;
                    self.metrics.files_processed.add(1);
                    self.metrics.time_scanning_until_data.stop();
                    self.metrics.time_scanning_total.stop();
                    return ScanAndReturn::Continue;
                }
            }
        }

        // No active reader, but a morsel is ready to become the reader.
        if let Some(morsel) = self.ready_morsels.pop_front() {
            self.metrics.time_opening.stop();
            self.metrics.time_scanning_until_data.start();
            self.metrics.time_scanning_total.start();
            self.reader = Some(morsel.into_stream());
            return ScanAndReturn::Continue;
        }

        // Do not start CPU planning or open another file while planner I/O is
        // still outstanding because they may need additional IO and ScanState
        // currently only permits a single outstanding IO
        if self.pending_planner.is_some() {
            if self.open_ahead {
                // The next file was claimed while a reader was active without
                // starting the opening timer; only the wait that is actually
                // exposed (no reader to drain) counts as opening time.
                self.metrics.time_opening.start_if_stopped();
            }
            return ScanAndReturn::Return(Poll::Pending);
        }

        // No reader or morsel, so try to produce more work via CPU planning.
        if let Some(planner) = self.ready_planners.pop_front() {
            return match planner.plan() {
                Ok(Some(mut plan)) => {
                    // Queue any newly-ready morsels, planners, or planner I/O.
                    self.ready_morsels.extend(plan.take_morsels());
                    self.ready_planners.extend(plan.take_ready_planners());
                    if let Some(pending_planner) = plan.take_pending_planner() {
                        // should not have planned if we have outstanding I/O
                        if self.pending_planner.is_some() {
                            return ScanAndReturn::Error(internal_datafusion_err!(
                                "Conflicting pending planner state in FileStream ScanState"
                            ));
                        }
                        self.pending_planner = Some(pending_planner);
                    }
                    ScanAndReturn::Continue
                }
                Ok(None) => {
                    self.metrics.files_processed.add(1);
                    self.metrics.time_opening.stop();
                    ScanAndReturn::Continue
                }
                Err(err) => {
                    self.metrics.file_open_errors.add(1);
                    self.metrics.time_opening.stop();
                    match self.on_error {
                        OnError::Skip => {
                            self.metrics.files_processed.add(1);
                            ScanAndReturn::Continue
                        }
                        OnError::Fail => ScanAndReturn::Error(err),
                    }
                }
            };
        }

        // No outstanding work remains, so begin planning the next unopened file.
        let part_file = match self.work_source.pop_front() {
            Some(part_file) => part_file,
            None => return ScanAndReturn::Done(None),
        };

        self.metrics.time_opening.start();
        match self.morselizer.plan_file(part_file) {
            Ok(planner) => {
                self.metrics.files_opened.add(1);
                self.ready_planners.push_back(planner);
                ScanAndReturn::Continue
            }
            Err(err) => match self.on_error {
                OnError::Skip => {
                    self.metrics.file_open_errors.add(1);
                    self.metrics.time_opening.stop();
                    self.metrics.files_processed.add(1);
                    ScanAndReturn::Continue
                }
                OnError::Fail => ScanAndReturn::Error(err),
            },
        }
    }
}

impl ScanState {
    /// Drives planning of the *next* file while a reader is active, keeping at
    /// most one file in flight ahead of it.
    ///
    /// Claims the next unopened file once nothing else is queued and runs its
    /// CPU planning until it either yields a ready morsel or blocks on I/O.
    /// The single-outstanding-I/O rule still holds: the pending planner is
    /// polled by [`Self::poll_scan`] before the reader on every iteration, so
    /// the file's open latency overlaps with the reader's data reads instead
    /// of following them.
    ///
    /// The opening timer is deliberately not started here: time spent opening
    /// a file behind an active reader is not exposed to the query. It is
    /// started by `poll_scan` only if the stream later has to wait for that
    /// open with no reader left to drain.
    ///
    /// Returns `Some` when the outer loop must return (an error surfaced by
    /// the look-ahead), `None` to continue with the active reader.
    fn advance_open_ahead(&mut self) -> Option<ScanAndReturn> {
        loop {
            // The next file is already either fully open or waiting on I/O.
            if self.pending_planner.is_some() || !self.ready_morsels.is_empty() {
                return None;
            }

            if let Some(planner) = self.ready_planners.pop_front() {
                match planner.plan() {
                    Ok(Some(mut plan)) => {
                        self.ready_morsels.extend(plan.take_morsels());
                        self.ready_planners.extend(plan.take_ready_planners());
                        if let Some(pending_planner) = plan.take_pending_planner() {
                            if self.pending_planner.is_some() {
                                return Some(ScanAndReturn::Error(
                                    internal_datafusion_err!(
                                        "Conflicting pending planner state in FileStream ScanState"
                                    ),
                                ));
                            }
                            self.pending_planner = Some(pending_planner);
                        }
                    }
                    // Nothing to read from this file (e.g. pruned late).
                    Ok(None) => {
                        self.metrics.files_processed.add(1);
                    }
                    Err(err) => {
                        self.metrics.file_open_errors.add(1);
                        match self.on_error {
                            OnError::Skip => {
                                self.metrics.files_processed.add(1);
                            }
                            OnError::Fail => return Some(ScanAndReturn::Error(err)),
                        }
                    }
                }
                continue;
            }

            // Nothing queued for the next file: claim it (`None` once the
            // work source is drained: keep serving the active reader).
            let part_file = self.work_source.pop_front()?;
            match self.morselizer.plan_file(part_file) {
                Ok(planner) => {
                    self.metrics.files_opened.add(1);
                    self.ready_planners.push_back(planner);
                }
                Err(err) => {
                    self.metrics.file_open_errors.add(1);
                    match self.on_error {
                        OnError::Skip => {
                            self.metrics.files_processed.add(1);
                        }
                        OnError::Fail => return Some(ScanAndReturn::Error(err)),
                    }
                }
            }
        }
    }
}

/// What should be done on the next iteration of [`ScanState::poll_scan`]?
pub(super) enum ScanAndReturn {
    /// Poll again.
    Continue,
    /// Return the provided result without changing the outer state.
    Return(Poll<Option<Result<RecordBatch>>>),
    /// Update the outer `FileStreamState` to `Done` and return the provided result.
    Done(Option<Result<RecordBatch>>),
    /// Update the outer `FileStreamState` to `Error` and return the provided error.
    Error(DataFusionError),
}
