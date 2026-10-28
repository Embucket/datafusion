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

//! [`EarlyStoppingStream`] terminates a Parquet file scan when a dynamic
//! filter narrows after the scan has already started.

use std::pin::Pin;
use std::task::{Context, Poll};

use arrow::array::RecordBatch;
use datafusion_common::Result;
use datafusion_physical_plan::metrics::{Count, PruningMetrics};
use datafusion_pruning::FilePruner;
use futures::{Stream, StreamExt, ready};

/// Wraps an inner RecordBatchStream and a [`FilePruner`]
///
/// This can terminate the scan early when some dynamic filters is updated after
/// the scan starts, so we discover after the scan starts that the file can be
/// pruned (can't have matching rows).
///
/// The pruner is consulted after every batch. With
/// [`Self::with_recheck_before_first_read`] it is also consulted once before
/// the inner stream is first polled, i.e. before the file's first data read:
/// a file that was opened while the filter was still loose (for example ahead
/// of time, by `datafusion.execution.file_stream_open_ahead`) is then skipped
/// without fetching or decoding any row group when the filter has tightened
/// since the file was opened.
pub(super) struct EarlyStoppingStream<S> {
    /// Has the stream finished processing? All subsequent polls will return
    /// None
    done: bool,
    file_pruner: FilePruner,
    files_ranges_pruned_statistics: PruningMetrics,
    /// `Some` until the pre-read check has run, when that check is enabled.
    /// Counts the file ranges it skipped.
    recheck_before_first_read: Option<Count>,
    /// The inner stream
    inner: S,
}

impl<S> EarlyStoppingStream<S> {
    pub(super) fn new(
        stream: S,
        file_pruner: FilePruner,
        files_ranges_pruned_statistics: PruningMetrics,
    ) -> Self {
        Self {
            done: false,
            inner: stream,
            file_pruner,
            files_ranges_pruned_statistics,
            recheck_before_first_read: None,
        }
    }

    /// Also check the pruner once before the first poll of the inner stream,
    /// counting the file ranges skipped this way in `files_pruned`.
    ///
    /// The pruner was already evaluated when the file was opened, so this
    /// check only rebuilds the pruning predicate when a watched dynamic filter
    /// has been updated since; otherwise it costs one atomic load per filter.
    pub(super) fn with_recheck_before_first_read(
        mut self,
        files_pruned: Option<Count>,
    ) -> Self {
        self.recheck_before_first_read = files_pruned;
        self
    }

    /// Stop the scan because the file can no longer contain matching rows.
    fn stop_pruned(&mut self) {
        self.files_ranges_pruned_statistics.add_pruned(1);
        // Previously this file range has been counted as matched
        self.files_ranges_pruned_statistics.subtract_matched(1);
        self.done = true;
    }
}

impl<S> EarlyStoppingStream<S>
where
    S: Stream<Item = Result<RecordBatch>> + Unpin,
{
    fn check_prune(&mut self, input: Result<RecordBatch>) -> Result<Option<RecordBatch>> {
        let batch = input?;

        // Since dynamic filters may have been updated, see if we can stop
        // reading this stream entirely.
        if self.file_pruner.should_prune()? {
            self.stop_pruned();
            Ok(None)
        } else {
            // Return the adapted batch
            Ok(Some(batch))
        }
    }

    /// Runs the optional check before the inner stream is first polled.
    /// Returns `true` when the file was pruned and the stream is done.
    fn check_prune_before_first_read(&mut self) -> Result<bool> {
        let Some(files_pruned) = self.recheck_before_first_read.take() else {
            return Ok(false);
        };
        if self.file_pruner.should_prune()? {
            // Counted exactly like a file stopped after its first batch, so
            // `files_ranges_pruned_statistics` is the same either way.
            self.stop_pruned();
            files_pruned.add(1);
            return Ok(true);
        }
        Ok(false)
    }
}

impl<S> Stream for EarlyStoppingStream<S>
where
    S: Stream<Item = Result<RecordBatch>> + Unpin,
{
    type Item = Result<RecordBatch>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        if self.done {
            return Poll::Ready(None);
        }
        match self.check_prune_before_first_read() {
            Ok(true) => return Poll::Ready(None),
            Ok(false) => {}
            Err(e) => return Poll::Ready(Some(Err(e))),
        }
        match ready!(self.inner.poll_next_unpin(cx)) {
            None => {
                // input done
                self.done = true;
                Poll::Ready(None)
            }
            Some(input_batch) => {
                let output = self.check_prune(input_batch);
                Poll::Ready(output.transpose())
            }
        }
    }
}
