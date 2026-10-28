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

//! End-to-end tests for
//! `datafusion.execution.recheck_dynamic_filter_before_read`.
//!
//! A TopK (`ORDER BY k LIMIT n`) pushes a dynamic filter into the Parquet
//! scan that starts as `true` and tightens once the TopK heap is full. A file
//! claimed before that first update (opened ahead by
//! `datafusion.execution.file_stream_open_ahead`, or claimed by a partition
//! whose first byte range was an empty sliver) passes file pruning and
//! row-group pruning against the empty filter. Without the option, such a
//! single-row-group file is only dropped by the per-batch early stop, after
//! its row group has been fetched. With the option, the scan re-checks the
//! filter right before the first data read and skips the file without
//! fetching it.
//!
//! The race is made deterministic with [`GatedStore`]: metadata reads
//! complete immediately while every data read first yields to the runtime.
//! `FileStream` polls the open-ahead file's pending metadata I/O before it
//! polls the active reader, so the open-ahead file is always planned (and
//! keeps its row group) before the first batch reaches the TopK.

use std::collections::{BTreeSet, HashMap};
use std::fmt::{Display, Formatter};
use std::ops::Range;
use std::sync::Arc;

use arrow::array::{ArrayRef, Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use async_trait::async_trait;
use bytes::Bytes;
use datafusion::datasource::physical_plan::FileScanConfig;
use datafusion::datasource::source::DataSourceExec;
use datafusion::physical_plan::{ExecutionPlan, collect};
use datafusion::prelude::{ParquetReadOptions, SessionConfig, SessionContext};
use datafusion_common::assert_batches_eq;
use datafusion_common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion_physical_plan::metrics::{MetricValue, MetricsSet};
use futures::stream::BoxStream;
use object_store::memory::InMemory;
use object_store::path::Path;
use object_store::{
    CopyOptions, GetOptions, GetRange, GetResult, ListResult, MultipartUpload,
    ObjectMeta, ObjectStore, ObjectStoreExt, PutMultipartOptions, PutOptions, PutPayload,
    PutResult,
};
use parking_lot::Mutex;
use parquet::arrow::ArrowWriter;
use parquet::file::metadata::ParquetMetaDataReader;
use url::Url;

use crate::parquet::utils::MetricsFinder;

/// How many times each data read yields to the runtime before it completes.
///
/// One yield is enough to let `FileStream` plan its open-ahead file before the
/// active file's first batch; more yields let every partition of a
/// multi-partition scan finish its initial claims first, which reproduces the
/// "everything was claimed before the first TopK update" shape seen on remote
/// object stores.
const DATA_READ_YIELDS: usize = 8;

/// [`ObjectStore`] that answers metadata reads (footer, page index)
/// immediately and holds each data read for [`DATA_READ_YIELDS`] yields,
/// recording which file every data read was for.
#[derive(Debug)]
struct GatedStore {
    inner: InMemory,
    /// End offset of the column chunks of each file. A read that ends at or
    /// before it fetches row-group data; any other read is metadata.
    data_end: HashMap<Path, u64>,
    data_reads: Mutex<Vec<Path>>,
}

impl Display for GatedStore {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "GatedStore")
    }
}

impl GatedStore {
    /// Returns `true` (and records the read) when `ranges` fetch row-group
    /// data of `location`.
    fn record_if_data_read<'a>(
        &self,
        location: &Path,
        mut ranges: impl Iterator<Item = &'a Range<u64>>,
    ) -> bool {
        let Some(data_end) = self.data_end.get(location) else {
            return false;
        };
        let is_data = ranges.all(|range| range.end <= *data_end);
        if is_data {
            self.data_reads.lock().push(location.clone());
        }
        is_data
    }

    async fn hold_data_read() {
        for _ in 0..DATA_READ_YIELDS {
            tokio::task::yield_now().await;
        }
    }

    fn take_data_reads(&self) -> Vec<Path> {
        std::mem::take(&mut *self.data_reads.lock())
    }
}

#[async_trait]
impl ObjectStore for GatedStore {
    async fn put_opts(
        &self,
        location: &Path,
        payload: PutPayload,
        opts: PutOptions,
    ) -> object_store::Result<PutResult> {
        self.inner.put_opts(location, payload, opts).await
    }

    async fn put_multipart_opts(
        &self,
        location: &Path,
        opts: PutMultipartOptions,
    ) -> object_store::Result<Box<dyn MultipartUpload>> {
        self.inner.put_multipart_opts(location, opts).await
    }

    async fn get_opts(
        &self,
        location: &Path,
        options: GetOptions,
    ) -> object_store::Result<GetResult> {
        if let Some(GetRange::Bounded(range)) = &options.range
            && self.record_if_data_read(location, std::iter::once(range))
        {
            Self::hold_data_read().await;
        }
        self.inner.get_opts(location, options).await
    }

    async fn get_ranges(
        &self,
        location: &Path,
        ranges: &[Range<u64>],
    ) -> object_store::Result<Vec<Bytes>> {
        if self.record_if_data_read(location, ranges.iter()) {
            Self::hold_data_read().await;
        }
        self.inner.get_ranges(location, ranges).await
    }

    fn list(
        &self,
        prefix: Option<&Path>,
    ) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        self.inner.list(prefix)
    }

    fn list_with_offset(
        &self,
        prefix: Option<&Path>,
        offset: &Path,
    ) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        self.inner.list_with_offset(prefix, offset)
    }

    async fn list_with_delimiter(
        &self,
        prefix: Option<&Path>,
    ) -> object_store::Result<ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, object_store::Result<Path>>,
    ) -> BoxStream<'static, object_store::Result<Path>> {
        self.inner.delete_stream(locations)
    }

    async fn copy_opts(
        &self,
        from: &Path,
        to: &Path,
        options: CopyOptions,
    ) -> object_store::Result<()> {
        self.inner.copy_opts(from, to, options).await
    }
}

/// A set of single-row-group files with disjoint, ascending `k` ranges:
/// file `i` holds `k` in `[i * 1000, i * 1000 + rows[i])`.
struct TestFiles {
    store: Arc<GatedStore>,
    /// Path and row-group data bytes of each file, in key order.
    files: Vec<(Path, u64)>,
}

impl TestFiles {
    async fn new(rows_per_file: &[i64]) -> Self {
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, false),
            Field::new("v", DataType::Utf8, false),
        ]));
        let inner = InMemory::new();
        let mut data_end = HashMap::new();
        let mut files = vec![];
        for (i, rows) in rows_per_file.iter().enumerate() {
            let base = i as i64 * 1000;
            let k: ArrayRef = Arc::new(Int64Array::from_iter_values(base..base + rows));
            let v: ArrayRef = Arc::new(StringArray::from_iter_values(
                (base..base + rows).map(|k| format!("payload-{k:08}")),
            ));
            let batch = RecordBatch::try_new(Arc::clone(&schema), vec![k, v]).unwrap();
            let mut buf = vec![];
            let mut writer =
                ArrowWriter::try_new(&mut buf, Arc::clone(&schema), None).unwrap();
            writer.write(&batch).unwrap();
            writer.close().unwrap();
            let bytes = Bytes::from(buf);

            let metadata = ParquetMetaDataReader::new()
                .parse_and_finish(&bytes)
                .unwrap();
            assert_eq!(metadata.num_row_groups(), 1);
            let (end, data_bytes) = metadata.row_group(0).columns().iter().fold(
                (0, 0),
                |(end, total), column| {
                    let (start, len) = column.byte_range();
                    (end.max(start + len), total + len)
                },
            );

            let path = Path::from(format!("data/file_{i:02}.parquet"));
            inner.put(&path, bytes.into()).await.unwrap();
            data_end.insert(path.clone(), end);
            files.push((path, data_bytes));
        }
        let store = Arc::new(GatedStore {
            inner,
            data_end,
            data_reads: Mutex::new(vec![]),
        });
        Self { store, files }
    }

    fn data_bytes(&self, path: &Path) -> u64 {
        self.files
            .iter()
            .find_map(|(p, bytes)| (p == path).then_some(*bytes))
            .unwrap()
    }
}

/// What one TopK run read and recorded.
#[derive(Debug)]
struct RunOutcome {
    batches: Vec<RecordBatch>,
    /// Files whose row group was fetched (each file has a single row group).
    fetched_files: BTreeSet<Path>,
    /// Number of data reads issued (one per fetched row group).
    data_reads: usize,
    bytes_scanned: usize,
    /// `(pruned, matched)` of `files_ranges_pruned_statistics`.
    files_ranges: (usize, usize),
    /// `(pruned, matched)` of `row_groups_pruned_statistics`.
    row_groups: (usize, usize),
    /// `files_ranges_pruned_before_read`; `None` when not registered.
    pruned_before_read: Option<usize>,
    /// `(ranges, empty slivers)` of the scan's file groups.
    ranges: (usize, usize),
}

fn pruning(metrics: &MetricsSet, name: &str) -> (usize, usize) {
    match metrics.sum_by_name(name) {
        Some(MetricValue::PruningMetrics {
            pruning_metrics, ..
        }) => (pruning_metrics.pruned(), pruning_metrics.matched()),
        other => panic!("unexpected {name}: {other:?}"),
    }
}

/// Count the byte ranges of the scan and how many of them are empty slivers
/// (the tail of a split single-row-group file, which holds no row group).
fn file_ranges(plan: &Arc<dyn ExecutionPlan>) -> (usize, usize) {
    let mut found = None;
    plan.apply(|node| {
        if let Some(exec) = node.downcast_ref::<DataSourceExec>()
            && let Some(config) = exec.data_source().downcast_ref::<FileScanConfig>()
        {
            let files = config.file_groups.iter().flat_map(|g| g.iter());
            let (mut ranges, mut slivers) = (0, 0);
            for file in files {
                ranges += 1;
                // Each file's only row group starts right after the 4-byte
                // magic, so every range that starts later is empty.
                if file.range.as_ref().is_some_and(|r| r.start > 4) {
                    slivers += 1;
                }
            }
            found = Some((ranges, slivers));
        }
        Ok(TreeNodeRecursion::Continue)
    })
    .unwrap();
    found.expect("plan has a file scan")
}

/// Run `sql` with open-ahead on, on `target_partitions` partitions.
///
/// File scans are only split into byte ranges when the table has more rows
/// than `batch_size`, so tests that need empty slivers pass a small one.
async fn run_topk(
    files: &TestFiles,
    target_partitions: usize,
    batch_size: usize,
    recheck: bool,
    sql: &str,
) -> RunOutcome {
    let mut config = SessionConfig::new()
        .with_target_partitions(target_partitions)
        .with_batch_size(batch_size);
    let options = config.options_mut();
    options.execution.file_stream_open_ahead = true;
    options.execution.recheck_dynamic_filter_before_read = recheck;
    // Split files into byte ranges even though they are small.
    options.optimizer.repartition_file_min_size = 0;
    let ctx = SessionContext::new_with_config(config);
    let url = Url::parse("gated://bucket").unwrap();
    ctx.register_object_store(&url, Arc::clone(&files.store) as _);
    ctx.register_parquet("t", "gated://bucket/data/", ParquetReadOptions::default())
        .await
        .unwrap();

    let plan = ctx
        .sql(sql)
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    let ranges = file_ranges(&plan);
    assert!(
        files.store.take_data_reads().is_empty(),
        "planning must not read row-group data"
    );
    let batches = collect(Arc::clone(&plan), ctx.task_ctx()).await.unwrap();
    let data_reads = files.store.take_data_reads();

    let metrics = MetricsFinder::find_metrics(plan.as_ref())
        .unwrap()
        .aggregate_by_name();
    RunOutcome {
        batches,
        fetched_files: data_reads.iter().cloned().collect(),
        data_reads: data_reads.len(),
        bytes_scanned: metrics.sum_by_name("bytes_scanned").unwrap().as_usize(),
        files_ranges: pruning(&metrics, "files_ranges_pruned_statistics"),
        row_groups: pruning(&metrics, "row_groups_pruned_statistics"),
        pruned_before_read: metrics
            .sum_by_name("files_ranges_pruned_before_read")
            .map(|m| m.as_usize()),
        ranges,
    }
}

const TOPK_SQL: &str = "SELECT * FROM t ORDER BY k LIMIT 3";

fn assert_topk_result(batches: &[RecordBatch]) {
    assert_batches_eq!(
        [
            "+---+------------------+",
            "| k | v                |",
            "+---+------------------+",
            "| 0 | payload-00000000 |",
            "| 1 | payload-00000001 |",
            "| 2 | payload-00000002 |",
            "+---+------------------+",
        ],
        batches
    );
}

/// One partition with open-ahead: while `file_00` is read, `file_01` is
/// claimed and planned against the still-empty filter. The TopK fills from
/// `file_00`'s first batch, and the tightened filter excludes every other
/// file. Without the re-check `file_01`'s row group is fetched before the
/// early stop drops it; with the re-check it is never fetched. The remaining
/// files are claimed after the update and pruned at open in both runs.
#[tokio::test]
async fn recheck_skips_open_ahead_file_single_partition() {
    let files = TestFiles::new(&[100; 6]).await;
    let file = |i: usize| files.files[i].0.clone();

    let off = run_topk(&files, 1, 8192, false, TOPK_SQL).await;
    let on = run_topk(&files, 1, 8192, true, TOPK_SQL).await;
    assert_topk_result(&off.batches);
    assert_topk_result(&on.batches);
    assert_eq!(off.ranges, (6, 0));
    assert_eq!(on.ranges, (6, 0));

    // Without the re-check: two row groups fetched, the open-ahead file is
    // stopped after its first batch.
    assert_eq!(
        off.fetched_files,
        BTreeSet::from([file(0), file(1)]),
        "{off:?}"
    );
    assert_eq!(off.data_reads, 2, "{off:?}");
    assert_eq!(
        off.bytes_scanned as u64,
        files.data_bytes(&file(0)) + files.data_bytes(&file(1))
    );
    assert_eq!(
        off.pruned_before_read, None,
        "metric must not be registered"
    );

    // With the re-check: only `file_00` is fetched.
    assert_eq!(on.fetched_files, BTreeSet::from([file(0)]), "{on:?}");
    assert_eq!(on.data_reads, 1, "{on:?}");
    assert_eq!(on.bytes_scanned as u64, files.data_bytes(&file(0)));
    assert_eq!(on.pruned_before_read, Some(1), "{on:?}");

    // Both runs end with only `file_00` matched: a file stopped before its
    // first read is counted like one stopped after its first batch.
    assert_eq!(off.files_ranges, (5, 1), "{off:?}");
    assert_eq!(on.files_ranges, (5, 1), "{on:?}");
    // Both open-ahead candidates were planned against the empty filter.
    assert_eq!(off.row_groups, (0, 2), "{off:?}");
    assert_eq!(on.row_groups, (0, 2), "{on:?}");
}

/// Five partitions over ten files of slightly different sizes: byte-range
/// splitting turns them into fourteen ranges, four of which are empty slivers
/// (the tails of split files). Every partition claims its first range, opens
/// a second one ahead, and partitions whose range was a sliver claim again,
/// all before the first TopK update. Without the re-check every claimed file
/// whose row group survived that empty filter is fetched; with it, only the
/// file each partition was already reading is fetched, and the files opened
/// ahead are skipped before their first read.
///
/// On the single-threaded test runtime this currently fetches 9 of the 10
/// row groups without the re-check and 5 with it; the assertions below check
/// the properties that hold for any interleaving.
#[tokio::test]
async fn recheck_skips_open_ahead_files_with_empty_slivers() {
    let rows: Vec<i64> = (0..10).map(|i| 100 + 13 * i).collect();
    let files = TestFiles::new(&rows).await;

    let off = run_topk(&files, 5, 100, false, TOPK_SQL).await;
    let on = run_topk(&files, 5, 100, true, TOPK_SQL).await;
    assert_topk_result(&off.batches);
    assert_topk_result(&on.batches);
    assert_eq!(off.ranges, (14, 4));
    assert_eq!(on.ranges, (14, 4));

    // Planning before the first TopK update is the same in both runs.
    assert_eq!(off.row_groups, on.row_groups);
    // Without the re-check every row group that survived the empty filter is
    // fetched.
    assert_eq!(off.fetched_files.len(), off.row_groups.1, "{off:?}");

    // Each fetched file is fetched once, and `bytes_scanned` is exactly the
    // row-group data of the fetched files.
    for outcome in [&off, &on] {
        assert_eq!(
            outcome.data_reads,
            outcome.fetched_files.len(),
            "{outcome:?}"
        );
        let fetched_bytes: u64 = outcome
            .fetched_files
            .iter()
            .map(|p| files.data_bytes(p))
            .sum();
        assert_eq!(outcome.bytes_scanned as u64, fetched_bytes, "{outcome:?}");
        let (pruned, matched) = outcome.files_ranges;
        assert_eq!(pruned + matched, 14, "{outcome:?}");
    }

    // With the re-check at most one file per partition is fetched: the one it
    // was reading when the filter was first updated.
    assert!(on.fetched_files.len() <= 5, "{on:?}");
    assert!(
        on.fetched_files.len() < off.fetched_files.len(),
        "re-check must skip files opened ahead\noff: {off:?}\non: {on:?}"
    );
    assert!(on.fetched_files.is_subset(&off.fetched_files));
    assert!(on.bytes_scanned < off.bytes_scanned);

    // Every file fetched without the re-check but not with it was skipped by
    // the re-check; empty slivers opened ahead may be skipped too.
    let skipped_real = off.fetched_files.len() - on.fetched_files.len();
    let pruned_before_read = on.pruned_before_read.expect("metric registered");
    assert!(pruned_before_read >= skipped_real, "{on:?}");
    assert!(pruned_before_read <= skipped_real + 4, "{on:?}");
    assert_eq!(off.pruned_before_read, None);
    assert!(on.files_ranges.1 <= off.files_ranges.1);
}

/// With the option on, scans without a dynamic filter keep their metrics and
/// I/O unchanged: the new counter is not registered and nothing is skipped.
#[tokio::test]
async fn recheck_does_not_touch_scans_without_dynamic_filter() {
    let files = TestFiles::new(&[100; 4]).await;
    for sql in [
        "SELECT * FROM t WHERE k >= 2000",
        "SELECT * FROM t",
        "SELECT * FROM t ORDER BY k",
    ] {
        let off = run_topk(&files, 1, 8192, false, sql).await;
        let on = run_topk(&files, 1, 8192, true, sql).await;
        assert_eq!(on.pruned_before_read, None, "{sql}");
        assert_eq!(off.pruned_before_read, None, "{sql}");
        assert_eq!(on.fetched_files, off.fetched_files, "{sql}");
        assert_eq!(on.bytes_scanned, off.bytes_scanned, "{sql}");
        assert_eq!(on.files_ranges, off.files_ranges, "{sql}");
        assert_eq!(on.batches, off.batches, "{sql}");
    }
}
