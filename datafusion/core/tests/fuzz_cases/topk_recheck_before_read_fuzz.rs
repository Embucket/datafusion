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

//! Randomized differential test for
//! `datafusion.execution.recheck_dynamic_filter_before_read`.
//!
//! Each case writes a random set of Parquet files to an in-memory object
//! store that delays metadata and data reads differently, then runs TopK
//! queries (`ORDER BY ... LIMIT ... [OFFSET ...]`) with the option off and on
//! on a multi-threaded runtime. Both runs are compared against an oracle that
//! sorts every input row with `lexsort_to_indices`:
//!
//! * with `rid` appended as the final sort key the result must equal the
//!   oracle exactly, in every column;
//! * without the tiebreaker the sort-key columns must equal the oracle row by
//!   row, every returned row must equal the input row with the same `rid` in
//!   every column, and no `rid` may repeat.
//!
//! The test asserts that the re-check actually skipped files in some cases
//! (`files_ranges_pruned_before_read` > 0), so it cannot pass vacuously.
//!
//! Float64 keys include -0.0, +0.0 and infinities in every file, but NaN only
//! in files written without statistics. Parquet min/max statistics exclude
//! NaN, and the scan currently trusts them in ways that are wrong for NaN
//! regardless of this option: a file whose non-NaN values are all equal is
//! treated as constant (its NaN rows are read back as that value), and a
//! `k > x` filter can prune a file whose only rows above `x` are NaN.
//!
//! Set `TOPK_RECHECK_FUZZ_SEED` to replay a seed and
//! `TOPK_RECHECK_FUZZ_CASES` to change the number of cases. Timing is not
//! replayed by a seed; a failing seed is best looped.

use std::collections::{HashMap, HashSet};
use std::fmt::{Display, Formatter};
use std::ops::Range;
use std::sync::Arc;
use std::time::Duration;

use arrow::array::{
    Array, ArrayRef, Date32Array, Decimal128Array, Float64Array, Int64Array, RecordBatch,
    StringArray,
};
use arrow::compute::{SortColumn, SortOptions, cast, concat_batches, lexsort_to_indices};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use async_trait::async_trait;
use bytes::Bytes;
use datafusion::datasource::source::DataSourceExec;
use datafusion::physical_plan::{ExecutionPlan, collect};
use datafusion::prelude::{ParquetReadOptions, SessionConfig, SessionContext};
use datafusion_common::tree_node::{TreeNode, TreeNodeRecursion};
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
use parquet::file::properties::{EnabledStatistics, WriterProperties};
use rand::prelude::IndexedRandom;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use url::Url;

const DEFAULT_CASES: usize = 64;

/// [`ObjectStore`] over [`InMemory`] that delays metadata reads (footer,
/// page index) and row-group data reads by different random amounts.
#[derive(Debug)]
struct DelayedStore {
    inner: InMemory,
    /// End offset of the column chunks of each file; reads ending at or
    /// before it fetch row-group data.
    data_end: HashMap<Path, u64>,
    metadata_delay_us: Range<u64>,
    data_delay_us: Range<u64>,
    rng: Mutex<StdRng>,
}

impl Display for DelayedStore {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "DelayedStore")
    }
}

impl DelayedStore {
    async fn delay<'a>(
        &self,
        location: &Path,
        mut ranges: impl Iterator<Item = &'a Range<u64>>,
    ) {
        let is_data = self
            .data_end
            .get(location)
            .is_some_and(|end| ranges.all(|range| range.end <= *end));
        let range = if is_data {
            &self.data_delay_us
        } else {
            &self.metadata_delay_us
        };
        let micros = if range.is_empty() {
            0
        } else {
            self.rng.lock().random_range(range.clone())
        };
        if micros == 0 {
            tokio::task::yield_now().await;
        } else {
            tokio::time::sleep(Duration::from_micros(micros)).await;
        }
    }
}

#[async_trait]
impl ObjectStore for DelayedStore {
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
        match &options.range {
            Some(GetRange::Bounded(range)) => {
                self.delay(location, std::iter::once(range)).await
            }
            _ => self.delay(location, std::iter::empty()).await,
        }
        self.inner.get_opts(location, options).await
    }

    async fn get_ranges(
        &self,
        location: &Path,
        ranges: &[Range<u64>],
    ) -> object_store::Result<Vec<Bytes>> {
        self.delay(location, ranges.iter()).await;
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

#[derive(Clone, Copy, Debug)]
enum KeyType {
    Int64,
    /// Includes -0.0, +0.0, infinities and (only without statistics) NaN.
    Float64,
    Utf8,
    Decimal128,
    Date32,
}

impl KeyType {
    fn data_type(self) -> DataType {
        match self {
            KeyType::Int64 => DataType::Int64,
            KeyType::Float64 => DataType::Float64,
            KeyType::Utf8 => DataType::Utf8,
            KeyType::Decimal128 => DataType::Decimal128(18, 3),
            KeyType::Date32 => DataType::Date32,
        }
    }

    /// Build a key column from order-preserving ordinals (`None` = NULL).
    ///
    /// `with_stats` is whether the file is written with statistics; NaN is
    /// only generated without them (see the module documentation).
    fn build(
        self,
        ordinals: &[Option<i64>],
        with_stats: bool,
        rng: &mut StdRng,
    ) -> ArrayRef {
        match self {
            KeyType::Int64 => Arc::new(Int64Array::from(ordinals.to_vec())),
            KeyType::Float64 => {
                let specials: &[f64] = if with_stats {
                    &[-0.0, 0.0, f64::INFINITY, f64::NEG_INFINITY]
                } else {
                    &[f64::NAN, -0.0, 0.0, f64::INFINITY, f64::NEG_INFINITY]
                };
                Arc::new(Float64Array::from_iter(ordinals.iter().map(|o| {
                    o.map(|o| {
                        if rng.random_bool(0.04) {
                            *specials.choose(rng).unwrap()
                        } else {
                            o as f64 / 4.0
                        }
                    })
                })))
            }
            KeyType::Utf8 => Arc::new(StringArray::from_iter(
                ordinals
                    .iter()
                    .map(|o| o.map(|o| format!("s{:07}", o + 1_000_000))),
            )),
            KeyType::Decimal128 => Arc::new(
                Decimal128Array::from_iter(
                    ordinals.iter().map(|o| o.map(|o| o as i128 * 1000 + 7)),
                )
                .with_precision_and_scale(18, 3)
                .unwrap(),
            ),
            KeyType::Date32 => Arc::new(Date32Array::from_iter(
                ordinals.iter().map(|o| o.map(|o| o as i32)),
            )),
        }
    }
}

/// How key ordinals are spread across files.
#[derive(Clone, Copy, Debug)]
enum Layout {
    /// File `i` holds ascending keys from its own disjoint range.
    SortedDisjoint,
    /// Neighbouring files' key ranges overlap.
    Overlapping,
    /// Uniformly random keys over a wide domain.
    Random,
    /// Few distinct keys, lots of ties.
    DuplicateHeavy,
}

fn ordinals(
    layout: Layout,
    file_index: usize,
    rows: usize,
    constant: bool,
    null_fraction: f64,
    rng: &mut StdRng,
) -> Vec<Option<i64>> {
    let file = file_index as i64;
    let constant_value = rng.random_range(-500..500);
    (0..rows as i64)
        .map(|row| {
            if rng.random_bool(null_fraction) {
                return None;
            }
            if constant {
                return Some(constant_value);
            }
            Some(match layout {
                Layout::SortedDisjoint => file * 1000 + row - 2000,
                Layout::Overlapping => file * 60 + rng.random_range(0..200) - 300,
                Layout::Random => rng.random_range(-5000..5000),
                Layout::DuplicateHeavy => rng.random_range(-4..4),
            })
        })
        .collect()
}

/// One generated data set: the files in the store and every row in `rid`
/// order (`rid` equals the row's index in `all_rows`).
struct DataSet {
    store: Arc<DelayedStore>,
    schema: SchemaRef,
    all_rows: RecordBatch,
    /// Row count of each file, in generation order.
    file_rows: Vec<usize>,
    description: String,
}

const KEY_TYPES: [KeyType; 5] = [
    KeyType::Int64,
    KeyType::Float64,
    KeyType::Utf8,
    KeyType::Decimal128,
    KeyType::Date32,
];
const LAYOUTS: [Layout; 4] = [
    Layout::SortedDisjoint,
    Layout::Overlapping,
    Layout::Random,
    Layout::DuplicateHeavy,
];

/// Generate a data set. A `race_prone` case uses sorted, disjoint `k1`
/// ranges, files with statistics and data reads much slower than metadata
/// reads, which is the shape where files opened ahead are skipped by the
/// re-check.
async fn generate_data(rng: &mut StdRng, race_prone: bool) -> DataSet {
    let k1_type = *KEY_TYPES.choose(rng).unwrap();
    let k2_type = *KEY_TYPES.choose(rng).unwrap();
    let k1_layout = if race_prone {
        Layout::SortedDisjoint
    } else {
        *LAYOUTS.choose(rng).unwrap()
    };
    let k2_layout = *LAYOUTS.choose(rng).unwrap();
    let num_files = rng.random_range(2..=12);
    let schema = Arc::new(Schema::new(vec![
        Field::new("rid", DataType::Int64, false),
        Field::new("k1", k1_type.data_type(), true),
        Field::new("k2", k2_type.data_type(), true),
        Field::new("payload", DataType::Utf8, false),
    ]));

    // Write files in a shuffled path order so that the scan's statistics
    // based file reordering has work to do.
    let mut path_order: Vec<usize> = (0..num_files).collect();
    for i in (1..path_order.len()).rev() {
        path_order.swap(i, rng.random_range(0..=i));
    }

    let inner = InMemory::new();
    let mut data_end = HashMap::new();
    let mut batches = vec![];
    let mut file_rows = vec![];
    let mut file_notes = vec![];
    let mut next_rid = 0i64;
    for (file_index, path_index) in path_order.iter().copied().enumerate() {
        let rows = if rng.random_bool(0.1) {
            0
        } else {
            rng.random_range(1..=300)
        };
        let constant = rng.random_bool(0.15);
        let with_stats = race_prone || rng.random_bool(0.75);
        let null_fraction = *[0.0, 0.0, 0.1, 0.5].choose(rng).unwrap();
        let row_group_rows = *[None, None, Some(64), Some(100)].choose(rng).unwrap();

        let k1 = ordinals(k1_layout, file_index, rows, constant, null_fraction, rng);
        let k2 = ordinals(k2_layout, file_index, rows, false, null_fraction, rng);
        let rid = next_rid..next_rid + rows as i64;
        next_rid += rows as i64;
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int64Array::from_iter_values(rid.clone())),
                k1_type.build(&k1, with_stats, rng),
                k2_type.build(&k2, with_stats, rng),
                Arc::new(StringArray::from_iter_values(
                    rid.map(|r| format!("payload-{r}")),
                )),
            ],
        )
        .unwrap();

        let mut props = WriterProperties::builder();
        if !with_stats {
            props = props.set_statistics_enabled(EnabledStatistics::None);
        }
        if let Some(row_group_rows) = row_group_rows {
            props = props.set_max_row_group_row_count(Some(row_group_rows));
        }
        let mut buf = vec![];
        let mut writer =
            ArrowWriter::try_new(&mut buf, Arc::clone(&schema), Some(props.build()))
                .unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        let bytes = Bytes::from(buf);
        let metadata = ParquetMetaDataReader::new()
            .parse_and_finish(&bytes)
            .unwrap();
        let end = metadata
            .row_groups()
            .iter()
            .flat_map(|rg| rg.columns())
            .map(|c| {
                let (start, len) = c.byte_range();
                start + len
            })
            .max()
            .unwrap_or(0);
        let path = Path::from(format!("data/file_{path_index:02}.parquet"));
        inner.put(&path, bytes.into()).await.unwrap();
        data_end.insert(path, end);

        file_notes.push(format!(
            "{rows}r{}{}{}",
            if constant { ",const" } else { "" },
            if with_stats { "" } else { ",nostats" },
            row_group_rows
                .map(|r| format!(",rg{r}"))
                .unwrap_or_default(),
        ));
        file_rows.push(rows);
        batches.push(batch);
    }

    // Different, random delays for metadata and data reads. Data reads are
    // usually slower than footers so files opened ahead are planned before
    // the TopK has seen any batch.
    let delays = if race_prone {
        2
    } else {
        rng.random_range(0..4)
    };
    let (metadata_delay_us, data_delay_us) = match delays {
        0 => (0..1, 0..1),
        1 => (0..200, 500..4000),
        2 => (0..1, 1000..3000),
        _ => (100..2000, 0..500),
    };
    let description = format!(
        "k1={k1_type:?}/{k1_layout:?} k2={k2_type:?}/{k2_layout:?} \
         files=[{}] metadata_delay_us={metadata_delay_us:?} \
         data_delay_us={data_delay_us:?}",
        file_notes.join(" "),
    );
    let store = Arc::new(DelayedStore {
        inner,
        data_end,
        metadata_delay_us,
        data_delay_us,
        rng: Mutex::new(StdRng::seed_from_u64(rng.random())),
    });
    let all_rows = concat_batches(&schema, &batches).unwrap();
    DataSet {
        store,
        schema,
        all_rows,
        file_rows,
        description,
    }
}

#[derive(Clone, Debug)]
struct OrderKey {
    column: &'static str,
    descending: bool,
    nulls_first: bool,
}

impl OrderKey {
    fn sql(&self) -> String {
        format!(
            "{} {} {}",
            self.column,
            if self.descending { "DESC" } else { "ASC" },
            if self.nulls_first {
                "NULLS FIRST"
            } else {
                "NULLS LAST"
            }
        )
    }
}

#[derive(Debug)]
struct QueryCase {
    keys: Vec<OrderKey>,
    limit: usize,
    offset: usize,
}

impl QueryCase {
    fn sql(&self, tiebreak: bool) -> String {
        let mut order: Vec<String> = self.keys.iter().map(OrderKey::sql).collect();
        if tiebreak {
            order.push("rid ASC".to_string());
        }
        let offset = if self.offset > 0 {
            format!(" OFFSET {}", self.offset)
        } else {
            String::new()
        };
        format!(
            "SELECT * FROM t ORDER BY {} LIMIT {}{offset}",
            order.join(", "),
            self.limit
        )
    }
}

/// Generate a query; a `race_prone` one orders by `k1 ASC` with a small
/// LIMIT (see [`generate_data`]).
fn generate_query(rng: &mut StdRng, file_rows: &[usize], race_prone: bool) -> QueryCase {
    let mut keys = vec![];
    let num_keys = if race_prone {
        1
    } else {
        rng.random_range(1..=2)
    };
    for column in ["k1", "k2"].into_iter().take(num_keys) {
        keys.push(OrderKey {
            column,
            descending: !race_prone && rng.random_bool(0.5),
            nulls_first: rng.random_bool(0.5),
        });
    }
    if race_prone {
        return QueryCase {
            keys,
            limit: *[1, 2, 3, 10].choose(rng).unwrap(),
            offset: 0,
        };
    }
    // LIMIT values around file boundaries (cumulative row counts), plus small
    // and whole-table limits.
    let total: usize = file_rows.iter().sum();
    let mut candidates = vec![1, 2, 3, 10, total.saturating_sub(1), total, total + 1];
    let mut cumulative = 0;
    for rows in file_rows {
        cumulative += rows;
        candidates.extend([cumulative.saturating_sub(1), cumulative, cumulative + 1]);
    }
    candidates.extend(file_rows.iter().map(|r| r + 1));
    let limit = (*candidates.choose(rng).unwrap()).max(1);
    let offset = if rng.random_bool(0.2) {
        rng.random_range(1..=5)
    } else {
        0
    };
    QueryCase {
        keys,
        limit,
        offset,
    }
}

#[derive(Debug)]
struct ScanConfig {
    target_partitions: usize,
    open_ahead: bool,
    pushdown_filters: bool,
    split_files: bool,
    work_stealing: bool,
    page_index: bool,
    batch_size: usize,
}

fn generate_config(rng: &mut StdRng, race_prone: bool) -> ScanConfig {
    ScanConfig {
        target_partitions: *[1, 2, 3, 5, 8].choose(rng).unwrap(),
        open_ahead: race_prone || rng.random_bool(0.75),
        pushdown_filters: rng.random_bool(0.3),
        split_files: rng.random_bool(0.6),
        work_stealing: rng.random_bool(0.8),
        page_index: rng.random_bool(0.5),
        batch_size: if rng.random_bool(0.3) {
            *[1, 2, 3, 7, 64, 8192].choose(rng).unwrap()
        } else {
            rng.random_range(1..=8192)
        },
    }
}

impl ScanConfig {
    fn session_config(&self, recheck: bool) -> SessionConfig {
        let mut config = SessionConfig::new()
            .with_target_partitions(self.target_partitions)
            .with_batch_size(self.batch_size);
        let options = config.options_mut();
        options.execution.file_stream_open_ahead = self.open_ahead;
        options.execution.recheck_dynamic_filter_before_read = recheck;
        options.execution.enable_file_stream_work_stealing = self.work_stealing;
        options.execution.parquet.pushdown_filters = self.pushdown_filters;
        options.execution.parquet.enable_page_index = self.page_index;
        if self.split_files {
            options.optimizer.repartition_file_min_size = 0;
        }
        config
    }
}

/// Sum a counter over every Parquet scan in `plan`.
fn sum_scan_metric(plan: &Arc<dyn ExecutionPlan>, name: &str) -> Option<usize> {
    let mut total = None;
    plan.apply(|node| {
        if node.downcast_ref::<DataSourceExec>().is_some()
            && let Some(value) = node
                .metrics()
                .and_then(|m| m.sum_by_name(name))
                .map(|v| v.as_usize())
        {
            *total.get_or_insert(0) += value;
        }
        Ok(TreeNodeRecursion::Continue)
    })
    .unwrap();
    total
}

async fn run_query(
    data: &DataSet,
    config: &ScanConfig,
    recheck: bool,
    sql: &str,
) -> (RecordBatch, Option<usize>) {
    let ctx = SessionContext::new_with_config(config.session_config(recheck));
    let url = Url::parse("delayed://bucket").unwrap();
    ctx.register_object_store(&url, Arc::clone(&data.store) as _);
    ctx.register_parquet("t", "delayed://bucket/data/", ParquetReadOptions::default())
        .await
        .unwrap();
    let plan = ctx
        .sql(sql)
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    let batches = collect(Arc::clone(&plan), ctx.task_ctx()).await.unwrap();
    let pruned_before_read = sum_scan_metric(&plan, "files_ranges_pruned_before_read");

    // Cast back to the input types (strings are read as views).
    let columns = data
        .schema
        .fields()
        .iter()
        .map(|field| {
            let parts = batches
                .iter()
                .map(|b| {
                    let column = b.column_by_name(field.name()).unwrap();
                    cast(column, field.data_type()).unwrap()
                })
                .collect::<Vec<_>>();
            let parts = parts.iter().map(|a| a.as_ref()).collect::<Vec<_>>();
            if parts.is_empty() {
                arrow::array::new_empty_array(field.data_type())
            } else {
                arrow::compute::concat(&parts).unwrap()
            }
        })
        .collect();
    let result = RecordBatch::try_new(Arc::clone(&data.schema), columns).unwrap();
    (result, pruned_before_read)
}

/// Oracle: every input row sorted by `query`'s keys (plus `rid` when
/// `tiebreak`), then the LIMIT/OFFSET window applied.
fn oracle(data: &DataSet, query: &QueryCase, tiebreak: bool) -> RecordBatch {
    let rows = &data.all_rows;
    let mut sort_columns: Vec<SortColumn> = query
        .keys
        .iter()
        .map(|key| SortColumn {
            values: Arc::clone(rows.column_by_name(key.column).unwrap()),
            options: Some(SortOptions {
                descending: key.descending,
                nulls_first: key.nulls_first,
            }),
        })
        .collect();
    if tiebreak {
        sort_columns.push(SortColumn {
            values: Arc::clone(rows.column_by_name("rid").unwrap()),
            options: Some(SortOptions::default()),
        });
    }
    let indices = lexsort_to_indices(&sort_columns, None).unwrap();
    let sorted = arrow::compute::take_record_batch(rows, &indices).unwrap();
    let start = query.offset.min(sorted.num_rows());
    let len = query.limit.min(sorted.num_rows() - start);
    sorted.slice(start, len)
}

/// Compare `actual` with `expected` exactly (bitwise for floats, so NaN and
/// the sign of zero matter).
fn check_exact(expected: &RecordBatch, actual: &RecordBatch) -> Result<(), String> {
    if expected.num_rows() != actual.num_rows() {
        return Err(format!(
            "row count {} != expected {}",
            actual.num_rows(),
            expected.num_rows()
        ));
    }
    for (i, field) in expected.schema().fields().iter().enumerate() {
        if expected.column(i).to_data() != actual.column(i).to_data() {
            return Err(format!(
                "column {} differs:\nexpected {:?}\nactual {:?}",
                field.name(),
                expected.column(i),
                actual.column(i)
            ));
        }
    }
    Ok(())
}

/// Check a result without a tiebreaker: sort keys equal the oracle row by row,
/// each row equals the input row with the same `rid`, and `rid`s are unique.
fn check_without_tiebreak(
    data: &DataSet,
    query: &QueryCase,
    expected: &RecordBatch,
    actual: &RecordBatch,
) -> Result<(), String> {
    if expected.num_rows() != actual.num_rows() {
        return Err(format!(
            "row count {} != expected {}",
            actual.num_rows(),
            expected.num_rows()
        ));
    }
    for key in &query.keys {
        let e = expected.column_by_name(key.column).unwrap();
        let a = actual.column_by_name(key.column).unwrap();
        if e.to_data() != a.to_data() {
            return Err(format!(
                "sort key {} differs:\nexpected {e:?}\nactual {a:?}",
                key.column
            ));
        }
    }
    let rids = actual
        .column_by_name("rid")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let mut seen = HashSet::new();
    for row in 0..actual.num_rows() {
        let rid = rids.value(row);
        if !seen.insert(rid) {
            return Err(format!("rid {rid} returned twice"));
        }
        let Ok(input_row) = usize::try_from(rid) else {
            return Err(format!("unknown rid {rid}"));
        };
        if input_row >= data.all_rows.num_rows() {
            return Err(format!("unknown rid {rid}"));
        }
        for (i, field) in data.schema.fields().iter().enumerate() {
            let input = data.all_rows.column(i).slice(input_row, 1);
            let output = actual.column(i).slice(row, 1);
            if input.to_data() != output.to_data() {
                return Err(format!(
                    "row with rid {rid} has wrong {}: input {input:?} output {output:?}",
                    field.name()
                ));
            }
        }
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_fuzz_topk_recheck_before_read() {
    let seed = std::env::var("TOPK_RECHECK_FUZZ_SEED")
        .ok()
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse::<u64>()
                .expect("TOPK_RECHECK_FUZZ_SEED must be a u64")
        })
        .unwrap_or_else(|| rand::rng().random());
    let cases = std::env::var("TOPK_RECHECK_FUZZ_CASES")
        .ok()
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse::<usize>()
                .expect("TOPK_RECHECK_FUZZ_CASES must be a usize")
        })
        .unwrap_or(DEFAULT_CASES);
    println!("topk recheck fuzz: TOPK_RECHECK_FUZZ_SEED={seed} cases={cases}");
    let mut rng = StdRng::seed_from_u64(seed);

    let mut engaged_cases = 0;
    let mut files_pruned_before_read = 0;
    let mut failures = vec![];
    for case in 0..cases {
        // Every fourth case is biased towards the open-ahead race so that the
        // re-check is reliably exercised; the others are fully random.
        let race_prone = case % 4 == 0;
        let data = generate_data(&mut rng, race_prone).await;
        let config = generate_config(&mut rng, race_prone);
        let query = generate_query(&mut rng, &data.file_rows, race_prone);
        let context = format!(
            "seed={seed} case={case} race_prone={race_prone} {}\n{config:?}\n{query:?}",
            data.description
        );
        println!("{context}");

        let mut case_pruned = 0;
        for tiebreak in [true, false] {
            let sql = query.sql(tiebreak);
            let expected = oracle(&data, &query, tiebreak);
            for recheck in [false, true] {
                let (actual, pruned) = run_query(&data, &config, recheck, &sql).await;
                if !recheck && pruned.is_some() {
                    failures.push(format!(
                        "{context}\n{sql}: metric registered with recheck off"
                    ));
                }
                case_pruned += pruned.unwrap_or(0);
                let check = if tiebreak {
                    check_exact(&expected, &actual)
                } else {
                    check_without_tiebreak(&data, &query, &expected, &actual)
                };
                if let Err(e) = check {
                    failures.push(format!("{context}\n{sql}\nrecheck={recheck}: {e}"));
                }
            }
        }
        if case_pruned > 0 {
            engaged_cases += 1;
            files_pruned_before_read += case_pruned;
        }
    }

    println!(
        "topk recheck fuzz: {engaged_cases}/{cases} cases skipped \
         {files_pruned_before_read} file ranges before their first read"
    );
    if !failures.is_empty() {
        for failure in &failures {
            println!("FAILURE:\n{failure}\n");
        }
        panic!(
            "{} TopK recheck fuzz failures (TOPK_RECHECK_FUZZ_SEED={seed})",
            failures.len()
        );
    }
    assert!(
        engaged_cases > 0,
        "the pre-read re-check never skipped a file (TOPK_RECHECK_FUZZ_SEED={seed}); \
         the test would pass without exercising it"
    );
}
