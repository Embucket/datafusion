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

//! Execution plan for reading CSV files

use datafusion_datasource::boundary_stream::AlignedBoundaryStream;
use datafusion_datasource::projection::{ProjectionOpener, SplitProjection};
use datafusion_physical_plan::projection::ProjectionExprs;
use std::fmt;
use std::io::{BufRead, BufReader, Read};
use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use datafusion_datasource::decoder::{DecoderDeserializer, deserialize_stream};
use datafusion_datasource::file_compression_type::FileCompressionType;
use datafusion_datasource::file_stream::{FileOpenFuture, FileOpener};
use datafusion_datasource::{
    FileRange, ListingTableUrl, PartitionedFile, TableSchema, as_file_source,
};

use arrow::csv;
use datafusion_common::config::CsvOptions;
use datafusion_common::tree_node::TreeNodeRecursion;
use datafusion_common::{DataFusionError, Result, exec_datafusion_err};
use datafusion_common_runtime::JoinSet;
use datafusion_datasource::file::FileSource;
use datafusion_datasource::file_scan_config::FileScanConfig;
use datafusion_execution::TaskContext;
use datafusion_physical_plan::metrics::{BaselineMetrics, ExecutionPlanMetricsSet};
use datafusion_physical_plan::{
    DisplayFormatType, ExecutionPlan, ExecutionPlanProperties,
};

use crate::file_format::{
    CsvDecoder, PhysicalLineSkip, skip_physical_lines_stream_prefix,
};
use futures::{StreamExt, TryStreamExt};
use object_store::buffered::BufWriter;
use object_store::path::Path;
use object_store::{GetOptions, GetResultPayload, ObjectStore};
use regex::Regex;
use tokio::io::AsyncWriteExt;

/// Creates a CSV record handler scoped to the file being opened.
pub trait CsvRecordErrorHandlerFactory: fmt::Debug + Send + Sync {
    /// Return the handler for `location`. The location is the object-store path
    /// of one file, before its CSV reader is created.
    fn for_file(&self, location: &Path) -> Arc<dyn csv::CsvRecordErrorHandler>;
}

/// Provides a per-file counter for physical blank CSV records skipped by the reader.
pub trait CsvBlankRecordCounterFactory: fmt::Debug + Send + Sync {
    /// Return the counter for an object-store file path.
    fn for_file(&self, location: &Path) -> Option<Arc<AtomicU64>>;
}

/// A Config for [`CsvOpener`]
///
/// # Example: create a `DataSourceExec` for CSV
/// ```
/// # use std::sync::Arc;
/// # use arrow::datatypes::Schema;
/// # use datafusion_datasource::file_scan_config::{FileScanConfig, FileScanConfigBuilder};
/// # use datafusion_datasource::PartitionedFile;
/// # use datafusion_datasource_csv::source::CsvSource;
/// # use datafusion_execution::object_store::ObjectStoreUrl;
/// # use datafusion_datasource::source::DataSourceExec;
/// # use datafusion_common::config::CsvOptions;
///
/// # let object_store_url = ObjectStoreUrl::local_filesystem();
/// # let file_schema = Arc::new(Schema::empty());
///
/// let options = CsvOptions {
///     has_header: Some(true),
///     delimiter: b',',
///     quote: b'"',
///     newlines_in_values: Some(true), // The file contains newlines in values
///     ..Default::default()
/// };
/// let source = Arc::new(CsvSource::new(file_schema.clone())
///     .with_csv_options(options)
///     .with_terminator(Some(b'#'))
/// );
/// // Create a DataSourceExec for reading the first 100MB of `file1.csv`
/// let config = FileScanConfigBuilder::new(object_store_url, source)
///     .with_file(PartitionedFile::new("file1.csv", 100*1024*1024))
///     .build();
/// let exec = (DataSourceExec::from_data_source(config));
/// ```
#[derive(Debug, Clone)]
pub struct CsvSource {
    options: CsvOptions,
    skip_physical_lines: usize,
    initial_source_lines: usize,
    initial_source_bytes: usize,
    numeric_boolean_values: bool,
    replace_invalid_utf8: bool,
    ignore_extra_fields: bool,
    record_error_handler: Option<Arc<dyn csv::CsvRecordErrorHandler>>,
    record_error_handler_factory: Option<Arc<dyn CsvRecordErrorHandlerFactory>>,
    blank_record_policy: csv::CsvBlankRecordPolicy,
    blank_record_counter_factory: Option<Arc<dyn CsvBlankRecordCounterFactory>>,
    blank_record_counter: Option<Arc<AtomicU64>>,
    batch_size: Option<usize>,
    table_schema: TableSchema,
    projection: SplitProjection,
    metrics: ExecutionPlanMetricsSet,
}

impl CsvSource {
    /// Returns a [`CsvSource`]
    pub fn new(table_schema: impl Into<TableSchema>) -> Self {
        let table_schema = table_schema.into();
        Self {
            options: CsvOptions::default(),
            skip_physical_lines: 0,
            initial_source_lines: 0,
            initial_source_bytes: 0,
            numeric_boolean_values: false,
            replace_invalid_utf8: false,
            ignore_extra_fields: false,
            record_error_handler: None,
            record_error_handler_factory: None,
            blank_record_policy: csv::CsvBlankRecordPolicy::Ignore,
            blank_record_counter_factory: None,
            blank_record_counter: None,
            projection: SplitProjection::unprojected(&table_schema),
            table_schema,
            batch_size: None,
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }

    #[cfg(feature = "proto")]
    fn ensure_serializable(&self) -> Result<()> {
        if self.blank_record_policy != csv::CsvBlankRecordPolicy::Ignore
            || self.blank_record_counter_factory.is_some()
            || self.blank_record_counter.is_some()
        {
            return datafusion_common::not_impl_err!(
                "CSV blank-record handling cannot be serialized"
            );
        }
        Ok(())
    }

    /// Sets the CSV options
    pub fn with_csv_options(mut self, options: CsvOptions) -> Self {
        self.options = options;
        self
    }

    /// Skip physical lines before CSV record parsing.
    pub fn with_skip_physical_lines(mut self, lines: usize) -> Self {
        self.skip_physical_lines = lines;
        self
    }

    /// Accept `0`/`1` and Snowflake-style text aliases (`t`/`f`, `yes`/`no`,
    /// `y`/`n`, `on`/`off`) in Boolean CSV columns.
    pub fn with_numeric_boolean_values(mut self, allow: bool) -> Self {
        self.numeric_boolean_values = allow;
        self
    }

    /// Replace malformed UTF-8 in decoded CSV fields with U+FFFD.
    pub fn with_replace_invalid_utf8(mut self, replace: bool) -> Self {
        self.replace_invalid_utf8 = replace;
        self
    }

    /// Ignore CSV fields beyond the explicitly supplied schema width.
    pub fn with_ignore_extra_fields(mut self, ignore: bool) -> Self {
        self.ignore_extra_fields = ignore;
        self
    }

    /// Whether numeric and Snowflake-style text aliases are accepted in Boolean CSV columns.
    pub fn numeric_boolean_values(&self) -> bool {
        self.numeric_boolean_values
    }

    /// Skip malformed field-count records and report them to `handler`.
    pub fn with_record_error_handler(
        mut self,
        handler: Arc<dyn csv::CsvRecordErrorHandler>,
    ) -> Self {
        self.record_error_handler = Some(handler);
        self.record_error_handler_factory = None;
        self
    }

    /// Select a record handler for each opened file. This disables byte-range
    /// repartitioning, as record offsets are file-relative.
    pub fn with_record_error_handler_factory(
        mut self,
        factory: Arc<dyn CsvRecordErrorHandlerFactory>,
    ) -> Self {
        self.record_error_handler = None;
        self.record_error_handler_factory = Some(factory);
        self
    }

    /// Select opt-in strict handling for physical blank CSV records.
    pub fn with_blank_record_policy(mut self, policy: csv::CsvBlankRecordPolicy) -> Self {
        self.blank_record_policy = policy;
        self
    }

    /// Select a per-file counter for skipped blank records.
    pub fn with_blank_record_counter_factory(
        mut self,
        factory: Arc<dyn CsvBlankRecordCounterFactory>,
    ) -> Self {
        self.blank_record_counter_factory = Some(factory);
        self
    }

    /// true if the first line of each file is a header
    pub fn has_header(&self) -> bool {
        self.options.has_header.unwrap_or(true)
    }

    // true if rows length support truncate
    pub fn truncate_rows(&self) -> bool {
        self.options.truncated_rows.unwrap_or(false)
    }

    /// The pattern matching CSV fields that should be read as null.
    pub fn null_regex(&self) -> Option<&str> {
        self.options.null_regex.as_deref()
    }

    /// Whether quoted empty fields bypass CSV null matching.
    pub fn preserve_quoted_empty(&self) -> bool {
        self.options.preserve_quoted_empty.unwrap_or(false)
    }

    /// A column delimiter
    pub fn delimiter(&self) -> u8 {
        self.options.delimiter
    }

    /// The quote character
    pub fn quote(&self) -> u8 {
        self.options.quote
    }

    /// The line terminator
    pub fn terminator(&self) -> Option<u8> {
        self.options.terminator
    }

    /// Lines beginning with this byte are ignored.
    pub fn comment(&self) -> Option<u8> {
        self.options.comment
    }

    /// The escape character
    pub fn escape(&self) -> Option<u8> {
        self.options.escape
    }

    /// Initialize a CsvSource with escape
    pub fn with_escape(&self, escape: Option<u8>) -> Self {
        let mut conf = self.clone();
        conf.options.escape = escape;
        conf
    }

    /// Initialize a CsvSource with terminator
    pub fn with_terminator(&self, terminator: Option<u8>) -> Self {
        let mut conf = self.clone();
        conf.options.terminator = terminator;
        conf
    }

    /// Initialize a CsvSource with comment
    pub fn with_comment(&self, comment: Option<u8>) -> Self {
        let mut conf = self.clone();
        conf.options.comment = comment;
        conf
    }

    /// Whether to support truncate rows when read csv file
    pub fn with_truncate_rows(&self, truncate_rows: bool) -> Self {
        let mut conf = self.clone();
        conf.options.truncated_rows = Some(truncate_rows);
        conf
    }

    /// Whether values may contain newline characters
    pub fn newlines_in_values(&self) -> bool {
        self.options.newlines_in_values.unwrap_or(false)
    }
}

fn skip_physical_lines_reader<R: Read>(
    source: R,
    lines: usize,
    terminator: Option<u8>,
) -> std::io::Result<(BufReader<R>, PhysicalLineSkip)> {
    let mut reader = BufReader::new(source);
    let mut skip = PhysicalLineSkip::new(lines, terminator);
    while !skip.is_done() {
        let available = reader.fill_buf()?;
        if available.is_empty() {
            break;
        }
        let consumed = skip.advance(available);
        reader.consume(consumed);
    }
    Ok((reader, skip))
}

impl CsvSource {
    fn open<R: Read>(&self, reader: R) -> Result<csv::Reader<R>> {
        Ok(self.builder()?.build(reader)?)
    }

    fn builder(&self) -> Result<csv::ReaderBuilder> {
        let mut builder =
            csv::ReaderBuilder::new(Arc::clone(self.table_schema.file_schema()))
                .with_delimiter(self.delimiter())
                .with_batch_size(
                    self.batch_size
                        .expect("Batch size must be set before initializing builder"),
                )
                .with_header(self.has_header())
                .with_initial_source_position(
                    self.initial_source_lines,
                    self.initial_source_bytes,
                )
                .with_quote(self.quote())
                .with_numeric_boolean_values(self.numeric_boolean_values)
                .with_replace_invalid_utf8(self.replace_invalid_utf8)
                .with_ignore_extra_fields(self.ignore_extra_fields)
                .with_preserve_quoted_empty(self.preserve_quoted_empty())
                .with_truncated_rows(self.truncate_rows());
        builder = builder.with_blank_record_policy(self.blank_record_policy);
        if let Some(counter) = &self.blank_record_counter {
            builder = builder.with_skipped_blank_records_counter(Arc::clone(counter));
        }
        if let Some(terminator) = self.terminator() {
            builder = builder.with_terminator(terminator);
        }
        builder = builder.with_projection(self.projection.file_indices.clone());
        if let Some(escape) = self.escape() {
            builder = builder.with_escape(escape)
        }
        if let Some(comment) = self.comment() {
            builder = builder.with_comment(comment);
        }
        if let Some(pattern) = &self.options.null_regex {
            if let Some(values) = crate::exact_null_values(pattern) {
                builder = builder.with_null_values(values);
            } else {
                let regex = Regex::new(pattern).map_err(|error| {
                    exec_datafusion_err!(
                        "Unable to parse CSV null regex '{pattern}': {error}"
                    )
                })?;
                builder = builder.with_null_regex(regex);
            }
        }
        if let Some(handler) = &self.record_error_handler {
            builder = builder.with_record_error_handler(Arc::clone(handler));
        }

        Ok(builder)
    }
}

/// A [`FileOpener`] that opens a CSV file and yields a [`FileOpenFuture`]
pub struct CsvOpener {
    config: Arc<CsvSource>,
    file_compression_type: FileCompressionType,
    object_store: Arc<dyn ObjectStore>,
    partition_index: usize,
}

impl CsvOpener {
    /// Returns a [`CsvOpener`]
    pub fn new(
        config: Arc<CsvSource>,
        file_compression_type: FileCompressionType,
        object_store: Arc<dyn ObjectStore>,
    ) -> Self {
        Self {
            config,
            file_compression_type,
            object_store,
            partition_index: 0,
        }
    }
}

impl From<CsvSource> for Arc<dyn FileSource> {
    fn from(source: CsvSource) -> Self {
        as_file_source(source)
    }
}

impl FileSource for CsvSource {
    fn create_file_opener(
        &self,
        object_store: Arc<dyn ObjectStore>,
        base_config: &FileScanConfig,
        partition_index: usize,
    ) -> Result<Arc<dyn FileOpener>> {
        let mut opener = Arc::new(CsvOpener {
            config: Arc::new(self.clone()),
            file_compression_type: base_config.file_compression_type,
            object_store,
            partition_index,
        }) as Arc<dyn FileOpener>;
        opener = ProjectionOpener::try_new(
            self.projection.clone(),
            Arc::clone(&opener),
            self.table_schema.file_schema(),
        )?;
        Ok(opener)
    }

    fn table_schema(&self) -> &TableSchema {
        &self.table_schema
    }

    fn with_batch_size(&self, batch_size: usize) -> Arc<dyn FileSource> {
        let mut conf = self.clone();
        conf.batch_size = Some(batch_size);
        Arc::new(conf)
    }

    fn try_pushdown_projection(
        &self,
        projection: &ProjectionExprs,
    ) -> Result<Option<Arc<dyn FileSource>>> {
        let mut source = self.clone();
        let new_projection = self.projection.source.try_merge(projection)?;
        let split_projection =
            SplitProjection::new(self.table_schema.file_schema(), &new_projection);
        source.projection = split_projection;
        Ok(Some(Arc::new(source)))
    }

    fn projection(&self) -> Option<&ProjectionExprs> {
        Some(&self.projection.source)
    }

    fn metrics(&self) -> &ExecutionPlanMetricsSet {
        &self.metrics
    }

    fn file_type(&self) -> &str {
        "csv"
    }

    fn supports_repartitioning(&self) -> bool {
        // Cannot repartition if values may contain newlines, as record
        // boundaries cannot be determined by byte offset alone
        self.record_error_handler.is_none()
            && self.record_error_handler_factory.is_none()
            && self.blank_record_policy == csv::CsvBlankRecordPolicy::Ignore
            && self.skip_physical_lines == 0
            && !self.options.newlines_in_values.unwrap_or(false)
    }

    fn fmt_extra(&self, t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(f, ", has_header={}", self.has_header())?;
                if self.skip_physical_lines > 0 {
                    write!(f, ", skip_physical_lines={}", self.skip_physical_lines)?;
                }
                Ok(())
            }
            DisplayFormatType::TreeRender => Ok(()),
        }
    }

    fn apply_expressions(
        &self,
        f: &mut dyn FnMut(
            &Arc<dyn datafusion_physical_plan::PhysicalExpr>,
        ) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        datafusion_physical_plan::apply_expression_roots(self.projection.source.iter(), f)
    }

    /// Emit a `CsvScan` node wrapping the shared base config and CSV options.
    #[cfg(feature = "proto")]
    fn try_to_proto(
        &self,
        base: &FileScanConfig,
        ctx: &datafusion_physical_plan::proto::ExecutionPlanEncodeCtx<'_>,
    ) -> Result<Option<datafusion_proto_models::protobuf::PhysicalPlanNode>> {
        use datafusion_proto_models::protobuf;
        use protobuf::physical_plan_node::PhysicalPlanType;

        self.ensure_serializable()?;

        if self.record_error_handler.is_some()
            || self.record_error_handler_factory.is_some()
        {
            return datafusion_common::not_impl_err!(
                "CSV record error handlers cannot be serialized"
            );
        }
        if self.replace_invalid_utf8 {
            return datafusion_common::not_impl_err!(
                "CSV UTF-8 replacement cannot be serialized"
            );
        }
        if self.ignore_extra_fields {
            return datafusion_common::not_impl_err!(
                "CSV extra-field ignoring cannot be serialized"
            );
        }
        if self.skip_physical_lines > 0 {
            return datafusion_common::not_impl_err!(
                "CSV physical-line skipping cannot be serialized"
            );
        }

        let node = protobuf::CsvScanExecNode {
            base_conf: Some(base.try_to_proto(ctx)?),
            has_header: self.has_header(),
            delimiter: proto_byte_to_string(self.delimiter(), "delimiter")?,
            quote: proto_byte_to_string(self.quote(), "quote")?,
            optional_escape: self
                .escape()
                .map(|escape| {
                    Ok::<_, DataFusionError>(
                        protobuf::csv_scan_exec_node::OptionalEscape::Escape(
                            proto_byte_to_string(escape, "escape")?,
                        ),
                    )
                })
                .transpose()?,
            optional_comment: self
                .comment()
                .map(|comment| {
                    Ok::<_, DataFusionError>(
                        protobuf::csv_scan_exec_node::OptionalComment::Comment(
                            proto_byte_to_string(comment, "comment")?,
                        ),
                    )
                })
                .transpose()?,
            newlines_in_values: self.newlines_in_values(),
            truncate_rows: self.truncate_rows(),
            null_regex: self.options.null_regex.clone(),
            preserve_quoted_empty: self.preserve_quoted_empty(),
            numeric_boolean_values: self.numeric_boolean_values(),
        };
        Ok(Some(protobuf::PhysicalPlanNode {
            physical_plan_type: Some(PhysicalPlanType::CsvScan(node)),
        }))
    }
}

impl FileOpener for CsvOpener {
    /// Open a partitioned CSV file.
    ///
    /// If `file_meta.range` is `None`, the entire file is opened.
    /// If `file_meta.range` is `Some(FileRange {start, end})`, this signifies that the partition
    /// corresponds to the byte range [start, end) within the file.
    ///
    /// Note: `start` or `end` might be in the middle of some lines. In such cases, the following rules
    /// are applied to determine which lines to read:
    /// 1. The first line of the partition is the line in which the index of the first character >= `start`.
    /// 2. The last line of the partition is the line in which the byte at position `end - 1` resides.
    ///
    /// Examples:
    /// Consider the following partitions enclosed by braces `{}`:
    ///
    /// {A,1,2,3,4,5,6,7,8,9\n
    ///  A,1,2,3,4,5,6,7,8,9\n}
    ///  A,1,2,3,4,5,6,7,8,9\n
    ///  The lines read would be: [0, 1]
    ///
    ///  A,{1,2,3,4,5,6,7,8,9\n
    ///  A,1,2,3,4,5,6,7,8,9\n
    ///  A},1,2,3,4,5,6,7,8,9\n
    ///  The lines read would be: [1, 2]
    fn open(&self, partitioned_file: PartitionedFile) -> Result<FileOpenFuture> {
        if self.config.skip_physical_lines > 0 && partitioned_file.range.is_some() {
            return datafusion_common::not_impl_err!(
                "CSV physical-line skipping cannot be combined with byte-range scans"
            );
        }
        // `self.config.has_header` controls whether to skip reading the 1st line header
        // If the .csv file is read in parallel and this `CsvOpener` is only reading some middle
        // partition, then don't skip first line
        let mut csv_has_header = self.config.has_header();
        if let Some(FileRange { start, .. }) = partitioned_file.range
            && start != 0
        {
            csv_has_header = false;
        }

        let mut config = (*self.config).clone();
        if let Some(factory) = &config.record_error_handler_factory {
            config.record_error_handler =
                Some(factory.for_file(&partitioned_file.object_meta.location));
        }
        if let Some(factory) = &config.blank_record_counter_factory {
            config.blank_record_counter = Some(
                factory
                    .for_file(&partitioned_file.object_meta.location)
                    .ok_or_else(|| {
                        DataFusionError::Execution(format!(
                            "CSV blank-record counter is missing for '{}'",
                            partitioned_file.object_meta.location
                        ))
                    })?,
            );
        }
        config.options.has_header = Some(csv_has_header);
        config.options.truncated_rows = Some(config.truncate_rows());

        let file_compression_type = self.file_compression_type.to_owned();

        if partitioned_file.range.is_some() {
            assert!(
                !file_compression_type.is_compressed(),
                "Reading compressed .csv in parallel is not supported"
            );
        }

        let store = Arc::clone(&self.object_store);
        let terminator = self.config.terminator();

        let baseline_metrics =
            BaselineMetrics::new(&self.config.metrics, self.partition_index);

        Ok(Box::pin(async move {
            // Current partition contains bytes [start_byte, end_byte) (might contain incomplete lines at boundaries)
            let file_size = partitioned_file.object_meta.size;
            let location = partitioned_file.object_meta.location;

            if let Some(file_range) = partitioned_file.range.as_ref() {
                let raw_start: u64 = file_range.start.try_into().map_err(|_| {
                    exec_datafusion_err!(
                        "Expected start range to fit in u64, got {}",
                        file_range.start
                    )
                })?;
                let raw_end: u64 = file_range.end.try_into().map_err(|_| {
                    exec_datafusion_err!(
                        "Expected end range to fit in u64, got {}",
                        file_range.end
                    )
                })?;

                let aligned_stream = AlignedBoundaryStream::new(
                    Arc::clone(&store),
                    location.clone(),
                    raw_start,
                    raw_end,
                    file_size,
                    terminator.unwrap_or(b'\n'),
                )
                .await?
                .map_err(DataFusionError::from);

                let input = file_compression_type
                    .convert_stream(aligned_stream.boxed())?
                    .fuse();
                let decoder = config.builder()?.build_decoder();
                let stream = deserialize_stream(
                    input,
                    DecoderDeserializer::new(CsvDecoder::new(decoder)),
                );
                return Ok(stream.map_err(Into::into).boxed());
            }

            // No range specified — read the entire file
            let options = GetOptions::default();
            let result = store.get_opts(&location, options).await?;

            match result.payload {
                #[cfg(not(target_arch = "wasm32"))]
                GetResultPayload::File(file, _) => {
                    let mut decoder = file_compression_type.convert_read(file)?;
                    if config.skip_physical_lines > 0 {
                        let (reader, skipped) = skip_physical_lines_reader(
                            decoder,
                            config.skip_physical_lines,
                            terminator,
                        )
                        .map_err(|error| DataFusionError::External(Box::new(error)))?;
                        config.initial_source_lines = skipped.skipped_lines();
                        config.initial_source_bytes = skipped.skipped_bytes();
                        decoder = Box::new(reader);
                    }
                    let mut reader = config.open(decoder)?;

                    // Use std::iter::from_fn to wrap execution of iterator's next() method.
                    let iterator = std::iter::from_fn(move || {
                        let mut timer = baseline_metrics.elapsed_compute().timer();
                        let result = reader.next();
                        timer.stop();
                        result
                    });

                    Ok(futures::stream::iter(iterator)
                        .map(|r| r.map_err(Into::into))
                        .boxed())
                }
                GetResultPayload::Stream(s) => {
                    let s = s.map_err(DataFusionError::from);
                    let input = file_compression_type.convert_stream(s.boxed())?.fuse();
                    if config.skip_physical_lines == 0 {
                        let decoder = config.builder()?.build_decoder();
                        let stream = deserialize_stream(
                            input,
                            DecoderDeserializer::new(CsvDecoder::new(decoder)),
                        );
                        Ok(stream.map_err(Into::into).boxed())
                    } else {
                        let (input, skipped) = skip_physical_lines_stream_prefix(
                            input.boxed(),
                            config.skip_physical_lines,
                            terminator,
                        )
                        .await?;
                        config.initial_source_lines = skipped.skipped_lines();
                        config.initial_source_bytes = skipped.skipped_bytes();
                        let decoder = config.builder()?.build_decoder();
                        let stream = deserialize_stream(
                            input,
                            DecoderDeserializer::new(CsvDecoder::new(decoder)),
                        );
                        Ok(stream.map_err(Into::into).boxed())
                    }
                }
            }
        }))
    }
}

pub async fn plan_to_csv(
    task_ctx: Arc<TaskContext>,
    plan: Arc<dyn ExecutionPlan>,
    path: impl AsRef<str>,
) -> Result<()> {
    let path = path.as_ref();
    let parsed = ListingTableUrl::parse(path)?;
    let object_store_url = parsed.object_store();
    let store = task_ctx.runtime_env().object_store(&object_store_url)?;
    let writer_buffer_size = task_ctx
        .session_config()
        .options()
        .execution
        .objectstore_writer_buffer_size;
    let mut join_set = JoinSet::new();
    for i in 0..plan.output_partitioning().partition_count() {
        let storeref = Arc::clone(&store);
        let plan: Arc<dyn ExecutionPlan> = Arc::clone(&plan);
        let filename = format!("{}/part-{i}.csv", parsed.prefix());
        let file = Path::parse(filename)?;

        let mut stream = plan.execute(i, Arc::clone(&task_ctx))?;
        join_set.spawn(async move {
            let mut buf_writer =
                BufWriter::with_capacity(storeref, file.clone(), writer_buffer_size);
            let mut buffer = Vec::with_capacity(1024);
            //only write headers on first iteration
            let mut write_headers = true;
            while let Some(batch) = stream.next().await.transpose()? {
                let mut writer = csv::WriterBuilder::new()
                    .with_header(write_headers)
                    .build(buffer);
                writer.write(&batch)?;
                buffer = writer.into_inner();
                buf_writer.write_all(&buffer).await?;
                buffer.clear();
                //prevent writing headers more than once
                write_headers = false;
            }
            buf_writer.shutdown().await.map_err(DataFusionError::from)
        });
    }

    while let Some(result) = join_set.join_next().await {
        match result {
            Ok(res) => res?, // propagate DataFusion error
            Err(e) => {
                if e.is_panic() {
                    std::panic::resume_unwind(e.into_panic());
                } else {
                    unreachable!();
                }
            }
        }
    }

    Ok(())
}

#[cfg(feature = "proto")]
fn proto_byte_to_string(b: u8, description: &str) -> Result<String> {
    let bytes = &[b];
    let s = std::str::from_utf8(bytes).map_err(|_| {
        datafusion_common::internal_datafusion_err!(
            "Invalid CSV {description}: can not represent {bytes:0x?} as utf8"
        )
    })?;
    Ok(s.to_owned())
}

#[cfg(feature = "proto")]
fn proto_str_to_byte(s: &str, description: &str) -> Result<u8> {
    datafusion_common::assert_eq_or_internal_err!(
        s.len(),
        1,
        "Invalid CSV {description}: expected single character, got {s}"
    );
    Ok(s.as_bytes()[0])
}

#[cfg(feature = "proto")]
impl CsvSource {
    /// Reconstructs a `DataSourceExec` from a protobuf `CsvScan`.
    ///
    /// Custom line terminators are not represented in the wire format.
    pub fn try_from_proto(
        node: &datafusion_proto_models::protobuf::PhysicalPlanNode,
        ctx: &datafusion_physical_plan::proto::ExecutionPlanDecodeCtx<'_>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        use datafusion_common::config::CsvOptions;
        use datafusion_datasource::file_compression_type::FileCompressionType;
        use datafusion_datasource::file_scan_config::{
            FileScanConfig, FileScanConfigBuilder,
        };
        use datafusion_datasource::source::DataSourceExec;
        use datafusion_proto_models::protobuf;

        let scan = match &node.physical_plan_type {
            Some(protobuf::physical_plan_node::PhysicalPlanType::CsvScan(scan)) => scan,
            _ => {
                return datafusion_common::internal_err!(
                    "PhysicalPlanNode is not a CsvScan"
                );
            }
        };

        let base_conf = scan.base_conf.as_ref().ok_or_else(|| {
            datafusion_common::internal_datafusion_err!(
                "CsvScanExecNode is missing required field 'base_conf'"
            )
        })?;

        let escape = match &scan.optional_escape {
            Some(protobuf::csv_scan_exec_node::OptionalEscape::Escape(escape)) => {
                Some(proto_str_to_byte(escape, "escape")?)
            }
            None => None,
        };
        let comment = match &scan.optional_comment {
            Some(protobuf::csv_scan_exec_node::OptionalComment::Comment(comment)) => {
                Some(proto_str_to_byte(comment, "comment")?)
            }
            None => None,
        };

        let table_schema = FileScanConfig::parse_table_schema_from_proto(base_conf)?;

        let csv_options = CsvOptions {
            has_header: Some(scan.has_header),
            delimiter: proto_str_to_byte(&scan.delimiter, "delimiter")?,
            quote: proto_str_to_byte(&scan.quote, "quote")?,
            newlines_in_values: Some(scan.newlines_in_values),
            truncated_rows: Some(scan.truncate_rows),
            null_regex: scan.null_regex.clone(),
            preserve_quoted_empty: Some(scan.preserve_quoted_empty),
            ..Default::default()
        };
        let source = Arc::new(
            CsvSource::new(table_schema)
                .with_csv_options(csv_options)
                .with_numeric_boolean_values(scan.numeric_boolean_values)
                .with_escape(escape)
                .with_comment(comment),
        );

        // The compression type is not on the wire; CSV scans always
        // deserialize as uncompressed.
        let conf = FileScanConfigBuilder::from(FileScanConfig::try_from_proto(
            base_conf, ctx, source,
        )?)
        .with_file_compression_type(FileCompressionType::UNCOMPRESSED)
        .build();
        Ok(DataSourceExec::from_data_source(conf))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::file_format::CsvFormat;
    use arrow::array::{AsArray, Int64Array, StringArray};
    use arrow::csv::{CsvRecordError, CsvRecordErrorHandler};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::error::ArrowError;
    use bytes::Bytes;
    use datafusion_datasource::file_format::FileFormat;
    use object_store::ObjectStoreExt;
    use object_store::memory::InMemory;
    use std::collections::HashMap;
    use std::io::Cursor;
    use std::sync::Mutex;
    use std::sync::atomic::Ordering;

    #[derive(Debug)]
    struct TestBlankCounters(HashMap<Path, Arc<AtomicU64>>);

    impl CsvBlankRecordCounterFactory for TestBlankCounters {
        fn for_file(&self, location: &Path) -> Option<Arc<AtomicU64>> {
            self.0.get(location).cloned()
        }
    }

    #[cfg(feature = "proto")]
    #[test]
    fn blank_record_policy_rejects_lossy_plan_serialization() {
        let schema =
            Arc::new(Schema::new(vec![Field::new("value", DataType::Utf8, true)]));
        assert!(
            CsvSource::new(Arc::clone(&schema))
                .ensure_serializable()
                .is_ok()
        );
        for policy in [
            csv::CsvBlankRecordPolicy::Reject,
            csv::CsvBlankRecordPolicy::EmitNullRow,
            csv::CsvBlankRecordPolicy::Skip,
        ] {
            let source =
                CsvSource::new(Arc::clone(&schema)).with_blank_record_policy(policy);
            assert!(
                source
                    .ensure_serializable()
                    .unwrap_err()
                    .to_string()
                    .contains("blank-record handling cannot be serialized")
            );
        }
        let source = CsvSource::new(schema).with_blank_record_counter_factory(Arc::new(
            TestBlankCounters(HashMap::new()),
        ));
        assert!(source.ensure_serializable().is_err());
    }

    #[tokio::test]
    async fn skipped_blank_record_counters_are_scoped_to_opened_file()
    -> Result<(), Box<dyn std::error::Error>> {
        let store = Arc::new(InMemory::new());
        let counters = ["first.csv", "second.csv"]
            .into_iter()
            .map(|name| (Path::from(name), Arc::new(AtomicU64::new(0))))
            .collect::<HashMap<_, _>>();
        let mut source = CsvSource::new(Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Utf8,
            true,
        )])))
        .with_csv_options(CsvOptions {
            has_header: Some(false),
            ..Default::default()
        })
        .with_blank_record_policy(csv::CsvBlankRecordPolicy::Skip)
        .with_blank_record_counter_factory(Arc::new(TestBlankCounters(counters.clone())));
        source.batch_size = Some(1024);
        assert!(!source.supports_repartitioning());
        let opener = CsvOpener::new(
            Arc::new(source),
            FileCompressionType::UNCOMPRESSED,
            store.clone(),
        );
        for (name, content) in
            [("first.csv", "first\n\n"), ("second.csv", "\nsecond\n\n")]
        {
            let path = Path::from(name);
            store.put(&path, Bytes::from(content).into()).await?;
            let meta = store.head(&path).await?;
            let batches = opener
                .open(PartitionedFile::new_from_meta(meta))?
                .await?
                .try_collect::<Vec<_>>()
                .await?;
            assert_eq!(
                batches.iter().map(|batch| batch.num_rows()).sum::<usize>(),
                1
            );
        }
        assert_eq!(
            counters[&Path::from("first.csv")].load(Ordering::Relaxed),
            1
        );
        assert_eq!(
            counters[&Path::from("second.csv")].load(Ordering::Relaxed),
            2
        );
        let missing = Path::from("unregistered.csv");
        store
            .put(&missing, Bytes::from_static(b"value\n").into())
            .await?;
        let meta = store.head(&missing).await?;
        assert!(opener.open(PartitionedFile::new_from_meta(meta)).is_err());
        Ok(())
    }

    #[derive(Debug, Default)]
    struct HeaderPositions(Mutex<Vec<(usize, usize, usize)>>);

    impl CsvRecordErrorHandler for HeaderPositions {
        fn handle(&self, error: &CsvRecordError<'_>) -> Result<(), ArrowError> {
            self.0.lock().unwrap().push((
                error.line_number,
                error.source_line_number,
                error.byte_offset,
            ));
            Ok(())
        }
    }

    #[tokio::test]
    async fn physical_header_skip_keeps_file_relative_error_positions()
    -> Result<(), Box<dyn std::error::Error>> {
        let store = Arc::new(InMemory::new());
        let path = Path::from("positions.csv");
        store
            .put(
                &path,
                Bytes::from_static(b"\"unclosed\n1,ok\n2,extra,ignored\n").into(),
            )
            .await?;
        let errors = Arc::new(HeaderPositions::default());
        let mut source = CsvSource::new(Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("value", DataType::Utf8, false),
        ])))
        .with_csv_options(CsvOptions {
            has_header: Some(false),
            ..Default::default()
        })
        .with_skip_physical_lines(1)
        .with_record_error_handler(errors.clone());
        source.batch_size = Some(1024);
        let opener = CsvOpener::new(
            Arc::new(source),
            FileCompressionType::UNCOMPRESSED,
            store.clone(),
        );
        let batches = opener
            .open(PartitionedFile::new_from_meta(store.head(&path).await?))?
            .await?
            .try_collect::<Vec<_>>()
            .await?;
        assert_eq!(
            batches.iter().map(|batch| batch.num_rows()).sum::<usize>(),
            1
        );
        assert_eq!(
            *errors.0.lock().unwrap(),
            [(2, 3, b"\"unclosed\n1,ok\n".len())]
        );
        Ok(())
    }

    #[tokio::test]
    async fn custom_terminator_skip_uses_physical_line_offset()
    -> Result<(), Box<dyn std::error::Error>> {
        let store = Arc::new(InMemory::new());
        let path = Path::from("custom-terminator.csv");
        store
            .put(
                &path,
                Bytes::from_static(b"meta\r\ncontinued#1,ok#2,extra,ignored#").into(),
            )
            .await?;
        let errors = Arc::new(HeaderPositions::default());
        let mut source = CsvSource::new(Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("value", DataType::Utf8, false),
        ])))
        .with_csv_options(CsvOptions {
            has_header: Some(false),
            ..Default::default()
        })
        .with_terminator(Some(b'#'))
        .with_skip_physical_lines(1)
        .with_record_error_handler(errors.clone());
        source.batch_size = Some(1024);
        let opener = CsvOpener::new(
            Arc::new(source),
            FileCompressionType::UNCOMPRESSED,
            store.clone(),
        );
        let batches = opener
            .open(PartitionedFile::new_from_meta(store.head(&path).await?))?
            .await?
            .try_collect::<Vec<_>>()
            .await?;
        assert_eq!(
            batches.iter().map(|batch| batch.num_rows()).sum::<usize>(),
            1
        );
        assert_eq!(
            *errors.0.lock().unwrap(),
            [(2, 2, b"meta\r\ncontinued#1,ok#".len())]
        );
        Ok(())
    }

    #[test]
    fn physical_header_skip_reader_handles_crlf_and_header_only_input() {
        let input = b"\"unfinished\r\nsecond header\r\n1,ok\r\n";
        let (mut reader, skipped) =
            skip_physical_lines_reader(Cursor::new(input), 2, None).unwrap();
        assert_eq!(skipped.skipped_lines(), 2);
        assert_eq!(
            skipped.skipped_bytes(),
            b"\"unfinished\r\nsecond header\r\n".len()
        );
        let mut data = String::new();
        reader.read_to_string(&mut data).unwrap();
        assert_eq!(data, "1,ok\r\n");

        let (mut reader, _) =
            skip_physical_lines_reader(Cursor::new(b"header\r\n"), 2, None).unwrap();
        data.clear();
        reader.read_to_string(&mut data).unwrap();
        assert!(data.is_empty());

        let (mut reader, skipped) =
            skip_physical_lines_reader(Cursor::new(b"header\r1,ok\r"), 1, None).unwrap();
        data.clear();
        reader.read_to_string(&mut data).unwrap();
        assert_eq!(data, "1,ok\r");
        assert_eq!(skipped.skipped_bytes(), b"header\r".len());
    }

    #[tokio::test]
    async fn physical_header_skip_precedes_csv_record_parsing()
    -> Result<(), Box<dyn std::error::Error>> {
        let store = Arc::new(InMemory::new());
        let path = Path::from("header.csv");
        store
            .put(
                &path,
                Bytes::from_static(b"\"unfinished\r\n1,ok\r\n2,good\r\n").into(),
            )
            .await?;
        let mut source = CsvSource::new(Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("value", DataType::Utf8, false),
        ])))
        .with_csv_options(CsvOptions {
            has_header: Some(false),
            ..Default::default()
        })
        .with_skip_physical_lines(1);
        source.batch_size = Some(1024);
        assert!(!source.supports_repartitioning());
        let opener = CsvOpener::new(
            Arc::new(source),
            FileCompressionType::UNCOMPRESSED,
            store.clone(),
        );
        let batches = opener
            .open(PartitionedFile::new_from_meta(store.head(&path).await?))?
            .await?
            .try_collect::<Vec<_>>()
            .await?;
        let ids = batches[0]
            .column(0)
            .as_primitive::<arrow::datatypes::Int64Type>();
        assert_eq!(ids.values(), &[1, 2]);

        let mut ranged = PartitionedFile::new_from_meta(store.head(&path).await?);
        ranged.range = Some(FileRange { start: 0, end: 12 });
        assert!(opener.open(ranged).is_err());

        let cr_path = Path::from("header-cr.csv");
        store
            .put(&cr_path, Bytes::from_static(b"\"unfinished\r1,ok\r").into())
            .await?;
        let batches = opener
            .open(PartitionedFile::new_from_meta(store.head(&cr_path).await?))?
            .await?
            .try_collect::<Vec<_>>()
            .await?;
        let ids = batches[0]
            .column(0)
            .as_primitive::<arrow::datatypes::Int64Type>();
        assert_eq!(ids.values(), &[1]);
        Ok(())
    }

    #[test]
    fn csv_source_ignore_extra_fields_is_opt_in() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "first",
            DataType::Utf8,
            false,
        )]));
        let mut source = CsvSource::new(schema).with_csv_options(CsvOptions {
            has_header: Some(false),
            ..Default::default()
        });
        source.batch_size = Some(1024);
        let csv = b"one,unused\ntwo,also-unused\n";
        assert!(
            source
                .open(Cursor::new(csv))
                .unwrap()
                .next()
                .unwrap()
                .is_err()
        );

        let batches = source
            .with_ignore_extra_fields(true)
            .open(Cursor::new(csv))
            .unwrap()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap();
        let values = batches[0].column(0).as_string::<i32>();
        assert_eq!(
            values.iter().collect::<Vec<_>>(),
            [Some("one"), Some("two")]
        );
    }

    #[test]
    fn csv_format_passes_ignore_extra_fields_to_reader() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "first",
            DataType::Utf8,
            false,
        )]));
        let format = CsvFormat::default()
            .with_has_header(false)
            .with_ignore_extra_fields(true);
        let source = format.file_source(schema.into());
        let source = source.downcast_ref::<CsvSource>().unwrap();
        let mut source = source.clone();
        source.batch_size = Some(1024);

        let batches = source
            .open(Cursor::new(b"one,unused\ntwo,also-unused\n"))
            .unwrap()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap();
        let values = batches[0].column(0).as_string::<i32>();
        assert_eq!(
            values.iter().collect::<Vec<_>>(),
            [Some("one"), Some("two")]
        );
    }

    #[test]
    fn csv_source_numeric_boolean_values_are_opt_in() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "flag",
            DataType::Boolean,
            false,
        )]));
        let mut source = CsvSource::new(schema).with_csv_options(CsvOptions {
            has_header: Some(false),
            ..Default::default()
        });
        source.batch_size = Some(1024);
        let strict_error = source
            .open(Cursor::new(b"true\n1\n0\nfalse\n"))
            .unwrap()
            .next()
            .unwrap()
            .unwrap_err();
        assert!(
            strict_error
                .to_string()
                .contains("value '1' as type 'Boolean'")
        );

        let batches = source
            .with_numeric_boolean_values(true)
            .open(Cursor::new(b"true\n1\n0\nfalse\n"))
            .unwrap()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap();
        let values = batches[0].column(0).as_boolean();
        assert_eq!(
            values.values().iter().collect::<Vec<_>>(),
            [true, true, false, false]
        );
    }

    #[test]
    fn csv_source_utf8_replacement_is_opt_in() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("msg", DataType::Utf8, false),
        ]));
        let csv = b"1,ok\n2,\xff\n3,after\n";
        let mut source = CsvSource::new(schema).with_csv_options(CsvOptions {
            has_header: Some(false),
            ..Default::default()
        });
        source.batch_size = Some(1024);
        assert!(
            source
                .open(Cursor::new(csv))
                .unwrap()
                .next()
                .unwrap()
                .is_err()
        );

        let batches = source
            .with_replace_invalid_utf8(true)
            .open(Cursor::new(csv))
            .unwrap()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(batches[0].num_rows(), 3);
        let values = batches[0].column(1).as_string::<i32>();
        assert_eq!(
            (values.value(0), values.value(1), values.value(2)),
            ("ok", "�", "after")
        );
    }

    #[test]
    fn csv_source_applies_null_regex_to_loaded_rows() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("value", DataType::Utf8, true),
        ]));
        let input = Cursor::new(b"1,\\N\n2,ok\n3,\n");
        let mut source = CsvSource::new(schema).with_csv_options(CsvOptions {
            has_header: Some(false),
            null_regex: Some(r"\A(?:\\N|)\z".to_owned()),
            ..Default::default()
        });
        source.batch_size = Some(1024);
        let batches = source
            .open(input)
            .unwrap()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap();
        let values = batches[0]
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(values.iter().collect::<Vec<_>>(), [None, Some("ok"), None]);

        source.options.null_regex = Some(r"\A\\N\z".to_owned());
        let batches = source
            .open(Cursor::new(b"1,\\N\n2,ok\n3,\n"))
            .unwrap()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap();
        let values = batches[0]
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(
            values.iter().collect::<Vec<_>>(),
            [None, Some("ok"), Some("")]
        );

        source.options.null_regex = Some("(".to_owned());
        assert!(source.open(Cursor::new(b"ok\n")).is_err());
    }

    #[test]
    fn csv_source_distinguishes_quoted_empty_from_null() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("value", DataType::Utf8, true),
        ]));
        let mut source = CsvSource::new(schema).with_csv_options(CsvOptions {
            has_header: Some(false),
            null_regex: Some(r"\A(?:NA|)\z".to_owned()),
            preserve_quoted_empty: Some(true),
            ..Default::default()
        });
        source.batch_size = Some(1024);
        let batches = source
            .open(Cursor::new(b"1,\n2,\"\"\n3,NA\n"))
            .unwrap()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap();
        let values = batches[0]
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(values.iter().collect::<Vec<_>>(), [None, Some(""), None]);
    }

    #[derive(Debug, Default)]
    struct CollectRecordErrors(Mutex<Vec<(usize, usize, usize)>>);

    impl CsvRecordErrorHandler for CollectRecordErrors {
        fn handle(&self, error: &CsvRecordError<'_>) -> Result<(), ArrowError> {
            self.0.lock().unwrap().push((
                error.line_number,
                error.byte_offset,
                error.actual_fields,
            ));
            Ok(())
        }
    }

    #[derive(Debug)]
    struct TaggedRecords {
        path: String,
        seen: Arc<Mutex<Vec<(String, String)>>>,
    }

    impl CsvRecordErrorHandler for TaggedRecords {
        fn handle(&self, error: &CsvRecordError<'_>) -> Result<(), ArrowError> {
            self.seen.lock().unwrap().push((
                self.path.clone(),
                format!("error:{}", String::from_utf8_lossy(error.record)),
            ));
            Ok(())
        }

        fn handle_record(&self, record: &csv::CsvRecord<'_>) -> Result<(), ArrowError> {
            self.seen.lock().unwrap().push((
                self.path.clone(),
                String::from_utf8_lossy(record.record).into_owned(),
            ));
            Ok(())
        }
    }

    #[derive(Debug)]
    struct TaggedRecordsFactory {
        seen: Arc<Mutex<Vec<(String, String)>>>,
    }

    impl CsvRecordErrorHandlerFactory for TaggedRecordsFactory {
        fn for_file(&self, location: &Path) -> Arc<dyn CsvRecordErrorHandler> {
            Arc::new(TaggedRecords {
                path: location.to_string(),
                seen: Arc::clone(&self.seen),
            })
        }
    }

    #[tokio::test]
    async fn record_error_handler_factory_is_scoped_to_opened_file()
    -> Result<(), Box<dyn std::error::Error>> {
        let store = Arc::new(InMemory::new());
        let seen = Arc::new(Mutex::new(Vec::new()));
        let mut source = CsvSource::new(Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Utf8,
            false,
        )])))
        .with_csv_options(CsvOptions {
            has_header: Some(false),
            ..Default::default()
        })
        .with_record_error_handler_factory(Arc::new(TaggedRecordsFactory {
            seen: Arc::clone(&seen),
        }));
        source.batch_size = Some(1024);
        assert!(!source.supports_repartitioning());
        let opener = CsvOpener::new(
            Arc::new(source),
            FileCompressionType::UNCOMPRESSED,
            store.clone(),
        );
        for (name, content) in [
            ("first.csv", "first\nbad,extra\n"),
            ("second.csv", "second\nwrong,extra\n"),
        ] {
            let path = Path::from(name);
            store.put(&path, Bytes::from(content).into()).await?;
            let meta = store.head(&path).await?;
            let batches = opener
                .open(PartitionedFile::new_from_meta(meta))?
                .await?
                .try_collect::<Vec<_>>()
                .await?;
            assert_eq!(
                batches.iter().map(|batch| batch.num_rows()).sum::<usize>(),
                1
            );
        }
        assert_eq!(
            *seen.lock().unwrap(),
            [
                ("first.csv".to_owned(), "first\n".to_owned()),
                ("first.csv".to_owned(), "error:bad,extra\n".to_owned()),
                ("second.csv".to_owned(), "second\n".to_owned()),
                ("second.csv".to_owned(), "error:wrong,extra\n".to_owned())
            ]
        );
        Ok(())
    }

    #[test]
    fn record_error_handler_skips_malformed_rows_and_disables_repartitioning() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("value", DataType::Utf8, false),
        ]));
        let handler = Arc::new(CollectRecordErrors::default());
        let options = CsvOptions {
            has_header: Some(false),
            ..Default::default()
        };
        let mut source = CsvSource::new(schema)
            .with_csv_options(options)
            .with_record_error_handler(handler.clone());
        source.batch_size = Some(1024);

        assert!(!source.supports_repartitioning());
        let batches = source
            .open(Cursor::new(b"1,ok\n2,extra,value\n3\n4,after\n"))
            .unwrap()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(batches.len(), 1);
        assert_eq!(
            batches[0]
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .values(),
            &[1, 4]
        );
        assert_eq!(
            batches[0]
                .column(1)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .iter()
                .collect::<Vec<_>>(),
            [Some("ok"), Some("after")]
        );
        assert_eq!(*handler.0.lock().unwrap(), [(2, 5, 3), (3, 19, 1)]);
    }
}
