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

//! [`CsvFormat`], Comma Separated Value (CSV) [`FileFormat`] abstractions

use std::collections::{HashMap, HashSet};
use std::fmt::{self, Debug};
use std::sync::Arc;

use crate::source::CsvSource;

use arrow::array::RecordBatch;
use arrow::csv::reader::SnowflakeCsvTypeState;
use arrow::csv::{CsvRecordErrorHandler, WriterBuilder};
use arrow::datatypes::{DataType, Field, Fields, Schema, SchemaRef};
use arrow::error::ArrowError;
use datafusion_common::config::{ConfigField, ConfigFileType, CsvOptions};
use datafusion_common::file_options::csv_writer::CsvWriterOptions;
use datafusion_common::{
    DEFAULT_CSV_EXTENSION, DataFusionError, GetExt, Result, Statistics, exec_err,
    not_impl_err,
};
use datafusion_common_runtime::SpawnedTask;
use datafusion_datasource::TableSchema;
use datafusion_datasource::decoder::Decoder;
use datafusion_datasource::display::FileGroupDisplay;
use datafusion_datasource::file::FileSource;
use datafusion_datasource::file_compression_type::FileCompressionType;
use datafusion_datasource::file_format::{
    DEFAULT_SCHEMA_INFER_MAX_RECORD, FileFormat, FileFormatFactory,
};
use datafusion_datasource::file_scan_config::{FileScanConfig, FileScanConfigBuilder};
use datafusion_datasource::file_sink_config::{FileSink, FileSinkConfig};
use datafusion_datasource::sink::{DataSink, DataSinkExec};
use datafusion_datasource::write::BatchSerializer;
use datafusion_datasource::write::demux::DemuxedStreamReceiver;
use datafusion_datasource::write::orchestration::spawn_writer_tasks_and_join;
use datafusion_execution::{SendableRecordBatchStream, TaskContext};
use datafusion_expr::dml::InsertOp;
use datafusion_physical_expr_common::sort_expr::LexRequirement;
use datafusion_physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan};
use datafusion_session::Session;

use async_trait::async_trait;
use bytes::{Buf, Bytes};
use datafusion_datasource::source::DataSourceExec;
use futures::stream::BoxStream;
use futures::{Stream, StreamExt, TryStreamExt, pin_mut};
use object_store::{
    ObjectMeta, ObjectStore, ObjectStoreExt, delimited::newline_delimited_stream,
};
use regex::Regex;

#[derive(Default)]
/// Factory used to create [`CsvFormat`]
pub struct CsvFormatFactory {
    /// the options for csv file read
    pub options: Option<CsvOptions>,
}

impl CsvFormatFactory {
    /// Creates an instance of [`CsvFormatFactory`]
    pub fn new() -> Self {
        Self { options: None }
    }

    /// Creates an instance of [`CsvFormatFactory`] with customized default options
    pub fn new_with_options(options: CsvOptions) -> Self {
        Self {
            options: Some(options),
        }
    }
}

impl Debug for CsvFormatFactory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CsvFormatFactory")
            .field("options", &self.options)
            .finish()
    }
}

impl FileFormatFactory for CsvFormatFactory {
    fn create(
        &self,
        state: &dyn Session,
        format_options: &HashMap<String, String>,
    ) -> Result<Arc<dyn FileFormat>> {
        let csv_options = match &self.options {
            None => {
                let mut table_options = state.default_table_options();
                table_options.set_config_format(ConfigFileType::CSV);
                table_options.alter_with_string_hash_map(format_options)?;
                table_options.csv
            }
            Some(csv_options) => {
                let mut csv_options = csv_options.clone();
                for (k, v) in format_options {
                    csv_options.set(k, v)?;
                }
                csv_options
            }
        };

        Ok(Arc::new(CsvFormat::default().with_options(csv_options)))
    }

    fn default(&self) -> Arc<dyn FileFormat> {
        Arc::new(CsvFormat::default())
    }
}

impl GetExt for CsvFormatFactory {
    fn get_ext(&self) -> String {
        // Removes the dot, i.e. ".csv" -> "csv"
        DEFAULT_CSV_EXTENSION[1..].to_string()
    }
}

/// Character Separated Value [`FileFormat`] implementation.
#[derive(Debug, Default)]
pub struct CsvFormat {
    options: CsvOptions,
    record_error_handler: Option<Arc<dyn CsvRecordErrorHandler>>,
}

impl CsvFormat {
    /// Return a newline delimited stream from the specified file on
    /// Stream, decompressing if necessary
    /// Each returned `Bytes` has a whole number of newline delimited rows
    async fn read_to_delimited_chunks<'a>(
        &self,
        store: &Arc<dyn ObjectStore>,
        object: &ObjectMeta,
    ) -> BoxStream<'a, Result<Bytes>> {
        // stream to only read as many rows as needed into memory
        let stream = store
            .get(&object.location)
            .await
            .map_err(|e| DataFusionError::ObjectStore(Box::new(e)));
        let stream = match stream {
            Ok(stream) => self
                .read_to_delimited_chunks_from_stream(
                    stream
                        .into_stream()
                        .map_err(|e| DataFusionError::ObjectStore(Box::new(e)))
                        .boxed(),
                )
                .map_err(DataFusionError::from)
                .left_stream(),
            Err(e) => {
                futures::stream::once(futures::future::ready(Err(e))).right_stream()
            }
        };
        stream.boxed()
    }

    /// Convert a stream of bytes into a stream of [`Bytes`] containing newline
    /// delimited CSV records, while accounting for `\` and `"`.
    pub fn read_to_delimited_chunks_from_stream<'a>(
        &self,
        stream: BoxStream<'a, Result<Bytes>>,
    ) -> BoxStream<'a, Result<Bytes>> {
        let file_compression_type: FileCompressionType = self.options.compression.into();
        let decoder = file_compression_type.convert_stream(stream);
        let stream = match decoder {
            Ok(decoded_stream) => {
                newline_delimited_stream(decoded_stream.map_err(|e| match e {
                    DataFusionError::ObjectStore(e) => *e,
                    err => object_store::Error::Generic {
                        store: "read to delimited chunks failed",
                        source: Box::new(err),
                    },
                }))
                .map_err(DataFusionError::from)
                .left_stream()
            }
            Err(e) => {
                futures::stream::once(futures::future::ready(Err(e))).right_stream()
            }
        };
        stream.boxed()
    }

    /// Set the csv options
    pub fn with_options(mut self, options: CsvOptions) -> Self {
        self.options = options;
        self
    }

    /// Skip malformed field-count records and report them to `handler`.
    ///
    /// This recovery path retains source record bytes and disables file range
    /// repartitioning. The default strict path is unchanged.
    pub fn with_record_error_handler(
        mut self,
        handler: Arc<dyn CsvRecordErrorHandler>,
    ) -> Self {
        self.record_error_handler = Some(handler);
        self
    }

    /// Retrieve the csv options
    pub fn options(&self) -> &CsvOptions {
        &self.options
    }

    /// Set a limit in terms of records to scan to infer the schema
    /// - default to `DEFAULT_SCHEMA_INFER_MAX_RECORD`
    ///
    /// # Behavior when set to 0
    ///
    /// When `max_rec` is set to 0, schema inference is disabled and all fields
    /// will be inferred as `Utf8` (string) type, regardless of their actual content.
    pub fn with_schema_infer_max_rec(mut self, max_rec: usize) -> Self {
        self.options.schema_infer_max_rec = Some(max_rec);
        self
    }

    /// Set true to indicate that the first line is a header.
    /// - default to true
    pub fn with_has_header(mut self, has_header: bool) -> Self {
        self.options.has_header = Some(has_header);
        self
    }

    pub fn with_truncated_rows(mut self, truncated_rows: bool) -> Self {
        self.options.truncated_rows = Some(truncated_rows);
        self
    }

    /// Set the regex to use for null values in the CSV reader.
    /// - default to treat empty values as null.
    pub fn with_null_regex(mut self, null_regex: Option<String>) -> Self {
        self.options.null_regex = null_regex;
        self
    }

    /// Keep quoted empty fields distinct from unquoted empty fields during CSV scans.
    ///
    /// If the null matcher includes an empty field, this requires an explicit
    /// table schema: CSV schema inference cannot recover quoting information.
    pub fn with_preserve_quoted_empty(mut self, preserve: bool) -> Self {
        self.options.preserve_quoted_empty = Some(preserve);
        self
    }

    /// Returns `Some(true)` if the first line is a header, `Some(false)` if
    /// it is not, and `None` if it is not specified.
    pub fn has_header(&self) -> Option<bool> {
        self.options.has_header
    }

    /// Lines beginning with this byte are ignored.
    pub fn with_comment(mut self, comment: Option<u8>) -> Self {
        self.options.comment = comment;
        self
    }

    /// The character separating values within a row.
    /// - default to ','
    pub fn with_delimiter(mut self, delimiter: u8) -> Self {
        self.options.delimiter = delimiter;
        self
    }

    /// The quote character in a row.
    /// - default to '"'
    pub fn with_quote(mut self, quote: u8) -> Self {
        self.options.quote = quote;
        self
    }

    /// The escape character in a row.
    /// - default is None
    pub fn with_escape(mut self, escape: Option<u8>) -> Self {
        self.options.escape = escape;
        self
    }

    /// The character used to indicate the end of a row.
    /// - default to None (CRLF)
    pub fn with_terminator(mut self, terminator: Option<u8>) -> Self {
        self.options.terminator = terminator;
        self
    }

    /// Specifies whether newlines in (quoted) values are supported.
    ///
    /// Parsing newlines in quoted values may be affected by execution behaviour such as
    /// parallel file scanning. Setting this to `true` ensures that newlines in values are
    /// parsed successfully, which may reduce performance.
    ///
    /// The default behaviour depends on the `datafusion.catalog.newlines_in_values` setting.
    pub fn with_newlines_in_values(mut self, newlines_in_values: bool) -> Self {
        self.options.newlines_in_values = Some(newlines_in_values);
        self
    }

    /// Set a `FileCompressionType` of CSV
    /// - defaults to `FileCompressionType::UNCOMPRESSED`
    pub fn with_file_compression_type(
        mut self,
        file_compression_type: FileCompressionType,
    ) -> Self {
        self.options.compression = file_compression_type.into();
        self
    }

    /// Set whether rows should be truncated to the column width
    /// - defaults to false
    pub fn with_truncate_rows(mut self, truncate_rows: bool) -> Self {
        self.options.truncated_rows = Some(truncate_rows);
        self
    }

    /// The delimiter character.
    pub fn delimiter(&self) -> u8 {
        self.options.delimiter
    }

    /// The quote character.
    pub fn quote(&self) -> u8 {
        self.options.quote
    }

    /// The escape character.
    pub fn escape(&self) -> Option<u8> {
        self.options.escape
    }
}

#[derive(Debug)]
pub struct CsvDecoder {
    inner: arrow::csv::reader::Decoder,
}

impl CsvDecoder {
    pub fn new(decoder: arrow::csv::reader::Decoder) -> Self {
        Self { inner: decoder }
    }
}

impl Decoder for CsvDecoder {
    fn decode(&mut self, buf: &[u8]) -> Result<usize, ArrowError> {
        self.inner.decode(buf)
    }

    fn flush(&mut self) -> Result<Option<RecordBatch>, ArrowError> {
        self.inner.flush()
    }

    fn can_flush_early(&self) -> bool {
        self.inner.capacity() == 0
    }
}

impl Debug for CsvSerializer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CsvSerializer")
            .field("header", &self.header)
            .finish()
    }
}

#[async_trait]
impl FileFormat for CsvFormat {
    fn get_ext(&self) -> String {
        CsvFormatFactory::new().get_ext()
    }

    fn get_ext_with_compression(
        &self,
        file_compression_type: &FileCompressionType,
    ) -> Result<String> {
        let ext = self.get_ext();
        Ok(format!("{}{}", ext, file_compression_type.get_ext()))
    }

    fn compression_type(&self) -> Option<FileCompressionType> {
        Some(self.options.compression.into())
    }

    async fn infer_schema(
        &self,
        state: &dyn Session,
        store: &Arc<dyn ObjectStore>,
        objects: &[ObjectMeta],
    ) -> Result<SchemaRef> {
        self.infer_schema_impl::<false, false>(state, store, objects)
            .await
    }

    async fn infer_stats(
        &self,
        _state: &dyn Session,
        _store: &Arc<dyn ObjectStore>,
        table_schema: SchemaRef,
        _object: &ObjectMeta,
    ) -> Result<Statistics> {
        Ok(Statistics::new_unknown(&table_schema))
    }

    async fn create_physical_plan(
        &self,
        state: &dyn Session,
        conf: FileScanConfig,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        // Consult configuration options for default values
        let has_header = self
            .options
            .has_header
            .unwrap_or_else(|| state.config_options().catalog.has_header);
        let newlines_in_values = self
            .options
            .newlines_in_values
            .unwrap_or_else(|| state.config_options().catalog.newlines_in_values);

        let mut csv_options = self.options.clone();
        csv_options.has_header = Some(has_header);
        csv_options.newlines_in_values = Some(newlines_in_values);

        // Get the existing CsvSource and update its options
        // We need to preserve the table_schema from the original source (which includes partition columns)
        let csv_source = conf
            .file_source
            .downcast_ref::<CsvSource>()
            .expect("file_source should be a CsvSource");
        let source = Arc::new(csv_source.clone().with_csv_options(csv_options));

        let config = FileScanConfigBuilder::from(conf)
            .with_file_compression_type(self.options.compression.into())
            .with_source(source)
            .build();

        Ok(DataSourceExec::from_data_source(config))
    }

    async fn create_writer_physical_plan(
        &self,
        input: Arc<dyn ExecutionPlan>,
        state: &dyn Session,
        conf: FileSinkConfig,
        order_requirements: Option<LexRequirement>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if conf.insert_op != InsertOp::Append {
            return not_impl_err!("Overwrites are not implemented yet for CSV");
        }

        // `has_header` and `newlines_in_values` fields of CsvOptions may inherit
        // their values from session from configuration settings. To support
        // this logic, writer options are built from the copy of `self.options`
        // with updated values of these special fields.
        let has_header = self
            .options()
            .has_header
            .unwrap_or_else(|| state.config_options().catalog.has_header);
        let newlines_in_values = self
            .options()
            .newlines_in_values
            .unwrap_or_else(|| state.config_options().catalog.newlines_in_values);

        let options = self
            .options()
            .clone()
            .with_has_header(has_header)
            .with_newlines_in_values(newlines_in_values);

        let writer_options = CsvWriterOptions::try_from(&options)?;

        let sink = Arc::new(CsvSink::new(conf, writer_options));

        Ok(Arc::new(DataSinkExec::new(input, sink, order_requirements)) as _)
    }

    fn file_source(&self, table_schema: TableSchema) -> Arc<dyn FileSource> {
        let mut csv_options = self.options.clone();
        if csv_options.has_header.is_none() {
            csv_options.has_header = Some(true);
        }
        let mut source = CsvSource::new(table_schema).with_csv_options(csv_options);
        if let Some(handler) = &self.record_error_handler {
            source = source.with_record_error_handler(Arc::clone(handler));
        }
        Arc::new(source)
    }
}

impl CsvFormat {
    /// Infer exact fixed-point CSV numbers as Arrow decimals. The ordinary
    /// `FileFormat::infer_schema` path retains its existing inference rules.
    pub async fn infer_schema_with_decimal(
        &self,
        state: &dyn Session,
        store: &Arc<dyn ObjectStore>,
        object: &ObjectMeta,
    ) -> Result<SchemaRef> {
        self.infer_schema_impl::<true, false>(state, store, std::slice::from_ref(object))
            .await
    }

    /// Infer Snowflake-compatible, order-sensitive types from one CSV object.
    /// Ordinary CSV scans and order-independent decimal inference are unchanged.
    pub async fn infer_schema_with_snowflake_types(
        &self,
        state: &dyn Session,
        store: &Arc<dyn ObjectStore>,
        object: &ObjectMeta,
    ) -> Result<SchemaRef> {
        self.infer_schema_impl::<true, true>(state, store, std::slice::from_ref(object))
            .await
    }

    async fn infer_schema_impl<const DECIMAL: bool, const SNOWFLAKE: bool>(
        &self,
        state: &dyn Session,
        store: &Arc<dyn ObjectStore>,
        objects: &[ObjectMeta],
    ) -> Result<SchemaRef> {
        let mut schemas = vec![];
        let mut records_to_read = self
            .options
            .schema_infer_max_rec
            .unwrap_or(DEFAULT_SCHEMA_INFER_MAX_RECORD);

        for object in objects {
            let stream = self.read_to_delimited_chunks(store, object).await;
            let (schema, records_read) = self
                .infer_schema_from_stream_impl::<_, DECIMAL, SNOWFLAKE>(
                    state,
                    records_to_read,
                    stream,
                )
                .await
                .map_err(|err| {
                    DataFusionError::Context(
                        format!("Error when processing CSV file {}", object.location),
                        Box::new(err),
                    )
                })?;
            records_to_read -= records_read;
            schemas.push(schema);
            if records_to_read == 0 {
                break;
            }
        }

        let merged_schema = Schema::try_merge(schemas)?;
        Ok(Arc::new(merged_schema))
    }

    /// Return the inferred schema reading up to records_to_read from a
    /// stream of delimited chunks returning the inferred schema and the
    /// number of lines that were read.
    ///
    /// This method can handle CSV files with different numbers of columns.
    /// The inferred schema will be the union of all columns found across all files.
    /// Files with fewer columns will have missing columns filled with null values.
    ///
    /// # Example
    ///
    /// If you have two CSV files:
    /// - `file1.csv`: `col1,col2,col3`
    /// - `file2.csv`: `col1,col2,col3,col4,col5`
    ///
    /// The inferred schema will contain all 5 columns, with files that don't
    /// have columns 4 and 5 having null values for those columns.
    pub async fn infer_schema_from_stream(
        &self,
        state: &dyn Session,
        records_to_read: usize,
        stream: impl Stream<Item = Result<Bytes>>,
    ) -> Result<(Schema, usize)> {
        self.infer_schema_from_stream_impl::<_, false, false>(
            state,
            records_to_read,
            stream,
        )
        .await
    }

    async fn infer_schema_from_stream_impl<
        S: Stream<Item = Result<Bytes>>,
        const DECIMAL: bool,
        const SNOWFLAKE: bool,
    >(
        &self,
        state: &dyn Session,
        mut records_to_read: usize,
        stream: S,
    ) -> Result<(Schema, usize)> {
        let mut total_records_read = 0;
        let mut column_names = vec![];
        let mut column_type_possibilities = vec![];
        let mut decimal_integral_digits = (DECIMAL && !SNOWFLAKE).then(Vec::new);
        let mut ordered_types = SNOWFLAKE.then(Vec::<SnowflakeCsvTypeState>::new);
        let mut record_number = -1;
        let initial_records_to_read = records_to_read;

        pin_mut!(stream);

        while let Some(chunk) = stream.next().await.transpose()? {
            record_number += 1;
            let first_chunk = record_number == 0;
            let mut format = arrow::csv::reader::Format::default()
                .with_header(
                    first_chunk
                        && self
                            .options
                            .has_header
                            .unwrap_or_else(|| state.config_options().catalog.has_header),
                )
                .with_delimiter(self.options.delimiter)
                .with_quote(self.options.quote)
                .with_preserve_quoted_empty(
                    self.options.preserve_quoted_empty.unwrap_or(false),
                )
                .with_truncated_rows(self.options.truncated_rows.unwrap_or(false));

            if let Some(null_regex) = &self.options.null_regex {
                if let Some(values) = crate::exact_null_values(null_regex) {
                    format = format.with_null_values(values);
                } else {
                    let regex = Regex::new(null_regex.as_str())
                        .expect("Unable to parse CSV null regex.");
                    format = format.with_null_regex(regex);
                }
            }

            if let Some(escape) = self.options.escape {
                format = format.with_escape(escape);
            }

            if let Some(comment) = self.options.comment {
                format = format.with_comment(comment);
            }

            let (Schema { fields, .. }, records_read, chunk_states) = if SNOWFLAKE {
                let (schema, states, records) = format
                    .infer_schema_with_snowflake_types(
                        chunk.reader(),
                        Some(records_to_read),
                    )?;
                (schema, records, states)
            } else if DECIMAL {
                let (schema, records) = format
                    .infer_schema_with_decimal(chunk.reader(), Some(records_to_read))?;
                (schema, records, Vec::new())
            } else {
                let (schema, records) =
                    format.infer_schema(chunk.reader(), Some(records_to_read))?;
                (schema, records, Vec::new())
            };

            if let Some(ordered) = ordered_types.as_mut() {
                for (current, next) in ordered.iter_mut().zip(&chunk_states) {
                    current.merge(next);
                }
                ordered.extend(chunk_states.into_iter().skip(ordered.len()));
            }

            records_to_read -= records_read;
            total_records_read += records_read;

            if SNOWFLAKE {
                if first_chunk {
                    column_names =
                        fields.iter().map(|field| field.name().clone()).collect();
                } else {
                    if fields.len() != column_names.len()
                        && !self.options.truncated_rows.unwrap_or(false)
                    {
                        return exec_err!(
                            "Encountered unequal lengths between records on CSV file whilst inferring schema. \
                             Expected {} fields, found {} fields at record {}",
                            column_names.len(),
                            fields.len(),
                            record_number + 1
                        );
                    }
                    column_names.extend(
                        fields
                            .iter()
                            .skip(column_names.len())
                            .map(|field| field.name().clone()),
                    );
                }
            } else if first_chunk {
                // set up initial structures for recording inferred schema across chunks
                (column_names, column_type_possibilities) = fields
                    .into_iter()
                    .map(|field| {
                        let mut possibilities = HashSet::new();
                        if let Some(digits) = decimal_integral_digits.as_mut() {
                            digits.push(csv_decimal_integral_digits(field).unwrap_or(0));
                        }
                        if records_read > 0 {
                            // at least 1 data row read, record the inferred datatype
                            possibilities.insert(field.data_type().clone());
                        }
                        (field.name().clone(), possibilities)
                    })
                    .unzip();
            } else {
                if fields.len() != column_type_possibilities.len()
                    && !self.options.truncated_rows.unwrap_or(false)
                {
                    return exec_err!(
                        "Encountered unequal lengths between records on CSV file whilst inferring schema. \
                         Expected {} fields, found {} fields at record {}",
                        column_type_possibilities.len(),
                        fields.len(),
                        record_number + 1
                    );
                }

                // First update type possibilities for existing columns using zip
                for (index, (possibilities, field)) in column_type_possibilities
                    .iter_mut()
                    .zip(&fields)
                    .enumerate()
                {
                    if let Some(digits) = decimal_integral_digits.as_mut()
                        && let Some(field_digits) = csv_decimal_integral_digits(field)
                    {
                        digits[index] = digits[index].max(field_digits);
                    }
                    possibilities.insert(field.data_type().clone());
                }

                // Handle files with different numbers of columns by extending the schema
                if fields.len() > column_type_possibilities.len() {
                    // New columns found - extend our tracking structures
                    for field in fields.iter().skip(column_type_possibilities.len()) {
                        column_names.push(field.name().clone());
                        let mut possibilities = HashSet::new();
                        if let Some(digits) = decimal_integral_digits.as_mut() {
                            digits.push(csv_decimal_integral_digits(field).unwrap_or(0));
                        }
                        if records_read > 0 {
                            possibilities.insert(field.data_type().clone());
                        }
                        column_type_possibilities.push(possibilities);
                    }
                }
            }

            if records_to_read == 0 {
                break;
            }
        }

        let schema = if let Some(ordered) = ordered_types {
            Schema::new(
                column_names
                    .into_iter()
                    .zip(ordered)
                    .map(|(name, inferred)| {
                        let data_type = if initial_records_to_read == 0 {
                            DataType::Utf8
                        } else {
                            inferred.data_type()
                        };
                        Field::new(name, data_type, true)
                    })
                    .collect::<Fields>(),
            )
        } else {
            build_schema_helper::<DECIMAL>(
                column_names,
                column_type_possibilities,
                decimal_integral_digits.as_deref(),
                initial_records_to_read == 0,
            )
        };
        Ok((schema, total_records_read))
    }
}

fn csv_decimal_integral_digits(field: &Field) -> Option<u16> {
    let DataType::Decimal128(precision, scale) = field.data_type() else {
        return None;
    };
    if field
        .metadata()
        .contains_key(arrow::csv::reader::CSV_DECIMAL_ZERO_INTEGRAL_METADATA_KEY)
    {
        Some(0)
    } else {
        u16::from(*precision).checked_sub(u16::try_from(*scale).ok()?)
    }
}

/// Builds a schema from column names and their possible data types.
///
/// # Arguments
///
/// * `names` - Vector of column names
/// * `types` - Vector of possible data types for each column (as HashSets)
/// * `disable_inference` - When true, forces all columns with no inferred types to be Utf8.
///   This should be set to true when `schema_infer_max_rec` is explicitly
///   set to 0, indicating the user wants to skip type inference and treat
///   all fields as strings. When false, columns with no inferred types
///   will be set to Null, allowing schema merging to work properly.
fn build_schema_helper<const DECIMAL: bool>(
    names: Vec<String>,
    types: Vec<HashSet<DataType>>,
    decimal_integral_digits: Option<&[u16]>,
    disable_inference: bool,
) -> Schema {
    let fields = names
        .into_iter()
        .zip(types)
        .enumerate()
        .map(|(index, (field_name, mut data_type_possibilities))| {
            // ripped from arrow::csv::reader::infer_reader_schema_with_csv_options
            // determine data type based on possible types
            // if there are incompatible types, use DataType::Utf8

            // ignore nulls, to avoid conflicting datatypes (e.g. [nulls, int]) being inferred as Utf8.
            data_type_possibilities.remove(&DataType::Null);

            if DECIMAL
                && let Some(decimal_type) = merge_csv_decimal_types(
                    &data_type_possibilities,
                    decimal_integral_digits
                        .and_then(|digits| digits.get(index))
                        .copied(),
                )
            {
                return Field::new(field_name, decimal_type, true);
            }

            match data_type_possibilities.len() {
                // When no types were inferred (empty HashSet):
                // - If schema_infer_max_rec was explicitly set to 0, return Utf8
                // - Otherwise return Null (whether from reading null values or empty files)
                //   This allows schema merging to work when reading folders with empty files
                0 => {
                    if disable_inference {
                        Field::new(field_name, DataType::Utf8, true)
                    } else {
                        Field::new(field_name, DataType::Null, true)
                    }
                }
                1 => Field::new(
                    field_name,
                    data_type_possibilities.iter().next().unwrap().clone(),
                    true,
                ),
                2 => {
                    if data_type_possibilities.contains(&DataType::Int64)
                        && data_type_possibilities.contains(&DataType::Float64)
                    {
                        // we have an integer and double, fall down to double
                        Field::new(field_name, DataType::Float64, true)
                    } else {
                        // default to Utf8 for conflicting datatypes (e.g bool and int)
                        Field::new(field_name, DataType::Utf8, true)
                    }
                }
                _ => Field::new(field_name, DataType::Utf8, true),
            }
        })
        .collect::<Fields>();
    Schema::new(fields)
}

fn merge_csv_decimal_types(
    types: &HashSet<DataType>,
    actual_integral_digits: Option<u16>,
) -> Option<DataType> {
    if !types
        .iter()
        .any(|t| matches!(t, DataType::Decimal128(_, _)))
        || !types
            .iter()
            .all(|t| matches!(t, DataType::Decimal128(_, _) | DataType::Float64))
    {
        return None;
    }
    let mut integral_digits = 0_u16;
    let mut scale = 0_u16;
    for data_type in types {
        match data_type {
            DataType::Decimal128(precision, field_scale) if *field_scale >= 0 => {
                let field_scale = u16::try_from(*field_scale).ok()?;
                integral_digits =
                    integral_digits.max(u16::from(*precision).checked_sub(field_scale)?);
                scale = scale.max(field_scale);
            }
            DataType::Float64 => return Some(DataType::Float64),
            _ => return None,
        }
    }
    let integral_digits = actual_integral_digits.unwrap_or(integral_digits);
    let precision = integral_digits + scale;
    let precision = if integral_digits == 0 && scale > 0 && precision < 38 {
        precision + 1
    } else {
        precision.max(1)
    };
    if precision > 38 {
        Some(DataType::Float64)
    } else {
        Some(DataType::Decimal128(precision as u8, scale as i8))
    }
}

impl Default for CsvSerializer {
    fn default() -> Self {
        Self::new()
    }
}

/// Define a struct for serializing CSV records to a stream
pub struct CsvSerializer {
    // CSV writer builder
    builder: WriterBuilder,
    // Flag to indicate whether there will be a header
    header: bool,
}

impl CsvSerializer {
    /// Constructor for the CsvSerializer object
    pub fn new() -> Self {
        Self {
            builder: WriterBuilder::new(),
            header: true,
        }
    }

    /// Method for setting the CSV writer builder
    pub fn with_builder(mut self, builder: WriterBuilder) -> Self {
        self.builder = builder;
        self
    }

    /// Method for setting the CSV writer header status
    pub fn with_header(mut self, header: bool) -> Self {
        self.header = header;
        self
    }
}

impl BatchSerializer for CsvSerializer {
    fn serialize(&self, batch: RecordBatch, initial: bool) -> Result<Bytes> {
        let mut buffer = Vec::with_capacity(4096);
        let builder = self.builder.clone();
        let header = self.header && initial;
        let mut writer = builder.with_header(header).build(&mut buffer);
        writer.write(&batch)?;
        drop(writer);
        Ok(Bytes::from(buffer))
    }
}

/// Implements [`DataSink`] for writing to a CSV file.
pub struct CsvSink {
    /// Config options for writing data
    config: FileSinkConfig,
    writer_options: CsvWriterOptions,
}

impl Debug for CsvSink {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CsvSink").finish()
    }
}

impl DisplayAs for CsvSink {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(f, "CsvSink(file_groups=",)?;
                FileGroupDisplay(&self.config.file_group).fmt_as(t, f)?;
                write!(f, ")")
            }
            DisplayFormatType::TreeRender => {
                writeln!(f, "format: csv")?;
                write!(f, "file={}", self.config.original_url)
            }
        }
    }
}

impl CsvSink {
    /// Create from config.
    pub fn new(config: FileSinkConfig, writer_options: CsvWriterOptions) -> Self {
        Self {
            config,
            writer_options,
        }
    }

    /// Retrieve the writer options
    pub fn writer_options(&self) -> &CsvWriterOptions {
        &self.writer_options
    }
}

#[async_trait]
impl FileSink for CsvSink {
    fn config(&self) -> &FileSinkConfig {
        &self.config
    }

    async fn spawn_writer_tasks_and_join(
        &self,
        context: &Arc<TaskContext>,
        demux_task: SpawnedTask<Result<()>>,
        file_stream_rx: DemuxedStreamReceiver,
        object_store: Arc<dyn ObjectStore>,
    ) -> Result<u64> {
        let builder = self.writer_options.writer_options.clone();
        let header = builder.header();
        let serializer = Arc::new(
            CsvSerializer::new()
                .with_builder(builder)
                .with_header(header),
        ) as _;
        spawn_writer_tasks_and_join(
            context,
            serializer,
            self.writer_options.compression.into(),
            self.writer_options.compression_level,
            object_store,
            demux_task,
            file_stream_rx,
        )
        .await
    }
}

#[async_trait]
impl DataSink for CsvSink {
    fn schema(&self) -> &SchemaRef {
        self.config.output_schema()
    }

    async fn write_all(
        &self,
        data: SendableRecordBatchStream,
        context: &Arc<TaskContext>,
    ) -> Result<u64> {
        FileSink::write_all(self, data, context).await
    }

    #[cfg(feature = "proto")]
    fn try_to_proto(
        &self,
        exec: &DataSinkExec,
        ctx: &datafusion_physical_plan::proto::ExecutionPlanEncodeCtx<'_>,
    ) -> Result<Option<datafusion_proto_models::protobuf::PhysicalPlanNode>> {
        use datafusion_proto_models::protobuf;
        use protobuf::physical_plan_node::PhysicalPlanType;

        let input = ctx.encode_child(exec.input())?;
        let sort_order = exec.encode_sort_order(ctx)?;
        let sink = protobuf::CsvSink::try_from(self)?;
        let node = protobuf::CsvSinkExecNode {
            input: Some(Box::new(input)),
            sink: Some(sink),
            sink_schema: Some(exec.schema().as_ref().try_into()?),
            sort_order,
        };
        Ok(Some(protobuf::PhysicalPlanNode {
            physical_plan_type: Some(PhysicalPlanType::CsvSink(Box::new(node))),
        }))
    }
}

#[cfg(feature = "proto")]
impl TryFrom<&CsvSink> for datafusion_proto_models::protobuf::CsvSink {
    type Error = DataFusionError;

    fn try_from(value: &CsvSink) -> Result<Self> {
        Ok(Self {
            config: Some(value.config().try_into()?),
            writer_options: Some(value.writer_options().try_into()?),
        })
    }
}

#[cfg(feature = "proto")]
impl TryFrom<&datafusion_proto_models::protobuf::CsvSink> for CsvSink {
    type Error = DataFusionError;

    fn try_from(value: &datafusion_proto_models::protobuf::CsvSink) -> Result<Self> {
        let config =
            FileSinkConfig::try_from(value.config.as_ref().ok_or_else(|| {
                datafusion_common::internal_datafusion_err!(
                    "CsvSink is missing required field 'config'"
                )
            })?)?;
        let writer_options = value
            .writer_options
            .as_ref()
            .ok_or_else(|| {
                datafusion_common::internal_datafusion_err!(
                    "CsvSink is missing required field 'writer_options'"
                )
            })?
            .try_into()?;

        Ok(Self::new(config, writer_options))
    }
}

#[cfg(feature = "proto")]
impl CsvSink {
    /// Reconstructs a [`DataSinkExec`] containing a `CsvSink` from protobuf.
    pub fn try_from_proto(
        node: &datafusion_proto_models::protobuf::PhysicalPlanNode,
        ctx: &datafusion_physical_plan::proto::ExecutionPlanDecodeCtx<'_>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        use datafusion_proto_models::protobuf;

        let sink_node = datafusion_physical_plan::expect_plan_variant!(
            node,
            protobuf::physical_plan_node::PhysicalPlanType::CsvSink,
            "CsvSink",
        );
        let input = ctx.decode_required_child(
            sink_node.input.as_deref(),
            "CsvSinkExecNode",
            "input",
        )?;
        let proto_sink = sink_node.sink.as_ref().ok_or_else(|| {
            datafusion_common::internal_datafusion_err!(
                "CsvSinkExecNode is missing required field 'sink'"
            )
        })?;
        let data_sink = CsvSink::try_from(proto_sink)?;
        let sort_order = DataSinkExec::decode_sort_order(
            sink_node.sort_order.as_ref(),
            ctx,
            input.schema().as_ref(),
        )?;

        Ok(Arc::new(DataSinkExec::new(
            input,
            Arc::new(data_sink),
            sort_order,
        )))
    }
}

/// Encode a [`CsvFormatFactory`]'s options as their protobuf form.
///
/// The reverse direction is `From<&protobuf::CsvOptions> for CsvOptions` in
/// `datafusion-proto-models`: `CsvOptions` is a `datafusion-common` type, so
/// that half cannot live here.
#[cfg(feature = "proto")]
impl From<&CsvFormatFactory> for datafusion_proto_models::protobuf::CsvOptions {
    fn from(factory: &CsvFormatFactory) -> Self {
        if let Some(options) = &factory.options {
            datafusion_proto_models::protobuf::CsvOptions {
                has_header: options.has_header.map_or(vec![], |v| vec![v as u8]),
                delimiter: vec![options.delimiter],
                quote: vec![options.quote],
                terminator: options.terminator.map_or(vec![], |v| vec![v]),
                escape: options.escape.map_or(vec![], |v| vec![v]),
                double_quote: options.double_quote.map_or(vec![], |v| vec![v as u8]),
                compression: options.compression as i32,
                schema_infer_max_rec: options.schema_infer_max_rec.map(|v| v as u64),
                date_format: options.date_format.clone().unwrap_or_default(),
                datetime_format: options.datetime_format.clone().unwrap_or_default(),
                timestamp_format: options.timestamp_format.clone().unwrap_or_default(),
                timestamp_tz_format: options
                    .timestamp_tz_format
                    .clone()
                    .unwrap_or_default(),
                time_format: options.time_format.clone().unwrap_or_default(),
                null_value: options.null_value.clone().unwrap_or_default(),
                null_regex: options.null_regex.clone().unwrap_or_default(),
                preserve_quoted_empty: options.preserve_quoted_empty,
                comment: options.comment.map_or(vec![], |v| vec![v]),
                newlines_in_values: options
                    .newlines_in_values
                    .map_or(vec![], |v| vec![v as u8]),
                truncated_rows: options.truncated_rows.map_or(vec![], |v| vec![v as u8]),
                compression_level: options.compression_level,
                quote_style: options.quote_style as i32,
                ignore_leading_whitespace: options
                    .ignore_leading_whitespace
                    .map_or(vec![], |v| vec![v as u8]),
                ignore_trailing_whitespace: options
                    .ignore_trailing_whitespace
                    .map_or(vec![], |v| vec![v as u8]),
            }
        } else {
            datafusion_proto_models::protobuf::CsvOptions::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{CsvFormat, build_schema_helper};
    use arrow::datatypes::DataType;
    use bytes::Bytes;
    use datafusion_common::DFSchema;
    use datafusion_common::config::TableOptions;
    use datafusion_execution::TaskContext;
    use datafusion_execution::config::SessionConfig;
    use datafusion_execution::runtime_env::RuntimeEnv;
    use datafusion_expr::execution_props::ExecutionProps;
    use datafusion_expr::registry::ExtensionTypeRegistryRef;
    use datafusion_expr::{
        AggregateUDF, Expr, HigherOrderUDF, LogicalPlan, ScalarUDF, WindowUDF,
    };
    use datafusion_physical_expr_common::physical_expr::PhysicalExpr;
    use datafusion_physical_plan::ExecutionPlan;
    use datafusion_session::{CatalogProviderList, EmptyCatalogProviderList, Session};
    use futures::stream;
    use std::any::Any;
    use std::collections::{HashMap, HashSet};
    use std::sync::Arc;

    struct InferenceSession {
        config: SessionConfig,
        runtime_env: Arc<RuntimeEnv>,
    }

    impl InferenceSession {
        fn new() -> Self {
            Self {
                config: SessionConfig::new(),
                runtime_env: Arc::new(RuntimeEnv::default()),
            }
        }
    }

    #[async_trait::async_trait]
    impl Session for InferenceSession {
        fn session_id(&self) -> &str {
            unimplemented!()
        }
        fn config(&self) -> &SessionConfig {
            &self.config
        }
        fn catalog_list(&self) -> Arc<dyn CatalogProviderList> {
            Arc::new(EmptyCatalogProviderList)
        }
        async fn create_physical_plan(
            &self,
            _: &LogicalPlan,
        ) -> datafusion_common::Result<Arc<dyn ExecutionPlan>> {
            unimplemented!()
        }
        fn create_physical_expr(
            &self,
            _: Expr,
            _: &DFSchema,
        ) -> datafusion_common::Result<Arc<dyn PhysicalExpr>> {
            unimplemented!()
        }
        fn scalar_functions(&self) -> &HashMap<String, Arc<ScalarUDF>> {
            unimplemented!()
        }
        fn higher_order_functions(&self) -> &HashMap<String, Arc<HigherOrderUDF>> {
            unimplemented!()
        }
        fn aggregate_functions(&self) -> &HashMap<String, Arc<AggregateUDF>> {
            unimplemented!()
        }
        fn window_functions(&self) -> &HashMap<String, Arc<WindowUDF>> {
            unimplemented!()
        }
        fn extension_type_registry(&self) -> &ExtensionTypeRegistryRef {
            unimplemented!()
        }
        fn runtime_env(&self) -> &Arc<RuntimeEnv> {
            &self.runtime_env
        }
        fn execution_props(&self) -> &ExecutionProps {
            unimplemented!()
        }
        fn as_any(&self) -> &dyn Any {
            unimplemented!()
        }
        fn table_options(&self) -> &TableOptions {
            unimplemented!()
        }
        fn table_options_mut(&mut self) -> &mut TableOptions {
            unimplemented!()
        }
        fn task_ctx(&self) -> Arc<TaskContext> {
            unimplemented!()
        }
    }

    #[test]
    fn test_build_schema_helper_different_column_counts() {
        // Test the core schema building logic with different column counts
        let mut column_names =
            vec!["col1".to_string(), "col2".to_string(), "col3".to_string()];

        // Simulate adding two more columns from another file
        column_names.push("col4".to_string());
        column_names.push("col5".to_string());

        let column_type_possibilities = vec![
            HashSet::from([DataType::Int64]),
            HashSet::from([DataType::Utf8]),
            HashSet::from([DataType::Float64]),
            HashSet::from([DataType::Utf8]), // col4
            HashSet::from([DataType::Utf8]), // col5
        ];

        let schema = build_schema_helper::<false>(
            column_names,
            column_type_possibilities,
            None,
            false,
        );

        // Verify schema has 5 columns
        assert_eq!(schema.fields().len(), 5);
        assert_eq!(schema.field(0).name(), "col1");
        assert_eq!(schema.field(1).name(), "col2");
        assert_eq!(schema.field(2).name(), "col3");
        assert_eq!(schema.field(3).name(), "col4");
        assert_eq!(schema.field(4).name(), "col5");

        // All fields should be nullable
        for field in schema.fields() {
            assert!(
                field.is_nullable(),
                "Field {} should be nullable",
                field.name()
            );
        }
    }

    #[test]
    fn test_build_schema_helper_type_merging() {
        // Test type merging logic
        let column_names = vec!["col1".to_string(), "col2".to_string()];

        let column_type_possibilities = vec![
            HashSet::from([DataType::Int64, DataType::Float64]), // Should resolve to Float64
            HashSet::from([DataType::Utf8]),                     // Should remain Utf8
        ];

        let schema = build_schema_helper::<false>(
            column_names,
            column_type_possibilities,
            None,
            false,
        );

        // col1 should be Float64 due to Int64 + Float64 = Float64
        assert_eq!(*schema.field(0).data_type(), DataType::Float64);

        // col2 should remain Utf8
        assert_eq!(*schema.field(1).data_type(), DataType::Utf8);
    }

    #[test]
    fn test_build_schema_helper_conflicting_types() {
        // Test when we have incompatible types - should default to Utf8
        let column_names = vec!["col1".to_string()];

        let column_type_possibilities = vec![
            HashSet::from([DataType::Boolean, DataType::Int64, DataType::Utf8]), // Should resolve to Utf8 due to conflicts
        ];

        let schema = build_schema_helper::<false>(
            column_names,
            column_type_possibilities,
            None,
            false,
        );

        // Should default to Utf8 for conflicting types
        assert_eq!(*schema.field(0).data_type(), DataType::Utf8);
    }

    #[test]
    fn test_opt_in_decimal_chunk_merging() {
        let names = vec![
            "DECIMAL".to_owned(),
            "SCIENTIFIC".to_owned(),
            "TEXT".to_owned(),
            "MIXED_TEXT".to_owned(),
        ];
        let types = vec![
            HashSet::from([DataType::Decimal128(3, 1), DataType::Decimal128(6, 3)]),
            HashSet::from([DataType::Decimal128(3, 1), DataType::Float64]),
            HashSet::from([DataType::Decimal128(3, 1), DataType::Utf8]),
            HashSet::from([
                DataType::Decimal128(3, 1),
                DataType::Float64,
                DataType::Utf8,
            ]),
        ];
        let schema =
            build_schema_helper::<true>(names.clone(), types.clone(), None, false);
        assert_eq!(schema.field(0).data_type(), &DataType::Decimal128(6, 3));
        assert_eq!(schema.field(1).data_type(), &DataType::Float64);
        assert_eq!(schema.field(2).data_type(), &DataType::Utf8);
        assert_eq!(schema.field(3).data_type(), &DataType::Utf8);

        let legacy = build_schema_helper::<false>(names, types, None, false);
        assert_eq!(legacy.field(0).data_type(), &DataType::Utf8);
    }

    #[test]
    fn test_opt_in_decimal_chunk_precision_overflow() {
        let schema = build_schema_helper::<true>(
            vec!["VALUE".to_owned()],
            vec![HashSet::from([
                DataType::Decimal128(38, 0),
                DataType::Decimal128(38, 38),
            ])],
            None,
            false,
        );
        assert_eq!(schema.field(0).data_type(), &DataType::Float64);
    }

    #[tokio::test]
    async fn test_opt_in_decimal_stream_preserves_zero_only_chunks() {
        let tiny = format!("0.{}1", "0".repeat(37));
        let chunks = stream::iter(vec![
            Ok(Bytes::from_static(b"EXACT,MIXED\n0.1,1.20\n")),
            Ok(Bytes::from(format!("{tiny},1e2\n"))),
            Ok(Bytes::from(format!("{tiny},hello\n"))),
        ]);
        let (schema, records) = CsvFormat::default()
            .with_has_header(true)
            .infer_schema_from_stream_impl::<_, true, false>(
                &InferenceSession::new(),
                3,
                chunks,
            )
            .await
            .unwrap();
        assert_eq!(records, 3);
        assert_eq!(schema.field(0).data_type(), &DataType::Decimal128(38, 38));
        assert_eq!(schema.field(1).data_type(), &DataType::Utf8);
        assert!(schema.field(0).metadata().is_empty());

        let zeros = stream::iter(vec![
            Ok(Bytes::from_static(b"VALUE\n0\n")),
            Ok(Bytes::from_static(b"-0.000\n")),
        ]);
        let (schema, records) = CsvFormat::default()
            .with_has_header(true)
            .infer_schema_from_stream_impl::<_, true, false>(
                &InferenceSession::new(),
                2,
                zeros,
            )
            .await
            .unwrap();
        assert_eq!(records, 2);
        assert_eq!(schema.field(0).data_type(), &DataType::Decimal128(1, 0));

        let integer_and_tiny = stream::iter(vec![
            Ok(Bytes::from_static(b"VALUE\n1\n")),
            Ok(Bytes::from(format!("{tiny}\n"))),
        ]);
        let (schema, _) = CsvFormat::default()
            .with_has_header(true)
            .infer_schema_from_stream_impl::<_, true, false>(
                &InferenceSession::new(),
                2,
                integer_and_tiny,
            )
            .await
            .unwrap();
        assert_eq!(schema.field(0).data_type(), &DataType::Float64);
    }

    #[tokio::test]
    async fn test_snowflake_ordered_inference_across_csv_chunks() {
        let tiny = format!("0.{}1", "0".repeat(37));
        let chunks = stream::iter(vec![
            Ok(Bytes::from(format!(
                "INT_EXP,EXP_INT,ZERO_TINY,TINY_ZERO,DATE_TS,TS_DATE,EXACT\n1,1e2,0,{tiny},2024-01-02,2024-01-02 01:02:03,1.20\n"
            ))),
            Ok(Bytes::from(format!(
                "1e2,1,{tiny},0,2024-01-02 01:02:03,2024-01-02,123.400\n"
            ))),
        ]);
        let (schema, records) = CsvFormat::default()
            .with_has_header(true)
            .infer_schema_from_stream_impl::<_, true, true>(
                &InferenceSession::new(),
                2,
                chunks,
            )
            .await
            .unwrap();
        assert_eq!(records, 2);
        let actual = schema
            .fields()
            .iter()
            .map(|field| field.data_type().clone())
            .collect::<Vec<_>>();
        assert_eq!(
            actual,
            [
                DataType::Utf8,
                DataType::Float64,
                DataType::Utf8,
                DataType::Float64,
                DataType::Date32,
                DataType::Utf8,
                DataType::Decimal128(6, 3),
            ]
        );

        for (first, later, expected) in [
            (
                "2024-01-02",
                "2024-01-02 01:02:03\n2024-01-02",
                DataType::Date32,
            ),
            ("1e2", "1\n1e2", DataType::Float64),
            ("true", "1\nfalse", DataType::Boolean),
            ("1", "true\n0", DataType::Utf8),
            ("true", "1\n2", DataType::Utf8),
            (
                "1234567890123456789012345678901234567",
                "0.01",
                DataType::Decimal128(38, 1),
            ),
            (
                "12345678901234567890123456789012345678",
                "0.1",
                DataType::Decimal128(38, 0),
            ),
        ] {
            let chunks = stream::iter(vec![
                Ok(Bytes::from(format!("VALUE\n{first}\n"))),
                Ok(Bytes::from(format!("{later}\n"))),
            ]);
            let (schema, _) = CsvFormat::default()
                .with_has_header(true)
                .infer_schema_from_stream_impl::<_, true, true>(
                    &InferenceSession::new(),
                    3,
                    chunks,
                )
                .await
                .unwrap();
            assert_eq!(schema.field(0).data_type(), &expected, "{first}; {later}");
        }
    }
}
