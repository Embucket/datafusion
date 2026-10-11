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

use std::borrow::Cow;
use std::fmt;
use std::sync::Arc;

use arrow::array::{RecordBatch, RecordBatchOptions, StringBuilder};
use arrow::datatypes::{DataType, SchemaRef};
use arrow::error::ArrowError;
use datafusion_datasource::decoder::Decoder;
use object_store::path::Path;

const MAX_RECORD_BYTES: usize = 64 * 1024 * 1024;
const MAX_BATCH_BYTES: usize = 16 * 1024 * 1024;

/// Input normalization applied only by the opt-in raw-record decoder.
#[derive(Clone, Copy, Debug)]
pub struct RawJsonOptions {
    pub skip_bom: bool,
    pub replace_invalid_utf8: bool,
}

impl Default for RawJsonOptions {
    fn default() -> Self {
        Self {
            skip_bom: true,
            replace_invalid_utf8: false,
        }
    }
}

/// One physical line in a newline-delimited JSON file.
pub struct JsonRecord<'a> {
    pub line_number: usize,
    pub byte_offset: usize,
    /// Absolute byte offset immediately after the record content, excluding CR/LF.
    pub end_byte_offset: usize,
    pub record: &'a [u8],
    pub error: Option<&'a ArrowError>,
}

/// Called for each non-empty raw record. Invalid UTF-8 and oversized records
/// have `error` set and are skipped when the handler returns `Ok(())`.
/// JSON syntax and type conversion are the consumer's responsibility.
pub trait JsonRecordHandler: fmt::Debug + Send + Sync {
    fn handle(&self, record: &JsonRecord<'_>) -> Result<(), ArrowError>;
}

/// Creates a record handler scoped to the file being opened.
pub trait JsonRecordHandlerFactory: fmt::Debug + Send + Sync {
    fn for_file(&self, location: &Path) -> Arc<dyn JsonRecordHandler>;
}

/// Bounded framing for newline-delimited raw JSON records.
#[derive(Debug)]
pub(crate) struct JsonRecoveryDecoder {
    schema: SchemaRef,
    values: Option<StringBuilder>,
    rows: usize,
    handler: Option<Arc<dyn JsonRecordHandler>>,
    pending: Vec<u8>,
    line_number: usize,
    byte_offset: usize,
    batch_size: usize,
    batch_bytes: usize,
    options: RawJsonOptions,
    oversized_start: Option<usize>,
    oversized_last_byte: Option<u8>,
}

impl JsonRecoveryDecoder {
    pub(crate) fn new_raw(
        schema: SchemaRef,
        handler: Option<Arc<dyn JsonRecordHandler>>,
        batch_size: usize,
        options: RawJsonOptions,
    ) -> Result<Self, ArrowError> {
        if !schema.fields().is_empty()
            && !matches!(schema.fields().as_ref(), [field] if field.data_type() == &DataType::Utf8)
        {
            return Err(ArrowError::JsonError(
                "Raw JSON records require zero or one Utf8 field".to_string(),
            ));
        }
        let values = (!schema.fields().is_empty()).then(StringBuilder::new);
        Ok(Self {
            schema,
            values,
            rows: 0,
            handler,
            pending: Vec::new(),
            line_number: 1,
            byte_offset: 0,
            batch_size,
            batch_bytes: 0,
            options,
            oversized_start: None,
            oversized_last_byte: None,
        })
    }

    fn process_record(
        &mut self,
        record: &[u8],
        byte_offset: usize,
    ) -> Result<(), ArrowError> {
        let (record, byte_offset) = if byte_offset == 0
            && self.options.skip_bom
            && record.starts_with(&[0xEF, 0xBB, 0xBF])
        {
            (&record[3..], 3)
        } else {
            (record, byte_offset)
        };
        if !record
            .iter()
            .all(|byte| matches!(byte, b' ' | b'\t' | b'\r' | b'\n'))
        {
            let result: Result<Cow<'_, str>, ArrowError> =
                if self.options.replace_invalid_utf8 {
                    Ok(String::from_utf8_lossy(record))
                } else {
                    std::str::from_utf8(record)
                        .map(Cow::Borrowed)
                        .map_err(|error| {
                            ArrowError::JsonError(format!(
                                "Invalid UTF-8 in JSON record: {error}"
                            ))
                        })
                };
            if let Some(handler) = &self.handler {
                let content = record.strip_suffix(b"\n").unwrap_or(record);
                let content = content.strip_suffix(b"\r").unwrap_or(content);
                handler.handle(&JsonRecord {
                    line_number: self.line_number,
                    byte_offset,
                    end_byte_offset: byte_offset.saturating_add(content.len()),
                    record,
                    error: result.as_ref().err(),
                })?;
            }
            match result {
                Ok(value) => {
                    if let Some(values) = &mut self.values {
                        values.append_value(value.as_ref());
                    }
                    self.rows += 1;
                    self.batch_bytes = self.batch_bytes.saturating_add(record.len());
                }
                Err(error) if self.handler.is_none() => return Err(error),
                Err(_) => {}
            }
        }
        self.line_number = self.line_number.saturating_add(1);
        Ok(())
    }

    fn check_record_size(size: usize) -> Result<(), ArrowError> {
        if size > MAX_RECORD_BYTES {
            return Err(ArrowError::JsonError(format!(
                "JSON record exceeds the {MAX_RECORD_BYTES}-byte recovery limit"
            )));
        }
        Ok(())
    }

    fn report_oversized(
        &self,
        byte_offset: usize,
        end_byte_offset: usize,
    ) -> Result<(), ArrowError> {
        let error = ArrowError::JsonError(format!(
            "JSON record exceeds the {MAX_RECORD_BYTES}-byte recovery limit"
        ));
        match &self.handler {
            Some(handler) => handler.handle(&JsonRecord {
                line_number: self.line_number,
                byte_offset,
                end_byte_offset,
                record: &[],
                error: Some(&error),
            }),
            None => Err(error),
        }
    }
}

impl Decoder for JsonRecoveryDecoder {
    fn decode(&mut self, buf: &[u8]) -> Result<usize, ArrowError> {
        if self.can_flush_early() {
            return Ok(0);
        }

        let mut consumed = 0;
        while consumed < buf.len() {
            let remaining = &buf[consumed..];
            if let Some(start) = self.oversized_start {
                if let Some(end) = remaining.iter().position(|&byte| byte == b'\n') {
                    let last_byte = if end == 0 {
                        self.oversized_last_byte
                    } else {
                        Some(remaining[end - 1])
                    };
                    let size = end + 1;
                    consumed += size;
                    self.byte_offset = self.byte_offset.saturating_add(size);
                    let content_end = self
                        .byte_offset
                        .saturating_sub(1 + usize::from(last_byte == Some(b'\r')));
                    self.report_oversized(start, content_end)?;
                    self.oversized_start = None;
                    self.oversized_last_byte = None;
                    self.line_number = self.line_number.saturating_add(1);
                    continue;
                }
                self.oversized_last_byte = remaining.last().copied();
                self.byte_offset = self.byte_offset.saturating_add(remaining.len());
                return Ok(buf.len());
            }
            let Some(end) = remaining.iter().position(|&byte| byte == b'\n') else {
                if self.pending.len().saturating_add(remaining.len()) > MAX_RECORD_BYTES {
                    let start = self.byte_offset.saturating_sub(self.pending.len());
                    if self.handler.is_none() {
                        Self::check_record_size(MAX_RECORD_BYTES + 1)?;
                    }
                    self.oversized_last_byte = remaining
                        .last()
                        .copied()
                        .or_else(|| self.pending.last().copied());
                    self.pending = Vec::new();
                    self.oversized_start = Some(start);
                    self.byte_offset = self.byte_offset.saturating_add(remaining.len());
                    return Ok(buf.len());
                }
                self.pending.extend_from_slice(remaining);
                self.byte_offset = self.byte_offset.saturating_add(remaining.len());
                return Ok(buf.len());
            };

            let line = &remaining[..=end];
            let record_start = self.byte_offset.saturating_sub(self.pending.len());
            if self.pending.len().saturating_add(line.len()) > MAX_RECORD_BYTES {
                if self.handler.is_none() {
                    Self::check_record_size(MAX_RECORD_BYTES + 1)?;
                }
                let last_byte = if end == 0 {
                    self.pending.last().copied()
                } else {
                    Some(line[end - 1])
                };
                self.pending = Vec::new();
                consumed += line.len();
                self.byte_offset = self.byte_offset.saturating_add(line.len());
                let content_end = self
                    .byte_offset
                    .saturating_sub(1 + usize::from(last_byte == Some(b'\r')));
                self.report_oversized(record_start, content_end)?;
                self.line_number = self.line_number.saturating_add(1);
                continue;
            }
            if self.pending.is_empty() {
                self.process_record(line, record_start)?;
            } else {
                self.pending.extend_from_slice(line);
                let record = std::mem::take(&mut self.pending);
                self.process_record(&record, record_start)?;
            }
            consumed += line.len();
            self.byte_offset = self.byte_offset.saturating_add(line.len());
            if self.can_flush_early() {
                break;
            }
        }
        Ok(consumed)
    }

    fn flush(&mut self) -> Result<Option<RecordBatch>, ArrowError> {
        if let Some(start) = self.oversized_start.take() {
            let content_end = self
                .byte_offset
                .saturating_sub(usize::from(self.oversized_last_byte == Some(b'\r')));
            self.report_oversized(start, content_end)?;
            self.oversized_last_byte = None;
            self.line_number = self.line_number.saturating_add(1);
        }
        if !self.pending.is_empty() {
            let record_start = self.byte_offset.saturating_sub(self.pending.len());
            let record = std::mem::take(&mut self.pending);
            self.process_record(&record, record_start)?;
        }
        let batch = if self.rows == 0 {
            None
        } else {
            let count = std::mem::take(&mut self.rows);
            Some(if let Some(values) = &mut self.values {
                RecordBatch::try_new(
                    Arc::clone(&self.schema),
                    vec![Arc::new(values.finish())],
                )?
            } else {
                RecordBatch::try_new_with_options(
                    Arc::clone(&self.schema),
                    vec![],
                    &RecordBatchOptions::new().with_row_count(Some(count)),
                )?
            })
        };
        self.batch_bytes = 0;
        Ok(batch)
    }

    fn can_flush_early(&self) -> bool {
        self.rows >= self.batch_size || self.batch_bytes >= MAX_BATCH_BYTES
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::StringArray;
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Mutex;

    #[derive(Debug, Default)]
    struct CollectRecords(Mutex<Vec<(usize, usize, usize, bool)>>);

    impl JsonRecordHandler for CollectRecords {
        fn handle(&self, record: &JsonRecord<'_>) -> Result<(), ArrowError> {
            self.0.lock().unwrap().push((
                record.line_number,
                record.byte_offset,
                record.end_byte_offset,
                record.error.is_some(),
            ));
            Ok(())
        }
    }

    #[test]
    fn frames_split_records_and_final_line() {
        let schema =
            Arc::new(Schema::new(vec![Field::new("raw", DataType::Utf8, false)]));
        let handler = Arc::new(CollectRecords::default());
        let mut decoder = JsonRecoveryDecoder::new_raw(
            schema,
            Some(handler.clone()),
            2,
            RawJsonOptions::default(),
        )
        .unwrap();
        let input = b"{\"id\":1}\n{\"id\":oops}\n\n{\"id\":3}";
        let mut rows = 0;

        for chunk in input.chunks(3) {
            let mut remaining = chunk;
            while !remaining.is_empty() {
                let consumed = decoder.decode(remaining).unwrap();
                remaining = &remaining[consumed..];
                if decoder.can_flush_early() {
                    rows += decoder.flush().unwrap().unwrap().num_rows();
                }
            }
        }
        rows += decoder.flush().unwrap().unwrap().num_rows();
        assert_eq!(rows, 3);
        assert_eq!(
            *handler.0.lock().unwrap(),
            vec![(1, 0, 8, false), (2, 9, 20, false), (4, 22, 30, false)]
        );
    }

    #[test]
    fn rejects_unbounded_record() {
        assert!(JsonRecoveryDecoder::check_record_size(MAX_RECORD_BYTES).is_ok());
        assert!(JsonRecoveryDecoder::check_record_size(MAX_RECORD_BYTES + 1).is_err());
    }

    #[test]
    fn raw_records_support_zero_column_projection() {
        let schema = Arc::new(Schema::empty());
        let mut decoder =
            JsonRecoveryDecoder::new_raw(schema, None, 2, RawJsonOptions::default())
                .unwrap();
        let input = b"{\"id\":1}\n{\"id\":2}\n";
        assert_eq!(decoder.decode(input).unwrap(), input.len());
        let batch = decoder.flush().unwrap().unwrap();
        assert_eq!(batch.num_columns(), 0);
        assert_eq!(batch.num_rows(), 2);
    }

    #[test]
    fn raw_records_flush_before_the_diagnostic_buffer_can_fill() {
        let schema =
            Arc::new(Schema::new(vec![Field::new("raw", DataType::Utf8, false)]));
        let mut decoder =
            JsonRecoveryDecoder::new_raw(schema, None, 8192, RawJsonOptions::default())
                .unwrap();
        decoder.batch_bytes = MAX_BATCH_BYTES - 2;
        assert_eq!(decoder.decode(b"x\n").unwrap(), 2);
        assert!(decoder.can_flush_early());
        assert_eq!(decoder.flush().unwrap().unwrap().num_rows(), 1);
        assert_eq!(decoder.batch_bytes, 0);
    }

    #[test]
    fn raw_records_strip_initial_bom_and_replace_invalid_utf8_when_requested() {
        let schema =
            Arc::new(Schema::new(vec![Field::new("raw", DataType::Utf8, false)]));
        let handler = Arc::new(CollectRecords::default());
        let mut decoder = JsonRecoveryDecoder::new_raw(
            schema,
            Some(handler.clone()),
            8,
            RawJsonOptions {
                skip_bom: true,
                replace_invalid_utf8: true,
            },
        )
        .unwrap();
        let input = b"\xef\xbb\xbf{\"x\":\"\xff\"}\n";
        assert_eq!(decoder.decode(input).unwrap(), input.len());
        let batch = decoder.flush().unwrap().unwrap();
        let values = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(values.value(0), "{\"x\":\"\u{fffd}\"}\n");
        assert_eq!(*handler.0.lock().unwrap(), vec![(1, 3, 12, false)]);
    }

    #[test]
    fn oversized_record_is_skipped_without_losing_the_next_line() {
        let schema =
            Arc::new(Schema::new(vec![Field::new("raw", DataType::Utf8, false)]));
        let handler = Arc::new(CollectRecords::default());
        let mut decoder = JsonRecoveryDecoder::new_raw(
            schema,
            Some(handler.clone()),
            8,
            RawJsonOptions::default(),
        )
        .unwrap();
        decoder.pending.resize(MAX_RECORD_BYTES, b'x');
        decoder.byte_offset = MAX_RECORD_BYTES;
        assert_eq!(decoder.decode(b"x\n{\"id\":2}\n").unwrap(), 11);
        let batch = decoder.flush().unwrap().unwrap();
        assert_eq!(batch.num_rows(), 1);
        assert_eq!(
            *handler.0.lock().unwrap(),
            vec![
                (1, 0, MAX_RECORD_BYTES + 1, true),
                (2, MAX_RECORD_BYTES + 2, MAX_RECORD_BYTES + 10, false),
            ]
        );
    }

    #[test]
    fn oversized_record_spanning_chunks_does_not_buffer_the_remainder() {
        let schema =
            Arc::new(Schema::new(vec![Field::new("raw", DataType::Utf8, false)]));
        let handler = Arc::new(CollectRecords::default());
        let mut decoder = JsonRecoveryDecoder::new_raw(
            schema,
            Some(handler.clone()),
            8,
            RawJsonOptions::default(),
        )
        .unwrap();
        decoder.pending.resize(MAX_RECORD_BYTES - 1, b'x');
        decoder.byte_offset = MAX_RECORD_BYTES - 1;

        assert_eq!(decoder.decode(b"xx").unwrap(), 2);
        assert!(decoder.pending.is_empty());
        assert_eq!(decoder.pending.capacity(), 0);
        assert_eq!(decoder.decode(b"yyy").unwrap(), 3);
        assert!(decoder.pending.is_empty());
        assert_eq!(decoder.decode(b"\n{\"id\":2}\n").unwrap(), 10);

        let batch = decoder.flush().unwrap().unwrap();
        assert_eq!(batch.num_rows(), 1);
        assert_eq!(
            *handler.0.lock().unwrap(),
            vec![
                (1, 0, MAX_RECORD_BYTES + 4, true),
                (2, MAX_RECORD_BYTES + 5, MAX_RECORD_BYTES + 13, false),
            ]
        );
    }
}
