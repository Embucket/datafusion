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

use std::any::Any;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use arrow::datatypes::{Schema, SchemaRef, Field, DataType};
use arrow::record_batch::RecordBatch;
use datafusion_common::{internal_err, Result, ScalarValue, DataFusionError};
use datafusion_physical_expr::PhysicalExpr;
use datafusion_execution::TaskContext;
use futures::{Stream, StreamExt};
use crate::{ExecutionPlan, RecordBatchStream, SendableRecordBatchStream, DisplayFormatType};
use crate::execution_plan::ExecutionPlanProperties;
use crate::metrics::{BaselineMetrics, ExecutionPlanMetricsSet};
use super::{DisplayAs, PlanProperties};
use std::collections::HashMap;
use datafusion_expr::PivotConfig;
use arrow::array::{Array, ArrayRef, StringArray, Float64Array, Int64Array, BooleanArray, UInt64Array, NullArray};

/// Function to create an Arrow array from ScalarValue items
fn cast_scalar_value_to_array(values: &[ScalarValue], len: usize) -> Result<ArrayRef> {
    // Check for empty array
    if values.is_empty() {
        return Ok(Arc::new(NullArray::new(len)));
    }
    
    // Use the first non-null value's type
    let data_type = values.iter()
        .find(|v| !v.is_null())
        .map(|v| v.data_type())
        .unwrap_or(DataType::Null);
    
    match data_type {
        DataType::Float64 => {
            let array = values.iter()
                .map(|v| match v {
                    ScalarValue::Float64(Some(f)) => Ok(Some(*f)),
                    ScalarValue::Float64(None) => Ok(None),
                    _ => internal_err!("Expected Float64 ScalarValue, got {:?}", v),
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .collect::<Float64Array>();
            Ok(Arc::new(array) as ArrayRef)
        },
        DataType::Int64 => {
            let array = values.iter()
                .map(|v| match v {
                    ScalarValue::Int64(Some(i)) => Ok(Some(*i)),
                    ScalarValue::Int64(None) => Ok(None),
                    _ => internal_err!("Expected Int64 ScalarValue, got {:?}", v),
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .collect::<Int64Array>();
            Ok(Arc::new(array) as ArrayRef)
        },
        DataType::Boolean => {
            let array = values.iter()
                .map(|v| match v {
                    ScalarValue::Boolean(Some(b)) => Ok(Some(*b)),
                    ScalarValue::Boolean(None) => Ok(None),
                    _ => internal_err!("Expected Boolean ScalarValue, got {:?}", v),
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .collect::<BooleanArray>();
            Ok(Arc::new(array) as ArrayRef)
        },
        DataType::UInt64 => {
            let array = values.iter()
                .map(|v| match v {
                    ScalarValue::UInt64(Some(i)) => Ok(Some(*i)),
                    ScalarValue::UInt64(None) => Ok(None),
                    _ => internal_err!("Expected UInt64 ScalarValue, got {:?}", v),
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .collect::<UInt64Array>();
            Ok(Arc::new(array) as ArrayRef)
        },
        DataType::Utf8 => {
            let array = values.iter()
                .map(|v| match v {
                    ScalarValue::Utf8(Some(s)) => Ok(Some(s.as_str())),
                    ScalarValue::Utf8(None) => Ok(None),
                    _ => internal_err!("Expected Utf8 ScalarValue, got {:?}", v),
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .collect::<StringArray>();
            Ok(Arc::new(array) as ArrayRef)
        },
        _ => {
            // Default to converting each value to its array representation
            // and then concatenating them
            let mut arrays = Vec::with_capacity(values.len());
            for value in values {
                arrays.push(value.to_array_of_size(1)?);
            }
            
            // Use arrow compute to concatenate all the tiny arrays
            let final_array = arrow::compute::concat(arrays.iter().map(|a| a.as_ref()).collect::<Vec<_>>().as_slice())?;
            Ok(final_array)
        }
    }
}

/// Physical execution plan for pivot operation
#[derive(Debug, Clone)]
pub struct PivotExec {
    /// The input plan
    input: Arc<dyn ExecutionPlan>,
    /// The pivot column
    pivot_column: Arc<dyn PhysicalExpr>,
    /// Optional known pivot values
    pivot_values: Vec<ScalarValue>,
    /// Optional placeholder mapping
    pivot_config: Option<PivotConfig>,
    /// Output schema
    schema: SchemaRef,
    /// Plan properties cache
    cache: PlanProperties,
    /// Execution metrics
    metrics: ExecutionPlanMetricsSet,
}

impl PivotExec {
    /// Create a new PivotExec
    pub fn try_new(
        input: Arc<dyn ExecutionPlan>,
        pivot_column: Arc<dyn PhysicalExpr>,
        pivot_values: Vec<ScalarValue>,
        pivot_config: Option<PivotConfig>,
        schema: SchemaRef,
    ) -> Self {
        // Compute plan properties
        let cache = Self::compute_properties(&input, schema.clone()).unwrap();
        
        Self {
            input,
            pivot_column,
            pivot_values,
            pivot_config,
            schema,
            cache,
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }
    
    fn compute_properties(
        input: &Arc<dyn ExecutionPlan>,
        _schema: SchemaRef,
    ) -> Result<PlanProperties> {
        // Create plan properties based on input
        let input_eq_properties = input.equivalence_properties().clone();
        
        Ok(PlanProperties::new(
            input_eq_properties,
            input.output_partitioning().clone(),
            input.pipeline_behavior(),
            input.boundedness(),
        ))
    }
    
    /// Create output schema based on pivot values
    fn create_output_schema(
        &self,
        pivot_values: &[ScalarValue]
    ) -> Result<SchemaRef> {
        // Get the original schema
        let orig_schema = self.schema.clone();
        let fields = orig_schema.fields();
        
        // Find the placeholder field
        let placeholder_field = fields.iter().find(|f| {
            if let Some(config) = &self.pivot_config {
                f.name().starts_with(&config.placeholder_name)
            } else {
                f.name().starts_with("__pivot_placeholder")
            }
        });
        
        // If there's no placeholder, just return the original schema
        if placeholder_field.is_none() {
            return Ok(orig_schema);
        }
        
        let placeholder_field = placeholder_field.unwrap();
        let placeholder_idx = fields.iter().position(|f| f.name() == placeholder_field.name()).unwrap();
        
        // Create new fields list with placeholder replaced by pivot value columns
        let mut new_fields = Vec::new();
        
        // Add fields before placeholder
        for i in 0..placeholder_idx {
            new_fields.push(fields[i].clone());
        }
        
        // Get data type for pivot columns
        let data_type = if let Some(config) = &self.pivot_config {
            match config.column_type.as_str() {
                "Float64" => DataType::Float64,
                "Int64" => DataType::Int64,
                "UInt64" => DataType::UInt64,
                "Boolean" => DataType::Boolean,
                "Utf8" => DataType::Utf8,
                _ => DataType::Float64, // Default to Float64 for most common aggregate types
            }
        } else {
            DataType::Float64 // Default to Float64
        };
        
        // Add fields for each pivot value
        for value in pivot_values {
            let column_name = if let Some(config) = &self.pivot_config {
                format!("{}_{}", config.agg_expr_name, value)
            } else {
                format!("pivot_{}", value)
            };
            
            new_fields.push(Arc::new(Field::new(&column_name, data_type.clone(), true)));
        }
        
        // Add fields after placeholder
        for i in placeholder_idx + 1..fields.len() {
            new_fields.push(fields[i].clone());
        }
        
        // Create new schema
        Ok(Arc::new(Schema::new(new_fields)))
    }
}

impl DisplayAs for PivotExec {
    fn fmt_as(
        &self,
        t: DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(f, "PivotExec: pivot_column={}, pivot_values={}",
                        self.pivot_column, self.pivot_values.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", "))
            }
            DisplayFormatType::TreeRender => {
                writeln!(f, "PivotExec:")?;
                writeln!(f, "  pivot_column: {}", self.pivot_column)?;
                writeln!(f, "  pivot_values: {}", 
                         self.pivot_values.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", "))?;
                Ok(())
            }
        }
    }
}

impl ExecutionPlan for PivotExec {
    fn name(&self) -> &'static str {
        "PivotExec"
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
    
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
    
    fn properties(&self) -> &PlanProperties {
        &self.cache
    }
    
    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }
    
    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        match children.len() {
            1 => Ok(Arc::new(Self {
                input: children[0].clone(),
                ..(*self).clone()
            })),
            _ => internal_err!("PivotExec expects 1 child"),
        }
    }
    
    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        // Use a simpler implementation - we always have pivot values now
        let stream = self.input.execute(partition, context.clone())?;
        
        // Create output schema based on known pivot values
        let output_schema = self.create_output_schema(&self.pivot_values)?;
        
        // Create a map from pivot value to column index
        let mut pivot_index_map = HashMap::new();
        for (i, value) in self.pivot_values.iter().enumerate() {
            pivot_index_map.insert(value.clone(), i);
        }
        
        Ok(Box::pin(PivotStream {
            schema: output_schema.clone(),
            state: PivotStreamState::Processing {
                stream,
                pivot_values: self.pivot_values.clone(),
                pivot_index_map,
                output_schema,
            },
            pivot_exec: self.clone(),
            partition,
            context,
            baseline_metrics: BaselineMetrics::new(&self.metrics, partition),
        }))
    }
}

/// Stream that implements pivot functionality
struct PivotStream {
    // Stream state fields
    schema: SchemaRef,
    state: PivotStreamState,
    pivot_exec: PivotExec,
    partition: usize,
    context: Arc<TaskContext>,
    baseline_metrics: BaselineMetrics,
}

/// States for the pivot stream execution
enum PivotStreamState {
    /// Processing state - transforming input based on pivot values
    Processing {
        stream: SendableRecordBatchStream,
        pivot_values: Vec<ScalarValue>,
        pivot_index_map: HashMap<ScalarValue, usize>,
        output_schema: SchemaRef,
    },
    /// Final state - finished processing
    Done,
}

impl PivotStream {
    fn new(
        pivot_exec: PivotExec,
        partition: usize,
        context: Arc<TaskContext>,
        _pivot_values: Option<Vec<ScalarValue>>,
    ) -> Result<Self> {
        let _schema = pivot_exec.schema.clone();
        
        // Always use the pivot values from PivotExec
        let pivot_values = pivot_exec.pivot_values.clone();
        
        // Create a map from pivot value to column index
        let mut pivot_index_map = HashMap::new();
        for (i, value) in pivot_values.iter().enumerate() {
            pivot_index_map.insert(value.clone(), i);
        }
        
        // Create the output schema based on pivot values
        let output_schema = pivot_exec.create_output_schema(&pivot_values)?;
        
        // Get input stream for processing
        let stream = pivot_exec.input.execute(partition, context.clone())?;
        
        // Store metrics reference before moving pivot_exec
        let metrics = &pivot_exec.metrics;
        
        Ok(Self {
            schema: output_schema.clone(),
            state: PivotStreamState::Processing {
                stream,
                pivot_values,
                pivot_index_map,
                output_schema,
            },
            pivot_exec: pivot_exec.clone(),
            partition,
            context,
            baseline_metrics: BaselineMetrics::new(metrics, partition),
        })
    }
    
    /// Transform a batch based on pivot values
    fn transform_batch(
        &self,
        batch: &RecordBatch,
        pivot_values: &[ScalarValue],
        pivot_index_map: &HashMap<ScalarValue, usize>,
        output_schema: &SchemaRef,
    ) -> Result<RecordBatch> {
        // Find the placeholder column index
        let input_schema = batch.schema();
        let placeholder_idx = if let Some(config) = &self.pivot_exec.pivot_config {
            input_schema.fields().iter().position(|f| 
                f.name().starts_with(&config.placeholder_name)
            ).ok_or_else(|| DataFusionError::Internal(
                format!("Placeholder field not found: {}", config.placeholder_name)
            ))?
        } else {
            input_schema.fields().iter().position(|f| 
                f.name().starts_with("__pivot_placeholder")
            ).ok_or_else(|| DataFusionError::Internal(
                "Placeholder field not found".to_string()
            ))?
        };
        
        // Evaluate the pivot column expression to get values
        let column_values = self.pivot_exec.pivot_column.evaluate(batch)?;
        let pivot_column_array = column_values.into_array(batch.num_rows())?;
        
        // Get the aggregate values from the placeholder column
        let agg_values = batch.column(placeholder_idx);
        
        // Prepare new arrays for output
        let mut output_arrays = Vec::with_capacity(output_schema.fields().len());
        
        // Add columns before pivot columns
        for i in 0..placeholder_idx {
            output_arrays.push(batch.column(i).clone());
        }
        
        // Create arrays for each pivot value
        let row_count = batch.num_rows();
        
        // Determine the data type of the output columns
        match agg_values.data_type() {
            DataType::Float64 => {
                self.create_pivot_arrays::<Float64Array>(
                    row_count,
                    pivot_values.len(),
                    &pivot_column_array,
                    agg_values,
                    pivot_index_map,
                    &mut output_arrays,
                )?;
            },
            DataType::Int64 => {
                self.create_pivot_arrays::<Int64Array>(
                    row_count,
                    pivot_values.len(),
                    &pivot_column_array,
                    agg_values,
                    pivot_index_map,
                    &mut output_arrays,
                )?;
            },
            DataType::Boolean => {
                self.create_pivot_arrays::<BooleanArray>(
                    row_count,
                    pivot_values.len(),
                    &pivot_column_array,
                    agg_values,
                    pivot_index_map,
                    &mut output_arrays,
                )?;
            },
            DataType::UInt64 => {
                self.create_pivot_arrays::<UInt64Array>(
                    row_count,
                    pivot_values.len(),
                    &pivot_column_array,
                    agg_values,
                    pivot_index_map,
                    &mut output_arrays,
                )?;
            },
            DataType::Utf8 => {
                self.create_pivot_arrays::<StringArray>(
                    row_count,
                    pivot_values.len(),
                    &pivot_column_array,
                    agg_values,
                    pivot_index_map,
                    &mut output_arrays,
                )?;
            },
            _ => {
                // Default to Float64
                self.create_pivot_arrays::<Float64Array>(
                    row_count,
                    pivot_values.len(),
                    &pivot_column_array,
                    agg_values,
                    pivot_index_map,
                    &mut output_arrays,
                )?;
            }
        }
        
        // Add columns after pivot columns
        for i in placeholder_idx + 1..input_schema.fields().len() {
            output_arrays.push(batch.column(i).clone());
        }
        
        // Create new record batch
        RecordBatch::try_new(output_schema.clone(), output_arrays).map_err(Into::into)
    }
    
    /// Helper method to create pivot arrays of specific type
    fn create_pivot_arrays<T: Array + 'static>(
        &self,
        row_count: usize,
        pivot_count: usize,
        pivot_column_array: &ArrayRef,
        agg_values: &ArrayRef,
        pivot_index_map: &HashMap<ScalarValue, usize>,
        output_arrays: &mut Vec<ArrayRef>,
    ) -> Result<()> {
        // Create a vector of arrays, one for each pivot value
        let mut value_arrays: Vec<Vec<ScalarValue>> = vec![Vec::with_capacity(row_count); pivot_count];
        
        // Fill value arrays based on pivot column values
        for row in 0..row_count {
            if pivot_column_array.is_null(row) {
                continue;
            }
            
            // Get the pivot value for this row
            let pivot_value = ScalarValue::try_from_array(pivot_column_array, row)?;
            
            // Find which pivot column this value belongs to
            if let Some(&pivot_idx) = pivot_index_map.get(&pivot_value) {
                // Check if there's a value for this pivot in this row
                if !agg_values.is_null(row) {
                    // Get value from the aggregate column
                    let value = ScalarValue::try_from_array(agg_values, row)?;
                    
                    // Extend all arrays up to the current row if needed
                    for vals in &mut value_arrays {
                        while vals.len() < row {
                            vals.push(ScalarValue::try_from(agg_values.data_type())?);
                        }
                    }
                    
                    // Add the value to the appropriate pivot column
                    value_arrays[pivot_idx].push(value);
                } else {
                    // No value for this pivot in this row, add null
                    if !value_arrays[pivot_idx].is_empty() {
                        value_arrays[pivot_idx].push(ScalarValue::try_from(agg_values.data_type())?);
                    }
                }
            }
        }
        
        // Convert value arrays to Arrow arrays
        for values in value_arrays {
            if values.is_empty() {
                // Empty array, create a null array of the right type and length
                output_arrays.push(Arc::new(NullArray::new(row_count)));
            } else {
                // Ensure the array has the right length
                let mut padded_values = values;
                while padded_values.len() < row_count {
                    padded_values.push(ScalarValue::try_from(agg_values.data_type())?);
                }
                
                // Convert to Arrow array
                let array = cast_scalar_value_to_array(
                    &padded_values,
                    row_count,
                )?;
                output_arrays.push(array);
            }
        }
        
        Ok(())
    }

    /// Separate implementation function that unpins the `PivotStream` so
    /// that partial borrows work correctly
    fn poll_next_impl(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<RecordBatch>>> {
        // Handle different states
        match &mut self.state {
            PivotStreamState::Processing { 
                stream, 
                pivot_values, 
                pivot_index_map, 
                output_schema 
            } => {
                // Poll for the next batch and transform it
                match self.baseline_metrics.record_poll(stream.poll_next_unpin(cx)) {
                    Poll::Ready(Some(Ok(batch))) => {
                        // Transform the batch
                        let pivot_values = pivot_values.clone();
                        let pivot_index_map = pivot_index_map.clone();
                        let output_schema = output_schema.clone();
                        match self.transform_batch(
                            &batch, 
                            &pivot_values, 
                            &pivot_index_map, 
                            &output_schema
                        ) {
                            Ok(transformed) => Poll::Ready(Some(Ok(transformed))),
                            Err(e) => Poll::Ready(Some(Err(e))),
                        }
                    },
                    Poll::Ready(Some(Err(e))) => Poll::Ready(Some(Err(e))),
                    Poll::Ready(None) => {
                        // Done processing, switch to Done state
                        self.state = PivotStreamState::Done;
                        Poll::Ready(None)
                    },
                    Poll::Pending => Poll::Pending,
                }
            },
            PivotStreamState::Done => Poll::Ready(None),
        }
    }
}

impl Stream for PivotStream {
    type Item = Result<RecordBatch>;
    
    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        self.poll_next_impl(cx)
    }
}

impl RecordBatchStream for PivotStream {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}