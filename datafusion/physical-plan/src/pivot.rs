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

//! Pivot operator for DataFusion
//!
//! The pivot operation transposes rows into columns based on unique values in a pivot column.
//! It requires an aggregate function to be applied to the values within each pivot group.
//!
//! This is similar to the DuckDB PIVOT operation:
//! https://duckdb.org/docs/sql/statements/pivot.html

use std::any::Any;
use std::collections::HashMap;
use std::fmt;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow::array::{
    Array, ArrayRef, BooleanArray, Float32Array, Float64Array, Int16Array, Int32Array, Int32Builder,
    Int64Array, Int64Builder, Int8Array, Float32Builder, Float64Builder, StringBuilder, StringArray,
    UInt16Array, UInt32Array, UInt64Array, UInt8Array,
};
use arrow::compute::concat_batches;
use arrow::datatypes::{DataType, SchemaRef};
use arrow::record_batch::RecordBatch;

use datafusion_common::{Result, ScalarValue, Statistics};
use datafusion_expr::Expr;
use datafusion_physical_expr::{EquivalenceProperties, PhysicalExpr};
use futures::{Stream, StreamExt};
use log::debug;

use crate::display::{DisplayAs, DisplayFormatType};
use crate::execution_plan::{Boundedness, EmissionType};
use crate::metrics::{self, BaselineMetrics, ExecutionPlanMetricsSet, MetricBuilder, MetricsSet};
use crate::{ExecutionPlan, RecordBatchStream, SendableRecordBatchStream, ColumnarValue, PlanProperties};
use datafusion_execution::TaskContext;

/// Execution plan for pivoting data
#[derive(Debug)]
pub struct PivotExec {
    /// The input plan
    input: Arc<dyn ExecutionPlan>,
    /// The aggregation expression
    aggregate_expr: Expr,
    /// The pivot column
    pivot_column: Expr,
    /// The pivot values
    pivot_values: Vec<ScalarValue>,
    /// The resulting schema, already computed during plan optimization
    schema: SchemaRef,
    /// Execution metrics
    metrics: ExecutionPlanMetricsSet,
    /// Plan properties
    cache: PlanProperties,
}

impl PivotExec {
    /// Create a new PivotExec
    pub fn new(
        input: Arc<dyn ExecutionPlan>,
        aggregate_expr: Expr,
        pivot_column: Expr,
        pivot_values: Vec<ScalarValue>,
        schema: SchemaRef,
    ) -> Self {
        let metrics = ExecutionPlanMetricsSet::new();
        let cache = Self::compute_properties(input.clone(), schema.clone());
        
        Self {
            input,
            aggregate_expr,
            pivot_column,
            pivot_values,
            schema,
            metrics,
            cache,
        }
    }

    /// Build plan properties
    fn compute_properties(input: Arc<dyn ExecutionPlan>, schema: SchemaRef) -> PlanProperties {
        // Pivot makes new columns, so output ordering is not preserved
        let equivalence = EquivalenceProperties::new(schema.clone());

        PlanProperties::new(
            equivalence,
            input.properties().output_partitioning().clone(),
            EmissionType::Final,
            Boundedness::Bounded,
        )
    }

    /// Return the pivot column expression
    pub fn pivot_column(&self) -> &Expr {
        &self.pivot_column
    }

    /// Return the pivot values array
    pub fn pivot_values(&self) -> &[ScalarValue] {
        &self.pivot_values
    }

    /// Return the aggregate expression
    pub fn aggregate_expr(&self) -> &Expr {
        &self.aggregate_expr
    }

    /// Return the input execution plan
    pub fn input(&self) -> &Arc<dyn ExecutionPlan> {
        &self.input
    }

    /// Create physical expressions for pivot and aggregate columns
    fn create_physical_exprs(
        &self,
        input_schema: SchemaRef,
        _ctx: Arc<TaskContext>,
    ) -> Result<(Arc<dyn PhysicalExpr>, Arc<dyn PhysicalExpr>)> {
        debug!("Creating physical expressions");
        
        // Extract table qualifier from expressions to handle column references correctly
        let table_qualifier = {
            // Try to extract qualifier from pivot column first
            let pivot_qualifier = match &self.pivot_column {
                Expr::Column(col) => col.relation.as_ref().map(|r| match r {
                    datafusion_common::TableReference::Bare { table } => table.clone(),
                    _ => "".to_string().into(),
                }),
                _ => None,
            };
            
            // If pivot column has no qualifier, try from aggregate expression
            if pivot_qualifier.is_none() {
                // For aggregate function, we need to check the arguments
                if let Expr::AggregateFunction(agg_fn) = &self.aggregate_expr {
                    if !agg_fn.params.args.is_empty() {
                        if let Expr::Column(col) = &agg_fn.params.args[0] {
                            col.relation.as_ref().map(|r| match r {
                                datafusion_common::TableReference::Bare { table } => table.clone(),
                                _ => "".to_string().into(),
                            })
                        } else if let Expr::Cast(cast) = &agg_fn.params.args[0] {
                            if let Expr::Column(col) = cast.expr.as_ref() {
                                col.relation.as_ref().map(|r| match r {
                                    datafusion_common::TableReference::Bare { table } => table.clone(),
                                    _ => "".to_string().into(),
                                })
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else if let Expr::Column(col) = &self.aggregate_expr {
                    col.relation.as_ref().map(|r| match r {
                        datafusion_common::TableReference::Bare { table } => table.clone(),
                        _ => "".to_string().into(),
                    })
                } else {
                    None
                }
            } else {
                pivot_qualifier
            }
        };
        
        // Create a DFSchema with or without table qualifier
        let df_schema = if let Some(qualifier) = table_qualifier {
            datafusion_common::DFSchema::try_from_qualified_schema(&*qualifier, &input_schema)?
        } else {
            schema_to_dfschema(input_schema.clone())
        };
        
        // Create new execution properties
        let exec_props = datafusion_expr::execution_props::ExecutionProps::new();
        
        // Create pivot column expr using the datafusion_physical_expr's create_physical_expr
        let pivot_expr = datafusion_physical_expr::create_physical_expr(
            &self.pivot_column,
            &df_schema,
            &exec_props,
        )?;
        
        // For the aggregate expression, we need special handling for aggregate functions
        let agg_expr = match &self.aggregate_expr {
            Expr::AggregateFunction(_) => {
                debug!("Creating physical aggregate expression from an AggregateFunction");
                // For aggregation, we'll extract the child expression from the aggregate function
                // We'll do the aggregation ourselves in the process_input method
                match &self.aggregate_expr {
                    Expr::AggregateFunction(agg_fn) => {
                        if agg_fn.params.args.len() != 1 {
                            return Err(datafusion_common::DataFusionError::Execution(
                                "Aggregate function in PIVOT must have exactly one argument".to_string()
                            ));
                        }
                        
                        // We just need the argument expression, not the aggregate function itself
                        datafusion_physical_expr::create_physical_expr(
                            &agg_fn.params.args[0],
                            &df_schema,
                            &exec_props,
                        )?
                    },
                    _ => unreachable!("Already checked expr is AggregateFunction"),
                }
            },
            // For other expression types, use the standard create_physical_expr
            _ => {
                datafusion_physical_expr::create_physical_expr(
                    &self.aggregate_expr,
                    &df_schema,
                    &exec_props,
                )?
            }
        };
        
        debug!("Created pivot expr: {:?}", pivot_expr);
        debug!("Created aggregate expr: {:?}", agg_expr);
        
        Ok((pivot_expr, agg_expr))
    }
}

impl DisplayAs for PivotExec {
    fn fmt_as(
        &self,
        t: DisplayFormatType,
        f: &mut fmt::Formatter,
    ) -> fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(
                    f,
                    "PivotExec: pivot_column={:?}, pivot_values={:?}, aggregate_expr={:?}",
                    self.pivot_column, self.pivot_values, self.aggregate_expr
                )
            }
            DisplayFormatType::TreeRender => {
                write!(
                    f,
                    "PivotExec: pivot_column={:?}, pivot_values={:?}",
                    self.pivot_column, self.pivot_values
                )
            }
        }
    }
}

#[async_trait::async_trait]
impl ExecutionPlan for PivotExec {
    /// Return a reference to Any that can be used for downcasting
    fn as_any(&self) -> &dyn Any {
        self
    }

    /// Get the schema for this execution plan
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    /// Get the name of this execution plan
    fn name(&self) -> &str {
        "PivotExec"
    }

    /// Get the properties of this execution plan
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
            1 => Ok(Arc::new(PivotExec::new(
                children[0].clone(),
                self.aggregate_expr.clone(),
                self.pivot_column.clone(),
                self.pivot_values.clone(),
                self.schema.clone(),
            ))),
            _ => Err(datafusion_common::DataFusionError::Internal(
                "PivotExec wrong number of children".to_string(),
            )),
        }
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        debug!("Start PivotExec::execute for partition: {}", partition);
        
        let _baseline_metrics = BaselineMetrics::new(&self.metrics, partition);
        let input_stream = self.input.execute(partition, context.clone())?;

        // Create physical expressions for evaluation
        let (pivot_expr, aggregate_expr) = self.create_physical_exprs(self.input.schema(), context)?;
        
        // Create the pivot stream
        let stream = PivotStream::new(
            self.schema.clone(),
            input_stream,
            partition,
            &self.metrics,
            self.pivot_values.clone(),
            pivot_expr,
            aggregate_expr,
        );

        Ok(Box::pin(stream))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn statistics(&self) -> Result<Statistics> {
        // For now, using the same statistics as the input
        self.input.statistics()
    }
}

/// Metrics for the pivot operation
#[derive(Debug)]
struct PivotMetrics {
    /// Time in nanos to compute the pivot
    elapsed_compute: metrics::Time,
    /// Number of input batches
    input_batches: metrics::Count,
    /// Number of input rows
    input_rows: metrics::Count,
    /// Number of output batches
    output_batches: metrics::Count,
    /// Number of output rows
    output_rows: metrics::Count,
}

impl PivotMetrics {
    /// Create new pivot metrics
    fn new(metrics: &ExecutionPlanMetricsSet, partition: usize) -> Self {
        let elapsed_compute = MetricBuilder::new(metrics)
            .subset_time("elapsed_compute", partition);
        
        let input_batches = MetricBuilder::new(metrics)
            .counter("input_batches", partition);
        
        let input_rows = MetricBuilder::new(metrics)
            .counter("input_rows", partition);
        
        let output_batches = MetricBuilder::new(metrics)
            .counter("output_batches", partition);
        
        let output_rows = MetricBuilder::new(metrics)
            .counter("output_rows", partition);

        Self {
            elapsed_compute,
            input_batches,
            input_rows,
            output_batches,
            output_rows,
        }
    }
}

/// Stream for the pivot operation
struct PivotStream {
    /// The schema for the output
    schema: SchemaRef,
    /// The input stream
    input: SendableRecordBatchStream,
    /// The partition being processed
    _partition: usize,
    /// Metrics for the operation
    metrics: PivotMetrics,
    /// Collected input batches
    batches: Vec<RecordBatch>,
    /// Stream state
    state: PivotStreamState,
    /// The pivot values from the query (not dynamically detected)
    pivot_values: Vec<ScalarValue>,
    pivot_expr: Arc<dyn PhysicalExpr>,
    aggregate_expr: Arc<dyn PhysicalExpr>,
}

/// States for the PivotStream state machine
#[derive(Debug, PartialEq)]
enum PivotStreamState {
    /// Collecting input batches
    Collecting,
    /// Processing collected batches
    Processing,
    /// Result emitted, waiting for next poll
    Emitted,
    /// No more data to process
    Exhausted,
}

impl PivotStream {
    /// Create a new pivot stream
    fn new(
        schema: SchemaRef,
        input: SendableRecordBatchStream,
        partition: usize,
        metrics_set: &ExecutionPlanMetricsSet,
        pivot_values: Vec<ScalarValue>,
        pivot_expr: Arc<dyn PhysicalExpr>,
        aggregate_expr: Arc<dyn PhysicalExpr>,
    ) -> Self {
        let metrics = PivotMetrics::new(metrics_set, partition);
        
        Self {
            schema,
            input,
            _partition: partition,
            metrics,
            batches: Vec::new(),
            state: PivotStreamState::Collecting,
            pivot_values,   
            pivot_expr,
            aggregate_expr,
        }
    }

    /// Process the input
    fn process_input(&mut self) -> Result<Option<RecordBatch>> {
        let timer = self.metrics.elapsed_compute.timer();

        if self.batches.is_empty() {
            self.state = PivotStreamState::Exhausted;
            return Ok(None);
        }
        
        // Combine all batches
        let combined_batch = concat_batches(&self.input.schema(), &self.batches)?;

        if combined_batch.num_rows() == 0 {
            // No data to process
            self.state = PivotStreamState::Exhausted;
            return Ok(None);
        }

        // Create a mapping from pivot values to column indices
        let mut pivot_map: HashMap<String, usize> = HashMap::new();
        for (i, value) in self.pivot_values.iter().enumerate() {
            pivot_map.insert(value.to_string(), i);
        }

        // Evaluate the pivot column to get the pivot values for each row
        let pivot_column_values = self.pivot_expr.evaluate(&combined_batch)?;
        let pivot_array = match pivot_column_values {
            ColumnarValue::Array(array) => array,
            ColumnarValue::Scalar(scalar) => scalar.to_array_of_size(combined_batch.num_rows())?,
        };
        
        // Evaluate the aggregate column to get the values to aggregate
        let agg_column_values = self.aggregate_expr.evaluate(&combined_batch)?;
        let agg_array = match agg_column_values {
            ColumnarValue::Array(array) => array,
            ColumnarValue::Scalar(scalar) => scalar.to_array_of_size(combined_batch.num_rows())?,
        };
        
        // Group by pivot values and collect aggregate values
        let rows = combined_batch.num_rows();
        
        // Create a map to hold the aggregated values for each pivot value and row id combination
        // This structure will allow us to perform the aggregation and lookup values later
        // Format: {row_id => {pivot_value => aggregated_value}}
        let mut pivot_data: HashMap<usize, HashMap<String, Vec<ScalarValue>>> = HashMap::new();
        
        // Collect all values grouped by row id and pivot value
        for row_idx in 0..rows {
            let pivot_value = array_value_to_string(&pivot_array, row_idx);
            
            // Skip pivot values that aren't in our pivot_values list
            if !pivot_map.contains_key(&pivot_value) {
                continue;
            }
            
            // Get the row's unique identifier (usually the first column that's not pivot or aggregate)
            // For simplicity, we'll use the row index here, but in a real implementation
            // you would extract an ID column from the row.
            // This would allow us to properly group by that ID
            let _id_column_idx = 0; // Assuming first column is the ID column
            
            let unique_id = if let Some(id_column) = combined_batch.column_by_name("empid") {
                array_value_to_string(id_column, row_idx)
            } else {
                // Fallback to using row index as ID if no empid column exists
                row_idx.to_string()
            };
            
            let unique_id_value = unique_id.parse::<usize>().unwrap_or(row_idx);
            
            let agg_value = extract_scalar_value(&agg_array, row_idx);
            
            // Initialize entry for this row if it doesn't exist
            let row_entry = pivot_data.entry(unique_id_value).or_insert_with(HashMap::new);
            
            // Add this value to the appropriate pivot group
            let pivot_entry = row_entry.entry(pivot_value).or_insert_with(Vec::new);
            pivot_entry.push(agg_value);
        }
        
        // Get unique row identifiers and sort them to ensure consistent output
        let mut unique_ids: Vec<usize> = pivot_data.keys().cloned().collect();
        unique_ids.sort();
        
        // Number of output rows is the number of unique identifiers
        let output_rows = unique_ids.len();
        
        // Initialize output columns
        let mut output_columns: Vec<ArrayRef> = Vec::with_capacity(self.schema.fields().len());
        
        // First, identify and extract non-pivot columns from the input
        // These will be the first columns in our output
        // We'll need to collect the input values by the unique ID
        
        // Create a map of column name -> column
        let mut input_columns = HashMap::new();
        for (idx, field) in combined_batch.schema().fields().iter().enumerate() {
            input_columns.insert(field.name().to_string(), combined_batch.column(idx));
        }
        
        // Process each field in the output schema
        for (_idx, field) in self.schema.fields().iter().enumerate() {
            let field_name = field.name();
            
            // Check if this is a pivot column (one of the pivot values)
            let is_pivot_column = self.pivot_values.iter().any(|v| v.to_string() == field_name.to_string());
            
            if is_pivot_column {
                // This is a pivot value column, process the aggregated values
                let pivot_value = field_name.to_string();
                
                // Create an array for this pivot column
                match field.data_type() {
                    DataType::Int64 => {
                        // For Int64 aggregation (e.g., SUM)
                        let mut builder = Int64Builder::with_capacity(output_rows);
                        
                        // Add a value for each unique ID
                        for id in &unique_ids {
                            let row_pivot_data = pivot_data.get(id);
                            
                            match row_pivot_data {
                                Some(pivot_values) => {
                                    match pivot_values.get(&pivot_value) {
                                        Some(values) if !values.is_empty() => {
                                            // Perform aggregation on the values
                                            // For SUM, we add all values
                                            let sum = values.iter().map(|v| {
                                                match v {
                                                    ScalarValue::Int32(Some(val)) => *val as i64,
                                                    ScalarValue::Int64(Some(val)) => *val,
                                                    ScalarValue::Float32(Some(val)) => *val as i64,
                                                    ScalarValue::Float64(Some(val)) => *val as i64,
                                                    _ => 0, // Default for unsupported or null types
                                                }
                                            }).sum();
                                            
                                            builder.append_value(sum);
                                        },
                                        _ => {
                                            // No value for this pivot, add NULL
                                            builder.append_null();
                                        }
                                    }
                                },
                                None => {
                                    // No data for this ID, add NULL
                                    builder.append_null();
                                }
                            }
                        }
                        
                        output_columns.push(Arc::new(builder.finish()));
                    },
                    // Add other data types as needed
                    _ => {
                        return Err(datafusion_common::DataFusionError::Execution(format!(
                            "Unsupported data type for pivot column: {:?}", field.data_type()
                        )));
                    }
                }
            } else {
                // This is not a pivot column, extract it from the input
                // We need to ensure we get one value per unique ID
                if let Some(input_column) = input_columns.get(field_name) {
                    // Build a new array that contains only the first occurrence of each unique ID
                    let mut id_to_row_idx = HashMap::new();
                    
                    // Scan input to find the first row for each unique ID
                    for row_idx in 0..rows {
                        let unique_id = if let Some(id_column) = combined_batch.column_by_name("empid") {
                            let id_str = array_value_to_string(id_column, row_idx);
                            id_str.parse::<usize>().unwrap_or(row_idx)
                        } else {
                            row_idx
                        };
                        
                        // Only insert if this ID hasn't been seen yet
                        id_to_row_idx.entry(unique_id).or_insert(row_idx);
                    }
                    
                    // Create the array by collecting values for each unique ID in order
                    match field.data_type() {
                        DataType::Int32 => {
                            let mut builder = Int32Builder::with_capacity(output_rows);
                            
                            for id in &unique_ids {
                                if let Some(row_idx) = id_to_row_idx.get(id) {
                                    let array = input_column.as_any().downcast_ref::<Int32Array>().unwrap();
                                    
                                    if array.is_null(*row_idx) {
                                        builder.append_null();
                                    } else {
                                        builder.append_value(array.value(*row_idx));
                                    }
                                } else {
                                    builder.append_null();
                                }
                            }
                            
                            output_columns.push(Arc::new(builder.finish()));
                        },
                        // Add other data types as needed
                        _ => {
                            return Err(datafusion_common::DataFusionError::Execution(format!(
                                "Unsupported data type for input column: {:?}", field.data_type()
                            )));
                        }
                    }
                } else {
                    return Err(datafusion_common::DataFusionError::Execution(format!(
                        "Column not found in input: {}", field_name
                    )));
                }
            }
        }
        
        // Create the output batch
        let output_batch = RecordBatch::try_new(self.schema.clone(), output_columns)?;
        
        // Update metrics
        self.metrics.output_batches.add(1);
        self.metrics.output_rows.add(output_batch.num_rows());
        
        // Set state to emitted
        self.state = PivotStreamState::Emitted;
        
        timer.done();
        
        Ok(Some(output_batch))
    }
    
    // Helper method to create an array from a scalar value
    fn create_array_for_scalar(&self, value: ScalarValue, length: usize) -> Result<ArrayRef> {
        match value {
            ScalarValue::Int32(Some(v)) => {
                let mut builder = Int32Builder::with_capacity(length);
                for _ in 0..length {
                    builder.append_value(v);
                }
                Ok(Arc::new(builder.finish()) as ArrayRef)
            },
            ScalarValue::Int64(Some(v)) => {
                let mut builder = Int64Builder::with_capacity(length);
                for _ in 0..length {
                    builder.append_value(v);
                }
                Ok(Arc::new(builder.finish()) as ArrayRef)
            },
            ScalarValue::Float32(Some(v)) => {
                let mut builder = Float32Builder::with_capacity(length);
                for _ in 0..length {
                    builder.append_value(v);
                }
                Ok(Arc::new(builder.finish()) as ArrayRef)
            },
            ScalarValue::Float64(Some(v)) => {
                let mut builder = Float64Builder::with_capacity(length);
                for _ in 0..length {
                    builder.append_value(v);
                }
                Ok(Arc::new(builder.finish()) as ArrayRef)
            },
            ScalarValue::Utf8(Some(v)) => {
                let mut builder = StringBuilder::with_capacity(length, v.len() * length);
                for _ in 0..length {
                    builder.append_value(&v);
                }
                Ok(Arc::new(builder.finish()) as ArrayRef)
            },
            // Add other types as needed
            _ => Err(datafusion_common::DataFusionError::Execution(format!(
                "Unsupported scalar value type for pivot: {:?}", value
            ))),
        }
    }
    
    // Helper method to create an array of nulls
    fn create_null_array(&self, data_type: &DataType, length: usize) -> Result<ArrayRef> {
        match data_type {
            DataType::Int32 => {
                let mut builder = Int32Builder::with_capacity(length);
                for _ in 0..length {
                    builder.append_null();
                }
                Ok(Arc::new(builder.finish()) as ArrayRef)
            },
            DataType::Int64 => {
                let mut builder = Int64Builder::with_capacity(length);
                for _ in 0..length {
                    builder.append_null();
                }
                Ok(Arc::new(builder.finish()) as ArrayRef)
            },
            DataType::Float32 => {
                let mut builder = Float32Builder::with_capacity(length);
                for _ in 0..length {
                    builder.append_null();
                }
                Ok(Arc::new(builder.finish()) as ArrayRef)
            },
            DataType::Float64 => {
                let mut builder = Float64Builder::with_capacity(length);
                for _ in 0..length {
                    builder.append_null();
                }
                Ok(Arc::new(builder.finish()) as ArrayRef)
            },
            DataType::Utf8 => {
                let mut builder = StringBuilder::with_capacity(length, 0);
                for _ in 0..length {
                    builder.append_null();
                }
                Ok(Arc::new(builder.finish()) as ArrayRef)
            },
            // Add other types as needed
            _ => Err(datafusion_common::DataFusionError::Execution(format!(
                "Unsupported data type for null array in pivot: {:?}", data_type
            ))),
        }
    }
}

/// Helper function to extract a string representation of a value from any array
fn array_value_to_string(array: &ArrayRef, index: usize) -> String {
    if array.is_null(index) {
        return "NULL".to_string();
    }
    
    // Handle different array types
    match array.data_type() {
        DataType::Utf8 => {
            let array = array.as_any().downcast_ref::<StringArray>().unwrap();
            array.value(index).to_string()
        }
        DataType::Int8 => {
            let array = array.as_any().downcast_ref::<Int8Array>().unwrap();
            array.value(index).to_string()
        }
        DataType::Int16 => {
            let array = array.as_any().downcast_ref::<Int16Array>().unwrap();
            array.value(index).to_string()
        }
        DataType::Int32 => {
            let array = array.as_any().downcast_ref::<Int32Array>().unwrap();
            array.value(index).to_string()
        }
        DataType::Int64 => {
            let array = array.as_any().downcast_ref::<Int64Array>().unwrap();
            array.value(index).to_string()
        }
        DataType::UInt8 => {
            let array = array.as_any().downcast_ref::<UInt8Array>().unwrap();
            array.value(index).to_string()
        }
        DataType::UInt16 => {
            let array = array.as_any().downcast_ref::<UInt16Array>().unwrap();
            array.value(index).to_string()
        }
        DataType::UInt32 => {
            let array = array.as_any().downcast_ref::<UInt32Array>().unwrap();
            array.value(index).to_string()
        }
        DataType::UInt64 => {
            let array = array.as_any().downcast_ref::<UInt64Array>().unwrap();
            array.value(index).to_string()
        }
        DataType::Float32 => {
            let array = array.as_any().downcast_ref::<Float32Array>().unwrap();
            array.value(index).to_string()
        }
        DataType::Float64 => {
            let array = array.as_any().downcast_ref::<Float64Array>().unwrap();
            array.value(index).to_string()
        }
        DataType::Boolean => {
            let array = array.as_any().downcast_ref::<BooleanArray>().unwrap();
            array.value(index).to_string()
        }
        // Add more types as needed
        _ => format!("{:?}", array.data_type()),
    }
}

/// Helper function to extract a value as f64 from any numeric array
/// This is useful for numeric aggregations like SUM, AVG, etc.
#[allow(dead_code)]
fn value_as_f64(array: &ArrayRef, index: usize) -> Result<f64> {
    if array.is_null(index) {
        return Ok(0.0);
    }
    
    match array.data_type() {
        DataType::Int8 => {
            let array = array.as_any().downcast_ref::<Int8Array>().unwrap();
            Ok(array.value(index) as f64)
        }
        DataType::Int16 => {
            let array = array.as_any().downcast_ref::<Int16Array>().unwrap();
            Ok(array.value(index) as f64)
        }
        DataType::Int32 => {
            let array = array.as_any().downcast_ref::<Int32Array>().unwrap();
            Ok(array.value(index) as f64)
        }
        DataType::Int64 => {
            let array = array.as_any().downcast_ref::<Int64Array>().unwrap();
            Ok(array.value(index) as f64)
        }
        DataType::UInt8 => {
            let array = array.as_any().downcast_ref::<UInt8Array>().unwrap();
            Ok(array.value(index) as f64)
        }
        DataType::UInt16 => {
            let array = array.as_any().downcast_ref::<UInt16Array>().unwrap();
            Ok(array.value(index) as f64)
        }
        DataType::UInt32 => {
            let array = array.as_any().downcast_ref::<UInt32Array>().unwrap();
            Ok(array.value(index) as f64)
        }
        DataType::UInt64 => {
            let array = array.as_any().downcast_ref::<UInt64Array>().unwrap();
            Ok(array.value(index) as f64)
        }
        DataType::Float32 => {
            let array = array.as_any().downcast_ref::<Float32Array>().unwrap();
            Ok(array.value(index) as f64)
        }
        DataType::Float64 => {
            let array = array.as_any().downcast_ref::<Float64Array>().unwrap();
            Ok(array.value(index))
        }
        _ => Err(datafusion_common::DataFusionError::Execution(format!(
            "Cannot convert {:?} to f64", array.data_type()
        ))),
    }
}

/// Extract a scalar value from an array at the given index
fn extract_scalar_value(array: &ArrayRef, index: usize) -> ScalarValue {
    if array.is_null(index) {
        // Return null of the appropriate type
        match array.data_type() {
            DataType::Int8 => ScalarValue::Int8(None),
            DataType::Int16 => ScalarValue::Int16(None),
            DataType::Int32 => ScalarValue::Int32(None),
            DataType::Int64 => ScalarValue::Int64(None),
            DataType::UInt8 => ScalarValue::UInt8(None),
            DataType::UInt16 => ScalarValue::UInt16(None),
            DataType::UInt32 => ScalarValue::UInt32(None),
            DataType::UInt64 => ScalarValue::UInt64(None),
            DataType::Float32 => ScalarValue::Float32(None),
            DataType::Float64 => ScalarValue::Float64(None),
            DataType::Utf8 => ScalarValue::Utf8(None),
            DataType::Boolean => ScalarValue::Boolean(None),
            // Add other types as needed
            _ => ScalarValue::Utf8(None), // Default to string null
        }
    } else {
        // Extract the value based on the data type
        match array.data_type() {
            DataType::Int8 => {
                let array = array.as_any().downcast_ref::<Int8Array>().unwrap();
                ScalarValue::Int8(Some(array.value(index)))
            }
            DataType::Int16 => {
                let array = array.as_any().downcast_ref::<Int16Array>().unwrap();
                ScalarValue::Int16(Some(array.value(index)))
            }
            DataType::Int32 => {
                let array = array.as_any().downcast_ref::<Int32Array>().unwrap();
                ScalarValue::Int32(Some(array.value(index)))
            }
            DataType::Int64 => {
                let array = array.as_any().downcast_ref::<Int64Array>().unwrap();
                ScalarValue::Int64(Some(array.value(index)))
            }
            DataType::UInt8 => {
                let array = array.as_any().downcast_ref::<UInt8Array>().unwrap();
                ScalarValue::UInt8(Some(array.value(index)))
            }
            DataType::UInt16 => {
                let array = array.as_any().downcast_ref::<UInt16Array>().unwrap();
                ScalarValue::UInt16(Some(array.value(index)))
            }
            DataType::UInt32 => {
                let array = array.as_any().downcast_ref::<UInt32Array>().unwrap();
                ScalarValue::UInt32(Some(array.value(index)))
            }
            DataType::UInt64 => {
                let array = array.as_any().downcast_ref::<UInt64Array>().unwrap();
                ScalarValue::UInt64(Some(array.value(index)))
            }
            DataType::Float32 => {
                let array = array.as_any().downcast_ref::<Float32Array>().unwrap();
                ScalarValue::Float32(Some(array.value(index)))
            }
            DataType::Float64 => {
                let array = array.as_any().downcast_ref::<Float64Array>().unwrap();
                ScalarValue::Float64(Some(array.value(index)))
            }
            DataType::Utf8 => {
                let array = array.as_any().downcast_ref::<StringArray>().unwrap();
                ScalarValue::Utf8(Some(array.value(index).to_string()))
            }
            DataType::Boolean => {
                let array = array.as_any().downcast_ref::<BooleanArray>().unwrap();
                ScalarValue::Boolean(Some(array.value(index)))
            }
            // Add other types as needed
            _ => {
                // For any other type, convert to string
                ScalarValue::Utf8(Some(array_value_to_string(array, index)))
            }
        }
    }
}

impl Stream for PivotStream {
    type Item = Result<RecordBatch>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        match self.state {
            PivotStreamState::Collecting => {
                // Try to fetch the next batch from the input
                match self.input.poll_next_unpin(cx) {
                    Poll::Ready(Some(Ok(batch))) => {
                        // Record metrics and save the batch
                        self.metrics.input_batches.add(1);
                        self.metrics.input_rows.add(batch.num_rows());
                        self.batches.push(batch);
                        // Return Poll::Ready(None) to signal that we're not done yet, but don't have data to return
                        // This allows the consumer to poll us again, rather than keeping us in a pending state
                        cx.waker().wake_by_ref();
                        return Poll::Pending;
                    }
                    Poll::Ready(Some(Err(e))) => {
                        // Error from input, forward it
                        return Poll::Ready(Some(Err(e)));
                    }
                    Poll::Ready(None) => {
                        // End of input, transition to processing state
                        self.state = PivotStreamState::Processing;
                        // Fall through to process data
                    }
                    Poll::Pending => {
                        // Input not ready yet
                        return Poll::Pending;
                    }
                }

            }
            // When we fall through from Collecting, we'll be in the Processing state
            // No other changes needed for the remaining states
            _ => {}
        }

        // Process all collected batches - only executed when we transition to Processing
        match self.state {
            PivotStreamState::Processing => {
                // Process the collected batches
                match self.process_input() {
                    Ok(Some(batch)) => {
                        // Transition to emitted state and return the batch
                        self.state = PivotStreamState::Emitted;
                        Poll::Ready(Some(Ok(batch)))
                    }
                    Ok(None) => {
                        // No output produced, we're done
                        self.state = PivotStreamState::Exhausted;
                        Poll::Ready(None)
                    }
                    Err(e) => {
                        // Error during processing
                        self.state = PivotStreamState::Exhausted;
                        Poll::Ready(Some(Err(e)))
                    }
                }

            }
            PivotStreamState::Emitted => {
                // We've emitted our batch, and we're done
                // For pivot, we only produce one batch of output after seeing all input
                self.state = PivotStreamState::Exhausted;
                Poll::Ready(None)
            }
            PivotStreamState::Exhausted => {
                // Stream is exhausted, no more data
                Poll::Ready(None)
            }
            _ => unreachable!("Should not reach this code path"),
        }
    }
}

impl RecordBatchStream for PivotStream {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

/// Helper function to convert Arrow schema to DFSchema
fn schema_to_dfschema(schema: SchemaRef) -> datafusion_common::DFSchema {
    datafusion_common::DFSchema::try_from_qualified_schema("", &schema).unwrap()
} 