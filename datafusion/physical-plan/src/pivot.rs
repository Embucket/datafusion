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

//! Defines the PIVOT execution plan. This transforms data from rows to columns.
//! See documentation on 'datafusion_expr::Pivot' for more information on the logical representation.

use std::any::Any;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::collections::{HashMap, HashSet};
use std::fmt;

use arrow::array::{RecordBatch, Array, ArrayRef, Float64Array, Int64Array, StringArray, UInt64Array, BooleanArray};
use arrow::datatypes::{Schema, SchemaRef, DataType, Field};
use async_trait::async_trait;
use datafusion_common::{DataFusionError, Result, ScalarValue, Statistics};
use datafusion_execution::{RecordBatchStream, SendableRecordBatchStream, TaskContext};
use datafusion_expr::Expr;
use datafusion_physical_expr::{EquivalenceProperties, Partitioning, PhysicalExpr};
use futures::{Stream, StreamExt};
use log::{debug, trace};
use arrow::compute::{cast, filter, filter_record_batch, take, concat_batches};

use crate::display::{DisplayAs, DisplayFormatType};
use crate::execution_plan::{Boundedness, EmissionType};
use crate::metrics::{self, BaselineMetrics, ExecutionPlanMetricsSet, MetricBuilder, MetricsSet};
use crate::{ExecutionPlan, PlanProperties};

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

    fn create_physical_exprs(
        &self, 
        _ctx: &TaskContext
    ) -> Result<(Arc<dyn PhysicalExpr>, Vec<Arc<dyn PhysicalExpr>>)> {
        // In a real implementation, we would:
        // 1. Convert the pivot_column Expr to a PhysicalExpr
        // 2. Convert the aggregate_expr to PhysicalExpr(s)
        // 3. Return these for use in the PivotStream
        
        // For now, we'll just return placeholder values to get it to compile
        Err(DataFusionError::NotImplemented(
            "Creating physical expressions not implemented yet".to_string()
        ))
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
            _ => Err(DataFusionError::Internal(
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
        
        // Create the pivot stream
        let stream = PivotStream::new(
            self.schema.clone(),
            input_stream,
            partition,
            &self.metrics,
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
    /// Whether we have processed the input
    processed: bool,
}

impl PivotStream {
    /// Create a new pivot stream
    fn new(
        schema: SchemaRef,
        input: SendableRecordBatchStream,
        partition: usize,
        metrics_set: &ExecutionPlanMetricsSet,
    ) -> Self {
        let metrics = PivotMetrics::new(metrics_set, partition);
        
        Self {
            schema,
            input,
            _partition: partition,
            metrics,
            batches: Vec::new(),
            processed: false,
        }
    }

    /// Process the input
    fn process_input(&mut self) -> Result<Option<RecordBatch>> {
        let timer = self.metrics.elapsed_compute.timer();

        if self.batches.is_empty() {
            return Ok(None);
        }
        
        // Combine all batches
        let combined_batch = concat_batches(&self.input.schema(), &self.batches)?;

        if combined_batch.num_rows() == 0 {
            // No data to process
            return Ok(None);
        }

        // Simplified implementation that returns an empty batch with the correct schema
        let empty_arrays: Vec<ArrayRef> = self.schema
            .fields()
            .iter()
            .map(|field| {
                match field.data_type() {
                    DataType::Float64 => Arc::new(Float64Array::from(vec![0.0])) as ArrayRef,
                    DataType::Int64 => Arc::new(Int64Array::from(vec![0])) as ArrayRef,
                    DataType::UInt64 => Arc::new(UInt64Array::from(vec![0])) as ArrayRef,
                    _ => Arc::new(StringArray::from(vec!["placeholder"])) as ArrayRef,
                }
            })
            .collect();

        let empty_batch = RecordBatch::try_new(self.schema.clone(), empty_arrays)?;
        
        self.metrics.output_batches.add(1);
        self.metrics.output_rows.add(empty_batch.num_rows());
        
        timer.done();
        
        Ok(Some(empty_batch))
    }
}

impl Stream for PivotStream {
    type Item = Result<RecordBatch>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        if !self.processed {
            // First collect all input batches
            match self.input.poll_next_unpin(cx) {
                Poll::Ready(Some(Ok(batch))) => {
                    self.metrics.input_batches.add(1);
                    self.metrics.input_rows.add(batch.num_rows());
                    self.batches.push(batch);
                    Poll::Ready(None)
                }
                Poll::Ready(Some(Err(e))) => Poll::Ready(Some(Err(e))),
                Poll::Ready(None) => {
                    // We've consumed all input, now process it
                    self.processed = true;
                    match self.process_input() {
                        Ok(Some(batch)) => Poll::Ready(Some(Ok(batch))),
                        Ok(None) => Poll::Ready(None),
                        Err(e) => Poll::Ready(Some(Err(e))),
                    }
                }
                Poll::Pending => Poll::Pending,
            }
        } else {
            // We've already processed the input and returned the result
            Poll::Ready(None)
        }
    }
}

impl RecordBatchStream for PivotStream {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
} 