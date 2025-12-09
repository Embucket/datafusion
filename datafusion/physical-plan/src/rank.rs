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

//! A spill-friendly Rank operator that assumes its input is already sorted by
//! `PARTITION BY` + `ORDER BY` and emits only the first K rows per partition.

use std::any::Any;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use crate::execution_plan::CardinalityEffect;
use crate::metrics::{BaselineMetrics, ExecutionPlanMetricsSet};
use crate::windows::{calc_requirements, window_equivalence_properties};
use crate::{
    ColumnStatistics, DisplayAs, DisplayFormatType, Distribution, ExecutionPlan,
    ExecutionPlanProperties, PlanProperties, RecordBatchStream,
    SendableRecordBatchStream, Statistics,
};

use arrow::array::{ArrayRef, UInt32Builder, UInt64Array};
use arrow::compute::{take, SortOptions};
use arrow::datatypes::{Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use arrow::row::{RowConverter, SortField};
use datafusion_common::Result;
use datafusion_execution::TaskContext;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr_common::sort_expr::{
    OrderingRequirements, PhysicalSortExpr,
};
use futures::{Stream, StreamExt};

/// Streaming rank operator that drops rows once rank exceeds `fetch` per
/// partition. Assumes input is ordered by `partition_by` followed by `order_by`.
#[derive(Debug, Clone)]
pub struct RankExec {
    /// Input execution plan
    input: Arc<dyn ExecutionPlan>,
    /// The window expression (expected to be ROW_NUMBER)
    window_expr: Arc<dyn datafusion_physical_expr::window::WindowExpr>,
    /// Partition expressions
    partition_by: Vec<Arc<dyn PhysicalExpr>>,
    /// Maximum number of rows to emit per partition
    fetch: usize,
    /// Output schema (input schema + row_number column)
    schema: SchemaRef,
    /// Execution metrics
    metrics: ExecutionPlanMetricsSet,
    /// Cached plan properties
    cache: PlanProperties,
}

impl RankExec {
    /// Create a new RankExec
    pub fn try_new(
        input: Arc<dyn ExecutionPlan>,
        window_expr: Arc<dyn datafusion_physical_expr::window::WindowExpr>,
        fetch: usize,
    ) -> Result<Self> {
        let input_schema = input.schema();
        let mut fields = input_schema
            .fields()
            .iter()
            .map(|f| f.as_ref().clone())
            .collect::<Vec<_>>();
        fields.push(window_expr.field()?.as_ref().clone());
        let schema = Arc::new(Schema::new(fields));

        let eq_properties = window_equivalence_properties(
            &schema,
            &input,
            std::slice::from_ref(&window_expr),
        )?;

        let partition_by = window_expr.partition_by().to_vec();
        let cache = PlanProperties::new(
            eq_properties,
            input.output_partitioning().clone(),
            input.pipeline_behavior(),
            input.boundedness(),
        );

        Ok(Self {
            input,
            window_expr,
            partition_by,
            fetch,
            schema,
            metrics: ExecutionPlanMetricsSet::new(),
            cache,
        })
    }

    /// Input execution plan
    pub fn input(&self) -> &Arc<dyn ExecutionPlan> {
        &self.input
    }

    /// K value (maximum rows per partition)
    pub fn fetch(&self) -> usize {
        self.fetch
    }

    /// Partition by expressions
    fn partition_by(&self) -> &[Arc<dyn PhysicalExpr>] {
        &self.partition_by
    }

    /// Order-by expressions
    fn order_by(&self) -> &[PhysicalSortExpr] {
        self.window_expr.order_by()
    }

    /// Number of input columns (excluding row_number)
    fn input_col_count(&self) -> usize {
        self.input.schema().fields().len()
    }

    /// Build the row converter used to detect partition boundaries.
    fn build_row_converter(&self, input_schema: &Schema) -> Result<Option<RowConverter>> {
        if self.partition_by().is_empty() {
            return Ok(None);
        }

        let sort_fields = self
            .partition_by()
            .iter()
            .map(|expr| {
                Ok(SortField::new_with_options(
                    expr.data_type(input_schema)?,
                    SortOptions::default(),
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let converter = RowConverter::new(sort_fields)?;
        Ok(Some(converter))
    }
}

impl DisplayAs for RankExec {
    fn fmt_as(
        &self,
        t: DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => write!(
                f,
                "RankExec: k={}, partition_by=[{}], order_by=[{}]",
                self.fetch,
                fmt_expr_list(self.partition_by()),
                fmt_sort_list(self.order_by())
            ),
            DisplayFormatType::TreeRender => {
                writeln!(f, "k={}", self.fetch)?;
                writeln!(f, "partition_by=[{}]", fmt_expr_list(self.partition_by()))?;
                write!(f, "order_by=[{}]", fmt_sort_list(self.order_by()))
            }
        }
    }
}

impl ExecutionPlan for RankExec {
    fn name(&self) -> &'static str {
        "RankExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &PlanProperties {
        &self.cache
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true]
    }

    fn required_input_ordering(&self) -> Vec<Option<OrderingRequirements>> {
        vec![calc_requirements(self.partition_by(), self.order_by())]
    }

    fn required_input_distribution(&self) -> Vec<Distribution> {
        if self.partition_by().is_empty() {
            vec![Distribution::SinglePartition]
        } else {
            vec![Distribution::HashPartitioned(self.partition_by().to_vec())]
        }
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![true]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(Self::try_new(
            Arc::clone(&children[0]),
            Arc::clone(&self.window_expr),
            self.fetch,
        )?))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let input = self.input.execute(partition, context)?;
        let baseline = BaselineMetrics::new(&self.metrics, partition);
        let row_converter = self.build_row_converter(input.schema().as_ref())?;
        let stream = RankStream::new(
            input,
            self.fetch,
            self.input_col_count(),
            self.partition_by.clone(),
            row_converter,
            Arc::clone(&self.schema),
            baseline,
        );
        Ok(Box::pin(stream))
    }

    fn metrics(&self) -> Option<crate::metrics::MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn statistics(&self) -> Result<Statistics> {
        self.partition_statistics(None)
    }

    fn partition_statistics(&self, partition: Option<usize>) -> Result<Statistics> {
        let mut stats = self.input.partition_statistics(partition)?;
        stats
            .column_statistics
            .resize(self.schema.fields().len(), ColumnStatistics::new_unknown());
        Ok(stats)
    }

    fn cardinality_effect(&self) -> CardinalityEffect {
        CardinalityEffect::LowerEqual
    }
}

/// RecordBatchStream that enforces per-partition top-K over a sorted input.
struct RankStream {
    input: SendableRecordBatchStream,
    fetch: usize,
    input_col_count: usize,
    partition_by: Vec<Arc<dyn PhysicalExpr>>,
    row_converter: Option<RowConverter>,
    schema: SchemaRef,
    baseline_metrics: BaselineMetrics,
    current_key: Option<Vec<u8>>,
    current_rank: usize,
}

impl RankStream {
    fn new(
        input: SendableRecordBatchStream,
        fetch: usize,
        input_col_count: usize,
        partition_by: Vec<Arc<dyn PhysicalExpr>>,
        row_converter: Option<RowConverter>,
        schema: SchemaRef,
        baseline_metrics: BaselineMetrics,
    ) -> Self {
        Self {
            input,
            fetch,
            input_col_count,
            partition_by,
            row_converter,
            schema,
            baseline_metrics,
            current_key: None,
            current_rank: 0,
        }
    }

    fn process_batch(&mut self, batch: RecordBatch) -> Result<Option<RecordBatch>> {
        let _timer = self.baseline_metrics.elapsed_compute().timer();
        let num_rows = batch.num_rows();
        if num_rows == 0 {
            return Ok(None);
        }

        let partition_arrays = if self.row_converter.is_some() {
            Some(self.partition_arrays(&batch)?)
        } else {
            None
        };
        let partition_keys = match (self.row_converter.as_mut(), partition_arrays) {
            (Some(conv), Some(arrays)) => Some(conv.convert_columns(&arrays)?),
            _ => None,
        };

        let mut indices = UInt32Builder::with_capacity(num_rows);
        let mut row_numbers = Vec::new();
        let mut kept = 0usize;

        for row_idx in 0..num_rows {
            if let Some(conv) = partition_keys.as_ref() {
                let row = conv.row(row_idx);
                let row_bytes = row.as_ref().to_vec();
                let is_same = self
                    .current_key
                    .as_ref()
                    .map(|key| key == &row_bytes)
                    .unwrap_or(false);
                if !is_same {
                    self.current_key = Some(row_bytes);
                    self.current_rank = 1;
                } else {
                    self.current_rank += 1;
                }
            } else {
                // Global ranking with no partition keys
                if row_idx == 0 && self.current_key.is_none() {
                    self.current_key = Some(Vec::new());
                    self.current_rank = 1;
                } else {
                    self.current_rank += 1;
                }
            };

            let rank = self.current_rank;
            // keep first K per partition
            if rank <= self.fetch {
                indices.append_value(row_idx as u32);
                row_numbers.push(rank as u64);
                kept += 1;
            }
        }

        if kept == 0 {
            return Ok(None);
        }

        let indices = indices.finish();
        let mut columns = Vec::with_capacity(self.input_col_count + 1);
        for col_idx in 0..self.input_col_count {
            let col = batch.column(col_idx);
            let taken = take(col, &indices, None)?;
            columns.push(taken);
        }

        let row_num_array: UInt64Array = row_numbers.into();
        columns.push(Arc::new(row_num_array));

        let output = RecordBatch::try_new(Arc::clone(&self.schema), columns)?;
        self.baseline_metrics.record_output(output.num_rows());
        Ok(Some(output))
    }

    fn partition_arrays(&self, batch: &RecordBatch) -> Result<Vec<ArrayRef>> {
        let mut arrays = Vec::with_capacity(self.partition_by.len());
        for expr in &self.partition_by {
            let col = expr.evaluate(batch)?.into_array(batch.num_rows())?;
            arrays.push(col);
        }
        Ok(arrays)
    }
}

impl Stream for RankStream {
    type Item = Result<RecordBatch>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        loop {
            match self.input.poll_next_unpin(cx) {
                Poll::Ready(Some(Ok(batch))) => {
                    if let Some(out) = self.process_batch(batch)? {
                        return Poll::Ready(Some(Ok(out)));
                    }
                    continue;
                }
                Poll::Ready(Some(Err(e))) => return Poll::Ready(Some(Err(e))),
                Poll::Ready(None) => return Poll::Ready(None),
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

impl RecordBatchStream for RankStream {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}

fn fmt_expr_list(exprs: &[Arc<dyn PhysicalExpr>]) -> String {
    exprs
        .iter()
        .map(|e| e.to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

fn fmt_sort_list(exprs: &[PhysicalSortExpr]) -> String {
    exprs
        .iter()
        .map(|e| e.to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Int32Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion_common::assert_batches_sorted_eq;
    use datafusion_execution::TaskContext;
    use datafusion_expr::{WindowFrame, WindowFunctionDefinition};
    use datafusion_physical_expr::expressions::Column;
    use datafusion_physical_plan::test::exec::MockExec;
    use datafusion_physical_plan::windows::create_window_expr;
    use datafusion_physical_plan::PhysicalExpr;
    use std::sync::Arc;

    #[tokio::test]
    async fn rank_exec_limits_rows_per_partition() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
        ]));

        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(vec![1, 1, 1, 2, 2])) as ArrayRef,
                Arc::new(Int32Array::from(vec![10, 20, 30, 5, 15])) as ArrayRef,
            ],
        )?;

        let input: Arc<dyn ExecutionPlan> =
            Arc::new(MockExec::new(vec![Ok(batch)], Arc::clone(&schema)));

        let partition_by: Vec<Arc<dyn PhysicalExpr>> =
            vec![Arc::new(Column::new("a", 0))];
        let order_by = vec![PhysicalSortExpr {
            expr: Arc::new(Column::new("b", 1)),
            options: Default::default(),
        }];
        let window_expr = create_window_expr(
            &WindowFunctionDefinition::WindowUDF(
                datafusion_functions_window::row_number::row_number_udwf(),
            ),
            "row_number".to_string(),
            &[],
            &partition_by,
            &order_by,
            Arc::new(WindowFrame::default()),
            schema.as_ref(),
            false,
            false,
            None,
        )?;

        let rank_plan = Arc::new(RankExec::try_new(input, window_expr, 2)?);

        let batches = crate::collect(rank_plan, Arc::new(TaskContext::default())).await?;

        assert_batches_sorted_eq!(
            vec![
                "+---+----+------------+",
                "| a | b  | row_number |",
                "+---+----+------------+",
                "| 1 | 10 | 1          |",
                "| 1 | 20 | 2          |",
                "| 2 | 5  | 1          |",
                "| 2 | 15 | 2          |",
                "+---+----+------------+",
            ],
            &batches
        );
        Ok(())
    }
}
