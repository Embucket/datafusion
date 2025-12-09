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

use crate::PhysicalOptimizerRule;
use datafusion_common::config::ConfigOptions;
use datafusion_common::tree_node::{Transformed, TreeNode};
use datafusion_common::{Result, ScalarValue};
use datafusion_expr::Operator;
use datafusion_physical_expr::expressions::{BinaryExpr, Column, Literal};
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_plan::filter::FilterExec;
use datafusion_physical_plan::projection::{ProjectionExec, ProjectionExpr};
use datafusion_physical_plan::rank::RankExec;
use datafusion_physical_plan::windows::{
    BoundedWindowAggExec, StandardWindowExpr, WindowUDFExpr,
};
use datafusion_physical_plan::{ExecutionPlan, WindowExpr};
use std::sync::Arc;

/// Rewrite `ROW_NUMBER() ... <= K` into a dedicated RankExec.
#[derive(Default, Clone, Debug)]
pub struct RankTopNPerPartition;

impl RankTopNPerPartition {
    pub fn new() -> Self {
        Self
    }
}

impl PhysicalOptimizerRule for RankTopNPerPartition {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if !config.optimizer.enable_rank_rewrite {
            return Ok(plan);
        }

        plan.transform_down(|node| {
            let Some(filter) = node.as_any().downcast_ref::<FilterExec>() else {
                return Ok(Transformed::no(node));
            };

            let projection = filter.projection().cloned();

            let child = filter.input();
            let Some(window) = child.as_any().downcast_ref::<BoundedWindowAggExec>()
            else {
                return Ok(Transformed::no(node));
            };

            // Only handle a single ROW_NUMBER window column.
            if window.window_expr().len() != 1 {
                return Ok(Transformed::no(node));
            }
            let window_expr = Arc::clone(&window.window_expr()[0]);

            if !is_row_number(&window_expr) {
                return Ok(Transformed::no(node));
            }

            let row_number_index = window_schema_index(window)?;
            let Some(fetch) = extract_rank_limit(filter.predicate(), row_number_index)?
            else {
                return Ok(Transformed::no(node));
            };

            if fetch == 0 {
                return Ok(Transformed::no(node));
            }

            let input = Arc::clone(window.input());
            let rank = Arc::new(RankExec::try_new(input, window_expr, fetch)?);

            let plan: Arc<dyn ExecutionPlan> = match projection {
                Some(indices) => {
                    let schema = rank.schema();
                    let exprs: Vec<ProjectionExpr> = indices
                        .iter()
                        .map(|idx| {
                            let field = schema.field(*idx);
                            let name = field.name().to_string();
                            ProjectionExpr {
                                expr: Arc::new(Column::new(&name, *idx)),
                                alias: name,
                            }
                        })
                        .collect();
                    Arc::new(ProjectionExec::try_new(exprs, rank)?)
                }
                None => rank,
            };
            Ok(Transformed::yes(plan))
        })
        .map(|t| t.data)
    }

    fn name(&self) -> &str {
        "RankTopNPerPartition"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

fn window_schema_index(window: &BoundedWindowAggExec) -> Result<usize> {
    let schema = window.schema();
    let field = window.window_expr()[0].field()?;
    let idx = match schema.index_of(field.name()) {
        Ok(idx) => idx,
        Err(_) => schema.fields().len().saturating_sub(1),
    };
    Ok(idx)
}

fn extract_rank_limit(
    predicate: &Arc<dyn PhysicalExpr>,
    row_number_index: usize,
) -> Result<Option<usize>> {
    let binary = match predicate.as_any().downcast_ref::<BinaryExpr>() {
        Some(binary) => binary,
        None => return Ok(None),
    };

    let op = *binary.op();
    let left_col = as_column(binary.left().as_ref());
    let right_col = as_column(binary.right().as_ref());

    let (col, lit, flip) = if let Some(col) = left_col {
        (Some(col), binary.right(), false)
    } else if let Some(col) = right_col {
        (Some(col), binary.left(), true)
    } else {
        (None, binary.left(), false)
    };

    let Some(col) = col else {
        return Ok(None);
    };

    if col.index() != row_number_index {
        return Ok(None);
    }

    // We only handle `row_number OP literal` patterns.
    if flip && op != Operator::Eq {
        return Ok(None);
    }

    let lit = match lit.as_any().downcast_ref::<Literal>() {
        Some(lit) => lit,
        None => return Ok(None),
    };
    let Some(value) = scalar_to_usize(lit.value()) else {
        return Ok(None);
    };
    if value == 0 {
        return Ok(None);
    }

    let fetch = match op {
        Operator::LtEq => Some(value),
        Operator::Lt => value.checked_sub(1),
        Operator::Eq if !flip => {
            if value == 1 {
                Some(1)
            } else {
                None
            }
        }
        _ => None,
    };

    Ok(fetch)
}

fn is_row_number(expr: &Arc<dyn WindowExpr>) -> bool {
    // Row number is implemented as a StandardWindowExpr wrapping a WindowUDFExpr
    // using the row_number_udwf(). Match on the canonical name to avoid
    // confusion with aliases.
    if let Some(std) = expr.as_any().downcast_ref::<StandardWindowExpr>() {
        if let Some(udwf) = std
            .get_standard_func_expr()
            .as_any()
            .downcast_ref::<WindowUDFExpr>()
        {
            return udwf.fun().name().eq_ignore_ascii_case("row_number");
        }
    }
    false
}

fn as_column(expr: &dyn PhysicalExpr) -> Option<&Column> {
    expr.as_any().downcast_ref::<Column>().or_else(|| {
        // Look through a top-level CAST.
        expr.as_any()
            .downcast_ref::<datafusion_physical_expr::expressions::CastExpr>()
            .and_then(|cast| cast.expr().as_any().downcast_ref::<Column>())
    })
}

fn scalar_to_usize(value: &ScalarValue) -> Option<usize> {
    match value {
        ScalarValue::UInt64(Some(v)) => usize::try_from(*v).ok(),
        ScalarValue::UInt32(Some(v)) => usize::try_from(*v).ok(),
        ScalarValue::UInt16(Some(v)) => usize::try_from(*v).ok(),
        ScalarValue::UInt8(Some(v)) => usize::try_from(*v).ok(),
        ScalarValue::Int64(Some(v)) if *v >= 0 => usize::try_from(*v as u64).ok(),
        ScalarValue::Int32(Some(v)) if *v >= 0 => usize::try_from(*v as u64).ok(),
        ScalarValue::Int16(Some(v)) if *v >= 0 => usize::try_from(*v as u64).ok(),
        ScalarValue::Int8(Some(v)) if *v >= 0 => usize::try_from(*v as u64).ok(),
        ScalarValue::Decimal128(Some(v), _precision, scale) if *scale == 0 => {
            usize::try_from(*v).ok()
        }
        ScalarValue::Decimal256(Some(v), _precision, scale) if *scale == 0 => v
            .to_string()
            .parse::<i128>()
            .ok()
            .and_then(|i| usize::try_from(i).ok()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use datafusion_common::ScalarValue;
    use datafusion_expr::{WindowFrame, WindowFunctionDefinition};
    use datafusion_functions_window::row_number::row_number_udwf;
    use datafusion_physical_expr::expressions::{lit, CastExpr, Column};
    use datafusion_physical_expr_common::sort_expr::PhysicalSortExpr;
    use datafusion_physical_plan::filter::FilterExec;
    use datafusion_physical_plan::projection::ProjectionExec;
    use datafusion_physical_plan::test::exec::MockExec;
    use datafusion_physical_plan::windows::create_window_expr;
    use datafusion_physical_plan::{ExecutionPlan, InputOrderMode, PhysicalExpr};

    #[test]
    fn rewrite_row_number_filter() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
        ]));
        let batch = RecordBatch::new_empty(Arc::clone(&schema));
        let input: Arc<dyn ExecutionPlan> =
            Arc::new(MockExec::new(vec![Ok(batch)], Arc::clone(&schema)));

        let partition_by: Vec<Arc<dyn PhysicalExpr>> =
            vec![Arc::new(Column::new("a", 0))];
        let order_by = vec![PhysicalSortExpr {
            expr: Arc::new(Column::new("b", 1)),
            options: Default::default(),
        }];
        let window_expr = create_window_expr(
            &WindowFunctionDefinition::WindowUDF(row_number_udwf()),
            "row_number".to_string(),
            &[],
            &partition_by,
            &order_by,
            Arc::new(WindowFrame::new(Some(true))),
            schema.as_ref(),
            false,
            false,
            None,
        )?;
        let window = Arc::new(BoundedWindowAggExec::try_new(
            vec![window_expr],
            Arc::clone(&input),
            InputOrderMode::Sorted,
            true,
        )?);

        let row_idx = input.schema().fields().len();
        let predicate: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("row_number", row_idx)),
            Operator::LtEq,
            lit(ScalarValue::UInt64(Some(2))),
        ));
        let filter = Arc::new(FilterExec::try_new(predicate, window)?);

        let rule = RankTopNPerPartition::new();
        let optimized = rule.optimize(filter, &ConfigOptions::default())?;
        let rank = optimized
            .as_any()
            .downcast_ref::<RankExec>()
            .expect("expected RankExec");
        assert_eq!(rank.fetch(), 2);
        assert!(rank.input().as_any().downcast_ref::<MockExec>().is_some());
        Ok(())
    }

    #[test]
    fn rewrite_row_number_filter_with_projection() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
        ]));
        let batch = RecordBatch::new_empty(Arc::clone(&schema));
        let input: Arc<dyn ExecutionPlan> =
            Arc::new(MockExec::new(vec![Ok(batch)], Arc::clone(&schema)));

        let partition_by: Vec<Arc<dyn PhysicalExpr>> =
            vec![Arc::new(Column::new("a", 0))];
        let order_by = vec![PhysicalSortExpr {
            expr: Arc::new(Column::new("b", 1)),
            options: Default::default(),
        }];
        let window_expr = create_window_expr(
            &WindowFunctionDefinition::WindowUDF(row_number_udwf()),
            "row_number".to_string(),
            &[],
            &partition_by,
            &order_by,
            Arc::new(WindowFrame::new(Some(true))),
            schema.as_ref(),
            false,
            false,
            None,
        )?;
        let window = Arc::new(BoundedWindowAggExec::try_new(
            vec![window_expr],
            Arc::clone(&input),
            InputOrderMode::Sorted,
            true,
        )?);

        let row_idx = input.schema().fields().len();
        let predicate: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("row_number", row_idx)),
            Operator::LtEq,
            lit(ScalarValue::UInt64(Some(2))),
        ));
        let filter = FilterExec::try_new(predicate, window)?;
        let filter = Arc::new(filter.with_projection(Some(vec![0, 1]))?);

        let rule = RankTopNPerPartition::new();
        let optimized = rule.optimize(filter, &ConfigOptions::default())?;
        let proj = optimized
            .as_any()
            .downcast_ref::<ProjectionExec>()
            .expect("expected ProjectionExec");
        assert_eq!(proj.expr().len(), 2);
        let rank = proj
            .input()
            .as_any()
            .downcast_ref::<RankExec>()
            .expect("expected RankExec input");
        assert_eq!(rank.fetch(), 2);
        Ok(())
    }

    #[test]
    fn no_rewrite_for_non_row_number() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
        let batch = RecordBatch::new_empty(Arc::clone(&schema));
        let input: Arc<dyn ExecutionPlan> =
            Arc::new(MockExec::new(vec![Ok(batch)], Arc::clone(&schema)));

        let partition_by: Vec<Arc<dyn PhysicalExpr>> =
            vec![Arc::new(Column::new("a", 0))];
        let order_by = vec![PhysicalSortExpr {
            expr: Arc::new(Column::new("a", 0)),
            options: Default::default(),
        }];
        let window_expr = create_window_expr(
            &WindowFunctionDefinition::WindowUDF(row_number_udwf()),
            "row_number".to_string(),
            &[],
            &partition_by,
            &order_by,
            Arc::new(WindowFrame::new(Some(true))),
            schema.as_ref(),
            false,
            false,
            None,
        )?;
        let window = Arc::new(BoundedWindowAggExec::try_new(
            vec![window_expr],
            Arc::clone(&input),
            InputOrderMode::Sorted,
            true,
        )?);

        // Filter on an input column, not on the window column.
        let predicate: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("a", 0)),
            Operator::LtEq,
            lit(ScalarValue::Int32(Some(1))),
        ));
        let filter = Arc::new(FilterExec::try_new(predicate, window)?);

        let rule = RankTopNPerPartition::new();
        let optimized = rule.optimize(filter.clone(), &ConfigOptions::default())?;
        assert!(optimized.as_any().downcast_ref::<FilterExec>().is_some());
        Ok(())
    }

    #[test]
    fn rewrite_with_casted_row_number_filter() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
        let batch = RecordBatch::new_empty(Arc::clone(&schema));
        let input: Arc<dyn ExecutionPlan> =
            Arc::new(MockExec::new(vec![Ok(batch)], Arc::clone(&schema)));

        let partition_by: Vec<Arc<dyn PhysicalExpr>> =
            vec![Arc::new(Column::new("a", 0))];
        let order_by = vec![PhysicalSortExpr {
            expr: Arc::new(Column::new("a", 0)),
            options: Default::default(),
        }];
        let window_expr = create_window_expr(
            &WindowFunctionDefinition::WindowUDF(row_number_udwf()),
            "row_number".to_string(),
            &[],
            &partition_by,
            &order_by,
            Arc::new(WindowFrame::new(Some(true))),
            schema.as_ref(),
            false,
            false,
            None,
        )?;
        let window = Arc::new(BoundedWindowAggExec::try_new(
            vec![window_expr],
            Arc::clone(&input),
            InputOrderMode::Sorted,
            true,
        )?);

        let row_idx = input.schema().fields().len();
        let cast_col: Arc<dyn PhysicalExpr> = Arc::new(CastExpr::new(
            Arc::new(Column::new("row_number", row_idx)),
            DataType::Decimal128(20, 0),
            None,
        ));
        let predicate: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            cast_col,
            Operator::Eq,
            lit(ScalarValue::Decimal128(Some(1), 20, 0)),
        ));
        let filter = Arc::new(FilterExec::try_new(predicate, window)?);

        let rule = RankTopNPerPartition::new();
        let optimized = rule.optimize(filter, &ConfigOptions::default())?;
        assert!(optimized.as_any().downcast_ref::<RankExec>().is_some());
        Ok(())
    }
}
