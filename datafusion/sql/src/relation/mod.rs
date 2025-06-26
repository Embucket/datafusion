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

use std::sync::Arc;

use crate::planner::{ContextProvider, PlannerContext, SqlToRel};

use datafusion_common::tree_node::{Transformed, TreeNode};
use datafusion_common::{
    not_impl_err, plan_err, Column, DFSchema, Diagnostic, Result, ScalarValue, Span,
    Spans, TableReference,
};
use datafusion_expr::binary::comparison_coercion;
use datafusion_expr::builder::subquery_alias;
use datafusion_expr::{expr::Unnest, Expr, LogicalPlan, LogicalPlanBuilder};
use datafusion_expr::{Subquery, SubqueryAlias};
use sqlparser::ast::{FunctionArg, FunctionArgExpr, NullInclusion, Spanned, TableFactor};

mod join;

impl<S: ContextProvider> SqlToRel<'_, S> {
    /// Create a `LogicalPlan` that scans the named relation
    fn create_relation(
        &self,
        relation: TableFactor,
        planner_context: &mut PlannerContext,
    ) -> Result<LogicalPlan> {
        let relation_span = relation.span();
        let (plan, alias) = match relation {
            TableFactor::Table {
                name, alias, args, ..
            } => {
                if let Some(func_args) = args {
                    let tbl_func_name =
                        name.0.first().unwrap().as_ident().unwrap().to_string();
                    let args = func_args
                        .args
                        .into_iter()
                        .map(|arg| {
                            if let FunctionArg::Unnamed(FunctionArgExpr::Expr(expr)) = arg
                            {
                                self.sql_expr_to_logical_expr(
                                    expr,
                                    &DFSchema::empty(),
                                    planner_context,
                                )
                                .map(|expr| (expr, None))
                            } else if let FunctionArg::Named { name, arg, .. } = arg {
                                if let FunctionArgExpr::Expr(expr) = arg {
                                    self.sql_expr_to_logical_expr(
                                        expr,
                                        &DFSchema::empty(),
                                        planner_context,
                                    )
                                    .map(|expr| (expr, Some(name.to_string())))
                                } else {
                                    plan_err!(
                                        "Unsupported function argument type: {:?}",
                                        arg
                                    )
                                }
                            } else {
                                plan_err!("Unsupported function argument type: {:?}", arg)
                            }
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let provider = self
                        .context_provider
                        .get_table_function_source(&tbl_func_name, args)?;
                    let plan = LogicalPlanBuilder::scan(
                        TableReference::Bare {
                            table: "tmp_table".into(),
                        },
                        provider,
                        None,
                    )?
                    .build()?;
                    (plan, alias)
                } else {
                    // Normalize name and alias
                    let table_ref = self.object_name_to_table_reference(name)?;
                    let table_name = table_ref.to_string();
                    let cte = planner_context.get_cte(&table_name);
                    (
                        match (
                            cte,
                            self.context_provider.get_table_source(table_ref.clone()),
                        ) {
                            (Some(cte_plan), _) => Ok(cte_plan.clone()),
                            (_, Ok(provider)) => LogicalPlanBuilder::scan(
                                table_ref.clone(),
                                provider,
                                None,
                            )?
                            .build(),
                            (None, Err(e)) => {
                                let e = e.with_diagnostic(Diagnostic::new_error(
                                    format!("table '{}' not found", table_ref),
                                    Span::try_from_sqlparser_span(relation_span),
                                ));
                                Err(e)
                            }
                        }?,
                        alias,
                    )
                }
            }
            TableFactor::Derived {
                subquery, alias, ..
            } => {
                let logical_plan = self.query_to_plan(*subquery, planner_context)?;
                (logical_plan, alias)
            }
            TableFactor::NestedJoin {
                table_with_joins,
                alias,
            } => (
                self.plan_table_with_joins(*table_with_joins, planner_context)?,
                alias,
            ),
            TableFactor::UNNEST {
                alias,
                array_exprs,
                with_offset: false,
                with_offset_alias: None,
                with_ordinality,
            } => {
                if with_ordinality {
                    return not_impl_err!("UNNEST with ordinality is not supported yet");
                }

                // Unnest table factor has empty input
                let schema = DFSchema::empty();
                let input = LogicalPlanBuilder::empty(true).build()?;
                // Unnest table factor can have multiple arguments.
                // We treat each argument as a separate unnest expression.
                let unnest_exprs = array_exprs
                    .into_iter()
                    .map(|sql_expr| {
                        let expr = self.sql_expr_to_logical_expr(
                            sql_expr,
                            &schema,
                            planner_context,
                        )?;
                        Self::check_unnest_arg(&expr, &schema)?;
                        Ok(Expr::Unnest(Unnest::new(expr)))
                    })
                    .collect::<Result<Vec<_>>>()?;
                if unnest_exprs.is_empty() {
                    return plan_err!("UNNEST must have at least one argument");
                }
                let logical_plan = self.try_process_unnest(input, unnest_exprs)?;
                (logical_plan, alias)
            }
            TableFactor::UNNEST { .. } => {
                return not_impl_err!(
                    "UNNEST table factor with offset is not supported yet"
                );
            }
            TableFactor::Pivot {
                table,
                aggregate_functions,
                value_column,
                value_source,
                default_on_null,
                alias,
            } => {
                let input_plan = self.create_relation(*table, planner_context)?;

                if aggregate_functions.len() != 1 {
                    return plan_err!("PIVOT requires exactly one aggregate function");
                }

                let agg_expr = self.sql_expr_to_logical_expr(
                    aggregate_functions[0].expr.clone(),
                    input_plan.schema(),
                    planner_context,
                )?;

                if value_column.is_empty() {
                    return plan_err!("PIVOT value column is required");
                }

                let column_name = value_column.last().unwrap().value.clone();
                let pivot_column = Column::new(None::<&str>, column_name);

                let default_on_null_expr = default_on_null
                    .map(|expr| {
                        self.sql_expr_to_logical_expr(
                            expr,
                            input_plan.schema(), // Default expression should be context-independent or use input schema
                            planner_context,
                        )
                    })
                    .transpose()?;

                match value_source {
                    sqlparser::ast::PivotValueSource::List(exprs) => {
                        let pivot_values = exprs
                            .iter()
                            .map(|expr| {
                                let logical_expr = self.sql_expr_to_logical_expr(
                                    expr.expr.clone(),
                                    input_plan.schema(),
                                    planner_context,
                                )?;

                                match logical_expr {
                                    Expr::Literal(scalar) => Ok(scalar),
                                    _ => plan_err!("PIVOT values must be literals"),
                                }
                            })
                            .collect::<Result<Vec<_>>>()?;

                        let input_arc = Arc::new(input_plan);

                        let pivot_plan = datafusion_expr::Pivot::try_new(
                            input_arc,
                            agg_expr,
                            pivot_column,
                            pivot_values,
                            default_on_null_expr.clone(),
                        )?;

                        (LogicalPlan::Pivot(pivot_plan), alias)
                    }
                    sqlparser::ast::PivotValueSource::Any(order_by) => {
                        let input_arc = Arc::new(input_plan);

                        let mut subquery_builder =
                            LogicalPlanBuilder::from(input_arc.as_ref().clone())
                                .project(vec![Expr::Column(pivot_column.clone())])?
                                .distinct()?;

                        if !order_by.is_empty() {
                            let sort_exprs = order_by
                                .iter()
                                .map(|item| {
                                    let input_schema = subquery_builder.schema();

                                    let expr = self.sql_expr_to_logical_expr(
                                        item.expr.clone(),
                                        input_schema,
                                        planner_context,
                                    );

                                    expr.map(|e| {
                                        e.sort(
                                            item.options.asc.unwrap_or(true),
                                            item.options.nulls_first.unwrap_or(false),
                                        )
                                    })
                                })
                                .collect::<Result<Vec<_>>>()?;

                            subquery_builder = subquery_builder.sort(sort_exprs)?;
                        }

                        let subquery_plan = subquery_builder.build()?;

                        let pivot_plan = datafusion_expr::Pivot::try_new_with_subquery(
                            input_arc,
                            agg_expr,
                            pivot_column,
                            Arc::new(subquery_plan),
                            default_on_null_expr.clone(),
                        )?;

                        (LogicalPlan::Pivot(pivot_plan), alias)
                    }
                    sqlparser::ast::PivotValueSource::Subquery(subquery) => {
                        let subquery_plan =
                            self.query_to_plan(*subquery.clone(), planner_context)?;

                        let input_arc = Arc::new(input_plan);

                        let pivot_plan = datafusion_expr::Pivot::try_new_with_subquery(
                            input_arc,
                            agg_expr,
                            pivot_column,
                            Arc::new(subquery_plan),
                            default_on_null_expr.clone(),
                        )?;

                        (LogicalPlan::Pivot(pivot_plan), alias)
                    }
                }
            }
            TableFactor::Unpivot {
                table,
                null_inclusion,
                value,
                name,
                columns,
                alias,
            } => {
                let base_plan = self.create_relation(*table, planner_context)?;
                let base_schema = base_plan.schema();

                let value_column = value.value.clone();
                let name_column = name.value.clone();

                let mut unpivot_column_indices = Vec::new();
                let mut unpivot_column_names = Vec::new();

                let mut common_type = None;

                for column_ident in &columns {
                    let column_name = column_ident.value.clone();

                    let idx = if let Some(i) =
                        base_schema.index_of_column_by_name(None, &column_name)
                    {
                        i
                    } else {
                        return plan_err!("Column '{}' not found in input", column_name);
                    };

                    let field = base_schema.field(idx);
                    let field_type = field.data_type();

                    // Verify all unpivot columns have compatible types
                    if let Some(current_type) = &common_type {
                        if comparison_coercion(current_type, field_type).is_none() {
                            return plan_err!(
                                    "The type of column '{}' conflicts with the type of other columns in the UNPIVOT list.",
                                    column_name.to_uppercase()
                                );
                        }
                    } else {
                        common_type = Some(field_type.clone());
                    }

                    unpivot_column_indices.push(idx);
                    unpivot_column_names.push(column_name);
                }

                if unpivot_column_names.is_empty() {
                    return plan_err!("UNPIVOT requires at least one column to unpivot");
                }

                let non_pivot_exprs: Vec<Expr> = base_schema
                    .fields()
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| !unpivot_column_indices.contains(i))
                    .map(|(_, f)| Expr::Column(Column::new(None::<&str>, f.name())))
                    .collect();

                let mut union_inputs = Vec::with_capacity(unpivot_column_names.len());

                for col_name in &unpivot_column_names {
                    let mut projection_exprs = non_pivot_exprs.clone();

                    let name_expr =
                        Expr::Literal(ScalarValue::Utf8(Some(col_name.clone())))
                            .alias(name_column.clone());

                    let value_expr =
                        Expr::Column(Column::new(None::<&str>, col_name.clone()))
                            .alias(value_column.clone());

                    projection_exprs.push(name_expr);
                    projection_exprs.push(value_expr);

                    let mut builder = LogicalPlanBuilder::from(base_plan.clone())
                        .project(projection_exprs)?;

                    if let Some(NullInclusion::ExcludeNulls) | None = null_inclusion {
                        let col = Column::new(None::<&str>, value_column.clone());
                        builder = builder
                            .filter(Expr::IsNotNull(Box::new(Expr::Column(col))))?;
                    }

                    union_inputs.push(builder.build()?);
                }

                let first = union_inputs.remove(0);
                let mut union_builder = LogicalPlanBuilder::from(first);

                for plan in union_inputs {
                    union_builder = union_builder.union(plan)?;
                }

                let unpivot_plan = union_builder.build()?;

                (unpivot_plan, alias)
            }
            TableFactor::Function {
                name, args, alias, ..
            } => {
                let tbl_func_ref = self.object_name_to_table_reference(name)?;
                let schema = planner_context
                    .outer_query_schema()
                    .cloned()
                    .unwrap_or_else(DFSchema::empty);
                let func_args = args
                    .into_iter()
                    .map(|arg| match arg {
                        FunctionArg::Unnamed(FunctionArgExpr::Expr(expr)) => {
                            let expr = self.sql_expr_to_logical_expr(
                                expr,
                                &schema,
                                planner_context,
                            )?;
                            Ok((expr, None))
                        }
                        FunctionArg::Named {
                            name,
                            arg: FunctionArgExpr::Expr(expr),
                            ..
                        } => {
                            let expr = self.sql_expr_to_logical_expr(
                                expr,
                                &schema,
                                planner_context,
                            )?;
                            Ok((expr, Some(name.value.clone())))
                        }
                        _ => plan_err!("Unsupported function argument: {arg:?}"),
                    })
                    .collect::<Result<Vec<(Expr, Option<String>)>>>()?;
                let tbl_func_name = tbl_func_ref.table().to_ascii_lowercase();
                let provider = self
                    .context_provider
                    .get_table_function_source(&tbl_func_name, func_args)?;
                let plan =
                    LogicalPlanBuilder::scan(tbl_func_name, provider, None)?.build()?;
                (plan, alias)
            }
            // @todo: Support TableFactory::TableFunction
            _ => {
                return not_impl_err!(
                    "Unsupported ast node {relation:?} in create_relation"
                );
            }
        };

        let optimized_plan = optimize_subquery_sort(plan)?.data;
        if let Some(alias) = alias {
            self.apply_table_alias(optimized_plan, alias)
        } else {
            Ok(optimized_plan)
        }
    }

    pub(crate) fn create_relation_subquery(
        &self,
        subquery: TableFactor,
        planner_context: &mut PlannerContext,
    ) -> Result<LogicalPlan> {
        // At this point for a syntactically valid query the outer_from_schema is
        // guaranteed to be set, so the `.unwrap()` call will never panic. This
        // is the case because we only call this method for lateral table
        // factors, and those can never be the first factor in a FROM list. This
        // means we arrived here through the `for` loop in `plan_from_tables` or
        // the `for` loop in `plan_table_with_joins`.
        let old_from_schema = planner_context
            .set_outer_from_schema(None)
            .unwrap_or_else(|| Arc::new(DFSchema::empty()));
        let new_query_schema = match planner_context.outer_query_schema() {
            Some(old_query_schema) => {
                let mut new_query_schema = old_from_schema.as_ref().clone();
                new_query_schema.merge(old_query_schema);
                Some(Arc::new(new_query_schema))
            }
            None => Some(Arc::clone(&old_from_schema)),
        };
        let old_query_schema = planner_context.set_outer_query_schema(new_query_schema);

        let plan = self.create_relation(subquery, planner_context)?;
        let outer_ref_columns = plan.all_out_ref_exprs();

        planner_context.set_outer_query_schema(old_query_schema);
        planner_context.set_outer_from_schema(Some(old_from_schema));

        // We can omit the subquery wrapper if there are no columns
        // referencing the outer scope.
        if outer_ref_columns.is_empty() {
            return Ok(plan);
        }

        match plan {
            LogicalPlan::SubqueryAlias(SubqueryAlias { input, alias, .. }) => {
                subquery_alias(
                    LogicalPlan::Subquery(Subquery {
                        subquery: input,
                        outer_ref_columns,
                        spans: Spans::new(),
                    }),
                    alias,
                )
            }
            plan => Ok(LogicalPlan::Subquery(Subquery {
                subquery: Arc::new(plan),
                outer_ref_columns,
                spans: Spans::new(),
            })),
        }
    }
}

fn optimize_subquery_sort(plan: LogicalPlan) -> Result<Transformed<LogicalPlan>> {
    // When initializing subqueries, we examine sort options since they might be unnecessary.
    // They are only important if the subquery result is affected by the ORDER BY statement,
    // which can happen when we have:
    // 1. DISTINCT ON / ARRAY_AGG ... => Handled by an `Aggregate` and its requirements.
    // 2. RANK / ROW_NUMBER ... => Handled by a `WindowAggr` and its requirements.
    // 3. LIMIT => Handled by a `Sort`, so we need to search for it.
    let mut has_limit = false;
    let new_plan = plan.transform_down(|c| {
        if let LogicalPlan::Limit(_) = c {
            has_limit = true;
            return Ok(Transformed::no(c));
        }
        match c {
            LogicalPlan::Sort(s) => {
                if !has_limit {
                    has_limit = false;
                    return Ok(Transformed::yes(s.input.as_ref().clone()));
                }
                Ok(Transformed::no(LogicalPlan::Sort(s)))
            }
            _ => Ok(Transformed::no(c)),
        }
    });
    new_plan
}
