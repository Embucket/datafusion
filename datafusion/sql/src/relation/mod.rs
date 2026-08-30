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
    Column, DFSchema, Diagnostic, Result, ScalarValue, Span, Spans, TableReference,
    not_impl_err, plan_err,
};
use datafusion_expr::builder::subquery_alias;
use datafusion_expr::expr::{AggregateFunction, Alias, BinaryExpr, Case, Cast};
use datafusion_expr::planner::{
    PlannedRelation, RelationPlannerContext, RelationPlanning,
};
use datafusion_expr::type_coercion::binary::comparison_coercion;
use datafusion_expr::{Expr, LogicalPlan, LogicalPlanBuilder, Operator, expr::Unnest};
use datafusion_expr::{Subquery, SubqueryAlias};
use sqlparser::ast::{
    FunctionArg, FunctionArgExpr, NullInclusion, PivotValueSource, Spanned, TableFactor,
};

mod join;

struct SqlToRelRelationContext<'a, 'b, S: ContextProvider> {
    planner: &'a SqlToRel<'b, S>,
    planner_context: &'a mut PlannerContext,
}

// Implement RelationPlannerContext
impl<'a, 'b, S: ContextProvider> RelationPlannerContext
    for SqlToRelRelationContext<'a, 'b, S>
{
    fn context_provider(&self) -> &dyn ContextProvider {
        self.planner.context_provider
    }

    fn plan(&mut self, relation: TableFactor) -> Result<LogicalPlan> {
        self.planner.create_relation(relation, self.planner_context)
    }

    fn sql_to_expr(
        &mut self,
        expr: sqlparser::ast::Expr,
        schema: &DFSchema,
    ) -> Result<Expr> {
        self.planner.sql_to_expr(expr, schema, self.planner_context)
    }

    fn sql_expr_to_logical_expr(
        &mut self,
        expr: sqlparser::ast::Expr,
        schema: &DFSchema,
    ) -> Result<Expr> {
        self.planner
            .sql_expr_to_logical_expr(expr, schema, self.planner_context)
    }

    fn normalize_ident(&self, ident: sqlparser::ast::Ident) -> String {
        self.planner.ident_normalizer.normalize(ident)
    }

    fn object_name_to_table_reference(
        &self,
        name: sqlparser::ast::ObjectName,
    ) -> Result<TableReference> {
        self.planner.object_name_to_table_reference(name)
    }
}

impl<S: ContextProvider> SqlToRel<'_, S> {
    /// Create a `LogicalPlan` that scans the named relation.
    ///
    /// First tries any registered extension planners. If no extension handles
    /// the relation, falls back to the default planner.
    fn create_relation(
        &self,
        relation: TableFactor,
        planner_context: &mut PlannerContext,
    ) -> Result<LogicalPlan> {
        let planned_relation =
            match self.create_extension_relation(relation, planner_context)? {
                RelationPlanning::Planned(planned) => planned,
                RelationPlanning::Original(original) => {
                    Box::new(self.create_default_relation(*original, planner_context)?)
                }
            };

        let optimized_plan = optimize_subquery_sort(
            planned_relation.plan,
            self.context_provider
                .options()
                .sql_parser
                .enable_subquery_sort_elimination,
        )?
        .data;
        if let Some(alias) = planned_relation.alias {
            self.apply_table_alias(optimized_plan, alias)
        } else {
            Ok(optimized_plan)
        }
    }

    fn create_extension_relation(
        &self,
        relation: TableFactor,
        planner_context: &mut PlannerContext,
    ) -> Result<RelationPlanning> {
        let planners = self.context_provider.get_relation_planners();
        if planners.is_empty() {
            return Ok(RelationPlanning::Original(Box::new(relation)));
        }

        let mut current_relation = relation;
        for planner in planners.iter() {
            let mut context = SqlToRelRelationContext {
                planner: self,
                planner_context,
            };

            match planner.plan_relation(current_relation, &mut context)? {
                RelationPlanning::Planned(planned) => {
                    return Ok(RelationPlanning::Planned(planned));
                }
                RelationPlanning::Original(original) => {
                    current_relation = *original;
                }
            }
        }

        Ok(RelationPlanning::Original(Box::new(current_relation)))
    }

    fn create_default_relation(
        &self,
        relation: TableFactor,
        planner_context: &mut PlannerContext,
    ) -> Result<PlannedRelation> {
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
                        .map(|arg| match arg {
                            FunctionArg::Unnamed(FunctionArgExpr::Expr(expr)) => self
                                .sql_expr_to_logical_expr(
                                    expr,
                                    &DFSchema::empty(),
                                    planner_context,
                                )
                                .map(|expr| (expr, None)),
                            FunctionArg::Named {
                                name,
                                arg: FunctionArgExpr::Expr(expr),
                                ..
                            } => self
                                .sql_expr_to_logical_expr(
                                    expr,
                                    &DFSchema::empty(),
                                    planner_context,
                                )
                                .map(|expr| (expr, Some(name.to_string()))),
                            _ => plan_err!("Unsupported function argument type: {arg}"),
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let provider = self
                        .context_provider
                        .get_table_function_source(&tbl_func_name, args)?;
                    let plan = LogicalPlanBuilder::scan(
                        TableReference::Bare {
                            table: format!("{tbl_func_name}()").into(),
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
                                    format!("table '{table_ref}' not found"),
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
            TableFactor::Function {
                name, args, alias, ..
            } => {
                let tbl_func_ref = self.object_name_to_table_reference(name)?;
                let schema = planner_context
                    .outer_queries_schemas()
                    .last()
                    .cloned()
                    .unwrap_or_else(|| Arc::new(DFSchema::empty()));
                let func_args = args
                    .into_iter()
                    .map(|arg| match arg {
                        FunctionArg::Unnamed(FunctionArgExpr::Expr(expr)) => self
                            .sql_expr_to_logical_expr(expr, &schema, planner_context)
                            .map(|expr| (expr, None)),
                        FunctionArg::Named {
                            name,
                            arg: FunctionArgExpr::Expr(expr),
                            ..
                        } => self
                            .sql_expr_to_logical_expr(expr, &schema, planner_context)
                            .map(|expr| (expr, Some(name.to_string()))),
                        _ => plan_err!("Unsupported function argument: {arg:?}"),
                    })
                    .collect::<Result<Vec<_>>>()?;
                let provider = self
                    .context_provider
                    .get_table_function_source(tbl_func_ref.table(), func_args)?;
                let plan =
                    LogicalPlanBuilder::scan(tbl_func_ref.table(), provider, None)?
                        .build()?;
                (plan, alias)
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

                let aggregate_expr = self.sql_expr_to_logical_expr(
                    aggregate_functions[0].expr.clone(),
                    input_plan.schema(),
                    planner_context,
                )?;
                let pivot_ident = value_column.last().ok_or_else(|| {
                    datafusion_common::plan_datafusion_err!(
                        "PIVOT value column is required"
                    )
                })?;
                let column_name = match pivot_ident {
                    sqlparser::ast::Expr::Identifier(ident) => {
                        self.ident_normalizer.normalize(ident.clone())
                    }
                    sqlparser::ast::Expr::CompoundIdentifier(idents) => {
                        self.ident_normalizer.normalize(
                            idents
                                .last()
                                .ok_or_else(|| {
                                    datafusion_common::plan_datafusion_err!(
                                        "PIVOT value column is required"
                                    )
                                })?
                                .clone(),
                        )
                    }
                    other => {
                        return plan_err!("Unsupported PIVOT value column: {other}");
                    }
                };
                let pivot_column = Column::new(None::<&str>, column_name);
                let default_on_null_expr = default_on_null
                    .map(|expr| {
                        self.sql_expr_to_logical_expr(
                            expr,
                            input_plan.schema(),
                            planner_context,
                        )
                    })
                    .transpose()?;

                let PivotValueSource::List(values) = value_source else {
                    return plan_err!(
                        "PIVOT ANY and subquery values must be resolved before planning"
                    );
                };
                let pivot_values = values
                    .into_iter()
                    .map(|value| {
                        match self.sql_expr_to_logical_expr(
                            value.expr,
                            input_plan.schema(),
                            planner_context,
                        )? {
                            Expr::Literal(value, _) => Ok(value),
                            _ => plan_err!("PIVOT values must be literals"),
                        }
                    })
                    .collect::<Result<Vec<_>>>()?;
                let plan = transform_pivot_to_aggregate(
                    input_plan,
                    &aggregate_expr,
                    &pivot_column,
                    &pivot_values,
                    default_on_null_expr.as_ref(),
                )?;
                (plan, alias)
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
                let value_column = value.to_string();
                let name_column = name.value;
                let mut unpivot_column_indices = Vec::new();
                let mut unpivot_column_names = Vec::new();
                let mut common_type = None;

                for column_ident in columns {
                    let column_name = column_ident.expr.to_string();
                    let Some(index) =
                        base_schema.index_of_column_by_name(None, &column_name)
                    else {
                        return plan_err!("Column '{column_name}' not found in input");
                    };
                    let field_type = base_schema.field(index).data_type();
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
                    unpivot_column_indices.push(index);
                    unpivot_column_names.push(column_name);
                }

                if unpivot_column_names.is_empty() {
                    return plan_err!("UNPIVOT requires at least one column to unpivot");
                }
                let non_pivot_exprs = base_schema
                    .fields()
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| !unpivot_column_indices.contains(index))
                    .map(|(_, field)| Expr::Column(Column::from_name(field.name())))
                    .collect::<Vec<_>>();
                let mut union_inputs = Vec::with_capacity(unpivot_column_names.len());

                for column_name in unpivot_column_names {
                    let mut projection_exprs = non_pivot_exprs.clone();
                    projection_exprs.push(
                        Expr::Literal(
                            ScalarValue::Utf8(Some(column_name.to_uppercase())),
                            None,
                        )
                        .alias(name_column.clone()),
                    );
                    projection_exprs.push(
                        Expr::Column(Column::from_name(&column_name))
                            .alias(value_column.clone()),
                    );
                    let mut builder = LogicalPlanBuilder::from(base_plan.clone())
                        .project(projection_exprs)?;
                    if matches!(null_inclusion, None | Some(NullInclusion::ExcludeNulls))
                    {
                        builder = builder.filter(Expr::IsNotNull(Box::new(
                            Expr::Column(Column::from_name(&value_column)),
                        )))?;
                    }
                    union_inputs.push(builder.build()?);
                }

                let first = union_inputs.remove(0);
                let plan = union_inputs
                    .into_iter()
                    .try_fold(LogicalPlanBuilder::from(first), |builder, input| {
                        builder.union(input)
                    })?
                    .build()?;
                (plan, alias)
            }
            // @todo Support TableFactory::TableFunction?
            _ => {
                return not_impl_err!(
                    "Unsupported ast node {relation:?} in create_relation"
                );
            }
        };
        Ok(PlannedRelation::new(plan, alias))
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
        let outer_query_schema = planner_context.pop_outer_query_schema();
        let new_query_schema = match outer_query_schema {
            Some(ref old_query_schema) => {
                let mut new_query_schema = old_from_schema.as_ref().clone();
                new_query_schema.merge(old_query_schema.as_ref());
                Arc::new(new_query_schema)
            }
            None => Arc::clone(&old_from_schema),
        };
        planner_context.append_outer_query_schema(new_query_schema);

        let plan = self.create_relation(subquery, planner_context)?;
        let outer_ref_columns = plan.all_out_ref_exprs();

        planner_context.pop_outer_query_schema();
        if let Some(schema) = outer_query_schema {
            planner_context.append_outer_query_schema(schema);
        }
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

fn transform_pivot_to_aggregate(
    input: LogicalPlan,
    aggregate_expr: &Expr,
    pivot_column: &Column,
    pivot_values: &[ScalarValue],
    default_on_null_expr: Option<&Expr>,
) -> Result<LogicalPlan> {
    let input_schema = input.schema();
    let group_by = input_schema
        .columns()
        .into_iter()
        .filter(|column| {
            column.name != pivot_column.name
                && !aggregate_expr
                    .column_refs()
                    .iter()
                    .any(|aggregate_column| aggregate_column.name == column.name)
        })
        .map(Expr::Column)
        .collect::<Vec<_>>();
    let pivot_index = input_schema.index_of_column(pivot_column).map_err(|_| {
        datafusion_common::plan_datafusion_err!(
            "Pivot column '{pivot_column}' does not exist in input schema"
        )
    })?;
    let pivot_type = input_schema.field(pivot_index).data_type().clone();
    let aggregates = pivot_values
        .iter()
        .map(|value| {
            let filter = Expr::BinaryExpr(BinaryExpr::new(
                Box::new(Expr::Column(pivot_column.clone())),
                Operator::IsNotDistinctFrom,
                Box::new(Expr::Cast(Cast::new(
                    Box::new(Expr::Literal(value.clone(), None)),
                    pivot_type.clone(),
                ))),
            ));
            let Expr::AggregateFunction(aggregate) = aggregate_expr else {
                return plan_err!("PIVOT expression must be an aggregate function");
            };
            let mut params = aggregate.params.clone();
            params.filter = Some(Box::new(filter));
            let name = value.to_string().trim_matches('\'').to_string();
            Ok(Expr::Alias(Alias {
                expr: Box::new(Expr::AggregateFunction(AggregateFunction {
                    func: Arc::clone(&aggregate.func),
                    params,
                })),
                relation: None,
                name,
                metadata: None,
            }))
        })
        .collect::<Result<Vec<_>>>()?;
    let aggregate_plan = LogicalPlanBuilder::from(input)
        .aggregate(group_by, aggregates)?
        .build()?;

    let Some(default_expr) = default_on_null_expr else {
        return Ok(aggregate_plan);
    };
    let pivot_names = pivot_values
        .iter()
        .map(|value| value.to_string().trim_matches('\'').to_string())
        .collect::<Vec<_>>();
    let projection = aggregate_plan
        .schema()
        .fields()
        .iter()
        .map(|field| {
            let column = Expr::Column(Column::from_name(field.name()));
            if pivot_names.iter().any(|name| name == field.name()) {
                Expr::Alias(Alias {
                    expr: Box::new(Expr::Case(Case {
                        expr: None,
                        when_then_expr: vec![(
                            Box::new(Expr::IsNull(Box::new(column.clone()))),
                            Box::new(default_expr.clone()),
                        )],
                        else_expr: Some(Box::new(column)),
                    })),
                    relation: None,
                    name: field.name().clone(),
                    metadata: None,
                })
            } else {
                column
            }
        })
        .collect::<Vec<_>>();
    LogicalPlanBuilder::from(aggregate_plan)
        .project(projection)?
        .build()
}

fn optimize_subquery_sort(
    plan: LogicalPlan,
    enable_subquery_sort_elimination: bool,
) -> Result<Transformed<LogicalPlan>> {
    if !enable_subquery_sort_elimination {
        return Ok(Transformed::no(plan));
    }

    // When initializing subqueries, we examine sort options since they might be unnecessary.
    // They are only important if the subquery result is affected by the ORDER BY statement,
    // which can happen when we have:
    // 1. DISTINCT ON / ARRAY_AGG ... => Handled by an `Aggregate` and its requirements.
    // 2. RANK / ROW_NUMBER ... => Handled by a `WindowAggr` and its requirements.
    // 3. LIMIT => Handled by a `Sort`, so we need to search for it.
    let mut has_limit = false;

    plan.transform_down(|c| {
        if let LogicalPlan::Limit(_) = c {
            has_limit = true;
            return Ok(Transformed::no(c));
        }
        match c {
            LogicalPlan::Sort(s) => {
                if !has_limit {
                    has_limit = false;
                    return Ok(Transformed::yes(Arc::unwrap_or_clone(s.input)));
                }
                Ok(Transformed::no(LogicalPlan::Sort(s)))
            }
            _ => Ok(Transformed::no(c)),
        }
    })
}
