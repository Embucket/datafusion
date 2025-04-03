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
    not_impl_err, plan_err, DFSchema, Diagnostic, Result, Span, Spans, TableReference,
    Column, DFSchemaRef, ScalarValue,
};
use datafusion_expr::builder::subquery_alias;
use datafusion_expr::{expr::Unnest, Expr, LogicalPlan, LogicalPlanBuilder};
use datafusion_expr::expr::{AggregateFunction, BinaryExpr, Alias, AggregateFunctionParams};
use datafusion_expr::{Subquery, SubqueryAlias, Operator};
use sqlparser::ast::{FunctionArg, FunctionArgExpr, Spanned, TableFactor};
use arrow::datatypes::Field;

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
                    let tbl_func_name = name.0.first().unwrap().value.to_string();
                    let args = func_args
                        .args
                        .into_iter()
                        .flat_map(|arg| {
                            if let FunctionArg::Unnamed(FunctionArgExpr::Expr(expr)) = arg
                            {
                                self.sql_expr_to_logical_expr(
                                    expr,
                                    &DFSchema::empty(),
                                    planner_context,
                                )
                            } else {
                                plan_err!("Unsupported function argument type: {:?}", arg)
                            }
                        })
                        .collect::<Vec<_>>();
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
                // Process the inner table first
                let input_plan = self.create_relation(*table, planner_context)?;
                
                // Process the aggregate function
                if aggregate_functions.len() != 1 {
                    return plan_err!("PIVOT requires exactly one aggregate function");
                }
                
                // Get the aggregate function expression
                let agg_expr = self.sql_expr_to_logical_expr(
                    aggregate_functions[0].expr.clone(),
                    input_plan.schema(),
                    planner_context,
                )?;
                
                // Process the pivot column (value_column contains identifier parts)
                if value_column.is_empty() {
                    return plan_err!("PIVOT value column is required");
                }
                
                // Convert the identifier vector to a column reference
                let column_name = value_column.last().unwrap().value.clone();
                let pivot_column = Column::new(None::<&str>, column_name);
                
                // Process the pivot values based on PivotValueSource
                // For ANY and SUBQUERY sources, we'll determine values at execution time
                let pivot_values = match &value_source {
                    sqlparser::ast::PivotValueSource::List(exprs) => {
                        // Process each pivot value expression
                        Some(exprs.iter()
                            .map(|expr| {
                                let logical_expr = self.sql_expr_to_logical_expr(
                                    expr.expr.clone(),
                                    input_plan.schema(),
                                    planner_context,
                                )?;
                                
                                // Convert the expression to a scalar value
                                match logical_expr {
                                    Expr::Literal(scalar) => Ok(scalar),
                                    _ => plan_err!("PIVOT values must be literals"),
                                }
                            })
                            .collect::<Result<Vec<_>>>()?)
                    },
                    sqlparser::ast::PivotValueSource::Any(_) => {
                        // ANY means use all distinct values from the column
                        // We'll determine them at execution time
                        None
                    },
                    sqlparser::ast::PivotValueSource::Subquery(_) => {
                        // Subquery means use values from a subquery
                        // For now, support but require concrete values
                        return plan_err!("PIVOT with subquery value source is not supported yet");
                    },
                };
                
                // Option 1: Use our custom Pivot logical plan node
                // For custom operator implementation - not actively using this path yet
                if false {
                    // Create the schema
                    let schema = derive_pivot_schema(input_plan.schema(), &agg_expr, &pivot_column, &pivot_values.as_ref().unwrap_or(&Vec::new()))?;
                    
                    // Create the Pivot logical plan
                    let pivot_plan = LogicalPlan::Pivot(datafusion_expr::Pivot {
                        input: Arc::new(input_plan),
                        aggregate_expr: agg_expr,
                        pivot_column,
                        pivot_values: pivot_values.unwrap_or_default(),
                        schema: Arc::new(schema),
                    });
                    
                    (pivot_plan, alias)
                } else {
                    // Option 2: Transform PIVOT into a standard aggregate plan (better compatibility)
                    let pivot_plan = transform_pivot_to_aggregate(
                        Arc::new(input_plan), 
                        &agg_expr, 
                        &pivot_column, 
                        pivot_values
                    )?;
                    
                    (pivot_plan, alias)
                }
            }
           // @todo Support TableFactory::TableFunction?

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

/// Derive the schema for a pivot operation
/// This creates a schema including:
/// 1. All the original columns from the input (excluding pivot column and aggregate columns)
/// 2. New columns for each pivot value provided
///
/// When executing with ANY or SUBQUERY value sources, additional columns would be 
/// added dynamically during execution.
pub fn derive_pivot_schema(
    input_schema: &DFSchemaRef,
    agg_expr: &Expr,
    pivot_column: &Column,
    pivot_values: &[ScalarValue],
) -> Result<DFSchema> {
    use datafusion_expr::ExprSchemable;
    
    // First, get the expected field type from the aggregate expression
    let field_type = agg_expr.get_type(input_schema.as_ref())?;
    
    // Create a vector for our schema fields, starting with existing fields
    let mut new_fields = Vec::<(Option<TableReference>, Arc<Field>)>::new();


    // Add existing fields with their table references
    // Exclude pivot column and aggregate columns from the output
    for field in input_schema.fields().iter() {
        // Skip the pivot column and the aggregate column
        if field.name() != pivot_column.name() && 
           !agg_expr.column_refs().iter().any(|col| col.name() == field.name()) {
            new_fields.push((None, field.clone()));
        }
    }
    
    // For each explicit pivot value, create a new field
    for value in pivot_values {
        // Format column name as '{value}' to match Snowflake's style
        // We could also use {agg_func_name}_{value} to be more explicit
        let field_name = value.to_string().trim_matches('\'').to_string();
        
        // Create a new column in the schema
        let field = Field::new(field_name, field_type.clone(), true);
        new_fields.push((None, Arc::new(field)));
    }
    
    // Create the new schema with all fields
    DFSchema::new_with_metadata(new_fields, input_schema.metadata().clone())
}

/// Transform a PIVOT operation into a more standard Aggregate + Projection plan
///
/// This follows the DuckDB approach of using filter conditions with aggregates.
/// The general pattern is:
/// 1. For known pivot values, we create a projection that includes "IS NOT DISTINCT FROM" conditions
/// 2. For ANY, we'd have to do a first pass to collect the values (not implemented yet)
/// 3. We then create a hash group by with filtered aggregates for each pivot value
///
/// For example, for SUM(amount) PIVOT(quarter FOR quarter in ('2023_Q1', '2023_Q2')), we create:
/// - SUM(amount) FILTER (WHERE quarter IS NOT DISTINCT FROM '2023_Q1') AS "2023_Q1"
/// - SUM(amount) FILTER (WHERE quarter IS NOT DISTINCT FROM '2023_Q2') AS "2023_Q2"
pub fn transform_pivot_to_aggregate(
    input: Arc<LogicalPlan>,
    aggregate_expr: &Expr,
    pivot_column: &Column,
    pivot_values: Option<Vec<ScalarValue>>,
) -> Result<LogicalPlan> {
    // Get the schema from the input plan
    let df_schema = input.schema();
    
    // Extract all column names from the schema
    let all_columns: Vec<Column> = df_schema.columns();
    
    // Filter to include only columns we want for GROUP BY 
    // (exclude pivot column and aggregate expression columns)
    let group_by_columns: Vec<Expr> = all_columns
        .into_iter()
        .filter(|col| {
            col.name != pivot_column.name
            && !aggregate_expr.column_refs().iter().any(|agg_col| agg_col.name == col.name)
        })
        .map(|col| Expr::Column(col))
        .collect();
    
    // Create builder for our logical plan
    let mut builder = LogicalPlanBuilder::from(Arc::unwrap_or_clone(input));
    
    // If pivot values aren't provided (e.g., with ANY), we need to determine them
    // Currently, we only support explicit pivot values
    let pivot_values = match pivot_values {
        Some(values) => values,
        None => {
            // For ANY, we would need to run a query to get distinct values
            // Placeholder for future implementation
            return plan_err!("Dynamic pivot values (ANY) are not yet implemented");
        }
    };
    
    if pivot_values.is_empty() {
        return plan_err!("PIVOT requires at least one value");
    }
    
    // Create filtered aggregate expressions for each pivot value
    let mut aggregate_exprs = Vec::new();
    
    for value in &pivot_values {
        // Create the filter condition: pivot_column IS NOT DISTINCT FROM value
        let filter_condition = Expr::BinaryExpr(BinaryExpr::new(
            Box::new(Expr::Column(pivot_column.clone())),
            Operator::IsNotDistinctFrom,
            Box::new(Expr::Literal(value.clone()))
        ));
        
        // Create the aggregate function with filter
        let filtered_agg = match aggregate_expr {
            Expr::AggregateFunction(agg) => {
                // Clone the aggregate function but add our filter
                let mut new_params = agg.params.clone();
                new_params.filter = Some(Box::new(filter_condition));
                Expr::AggregateFunction(AggregateFunction {
                    func: agg.func.clone(),
                    params: new_params,
                })
            },
            _ => {
                return plan_err!("Unsupported aggregate expression should always be AggregateFunction");
            }
        };
        
        // Use the pivot value as the column name
        let field_name = value.to_string().trim_matches('\'').to_string();
        let aliased_agg = Expr::Alias(Alias {
            expr: Box::new(filtered_agg),
            relation: None,
            name: field_name,
            metadata: None,
        });
        
        aggregate_exprs.push(aliased_agg);
    }
    
    // Create the aggregate plan with GROUP BY and filtered aggregates
    let aggregate_plan = builder.aggregate(group_by_columns, aggregate_exprs)?.build()?;
    
    Ok(aggregate_plan)
}