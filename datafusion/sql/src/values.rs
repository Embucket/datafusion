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

use std::collections::HashMap;
use std::sync::Arc;

use crate::planner::{ContextProvider, PlannerContext, SqlToRel};
use arrow::datatypes::DataType;
use datafusion_common::{DFSchema, DataFusionError, Result, not_impl_err};
use datafusion_expr::expr::ScalarFunction;
use datafusion_expr::{Expr, ExprSchemable, LogicalPlan, LogicalPlanBuilder, ScalarUDF};
use sqlparser::ast::Values as SQLValues;

impl<S: ContextProvider> SqlToRel<'_, S> {
    pub(super) fn sql_values_to_plan(
        &self,
        values: SQLValues,
        planner_context: &mut PlannerContext,
    ) -> Result<LogicalPlan> {
        let SQLValues {
            explicit_row: _,
            rows,
            value_keyword,
        } = values;
        if value_keyword {
            return not_impl_err!(
                "`VALUE` keyword not supported. Did you mean `VALUES`?"
            )?;
        }

        let empty_schema = Arc::new(DFSchema::empty());
        let mut values = rows
            .into_iter()
            .map(|row| {
                row.content
                    .into_iter()
                    .map(|v| self.sql_to_expr(v, &empty_schema, planner_context))
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;

        let schema = planner_context.table_schema().unwrap_or(empty_schema);
        let array_columns = schema
            .fields()
            .iter()
            .enumerate()
            .filter_map(|(index, field)| {
                (field.extension_type_name() == Some("arrow.parquet.variant")
                    && field
                        .metadata()
                        .get("embucket.snowflake_logical_type")
                        .is_some_and(|value| value == "ARRAY"))
                .then_some(index)
            })
            .collect::<Vec<_>>();
        if !array_columns.is_empty() {
            let mut conversions: HashMap<&'static str, Arc<ScalarUDF>> = HashMap::new();
            for row in &mut values {
                for &index in &array_columns {
                    let Some(expr) = row.get_mut(index) else {
                        continue;
                    };
                    let function_name = match expr.get_type(&schema)? {
                        DataType::List(_)
                        | DataType::LargeList(_)
                        | DataType::FixedSizeList(_, _) => "to_variant",
                        DataType::Utf8 | DataType::Utf8View | DataType::LargeUtf8 => {
                            "__rustice_array_to_variant"
                        }
                        _ if matches!(expr, Expr::Placeholder(_)) => {
                            "__rustice_array_to_variant"
                        }
                        _ => continue,
                    };
                    let conversion = if let Some(conversion) =
                        conversions.get(function_name)
                    {
                        Arc::clone(conversion)
                    } else {
                        let conversion = self
                            .context_provider
                            .get_function_meta(function_name)
                            .ok_or_else(|| {
                                DataFusionError::Plan(format!(
                                    "{function_name} is required to insert ARRAY into VARIANT storage"
                                ))
                            })?;
                        conversions.insert(function_name, Arc::clone(&conversion));
                        conversion
                    };
                    let input = std::mem::take(expr);
                    *expr = Expr::ScalarFunction(ScalarFunction::new_udf(
                        conversion,
                        vec![input],
                    ));
                }
            }
        }
        if schema.fields().is_empty() {
            LogicalPlanBuilder::values(values)?.build()
        } else {
            LogicalPlanBuilder::values_with_schema(values, &schema)?.build()
        }
    }
}
