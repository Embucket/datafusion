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
use datafusion_common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion_common::{
    Column, DFSchema, DataFusionError, Result, not_impl_err, plan_err,
};
use datafusion_expr::expr::ScalarFunction;
use datafusion_expr::{Expr, ExprSchemable, LogicalPlan, LogicalPlanBuilder, ScalarUDF};
use sqlparser::ast::Values as SQLValues;

enum ObjectSource<'a> {
    Plan(&'a LogicalPlan, usize),
    Expr(&'a Expr, Option<&'a LogicalPlan>),
}

pub(super) fn object_output_has_placeholder(
    source: &LogicalPlan,
    output_index: usize,
) -> Result<bool> {
    object_source_has_placeholder(ObjectSource::Plan(source, output_index))
}

pub(super) fn object_output_is_object_variant_union(
    source: &LogicalPlan,
    output_index: usize,
) -> Result<bool> {
    Ok(object_union_source_kind(source, output_index, 0)? == ObjectUnionKind::Mixed)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ObjectUnionKind {
    Object,
    Variant,
    Null,
    Mixed,
    Other,
}

fn object_union_field_kind(field: &arrow::datatypes::Field) -> ObjectUnionKind {
    let logical_type = field.metadata().get("embucket.snowflake_logical_type");
    if logical_type.is_some_and(|kind| kind == "OBJECT") {
        ObjectUnionKind::Object
    } else if field.extension_type_name() == Some("arrow.parquet.variant")
        && logical_type.is_none_or(|kind| kind != "ARRAY")
    {
        ObjectUnionKind::Variant
    } else if field.data_type() == &DataType::Null {
        ObjectUnionKind::Null
    } else {
        ObjectUnionKind::Other
    }
}

fn object_union_source_kind(
    plan: &LogicalPlan,
    index: usize,
    depth: usize,
) -> Result<ObjectUnionKind> {
    if depth > 256 {
        return plan_err!("OBJECT INSERT source is too deeply nested");
    }
    let passthrough = |input: &LogicalPlan, column: &Column| {
        input
            .schema()
            .maybe_index_of_column(column)
            .map(|index| object_union_source_kind(input, index, depth + 1))
    };
    match plan {
        LogicalPlan::Union(node) => {
            let mut has_object = false;
            let mut has_variant = false;
            for input in &node.inputs {
                match object_union_source_kind(input, index, depth + 1)? {
                    ObjectUnionKind::Object => has_object = true,
                    ObjectUnionKind::Variant => has_variant = true,
                    ObjectUnionKind::Mixed => {
                        has_object = true;
                        has_variant = true;
                    }
                    ObjectUnionKind::Null => {}
                    ObjectUnionKind::Other => return Ok(ObjectUnionKind::Other),
                }
            }
            Ok(match (has_object, has_variant) {
                (true, true) => ObjectUnionKind::Mixed,
                (true, false) => ObjectUnionKind::Object,
                (false, true) => ObjectUnionKind::Variant,
                (false, false) => ObjectUnionKind::Null,
            })
        }
        LogicalPlan::Projection(node) => {
            let Some(mut expr) = node.expr.get(index) else {
                return Ok(ObjectUnionKind::Other);
            };
            while let Expr::Alias(alias) = expr {
                expr = alias.expr.as_ref();
            }
            if let Expr::Column(column) = expr
                && let Some(kind) = passthrough(node.input.as_ref(), column)
            {
                return kind;
            }
            Ok(object_union_field_kind(plan.schema().field(index)))
        }
        LogicalPlan::Filter(node) => {
            object_union_source_kind(&node.input, index, depth + 1)
        }
        LogicalPlan::Limit(node) => {
            object_union_source_kind(&node.input, index, depth + 1)
        }
        LogicalPlan::Sort(node) => {
            object_union_source_kind(&node.input, index, depth + 1)
        }
        LogicalPlan::SubqueryAlias(node) => {
            object_union_source_kind(&node.input, index, depth + 1)
        }
        LogicalPlan::Subquery(node) => {
            object_union_source_kind(&node.subquery, index, depth + 1)
        }
        LogicalPlan::Repartition(node) => {
            object_union_source_kind(&node.input, index, depth + 1)
        }
        LogicalPlan::Distinct(node) => {
            object_union_source_kind(node.input().as_ref(), index, depth + 1)
        }
        LogicalPlan::Join(node) => {
            let column = Column::from(plan.schema().qualified_field(index));
            if let Some(kind) = passthrough(node.left.as_ref(), &column) {
                kind
            } else if let Some(kind) = passthrough(node.right.as_ref(), &column) {
                kind
            } else {
                Ok(ObjectUnionKind::Other)
            }
        }
        LogicalPlan::Aggregate(node) => {
            if let Some(Expr::Column(column)) = node.group_expr.get(index)
                && let Some(kind) = passthrough(node.input.as_ref(), column)
            {
                return kind;
            }
            Ok(object_union_field_kind(plan.schema().field(index)))
        }
        LogicalPlan::Window(node) if index < node.input.schema().fields().len() => {
            object_union_source_kind(&node.input, index, depth + 1)
        }
        _ => Ok(object_union_field_kind(plan.schema().field(index))),
    }
}

fn object_value_has_placeholder(expr: &Expr) -> Result<bool> {
    object_source_has_placeholder(ObjectSource::Expr(expr, None))
}

fn object_source_has_placeholder(source: ObjectSource<'_>) -> Result<bool> {
    let mut pending = vec![(source, 0)];
    while let Some((source, depth)) = pending.pop() {
        if depth > 256 {
            return plan_err!("OBJECT INSERT source is too deeply nested");
        }
        match source {
            ObjectSource::Plan(plan, index) => match plan {
                LogicalPlan::Projection(node) => {
                    let expr = node.expr.get(index).ok_or_else(|| {
                        DataFusionError::Plan(
                            "OBJECT INSERT source column is missing".into(),
                        )
                    })?;
                    pending.push((
                        ObjectSource::Expr(expr, Some(node.input.as_ref())),
                        depth + 1,
                    ));
                }
                LogicalPlan::Values(node) => {
                    for row in &node.values {
                        if let Some(expr) = row.get(index)
                            && !matches!(expr, Expr::Placeholder(_))
                        {
                            pending.push((ObjectSource::Expr(expr, None), depth + 1));
                        }
                    }
                }
                LogicalPlan::Limit(node) => {
                    pending.push((
                        ObjectSource::Plan(node.input.as_ref(), index),
                        depth + 1,
                    ));
                }
                LogicalPlan::Sort(node) => {
                    pending.push((
                        ObjectSource::Plan(node.input.as_ref(), index),
                        depth + 1,
                    ));
                }
                LogicalPlan::Filter(node) => {
                    pending.push((
                        ObjectSource::Plan(node.input.as_ref(), index),
                        depth + 1,
                    ));
                }
                LogicalPlan::SubqueryAlias(node) => {
                    pending.push((
                        ObjectSource::Plan(node.input.as_ref(), index),
                        depth + 1,
                    ));
                }
                LogicalPlan::Subquery(node) => {
                    pending.push((
                        ObjectSource::Plan(node.subquery.as_ref(), index),
                        depth + 1,
                    ));
                }
                LogicalPlan::Repartition(node) => {
                    pending.push((
                        ObjectSource::Plan(node.input.as_ref(), index),
                        depth + 1,
                    ));
                }
                LogicalPlan::Distinct(node) => {
                    pending.push((
                        ObjectSource::Plan(node.input().as_ref(), index),
                        depth + 1,
                    ));
                }
                LogicalPlan::Join(node) => {
                    let column = Column::from(plan.schema().qualified_field(index));
                    if let Some(index) = node.left.schema().maybe_index_of_column(&column)
                    {
                        pending.push((
                            ObjectSource::Plan(node.left.as_ref(), index),
                            depth + 1,
                        ));
                    } else if let Some(index) =
                        node.right.schema().maybe_index_of_column(&column)
                    {
                        pending.push((
                            ObjectSource::Plan(node.right.as_ref(), index),
                            depth + 1,
                        ));
                    } else {
                        return Ok(true);
                    }
                }
                LogicalPlan::Aggregate(node) => {
                    let expr = node.group_expr.get(index).or_else(|| {
                        index
                            .checked_sub(node.group_expr.len())
                            .and_then(|index| node.aggr_expr.get(index))
                    });
                    let Some(expr) = expr else {
                        return Ok(true);
                    };
                    pending.push((
                        ObjectSource::Expr(expr, Some(node.input.as_ref())),
                        depth + 1,
                    ));
                }
                LogicalPlan::Window(node) => {
                    let input_len = node.input.schema().fields().len();
                    if index < input_len {
                        pending.push((
                            ObjectSource::Plan(node.input.as_ref(), index),
                            depth + 1,
                        ));
                    } else if let Some(expr) = node.window_expr.get(index - input_len) {
                        pending.push((
                            ObjectSource::Expr(expr, Some(node.input.as_ref())),
                            depth + 1,
                        ));
                    } else {
                        return Ok(true);
                    }
                }
                LogicalPlan::Union(node) => {
                    pending.extend(node.inputs.iter().map(|input| {
                        (ObjectSource::Plan(input.as_ref(), index), depth + 1)
                    }));
                }
                _ => {}
            },
            ObjectSource::Expr(expr, input) => match expr {
                Expr::Placeholder(_) => return Ok(true),
                Expr::Case(case) => {
                    pending.extend(case.when_then_expr.iter().map(|(_, value)| {
                        (ObjectSource::Expr(value.as_ref(), input), depth + 1)
                    }));
                    pending.extend(
                        case.else_expr
                            .as_deref()
                            .map(|expr| (ObjectSource::Expr(expr, input), depth + 1)),
                    );
                }
                Expr::Column(column) => {
                    if let Some(input) = input
                        && let Some(index) = input.schema().maybe_index_of_column(column)
                    {
                        pending.push((ObjectSource::Plan(input, index), depth + 1));
                    }
                }
                Expr::ScalarSubquery(node) => {
                    pending
                        .push((ObjectSource::Plan(node.subquery.as_ref(), 0), depth + 1));
                }
                _ => {
                    expr.apply_children(|child| {
                        pending.push((ObjectSource::Expr(child, input), depth + 1));
                        Ok(TreeNodeRecursion::Continue)
                    })?;
                }
            },
        }
    }
    Ok(false)
}

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
        let table_schema = planner_context.set_table_schema(None);
        let values = rows
            .into_iter()
            .map(|row| {
                row.content
                    .into_iter()
                    .map(|v| self.sql_to_expr(v, &empty_schema, planner_context))
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>();
        planner_context.set_table_schema(table_schema.clone());
        let mut values = values?;

        let has_target_schema = table_schema.is_some();
        let schema = table_schema.unwrap_or(empty_schema);
        let object_columns = schema
            .fields()
            .iter()
            .enumerate()
            .filter_map(|(index, field)| {
                (field.data_type() == &DataType::Utf8
                    && field
                        .metadata()
                        .get("embucket.snowflake_logical_type")
                        .is_some_and(|value| value == "OBJECT"))
                .then_some(index)
            })
            .collect::<Vec<_>>();
        for row in &values {
            for &index in &object_columns {
                let Some(expr) = row.get(index) else {
                    continue;
                };
                let (_, source_field) = expr.to_field(&schema)?;
                if source_field.data_type() == &DataType::Null
                    && !matches!(expr, Expr::Placeholder(_))
                    && object_value_has_placeholder(expr)?
                {
                    return plan_err!(
                        "Cannot insert an untyped bound expression into OBJECT"
                    );
                }
                if source_field.data_type() != &DataType::Null
                    && source_field.extension_type_name() != Some("arrow.parquet.variant")
                    && !source_field
                        .metadata()
                        .get("embucket.snowflake_logical_type")
                        .is_some_and(|value| value == "OBJECT")
                {
                    plan_err!(
                        "Cannot implicitly convert {} to OBJECT during INSERT; use TO_OBJECT",
                        source_field.data_type()
                    )?;
                }
            }
        }
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
        if array_columns.is_empty() {
            let plan = if schema.fields().is_empty() {
                LogicalPlanBuilder::values(values)?.build()
            } else {
                LogicalPlanBuilder::values_with_schema(values, &schema)?.build()
            };
            return self.normalize_values_column_names(plan?, has_target_schema);
        }
        for row in &values {
            for expr in row {
                let mut has_subquery = false;
                expr.apply(|node| {
                    if matches!(
                        node,
                        Expr::ScalarSubquery(_) | Expr::Exists(_) | Expr::InSubquery(_)
                    ) {
                        has_subquery = true;
                        Ok(TreeNodeRecursion::Stop)
                    } else {
                        Ok(TreeNodeRecursion::Continue)
                    }
                })?;
                if has_subquery {
                    return plan_err!(
                        "Subqueries in ARRAY INSERT VALUES are not supported; use INSERT ... SELECT"
                    );
                }
            }
        }
        let mut value_fields = schema.fields().to_vec();
        let mut per_row_columns = Vec::new();
        for index in array_columns.iter().copied() {
            let mut common_type = None;
            let mut mixed = false;
            for row in &values {
                let Some(expr) = row.get(index) else {
                    mixed = true;
                    break;
                };
                let (_, source_field) = expr.to_field(&schema)?;
                if source_field
                    .metadata()
                    .get("snowflake.structured_array")
                    .is_some_and(|value| value == "true")
                {
                    mixed = true;
                    break;
                }
                let data_type = source_field.data_type().clone();
                if data_type == DataType::Null {
                    continue;
                }
                if common_type
                    .as_ref()
                    .is_some_and(|previous| previous != &data_type)
                {
                    mixed = true;
                    break;
                }
                common_type = Some(data_type);
            }
            let batch_type =
                (!mixed)
                    .then_some(common_type)
                    .flatten()
                    .filter(|data_type| {
                        matches!(
                            data_type,
                            DataType::List(_)
                                | DataType::LargeList(_)
                                | DataType::FixedSizeList(_, _)
                                | DataType::Utf8
                                | DataType::Utf8View
                                | DataType::LargeUtf8
                        )
                    });
            if let Some(data_type) = batch_type {
                // Let the INSERT projection convert homogeneous rows as a batch.
                value_fields[index] = Arc::new(
                    schema
                        .field(index)
                        .as_ref()
                        .clone()
                        .with_data_type(data_type)
                        .with_metadata(HashMap::new()),
                );
            } else {
                per_row_columns.push(index);
            }
        }
        if !per_row_columns.is_empty() {
            let mut conversions: HashMap<&'static str, Arc<ScalarUDF>> = HashMap::new();
            for row in &mut values {
                for &index in &per_row_columns {
                    let Some(expr) = row.get_mut(index) else {
                        continue;
                    };
                    let function_name = if matches!(expr, Expr::Placeholder(_)) {
                        "__rustice_bound_array_to_variant"
                    } else {
                        match expr.get_type(&schema)? {
                            DataType::List(_)
                            | DataType::LargeList(_)
                            | DataType::FixedSizeList(_, _) => "to_variant",
                            DataType::Utf8 | DataType::Utf8View | DataType::LargeUtf8 => {
                                "__rustice_array_to_variant"
                            }
                            _ => continue,
                        }
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
        let values_schema = if per_row_columns.len() == array_columns.len() {
            schema
        } else {
            Arc::new(DFSchema::from_unqualified_fields(
                value_fields.into(),
                Default::default(),
            )?)
        };
        let plan = if values_schema.fields().is_empty() {
            LogicalPlanBuilder::values(values)?.build()
        } else {
            LogicalPlanBuilder::values_with_schema(values, &values_schema)?.build()
        };
        self.normalize_values_column_names(plan?, has_target_schema)
    }

    fn normalize_values_column_names(
        &self,
        mut plan: LogicalPlan,
        has_target_schema: bool,
    ) -> Result<LogicalPlan> {
        if has_target_schema
            || !self.options.enable_ident_normalization
            || !self
                .context_provider
                .options()
                .sql_parser
                .uppercase_unquoted_identifiers
        {
            return Ok(plan);
        }

        if let LogicalPlan::Values(values) = &mut plan {
            let fields = values
                .schema
                .fields()
                .iter()
                .map(|field| {
                    Arc::new(
                        field
                            .as_ref()
                            .clone()
                            .with_name(field.name().to_ascii_uppercase()),
                    )
                })
                .collect::<Vec<_>>();
            values.schema = Arc::new(DFSchema::from_unqualified_fields(
                fields.into(),
                values.schema.metadata().clone(),
            )?);
        }

        Ok(plan)
    }
}
