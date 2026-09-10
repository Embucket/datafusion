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

//! [`datafusion_expr::HigherOrderUDF`] definitions for array_reduce function.

use std::sync::Arc;

use arrow::{
    array::{Array, ArrayRef, AsArray, BooleanArray, UInt64Array, new_null_array},
    compute::{kernels::zip::zip, take, take_arrays},
    datatypes::{DataType, Field, FieldRef},
};
use datafusion_common::{
    Result, exec_err, internal_datafusion_err, plan_err,
    utils::{adjust_offsets_for_slice, list_values},
};
use datafusion_expr::{
    ColumnarValue, Documentation, HigherOrderFunctionArgs, HigherOrderReturnFieldArgs,
    HigherOrderSignature, HigherOrderUDFImpl, LambdaParametersProgress, ValueOrLambda,
    Volatility,
};
use datafusion_macros::user_doc;

make_higher_order_function_expr_and_func!(
    ArrayReduce,
    array_reduce,
    array initial merge,
    "reduces an array to a single value using a binary lambda",
    array_reduce_higher_order_function
);

#[user_doc(
    doc_section(label = "Array Functions"),
    description = "Reduces an array to a single value by applying a binary lambda to the accumulator and each array element in order.",
    syntax_example = "array_reduce(array, initial, (acc, value) -> expression)",
    sql_example = r#"```sql
> select array_reduce([1, 2, 3], 0, (acc, value) -> acc + value);
+----------------------------------------------------------------+
| array_reduce([1,2,3],0,(acc,value) -> acc + value)              |
+----------------------------------------------------------------+
| 6                                                              |
+----------------------------------------------------------------+
```"#,
    argument(name = "array", description = "Array expression to reduce."),
    argument(name = "initial", description = "Initial accumulator value."),
    argument(
        name = "merge",
        description = "Binary lambda whose parameters are the accumulator and current array element."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct ArrayReduce {
    signature: HigherOrderSignature,
}

impl Default for ArrayReduce {
    fn default() -> Self {
        Self::new()
    }
}

impl ArrayReduce {
    pub fn new() -> Self {
        Self {
            signature: HigherOrderSignature::exact(
                vec![
                    ValueOrLambda::Value(()),
                    ValueOrLambda::Value(()),
                    ValueOrLambda::Lambda(()),
                ],
                Volatility::Immutable,
            ),
        }
    }
}

impl HigherOrderUDFImpl for ArrayReduce {
    fn name(&self) -> &str {
        "array_reduce"
    }

    fn signature(&self) -> &HigherOrderSignature {
        &self.signature
    }

    fn coerce_value_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        let [list, initial] = arg_types else {
            return plan_err!(
                "{} requires two value arguments, got {}",
                self.name(),
                arg_types.len()
            );
        };

        let list = match list {
            DataType::List(_) | DataType::LargeList(_) => list.clone(),
            DataType::ListView(field) | DataType::FixedSizeList(field, _) => {
                DataType::List(Arc::clone(field))
            }
            DataType::LargeListView(field) => DataType::LargeList(Arc::clone(field)),
            DataType::Null => DataType::new_list(DataType::Null, true),
            other => {
                return plan_err!(
                    "{} expected a list as first argument, got {other}",
                    self.name()
                );
            }
        };

        Ok(vec![list, initial.clone()])
    }

    fn lambda_parameters(
        &self,
        step: usize,
        fields: &[ValueOrLambda<FieldRef, Option<FieldRef>>],
    ) -> Result<LambdaParametersProgress> {
        let [
            ValueOrLambda::Value(list),
            ValueOrLambda::Value(initial),
            ValueOrLambda::Lambda(merge),
        ] = fields
        else {
            return plan_err!(
                "{} expects an array, an initial value, and a lambda",
                self.name()
            );
        };

        let element = match list.data_type() {
            DataType::List(field) | DataType::LargeList(field) => Arc::clone(field),
            other => {
                return plan_err!(
                    "{} expected a list as first argument, got {other}",
                    self.name()
                );
            }
        };

        match (step, merge) {
            (0, None) => Ok(LambdaParametersProgress::Partial(vec![Some(vec![
                Arc::clone(initial),
                element,
            ])])),
            (0 | 1, Some(accumulator)) => Ok(LambdaParametersProgress::Complete(vec![
                vec![Arc::clone(accumulator), element],
            ])),
            _ => Err(internal_datafusion_err!(
                "{} could not resolve its accumulator type at step {step}",
                self.name()
            )),
        }
    }

    fn coerce_values_for_lambdas(
        &self,
        fields: &[ValueOrLambda<DataType, DataType>],
    ) -> Result<Option<Vec<DataType>>> {
        let [
            ValueOrLambda::Value(list),
            ValueOrLambda::Value(_initial),
            ValueOrLambda::Lambda(merge),
        ] = fields
        else {
            return plan_err!(
                "{} expects an array, an initial value, and a lambda",
                self.name()
            );
        };

        Ok(Some(vec![list.clone(), merge.clone()]))
    }

    fn return_field_from_args(
        &self,
        args: HigherOrderReturnFieldArgs,
    ) -> Result<FieldRef> {
        let [
            ValueOrLambda::Value(list),
            ValueOrLambda::Value(initial),
            ValueOrLambda::Lambda(merge),
        ] = args.arg_fields
        else {
            return plan_err!(
                "{} expects an array, an initial value, and a lambda",
                self.name()
            );
        };

        Ok(Arc::new(Field::new(
            "",
            merge.data_type().clone(),
            list.is_nullable() || initial.is_nullable() || merge.is_nullable(),
        )))
    }

    fn invoke_with_args(&self, args: HigherOrderFunctionArgs) -> Result<ColumnarValue> {
        let [list, initial, merge] = args.args.as_slice() else {
            return exec_err!(
                "{} expects an array, an initial value, and a lambda",
                self.name()
            );
        };
        let (
            ValueOrLambda::Value(list),
            ValueOrLambda::Value(initial),
            ValueOrLambda::Lambda(merge),
        ) = (list, initial, merge)
        else {
            return exec_err!(
                "{} expects an array, an initial value, and a lambda",
                self.name()
            );
        };

        let list = list.to_array(args.number_rows)?;
        let values = list_values(list.as_ref())?;
        let offsets: Vec<usize> = match list.data_type() {
            DataType::List(_) => adjust_offsets_for_slice(list.as_list::<i32>())
                .iter()
                .map(|offset| usize::try_from(*offset))
                .collect::<std::result::Result<_, _>>()
                .map_err(|error| {
                    internal_datafusion_err!("invalid list offset: {error}")
                })?,
            DataType::LargeList(_) => adjust_offsets_for_slice(list.as_list::<i64>())
                .iter()
                .map(|offset| usize::try_from(*offset))
                .collect::<std::result::Result<_, _>>()
                .map_err(|error| {
                    internal_datafusion_err!("invalid list offset: {error}")
                })?,
            other => return exec_err!("{} expected a list, got {other}", self.name()),
        };

        let mut accumulator = initial.clone().to_array(args.number_rows)?;
        if list.null_count() > 0 {
            let valid_lists = BooleanArray::from(
                (0..list.len())
                    .map(|row| list.is_valid(row))
                    .collect::<Vec<_>>(),
            );
            let null_accumulator =
                new_null_array(accumulator.data_type(), accumulator.len());
            accumulator = zip(&valid_lists, &accumulator, &null_accumulator)?;
        }

        let max_len = offsets
            .windows(2)
            .map(|pair| pair[1] - pair[0])
            .max()
            .unwrap_or(0);

        for position in 0..max_len {
            let mut source_indices = Vec::with_capacity(list.len());
            let mut row_indices = Vec::with_capacity(list.len());
            let mut scatter_indices = Vec::with_capacity(list.len());

            for row in 0..list.len() {
                let active = accumulator.is_valid(row)
                    && list.is_valid(row)
                    && position < offsets[row + 1] - offsets[row];
                if active {
                    source_indices.push(u64::try_from(offsets[row] + position).map_err(
                        |error| internal_datafusion_err!("invalid list index: {error}"),
                    )?);
                    row_indices.push(u64::try_from(row).map_err(|error| {
                        internal_datafusion_err!("invalid row index: {error}")
                    })?);
                    scatter_indices.push(Some(
                        u64::try_from(source_indices.len() - 1).map_err(|error| {
                            internal_datafusion_err!("invalid scatter index: {error}")
                        })?,
                    ));
                } else {
                    scatter_indices.push(None);
                }
            }

            if source_indices.is_empty() {
                break;
            }

            if source_indices.len() == list.len() {
                let elements =
                    take(values.as_ref(), &UInt64Array::from(source_indices), None)?;
                let accumulator_param = || Ok(Arc::clone(&accumulator));
                let element_param = || Ok(Arc::clone(&elements));
                accumulator = merge
                    .evaluate(&[&accumulator_param, &element_param], |arrays| {
                        Ok(arrays.to_vec())
                    })?
                    .into_array(list.len())?;
                continue;
            }

            let row_indices: ArrayRef = Arc::new(UInt64Array::from(row_indices));
            let active_accumulator = take(accumulator.as_ref(), &row_indices, None)?;
            let elements =
                take(values.as_ref(), &UInt64Array::from(source_indices), None)?;
            let accumulator_param = || Ok(Arc::clone(&active_accumulator));
            let element_param = || Ok(Arc::clone(&elements));
            let merged = merge
                .evaluate(&[&accumulator_param, &element_param], |arrays| {
                    Ok(take_arrays(arrays, &row_indices, None)?)
                })?
                .into_array(row_indices.len())?;

            let scatter_indices = UInt64Array::from(scatter_indices);
            let active_mask = BooleanArray::from(
                scatter_indices
                    .iter()
                    .map(|index| index.is_some())
                    .collect::<Vec<_>>(),
            );
            let expanded = take(merged.as_ref(), &scatter_indices, None)?;
            accumulator = zip(&active_mask, &expanded, &accumulator)?;
        }

        Ok(ColumnarValue::Array(accumulator))
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }
}
