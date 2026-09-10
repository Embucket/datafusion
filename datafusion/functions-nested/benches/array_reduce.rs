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

use std::{collections::HashMap, sync::Arc};

use arrow::{
    array::{Array, ArrayRef, Int64Array, ListArray, RecordBatch},
    buffer::OffsetBuffer,
    datatypes::{DataType, Field},
};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use datafusion_common::DFSchema;
use datafusion_expr::{
    Expr, col,
    execution_props::ExecutionProps,
    expr::{HigherOrderFunction, LambdaVariable},
    lambda, lit,
    physical_planning_context::PhysicalPlanningContext,
};
use datafusion_functions_nested::array_reduce::array_reduce_higher_order_function;
use datafusion_physical_expr::create_physical_expr;

const NUM_ROWS: usize = 8192;
const LIST_SIZE: usize = 16;

fn list_array(lengths: impl IntoIterator<Item = usize>) -> ListArray {
    let lengths = lengths.into_iter().collect::<Vec<_>>();
    let values_len = lengths.iter().sum();
    let values = [1_i64, 2, 3, 4]
        .into_iter()
        .cycle()
        .take(values_len)
        .collect::<Vec<_>>();
    ListArray::new(
        Arc::new(Field::new_list_field(DataType::Int64, false)),
        OffsetBuffer::from_lengths(lengths),
        Arc::new(Int64Array::from(values)),
        None,
    )
}

fn reduce_expression(
    list: &ListArray,
) -> (Arc<dyn datafusion_physical_expr::PhysicalExpr>, RecordBatch) {
    let schema = DFSchema::from_unqualified_fields(
        vec![Field::new("list", list.data_type().clone(), false)].into(),
        HashMap::new(),
    )
    .unwrap();
    let accumulator = Expr::LambdaVariable(LambdaVariable::new(
        "acc".to_string(),
        Some(Arc::new(Field::new("acc", DataType::Int64, false))),
    ));
    let value = Expr::LambdaVariable(LambdaVariable::new(
        "value".to_string(),
        Some(Arc::new(Field::new("value", DataType::Int64, false))),
    ));
    let expression = Expr::HigherOrderFunction(HigherOrderFunction::new(
        array_reduce_higher_order_function(),
        vec![
            col("list"),
            lit(0_i64),
            lambda(["acc", "value"], accumulator + value),
        ],
    ));
    let physical = create_physical_expr(
        &expression,
        &schema,
        &ExecutionProps::new(),
        &PhysicalPlanningContext::default(),
    )
    .unwrap();
    let batch = RecordBatch::try_new(
        Arc::clone(schema.inner()),
        vec![Arc::new(list.clone()) as ArrayRef],
    )
    .unwrap();
    (physical, batch)
}

fn criterion_benchmark(c: &mut Criterion) {
    let inputs = [
        (
            "uniform",
            list_array(std::iter::repeat_n(LIST_SIZE, NUM_ROWS)),
        ),
        (
            "varying",
            list_array((0..NUM_ROWS).map(|row| match row % 8 {
                0 => 0,
                1 | 2 => 1,
                _ => LIST_SIZE,
            })),
        ),
    ];

    for (name, list) in inputs {
        let (expression, batch) = reduce_expression(&list);
        c.bench_with_input(
            BenchmarkId::new("array_reduce", name),
            &batch,
            |b, batch| b.iter(|| expression.evaluate(batch).unwrap()),
        );
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
