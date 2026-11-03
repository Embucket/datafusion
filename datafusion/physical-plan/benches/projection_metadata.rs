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

//! Column-only projection chains, runnable unchanged before/after metadata collapse.

use std::collections::HashMap;
use std::hint::black_box;
use std::sync::Arc;
use std::time::Duration;

use arrow::array::{ArrayRef, Int64Array};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use criterion::async_executor::FuturesExecutor;
use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use datafusion_common::tree_node::TreeNode;
use datafusion_execution::TaskContext;
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::common::collect;
use datafusion_physical_plan::execution_plan::reset_plan_states;
use datafusion_physical_plan::projection::{
    ProjectionExec, ProjectionExpr, remove_unnecessary_projections,
};
use datafusion_physical_plan::test::TestMemoryExec;
use futures::executor::block_on;

fn schema(width: usize, layer: Option<usize>) -> SchemaRef {
    let fields: Vec<_> = (0..width)
        .map(|column| {
            let field = Field::new(format!("c{column}"), DataType::Int64, true);
            match layer {
                Some(layer) => field.with_metadata(HashMap::from([(
                    "field-layer".to_string(),
                    format!("{layer}:{column}"),
                )])),
                None => field,
            }
        })
        .collect();
    let metadata = layer
        .map(|layer| HashMap::from([("schema-layer".to_string(), layer.to_string())]))
        .unwrap_or_default();
    Arc::new(Schema::new_with_metadata(fields, metadata))
}

fn fixture(
    rows: usize,
    width: usize,
    depth: usize,
    metadata: bool,
) -> (RecordBatch, Arc<dyn ExecutionPlan>, SchemaRef) {
    // Adjacent duplicate rows and periodic NULLs exercise multiplicity/validity.
    let columns = (0..width)
        .map(|column| {
            Arc::new(Int64Array::from_iter((0..rows).map(|row| {
                ((row / 2 + column) % 31 != 7)
                    .then_some((column * 100 + row / 2 % 17) as i64)
            }))) as ArrayRef
        })
        .collect();
    let input = RecordBatch::try_new(schema(width, None), columns).unwrap();
    let mut plan: Arc<dyn ExecutionPlan> =
        TestMemoryExec::try_new_exec(&[vec![input.clone()]], input.schema(), None)
            .unwrap();
    for layer in 1..=depth {
        let expressions = (0..width).map(|column| ProjectionExpr {
            expr: Arc::new(Column::new(&format!("c{column}"), column)),
            alias: format!("c{column}"),
        });
        plan = Arc::new(if metadata {
            ProjectionExec::try_new_with_schema_metadata(
                expressions,
                plan,
                schema(width, Some(layer)).as_ref(),
            )
            .unwrap()
        } else {
            ProjectionExec::try_new(expressions, plan).unwrap()
        });
    }
    // Expected output comes from the input and explicit outer metadata, not a plan.
    (input, plan, schema(width, metadata.then_some(depth)))
}

fn optimize(plan: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
    plan.transform_down(remove_unnecessary_projections)
        .unwrap()
        .data
}

fn validate(
    plan: &Arc<dyn ExecutionPlan>,
    input: &RecordBatch,
    expected: &Schema,
    context: &Arc<TaskContext>,
) {
    assert_eq!(plan.schema().as_ref(), expected);
    let batches =
        block_on(collect(plan.execute(0, Arc::clone(context)).unwrap())).unwrap();
    let mut offset = 0;
    for batch in batches {
        assert_eq!(batch.schema().as_ref(), expected);
        for (actual, source) in batch.columns().iter().zip(input.columns()) {
            let actual = actual.as_any().downcast_ref::<Int64Array>().unwrap();
            let source = source.as_any().downcast_ref::<Int64Array>().unwrap();
            assert_eq!(actual, &source.slice(offset, batch.num_rows()));
        }
        offset += batch.num_rows();
    }
    assert_eq!(offset, input.num_rows());
}

fn projection_metadata(c: &mut Criterion) {
    let mut group = c.benchmark_group("projection_metadata");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(200));
    group.measurement_time(Duration::from_secs(1));
    // One ~1 MiB fixture is live; all plans share its Arrow buffers.
    for (rows, width, depth) in [(1, 1, 1), (8192, 1, 8), (8192, 16, 8)] {
        for metadata in [false, true] {
            let (input, original, expected) = fixture(rows, width, depth, metadata);
            let optimized = optimize(Arc::clone(&original));
            let context = Arc::new(TaskContext::default());
            validate(&original, &input, expected.as_ref(), &context);
            validate(&optimized, &input, expected.as_ref(), &context);
            let mode = if metadata { "overrides" } else { "unchanged" };
            let name = format!("{mode}/rows{rows}_width{width}_depth{depth}");
            for (phase, plan, apply_rule) in [
                ("optimize_and_collect", &original, true),
                ("execute_already_optimized", &optimized, false),
            ] {
                group.bench_function(BenchmarkId::new(phase, &name), |b| {
                    b.to_async(FuturesExecutor).iter_batched(
                        // Untimed: fresh metrics on the prebuilt topology; otherwise
                        // repeated execute() appends metrics without bound.
                        || reset_plan_states(Arc::clone(plan)).unwrap(),
                        |plan| {
                            let context = Arc::clone(&context);
                            async move {
                                let running = Arc::clone(&plan);
                                let running = if apply_rule {
                                    optimize(running)
                                } else {
                                    running
                                };
                                let batches =
                                    collect(running.execute(0, context).unwrap())
                                        .await
                                        .unwrap();
                                // Timed: Arc clones, optional rule, stream setup,
                                // collection/materialization, batch and plan drops.
                                drop(black_box(batches));
                            }
                        },
                        BatchSize::PerIteration,
                    );
                });
            }
        }
    }
    group.finish();
}

criterion_group!(benches, projection_metadata);
criterion_main!(benches);
