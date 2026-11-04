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

//! Planning-only screen for equivalent, unannotated CASE simplification paths.
//! Run on both revisions with the same compiler/profile and interleaved order.
//! Each invocation is timed separately; cloning and result checks are untimed.
//! Clock overhead remains included. These are not query execution measurements.

use std::collections::HashMap;
use std::hint::black_box;
use std::sync::Arc;

use arrow::datatypes::{DataType, Field};
use datafusion_common::instant::Instant;
use datafusion_common::{DFSchema, DataFusionError, Result, ScalarValue};
use datafusion_expr::{Case, Expr, ExprSchemable, col, lit};
use datafusion_optimizer::simplify_expressions::{ExprSimplifier, SimplifyContext};

fn case(branches: Vec<(Expr, Expr)>, otherwise: Option<Expr>) -> Expr {
    Expr::Case(Case::new(
        None,
        branches
            .into_iter()
            .map(|(when, then)| (Box::new(when), Box::new(then)))
            .collect(),
        otherwise.map(Box::new),
    ))
}

fn native_field_samples(iterations: u32) -> Result<()> {
    let marked = HashMap::from([("marker".to_owned(), "kept".to_owned())]);
    let schema = DFSchema::from_unqualified_fields(
        vec![
            Field::new("plain", DataType::Int64, true),
            Field::new("marked", DataType::Int64, true).with_metadata(marked.clone()),
        ]
        .into(),
        Default::default(),
    )?;
    for (column, metadata) in [("plain", HashMap::new()), ("marked", marked)] {
        for count in [1, 64, 256] {
            let expression = case(vec![(lit(false), col(column)); count], None);
            let expected = (
                None,
                Arc::new(
                    Field::new(
                        expression.schema_name().to_string(),
                        DataType::Int64,
                        true,
                    )
                    .with_metadata(metadata.clone()),
                ),
            );
            for _ in 0..256 {
                assert_eq!(expression.to_field(&schema)?, expected);
            }
            for repetition in 0..7 {
                let mut elapsed = 0_u128;
                for _ in 0..iterations {
                    let input = expression.clone();
                    let started = Instant::now();
                    let result = black_box(&input).to_field(black_box(&schema))?;
                    elapsed += started.elapsed().as_nanos();
                    assert_eq!(result, expected);
                    black_box(result);
                }
                println!("native_{column}_{count},{repetition},{iterations},{elapsed}");
            }
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let iterations = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "10000".to_owned())
        .parse::<u32>()
        .map_err(|error| DataFusionError::External(Box::new(error)))?;
    if !(1..=100_000).contains(&iterations) {
        return datafusion_common::exec_err!("iterations must be between 1 and 100000");
    }
    let schema = Arc::new(DFSchema::from_unqualified_fields(
        vec![
            Field::new("flag", DataType::Boolean, true),
            Field::new("value", DataType::Int64, true),
        ]
        .into(),
        Default::default(),
    )?);
    let simplifier =
        ExprSimplifier::new(SimplifyContext::builder().with_schema(schema).build());
    let dynamic = case(vec![(col("flag"), col("value"))], None);
    let mut cases = vec![
        (
            "scalar_false",
            case(vec![(lit(false), lit(42_i64))], None),
            lit(ScalarValue::Int64(None)),
        ),
        (
            "false_else_column",
            case(vec![(lit(false), col("value"))], Some(col("value"))),
            col("value"),
        ),
        (
            "true_column",
            case(vec![(lit(true), col("value"))], None),
            col("value"),
        ),
        ("dynamic_column", dynamic.clone(), dynamic.clone()),
    ];
    for count in [1, 64, 256] {
        let false_branches = vec![(lit(false), col("value")); count];
        cases.push((
            match count {
                1 => "false_column_1",
                64 => "false_column_64",
                _ => "false_column_256",
            },
            case(false_branches.clone(), None),
            lit(ScalarValue::Int64(None)),
        ));
        let mut mixed = false_branches;
        mixed.push((col("flag"), col("value")));
        cases.push((
            match count {
                1 => "false_prefix_1",
                64 => "false_prefix_64",
                _ => "false_prefix_256",
            },
            case(mixed, None),
            dynamic.clone(),
        ));
    }
    println!("case,repetition,iterations,elapsed_ns");
    for (name, expression, expected) in cases {
        for _ in 0..256 {
            assert_eq!(simplifier.simplify(expression.clone())?, expected);
        }
        for repetition in 0..7 {
            let mut elapsed = 0_u128;
            for _ in 0..iterations {
                let input = expression.clone();
                let started = Instant::now();
                let result = black_box(&simplifier).simplify(black_box(input))?;
                elapsed += started.elapsed().as_nanos();
                assert_eq!(result, expected);
                black_box(result);
            }
            println!("{name},{repetition},{iterations},{elapsed}");
        }
    }
    native_field_samples(iterations)
}
