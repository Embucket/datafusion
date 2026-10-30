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

//! Planning-only inference on prebuilt expressions, not SQL execution or an
//! Arrow 59 release query-performance benchmark. Copy unchanged to 78e3/current
//! for paired fresh-process runs with identical compiler/profile/features.
//! Only equivalent, correct outputs are compared; depth-9 UDF changes are excluded.

use std::collections::HashMap;
use std::hint::black_box;
use std::sync::Arc;

use arrow::datatypes::{DataType, Field, FieldRef};
use datafusion_common::instant::Instant;
use datafusion_common::{DFSchema, Result};
use datafusion_expr::{
    Case, Cast, ColumnarValue, Expr, ExprSchemable, ReturnFieldArgs, ScalarFunctionArgs,
    ScalarUDF, ScalarUDFImpl, Signature, Volatility, col, lit, when,
};

#[derive(Debug, PartialEq, Eq, Hash)]
struct MetadataIdentity {
    signature: Signature,
}

impl ScalarUDFImpl for MetadataIdentity {
    fn name(&self) -> &str {
        "metadata_identity"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Int32)
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        Ok(Arc::clone(&args.arg_fields[0]))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        Ok(args.args[0].clone())
    }
}

fn flat_case(
    width: usize,
    result: impl Fn(usize) -> Expr,
    otherwise: Option<Expr>,
) -> Expr {
    let branches = (0..width)
        .map(|i| (Box::new(lit(i == 0)), Box::new(result(i))))
        .collect();
    Expr::Case(Case::new(None, branches, otherwise.map(Box::new)))
}

fn nested_case(depth: usize, wrap: impl Fn(Expr) -> Expr) -> Expr {
    let mut expr = col("c0");
    for _ in 0..depth {
        expr = when(lit(true), col("c1")).otherwise(wrap(expr)).unwrap();
    }
    expr
}

fn measure(
    name: &str,
    expr: &Expr,
    schema: &DFSchema,
    nullable: bool,
    metadata: &HashMap<String, String>,
) {
    let (_, field) = expr.to_field(schema).unwrap();
    assert_eq!(field.data_type(), &DataType::Int32, "{name}: type");
    assert_eq!(field.is_nullable(), nullable, "{name}: nullable");
    assert_eq!(field.metadata(), metadata, "{name}: metadata");
    drop(field);
    for _ in 0..20 {
        black_box(black_box(expr).to_field(black_box(schema)).unwrap());
    }
    let mut samples = Vec::with_capacity(101);
    for _ in 0..101 {
        let start = Instant::now();
        let result = black_box(black_box(expr).to_field(black_box(schema)));
        let elapsed = start.elapsed();
        // Result checking, destruction, sample storage and JSON are not timed.
        black_box(result.unwrap());
        samples.push(u64::try_from(elapsed.as_nanos()).expect("duration exceeds u64 ns"));
    }
    samples.sort_unstable();
    println!(
        "CASE_FIELD_PERF {}",
        serde_json::json!({
            "name": name,
            "median_ns": samples[samples.len() / 2],
            "min_ns": samples[0],
            "max_ns": samples[samples.len() - 1],
            "samples": samples.len(),
        })
    );
}

#[test]
#[ignore = "planning microbenchmark; paired fresh processes with --ignored --nocapture"]
fn case_field_metadata_perf() {
    eprintln!(
        "CASE_FIELD_PERF_CONTEXT manifest_dir={} source={}",
        env!("CARGO_MANIFEST_DIR"),
        file!()
    );
    let empty = HashMap::new();
    let shared = HashMap::from([("source".to_owned(), "marked".to_owned())]);
    let make_schema = |metadata: &HashMap<String, String>| {
        let fields = (0..64)
            .map(|i| {
                Field::new(format!("c{i}"), DataType::Int32, false)
                    .with_metadata(metadata.clone())
            })
            .collect::<Vec<_>>();
        DFSchema::from_unqualified_fields(fields.into(), HashMap::new()).unwrap()
    };
    let plain = make_schema(&empty);
    let marked = make_schema(&shared);
    // Sizes are fixed: at most 1000 flat branches or 128 linear nesting levels.
    for width in [64, 1000] {
        let expr = flat_case(width, |i| lit(i32::try_from(i).unwrap()), None);
        measure(
            &format!("scalar_literals_{width}"),
            &expr,
            &plain,
            true,
            &empty,
        );
    }
    // Native strict merge: identical branch maps survive; otherwise output is empty.
    for (name, schema, metadata) in [
        ("plain_columns_64", &plain, &empty),
        ("all_marked_columns_64", &marked, &shared),
    ] {
        let expr = flat_case(64, |i| col(format!("c{i}")), Some(col("c0")));
        measure(name, &expr, schema, false, metadata);
    }
    let identity = ScalarUDF::from(MetadataIdentity {
        signature: Signature::exact(vec![DataType::Int32], Volatility::Immutable),
    });
    // Shallow controls stay below 78e3's depth-8 guard; the UDF is never evaluated.
    for depth in [1, 4] {
        let wrap = |mut expr| {
            for _ in 0..depth {
                expr = identity.call(vec![expr]);
            }
            expr
        };
        let expr = when(lit(true), wrap(col("c0")))
            .otherwise(wrap(col("c1")))
            .unwrap();
        measure(
            &format!("shallow_case_udf_{depth}"),
            &expr,
            &marked,
            false,
            &shared,
        );
    }
    for depth in [32, 128] {
        for (name, expr) in [
            (
                "nested_case_aliases",
                nested_case(depth, |e| e.alias("nested")),
            ),
            (
                "nested_case_casts",
                nested_case(depth, |e| {
                    Expr::Cast(Cast::new(Box::new(e), DataType::Int32))
                }),
            ),
        ] {
            measure(&format!("{name}_{depth}"), &expr, &marked, false, &shared);
        }
    }
    for depth in [8, 32, 128] {
        let mut expr = col("c0");
        for _ in 0..depth {
            // Binary arithmetic clears source metadata; ELSE is also unmarked.
            expr = when(lit(true), expr + col("c1"))
                .otherwise(lit(0_i32))
                .unwrap();
        }
        measure(
            &format!("nested_case_binary_{depth}"),
            &expr,
            &marked,
            false,
            &empty,
        );
    }
}
