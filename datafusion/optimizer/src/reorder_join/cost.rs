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

use datafusion_common::{Column, Result, plan_err, stats::Precision};
use datafusion_expr::{Expr, JoinType, LogicalPlan};

use super::join_graph::Edge;

pub trait JoinCostEstimator: std::fmt::Debug {
    /// Cardinality of `plan`.
    ///
    /// - `column = None`: number of output rows of `plan`.
    /// - `column = Some(c)`: number of distinct values of column `c`
    ///   in `plan`'s output (NDV).
    fn cardinality(&self, plan: &LogicalPlan, column: Option<&Column>) -> Option<f64> {
        estimate_cardinality(plan, column).ok()
    }

    /// Estimated selectivity of joining `left` with `right` via `edge`.
    ///
    /// Default: `1 / max(NDV(left.key), NDV(right.key))` for inner equi-joins
    /// when both NDVs are available; otherwise a per-join-type constant.
    fn selectivity(&self, edge: &Edge, left: &LogicalPlan, right: &LogicalPlan) -> f64 {
        let fallback = match edge.join_type {
            JoinType::Inner => 0.1,
            _ => 1.0,
        };
        if edge.join_type != JoinType::Inner || edge.on.is_empty() {
            return fallback;
        }
        // Use only the first equi-pair. Compounding pairwise selectivities
        // under independence assumptions overestimates selectivity when
        // composite-key columns are correlated, which is the common case.
        let (a, b) = &edge.on[0];
        let (Some(col_a), Some(col_b)) = (key_column(a), key_column(b)) else {
            return fallback;
        };
        let ndv_a = ndv_for(self, col_a, left, right);
        let ndv_b = ndv_for(self, col_b, left, right);
        match (ndv_a, ndv_b) {
            (Some(a), Some(b)) if a.max(b) > 0.0 => 1.0 / a.max(b),
            _ => fallback,
        }
    }

    fn cost(&self, selectivity: f64, cardinality: f64) -> f64 {
        selectivity * cardinality
    }
}

/// Default implementation of JoinCostEstimator
#[derive(Debug, Clone, Copy)]
pub struct DefaultCostEstimator;

impl JoinCostEstimator for DefaultCostEstimator {}

fn key_column(expr: &Expr) -> Option<&Column> {
    match expr {
        Expr::Column(c) => Some(c),
        _ => None,
    }
}

/// Look up NDV of `column` on whichever side (left or right) owns it.
fn ndv_for<E: JoinCostEstimator + ?Sized>(
    estimator: &E,
    column: &Column,
    left: &LogicalPlan,
    right: &LogicalPlan,
) -> Option<f64> {
    if left.schema().has_column(column) {
        estimator.cardinality(left, Some(column))
    } else if right.schema().has_column(column) {
        estimator.cardinality(right, Some(column))
    } else {
        None
    }
}

fn estimate_cardinality(plan: &LogicalPlan, column: Option<&Column>) -> Result<f64> {
    match plan {
        LogicalPlan::Filter(filter) => match column {
            None => {
                let input = estimate_cardinality(&filter.input, None)?;
                Ok(0.1 * input)
            }
            Some(c) => {
                // NDV is bounded above by the input's NDV and by the
                // surviving row count.
                let ndv_in = estimate_cardinality(&filter.input, Some(c))?;
                let rows_out = estimate_cardinality(plan, None).unwrap_or(ndv_in);
                Ok(ndv_in.min(rows_out))
            }
        },
        LogicalPlan::Aggregate(agg) => match column {
            None => {
                let input = estimate_cardinality(&agg.input, None)?;
                Ok(0.1 * input)
            }
            Some(c) => {
                // Group-by keys are unique in the aggregate's output, so
                // NDV(group_key) equals the post-aggregate row count.
                let is_group_key = agg.group_expr.iter().any(|e| match e {
                    Expr::Column(g) => g.name == c.name && g.relation == c.relation,
                    _ => false,
                });
                if is_group_key {
                    estimate_cardinality(plan, None)
                } else {
                    plan_err!(
                        "Cannot estimate NDV of non-group-by column \
                         `{}` through Aggregate",
                        c.name
                    )
                }
            }
        },
        LogicalPlan::TableScan(scan) => {
            let stats = scan.source.statistics().ok_or_else(|| {
                datafusion_common::DataFusionError::Plan(format!(
                    "TableSource for `{}` does not expose statistics",
                    scan.table_name
                ))
            })?;
            match column {
                None => match stats.num_rows {
                    Precision::Exact(n) | Precision::Inexact(n) => Ok(n as f64),
                    Precision::Absent => plan_err!(
                        "TableSource for `{}` does not provide a row count",
                        scan.table_name
                    ),
                },
                Some(c) => {
                    // `column_statistics` is indexed by the source schema
                    // (pre-projection), so resolve the column there.
                    let idx = scan.source.schema().index_of(&c.name).map_err(|_| {
                        datafusion_common::DataFusionError::Plan(format!(
                            "Column `{}` not found in source schema of `{}`",
                            c.name, scan.table_name
                        ))
                    })?;
                    let col_stats =
                        stats.column_statistics.get(idx).ok_or_else(|| {
                            datafusion_common::DataFusionError::Plan(format!(
                                "Column statistics missing for index {idx} \
                                 on `{}`",
                                scan.table_name
                            ))
                        })?;
                    match col_stats.distinct_count {
                        Precision::Exact(n) | Precision::Inexact(n) => Ok(n as f64),
                        Precision::Absent => plan_err!(
                            "Column `{}` on `{}` has no distinct-count statistic",
                            c.name,
                            scan.table_name
                        ),
                    }
                }
            }
        }
        x => {
            let inputs = x.inputs();
            if inputs.len() == 1 {
                estimate_cardinality(inputs[0], column)
            } else {
                plan_err!("Cannot estimate cardinality for plan with multiple inputs")
            }
        }
    }
}
