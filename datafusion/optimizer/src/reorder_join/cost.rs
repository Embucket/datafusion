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

/// Fraction of preserved-side rows estimated to survive a semi/anti join
/// when column NDV statistics are unavailable. Mirrors DuckDB's
/// `CardinalityEstimator::DEFAULT_SEMI_ANTI_SELECTIVITY = 1/5`.
const DEFAULT_SEMI_ANTI_SELECTIVITY: f64 = 0.2;

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
    /// Default: `1 / max(NDV(left.key), NDV(right.key))` for equi-joins
    /// (inner and semi/anti) when both NDVs are available; otherwise a
    /// per-join-type constant.
    fn selectivity(&self, edge: &Edge, left: &LogicalPlan, right: &LogicalPlan) -> f64 {
        let fallback = match edge.join_type {
            JoinType::Inner => 0.1,
            JoinType::LeftSemi
            | JoinType::LeftAnti
            | JoinType::RightSemi
            | JoinType::RightAnti => DEFAULT_SEMI_ANTI_SELECTIVITY,
            _ => 1.0,
        };
        let is_eq_join = matches!(
            edge.join_type,
            JoinType::Inner
                | JoinType::LeftSemi
                | JoinType::LeftAnti
                | JoinType::RightSemi
                | JoinType::RightAnti
        );
        if !is_eq_join || edge.on.is_empty() {
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
        match edge.join_type {
            JoinType::Inner => match (ndv_a, ndv_b) {
                (Some(a), Some(b)) if a.max(b) > 0.0 => 1.0 / a.max(b),
                _ => fallback,
            },
            // Semi/anti containment estimator: surviving fraction of the
            // preserved side ≈ `min(NDV_preserved, NDV_filtering) / NDV_preserved`.
            // Edges normalized by `flatten_joins_recursive` always have
            // `on = (preserved_key, filtering_key)`, so the preserved
            // NDV is `ndv_a` for Left{Semi,Anti}. RightSemi/RightAnti
            // shouldn't appear in graph edges (they get normalized) but
            // are handled defensively.
            JoinType::LeftSemi | JoinType::LeftAnti => match (ndv_a, ndv_b) {
                (Some(a), Some(b)) if a > 0.0 => (a.min(b) / a).min(1.0),
                _ => fallback,
            },
            JoinType::RightSemi | JoinType::RightAnti => match (ndv_a, ndv_b) {
                (Some(a), Some(b)) if b > 0.0 => (a.min(b) / b).min(1.0),
                _ => fallback,
            },
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
                // Ungrouped aggregate → exactly 1 row.
                if agg.group_expr.is_empty() {
                    return Ok(1.0);
                }
                let input = estimate_cardinality(&agg.input, None)?;
                // Per-group-key NDV from the child plan, where available.
                // Mirrors duckdb's `ExtractAggregationStats`
                // (relation_statistics_helper.cpp:380-415): start with the
                // product of per-key NDVs, apply a correlation correction,
                // then use the Occupancy-Problem formula to estimate the
                // number of group-key tuples actually occupied given
                // `input` rows.
                let ndvs: Vec<f64> = agg
                    .group_expr
                    .iter()
                    .filter_map(|e| match e {
                        Expr::Column(c) => Some(c),
                        _ => None,
                    })
                    .filter_map(|c| estimate_cardinality(&agg.input, Some(c)).ok())
                    .map(|n| if n <= 0.0 { 1.0 } else { n })
                    .collect();
                if ndvs.is_empty() || ndvs.len() < agg.group_expr.len() {
                    // No (or partial) per-key NDV. Half the input is a
                    // less-pessimistic default than `0.1 * input`, matching
                    // duckdb's fallback at relation_statistics_helper.cpp:394.
                    return Ok((input / 2.0).max(1.0));
                }
                let product: f64 = ndvs.iter().product();
                let correction = 0.95_f64.powi((ndvs.len() as i32) - 1);
                let product = product * correction;
                let mult = 1.0 - (-input / product).exp();
                let new_card = if mult == 0.0 { input } else { product * mult };
                Ok(new_card.min(input).max(1.0))
            }
            Some(c) => {
                // Group-by keys are unique in the aggregate's output, so
                // NDV(group_key) equals the post-aggregate row count.
                // Match by column name only — a SubqueryAlias wrapping the
                // aggregate rewrites the relation prefix, so a strict
                // `relation == relation` comparison would miss legitimate
                // group keys.
                let is_group_key = agg.group_expr.iter().any(|e| match e {
                    Expr::Column(g) => g.name == c.name,
                    _ => false,
                });
                if is_group_key {
                    estimate_cardinality(plan, None)
                } else {
                    // For non-group columns, the post-aggregate NDV is
                    // bounded by the row count (most one distinct value per
                    // output row). Return that as a loose upper bound
                    // instead of erroring, so callers (e.g.
                    // `selectivity()`) can still compute a fallback.
                    estimate_cardinality(plan, None)
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
        // Semi/anti joins do not grow rows: the output cardinality is
        // bounded by the preserved side. We size them via the
        // `DEFAULT_SEMI_ANTI_SELECTIVITY` heuristic. NDV queries on the
        // output route to whichever side is preserved.
        LogicalPlan::Join(j)
            if matches!(
                j.join_type,
                JoinType::LeftSemi
                    | JoinType::LeftAnti
                    | JoinType::RightSemi
                    | JoinType::RightAnti
            ) =>
        {
            let preserved = match j.join_type {
                JoinType::LeftSemi | JoinType::LeftAnti => &j.left,
                _ => &j.right,
            };
            match column {
                None => {
                    let rows = estimate_cardinality(preserved, None)?;
                    Ok(rows * DEFAULT_SEMI_ANTI_SELECTIVITY)
                }
                Some(c) => estimate_cardinality(preserved, Some(c)),
            }
        }
        // Inner joins (and the cross-product, encoded as Inner with empty
        // `on`) appear here when an upstream caller asks about a join
        // subtree that the flattener absorbed as an opaque graph node
        // (e.g. when a projection or other wrapper sits between joins).
        // Estimate via the same NDV-of-the-largest-side formula
        // `selectivity()` uses for inner equi-joins, falling back to 0.1
        // when NDV is unavailable.
        LogicalPlan::Join(j) if j.join_type == JoinType::Inner => {
            let left_card = estimate_cardinality(&j.left, None)?;
            let right_card = estimate_cardinality(&j.right, None)?;
            let cross = left_card * right_card;
            let sel = if let Some((a, b)) = j.on.first() {
                let ndv_max = match (a, b) {
                    (Expr::Column(ca), Expr::Column(cb)) => {
                        let na = estimate_cardinality(&j.left, Some(ca))
                            .ok()
                            .or_else(|| estimate_cardinality(&j.right, Some(ca)).ok());
                        let nb = estimate_cardinality(&j.right, Some(cb))
                            .ok()
                            .or_else(|| estimate_cardinality(&j.left, Some(cb)).ok());
                        match (na, nb) {
                            (Some(x), Some(y)) if x.max(y) > 0.0 => Some(x.max(y)),
                            _ => None,
                        }
                    }
                    _ => None,
                };
                ndv_max.map(|n| 1.0 / n).unwrap_or(0.1)
            } else {
                1.0
            };
            match column {
                None => Ok((sel * cross).max(1.0)),
                Some(c) => {
                    // NDV of a column on the join output is bounded by the
                    // child-side NDV (joins don't create new distinct values
                    // for already-existing columns).
                    estimate_cardinality(&j.left, Some(c))
                        .or_else(|_| estimate_cardinality(&j.right, Some(c)))
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
