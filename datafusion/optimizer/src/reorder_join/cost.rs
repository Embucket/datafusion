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

use datafusion_common::{Result, plan_err};
use datafusion_expr::{JoinType, LogicalPlan};

use super::join_graph::Edge;

pub trait JoinCostEstimator: std::fmt::Debug {
    fn cardinality(&self, plan: &LogicalPlan) -> Option<f64> {
        estimate_cardinality(plan).ok()
    }

    fn selectivity(&self, edge: &Edge) -> f64 {
        match edge.join_type {
            JoinType::Inner => 0.1,
            _ => 1.0,
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

fn estimate_cardinality(plan: &LogicalPlan) -> Result<f64> {
    match plan {
        LogicalPlan::Filter(filter) => {
            let input_cardinality = estimate_cardinality(&filter.input)?;
            Ok(0.1 * input_cardinality)
        }
        LogicalPlan::Aggregate(agg) => {
            let input_cardinality = estimate_cardinality(&agg.input)?;
            Ok(0.1 * input_cardinality)
        }
        LogicalPlan::TableScan(_) => {
            // The logical-plan-level `TableSource` trait doesn't expose row
            // statistics. To use cardinalities for base relations, override
            // `JoinCostEstimator::cardinality`.
            plan_err!(
                "Default JoinCostEstimator cannot size TableScan; \
                 override `cardinality` to provide statistics"
            )
        }
        x => {
            let inputs = x.inputs();
            if inputs.len() == 1 {
                estimate_cardinality(inputs[0])
            } else {
                plan_err!("Cannot estimate cardinality for plan with multiple inputs")
            }
        }
    }
}
