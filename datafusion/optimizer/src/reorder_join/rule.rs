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

//! `OptimizerRule` wrapper for join reordering.
//!
//! Append this to an `Optimizer`'s rule list (or to a
//! `SessionStateBuilder` via `with_optimizer_rule`) so the IK84 reorder
//! runs *after* `ExtractEquijoinPredicate` has lifted equi-conditions
//! into the joins' `on` clauses. Running it before that point leaves the
//! reorder with empty-`on` cross-products and a disconnected join graph.

use std::sync::Arc;

use datafusion_common::{Result, tree_node::Transformed};
use datafusion_expr::LogicalPlan;

use crate::{OptimizerConfig, OptimizerRule};

use super::{
    cost::{DefaultCostEstimator, JoinCostEstimator},
    left_deep_join_plan::optimal_left_deep_join_plan,
};

/// Optimizer-rule wrapper around [`optimal_left_deep_join_plan`].
#[derive(Debug)]
pub struct ReorderJoinRule {
    estimator: Arc<dyn JoinCostEstimator + Send + Sync>,
}

impl ReorderJoinRule {
    pub fn new(estimator: Arc<dyn JoinCostEstimator + Send + Sync>) -> Self {
        Self { estimator }
    }
}

impl Default for ReorderJoinRule {
    fn default() -> Self {
        Self::new(Arc::new(DefaultCostEstimator))
    }
}

impl OptimizerRule for ReorderJoinRule {
    fn name(&self) -> &str {
        "reorder_join"
    }

    // `optimal_left_deep_join_plan` does its own top-level traversal and
    // short-circuits when the plan has no joins, so we don't want the
    // framework to walk the tree on our behalf.
    fn apply_order(&self) -> Option<crate::optimizer::ApplyOrder> {
        None
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        let before = plan.clone();
        let after = optimal_left_deep_join_plan(plan, self.estimator.as_ref())?;
        // IK84 is deterministic on a stable graph, so a second pass over
        // an already-optimal plan reproduces the same chain and we
        // converge by reporting no change.
        if after == before {
            Ok(Transformed::no(after))
        } else {
            Ok(Transformed::yes(after))
        }
    }
}
