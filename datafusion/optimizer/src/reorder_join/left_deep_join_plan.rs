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

use std::{collections::HashSet, fmt::Debug, sync::Arc};

use datafusion_common::{Result, plan_datafusion_err, plan_err};
use datafusion_expr::{Expr, Filter, JoinConstraint, LogicalPlan};

use crate::reorder_join::{
    cost::JoinCostEstimator,
    join_graph::{JoinGraph, NodeId},
};

/// Generates an optimized left-deep join plan from a logical plan using the Ibaraki-Kameda algorithm.
///
/// This function is the main entry point for join reordering optimization. It takes a logical plan
/// that may contain joins along with wrapper operators (filters, sorts, aggregations, etc.) and
/// produces an optimized plan with reordered joins while preserving the wrapper operators.
///
/// # Algorithm Overview
///
/// The optimization process consists of several steps:
///
/// 1. **Extraction**: Separates the join subtree from wrapper operators (filters, sorts, limits, etc.)
/// 2. **Graph Conversion**: Converts the join subtree into a query graph representation where:
///    - Nodes represent base relations (or opaque subtrees containing non-reorderable operators)
///    - Edges represent equi-join conditions between relations
///    - Non-equi predicates are collected in a side-channel filter list
/// 3. **Optimization**: Uses the Ibaraki-Kameda algorithm to find the optimal left-deep join ordering
///    by trying each node as a potential root and selecting the plan with the lowest estimated cost
/// 4. **Reconstruction**: Rebuilds the complete logical plan by reapplying side-channel filters and
///    the wrapper operators on top of the optimized join plan
///
/// # Left-Deep Join Plans
///
/// A left-deep join plan is a join tree where:
/// - Each join has a relation or previous join result on the left side
/// - Each join has a single relation on the right side
/// - This creates a linear "chain" of joins processed left-to-right
///
/// Example: `((A ⋈ B) ⋈ C) ⋈ D` is left-deep, while `(A ⋈ B) ⋈ (C ⋈ D)` is not.
///
/// Left-deep plans are preferred because they:
/// - Allow pipelining of intermediate results
/// - Work well with hash join implementations
/// - Have predictable memory usage patterns
///
/// # Arguments
///
/// * `plan` - The logical plan to optimize. Must contain at least one join node.
/// * `cost_estimator` - Cost estimator for calculating join costs, cardinality, and selectivity.
///   Used to compare different join orderings and select the optimal one.
///
/// # Returns
///
/// Returns a `LogicalPlan` with optimized join ordering. The plan structure is:
/// - Wrapper operators (filters, sorts, etc.) in their original positions
/// - Joins reordered to minimize estimated execution cost
/// - Join semantics preserved (same result set as input plan)
pub fn optimal_left_deep_join_plan(
    plan: LogicalPlan,
    cost_estimator: &dyn JoinCostEstimator,
) -> Result<LogicalPlan> {
    // Convert join subtree to query graph
    let (query_graph, wrappers) = JoinGraph::try_from_logical_plan(plan)?;

    // Optimize the joins
    let mut optimized_joins =
        query_graph_to_optimal_left_deep_join_plan(&query_graph, cost_estimator)?;

    // Reapply side-channel filters (hoisted from `Join.filter` and from
    // `Filter` nodes that sat between joins) as a single Filter on top of
    // the reordered plan.
    if let Some(combined) = query_graph.filters().iter().cloned().reduce(Expr::and) {
        optimized_joins =
            LogicalPlan::Filter(Filter::try_new(combined, Arc::new(optimized_joins))?);
    }

    // Reconstruct the full plan with wrappers
    crate::reorder_join::join_graph::reconstruct_plan(optimized_joins, wrappers)
}

/// Generates an optimized linear join plan from a query graph using the Ibaraki-Kameda algorithm.
///
/// This function finds the optimal join ordering for a query by:
/// 1. Trying each node in the query graph as a potential root
/// 2. For each root, building a precedence tree and optimizing it through normalization/denormalization
/// 3. Selecting the plan with the lowest estimated cost
///
/// The optimization process uses the Ibaraki-Kameda algorithm, which arranges joins to minimize
/// intermediate result sizes by considering both cardinality and cost estimates.
///
/// # Algorithm Steps
///
/// For each candidate root node:
/// 1. **Construction**: Build a precedence tree from the query graph starting at that node
/// 2. **Normalization**: Transform the tree into a chain structure ordered by rank
/// 3. **Denormalization**: Split merged operations back into individual nodes while maintaining chain structure
/// 4. **Cost Comparison**: Compare the resulting plan's cost against the current best
pub fn query_graph_to_optimal_left_deep_join_plan(
    query_graph: &JoinGraph,
    cost_estimator: &dyn JoinCostEstimator,
) -> Result<LogicalPlan> {
    let mut best_graph: Option<PrecedenceTreeNode> = None;

    for (node_id, _) in query_graph.nodes() {
        let mut precedence_graph =
            PrecedenceTreeNode::from_query_graph(query_graph, node_id, cost_estimator)?;
        precedence_graph.normalize();
        precedence_graph.denormalize()?;

        best_graph = match best_graph.take() {
            Some(current) => {
                let new_cost = precedence_graph.cost()?;
                if new_cost < current.cost()? {
                    Some(precedence_graph)
                } else {
                    Some(current)
                }
            }
            None => Some(precedence_graph),
        };
    }

    best_graph
        .ok_or_else(|| plan_datafusion_err!("No valid precedence graph found"))?
        .into_logical_plan(query_graph)
}

#[derive(Debug)]
struct QueryNode {
    node_id: NodeId,
    // T in [IbarakiKameda84]
    selectivity: f64,
    // C in [IbarakiKameda84]
    cost: f64,
}

impl QueryNode {
    fn rank(&self) -> f64 {
        (self.selectivity - 1.0) / self.cost
    }
}

/// A node in the precedence tree for query optimization.
///
/// The precedence tree is a data structure used by the Ibaraki-Kameda algorithm for
/// optimizing join ordering in database queries. It can represent both arbitrary tree
/// structures and linear chain structures (where each node has at most one child).
struct PrecedenceTreeNode<'graph> {
    query_nodes: Vec<QueryNode>,
    children: Vec<PrecedenceTreeNode<'graph>>,
    query_graph: &'graph JoinGraph,
}

impl Debug for PrecedenceTreeNode<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrecedenceTreeNode")
            .field("query_nodes", &self.query_nodes)
            .field("children", &self.children)
            .finish()
    }
}

impl<'graph> PrecedenceTreeNode<'graph> {
    /// Creates a precedence tree from a query graph.
    pub(crate) fn from_query_graph(
        graph: &'graph JoinGraph,
        root_id: NodeId,
        cost_estimator: &dyn JoinCostEstimator,
    ) -> Result<Self> {
        let mut remaining: HashSet<NodeId> = graph.nodes().map(|(x, _)| x).collect();
        remaining.remove(&root_id);
        PrecedenceTreeNode::from_query_node(
            root_id,
            1.0,
            graph,
            &mut remaining,
            cost_estimator,
            true,
        )
    }

    /// Recursively constructs a precedence tree node from a query graph node.
    fn from_query_node(
        node_id: NodeId,
        selectivity: f64,
        query_graph: &'graph JoinGraph,
        remaining: &mut HashSet<NodeId>,
        cost_estimator: &dyn JoinCostEstimator,
        is_root: bool,
    ) -> Result<Self> {
        let node = query_graph
            .get_node(node_id)
            .ok_or_else(|| plan_datafusion_err!("Root node not found"))?;
        let input_cardinality = cost_estimator.cardinality(&node.plan).unwrap_or(1.0);

        let children = node
            .connections()
            .iter()
            .filter_map(|edge_id| {
                let edge = query_graph.get_edge(*edge_id)?;
                let other = edge
                    .nodes
                    .into_iter()
                    .find(|x| *x != node_id && remaining.contains(x))?;

                remaining.remove(&other);
                let child_selectivity = cost_estimator.selectivity(edge);
                Some(PrecedenceTreeNode::from_query_node(
                    other,
                    child_selectivity,
                    query_graph,
                    remaining,
                    cost_estimator,
                    false,
                ))
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(PrecedenceTreeNode {
            query_nodes: vec![QueryNode {
                node_id,
                selectivity: (selectivity * input_cardinality),
                cost: if is_root {
                    0.0
                } else {
                    cost_estimator.cost(selectivity, input_cardinality)
                },
            }],
            children,
            query_graph,
        })
    }

    /// Rank function according to IbarakiKameda84
    fn rank(&self) -> f64 {
        let (cardinality, cost) =
            self.query_nodes
                .iter()
                .fold((1.0, 0.0), |(cardinality, cost), node| {
                    let cost = cost + cardinality * node.cost;
                    let cardinality = cardinality * node.selectivity;
                    (cardinality, cost)
                });
        if cost == 0.0 {
            0.0
        } else {
            (cardinality - 1.0) / cost
        }
    }

    /// Normalizes the precedence tree into a linear chain structure.
    fn normalize(&mut self) {
        match self.children.len() {
            0 => (),
            1 => {
                if self.children[0].rank() < self.rank() {
                    let mut child = self.children.pop().unwrap();
                    self.query_nodes.append(&mut child.query_nodes);
                    self.children = child.children;
                    self.normalize();
                } else {
                    self.children[0].normalize();
                }
            }
            _ => {
                for child in &mut self.children {
                    child.normalize();
                }
                let child = std::mem::take(&mut self.children)
                    .into_iter()
                    .reduce(Self::merge)
                    .unwrap();
                self.children = vec![child];
            }
        }
    }

    /// Merges two precedence tree chains into a single chain.
    fn merge(self, other: PrecedenceTreeNode<'graph>) -> Self {
        let (mut first, second) = if self.rank() < other.rank() {
            (self, other)
        } else {
            (other, self)
        };
        if first.children.is_empty() {
            first.children = vec![second];
        } else {
            first.children = vec![first.children.pop().unwrap().merge(second)];
        }
        first
    }

    /// Denormalizes a normalized precedence tree by splitting merged query nodes.
    fn denormalize(&mut self) -> Result<()> {
        match self.children.len() {
            0 => (),
            1 => self.children[0].denormalize()?,
            _ => return plan_err!("Tree is not normalized"),
        }

        while self.query_nodes.len() > 1 {
            if self.children.is_empty() {
                let highest_rank_idx = self
                    .query_nodes
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.rank().partial_cmp(&b.rank()).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap();

                let node = self.query_nodes.remove(highest_rank_idx);

                self.children.push(PrecedenceTreeNode {
                    query_nodes: vec![node],
                    children: Vec::new(),
                    query_graph: self.query_graph,
                });
            } else {
                let child_id = self.children[0].query_nodes[0].node_id;
                let child_node = self.query_graph.get_node(child_id).unwrap();
                let neighbours = child_node.neighbours(child_id, self.query_graph);

                let highest_rank_idx = self
                    .query_nodes
                    .iter()
                    .enumerate()
                    .filter(|(_, node)| neighbours.contains(&node.node_id))
                    .max_by(|(_, a), (_, b)| a.rank().partial_cmp(&b.rank()).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap();

                let node = self.query_nodes.remove(highest_rank_idx);

                let child = std::mem::replace(
                    &mut self.children[0],
                    PrecedenceTreeNode {
                        query_nodes: vec![node],
                        children: Vec::new(),
                        query_graph: self.query_graph,
                    },
                );
                self.children[0].children = vec![child];
            };
        }
        Ok(())
    }

    /// Converts the precedence tree chain into a DataFusion `LogicalPlan`.
    pub(crate) fn into_logical_plan(
        self,
        query_graph: &JoinGraph,
    ) -> Result<LogicalPlan> {
        let current_node_id = self.query_nodes[0].node_id;
        let mut current_plan = query_graph
            .get_node(current_node_id)
            .ok_or_else(|| plan_datafusion_err!("Node {:?} not found", current_node_id))?
            .plan
            .as_ref()
            .clone();

        let mut processed_nodes = vec![current_node_id];

        let mut current_chain = &self;

        while !current_chain.children.is_empty() {
            let child = &current_chain.children[0];
            let next_node_id = child.query_nodes[0].node_id;

            let next_plan = query_graph
                .get_node(next_node_id)
                .ok_or_else(|| plan_datafusion_err!("Node {:?} not found", next_node_id))?
                .plan
                .as_ref()
                .clone();

            let next_node = query_graph.get_node(next_node_id).ok_or_else(|| {
                plan_datafusion_err!("Node {:?} not found", next_node_id)
            })?;

            let edge = processed_nodes
                .iter()
                .rev()
                .find_map(|&processed_id| {
                    next_node.connection_with(processed_id, query_graph)
                })
                .ok_or_else(|| {
                    plan_datafusion_err!(
                        "No edge found between {:?} and any processed nodes {:?}",
                        next_node_id,
                        processed_nodes
                    )
                })?;

            // Determine if the join order was swapped compared to the original edge.
            // We check whether qualified columns from the equi-join expressions
            // belong to the current (left) or next (right) schema.
            let current_schema = current_plan.schema();
            let next_schema = next_plan.schema();

            let join_order_swapped = if !edge.on.is_empty() {
                let (left_expr, right_expr) = &edge.on[0];
                let left_columns = left_expr.column_refs();
                let right_columns = right_expr.column_refs();

                let column_in_schema = |col: &datafusion_common::Column,
                                        schema: &datafusion_common::DFSchema|
                 -> bool {
                    if let Some(relation) = &col.relation {
                        schema.iter().any(|(qualifier, field)| {
                            qualifier == Some(relation) && field.name() == col.name()
                        })
                    } else {
                        schema.field_with_unqualified_name(&col.name).is_ok()
                    }
                };

                let left_in_current = left_columns
                    .iter()
                    .all(|c| column_in_schema(c, current_schema.as_ref()));
                let right_in_next = right_columns
                    .iter()
                    .all(|c| column_in_schema(c, next_schema.as_ref()));
                let left_in_next = left_columns
                    .iter()
                    .all(|c| column_in_schema(c, next_schema.as_ref()));
                let right_in_current = right_columns
                    .iter()
                    .all(|c| column_in_schema(c, current_schema.as_ref()));

                !(left_in_current && right_in_next) && (left_in_next && right_in_current)
            } else {
                false
            };

            // When the join order is swapped, swap on conditions and adjust
            // the join type (e.g., LeftSemi → RightSemi).
            let (on, join_type) = if join_order_swapped {
                let swapped_on = edge
                    .on
                    .iter()
                    .map(|(left, right)| (right.clone(), left.clone()))
                    .collect();
                (swapped_on, edge.join_type.swap())
            } else {
                (edge.on.clone(), edge.join_type)
            };

            // Build the join. Schema is auto-derived; non-equi predicates were
            // already hoisted into the side-channel and are reapplied at the
            // top level by `optimal_left_deep_join_plan`.
            let join = datafusion_expr::Join::try_new(
                Arc::new(current_plan),
                Arc::new(next_plan),
                on,
                None,
                join_type,
                JoinConstraint::On,
                edge.null_equality,
                false,
            )?;
            current_plan = LogicalPlan::Join(join);

            processed_nodes.push(next_node_id);
            current_chain = child;
        }

        Ok(current_plan)
    }

    fn cost(&self) -> Result<f64> {
        self.cost_recursive(self.query_nodes[0].selectivity, 0.0)
    }

    fn cost_recursive(&self, cardinality: f64, cost: f64) -> Result<f64> {
        let cost = match self.children.len() {
            0 => cost + cardinality * self.query_nodes[0].cost,
            1 => self.children[0].cost_recursive(
                cardinality * self.query_nodes[0].selectivity,
                cost + cardinality * self.query_nodes[0].cost,
            )?,
            _ => {
                return plan_err!(
                    "Cost calculation requires normalized tree with 0 or 1 children"
                );
            }
        };
        Ok(cost)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decorrelate_predicate_subquery::DecorrelatePredicateSubquery;
    use crate::eliminate_filter::EliminateFilter;
    use crate::extract_equijoin_predicate::ExtractEquijoinPredicate;
    use crate::filter_null_join_keys::FilterNullJoinKeys;
    use crate::optimizer::{Optimizer, OptimizerContext};
    use crate::push_down_filter::PushDownFilter;
    use crate::reorder_join::cost::JoinCostEstimator;
    use crate::scalar_subquery_to_join::ScalarSubqueryToJoin;
    use crate::simplify_expressions::SimplifyExpressions;
    use crate::test::*;
    use datafusion_expr::LogicalPlanBuilder;
    use datafusion_expr::logical_plan::JoinType;

    /// A simple cost estimator for testing — uses the trait defaults, which
    /// can't size base TableScans on this branch (TableSource no longer
    /// exposes statistics). The algorithm still runs; it just degrades to
    /// uniform cardinalities for leaves.
    #[derive(Debug)]
    struct TestCostEstimator;

    impl JoinCostEstimator for TestCostEstimator {}

    /// Test three-way join: customer -> orders -> lineitem
    #[test]
    fn test_three_way_join_customer_orders_lineitem() -> Result<()> {
        use datafusion_expr::test::function_stub::sum;
        use datafusion_expr::{col, in_subquery, lit};

        let customer = scan_tpch_table("customer");
        let orders = scan_tpch_table("orders");
        let lineitem = scan_tpch_table("lineitem");

        // Subquery: SELECT l_orderkey FROM lineitem GROUP BY l_orderkey HAVING sum(l_quantity) > 300
        let subquery = LogicalPlanBuilder::from(lineitem.clone())
            .aggregate(vec![col("l_orderkey")], vec![sum(col("l_quantity"))])?
            .filter(sum(col("l_quantity")).gt(lit(300)))?
            .project(vec![col("l_orderkey")])?
            .build()?;

        let plan = LogicalPlanBuilder::from(customer.clone())
            .join(
                orders.clone(),
                JoinType::Inner,
                (vec!["c_custkey"], vec!["o_custkey"]),
                None,
            )?
            .join(
                lineitem.clone(),
                JoinType::Inner,
                (vec!["o_orderkey"], vec!["l_orderkey"]),
                None,
            )?
            .filter(in_subquery(col("o_orderkey"), Arc::new(subquery)))?
            .aggregate(
                vec![
                    col("c_name"),
                    col("c_custkey"),
                    col("o_orderkey"),
                    col("o_totalprice"),
                ],
                vec![sum(col("l_quantity"))],
            )?
            .sort(vec![col("o_totalprice").sort(false, true)])?
            .limit(0, Some(100))?
            .build()?;

        // Run the standard optimizer rules that this pass depends on
        // (decorrelation, equi-join extraction, filter pushdown). We exclude
        // OptimizeProjections so joins remain consecutive.
        let config = OptimizerContext::new().with_skip_failing_rules(false);
        let optimizer = Optimizer::with_rules(vec![
            Arc::new(SimplifyExpressions::new()),
            Arc::new(DecorrelatePredicateSubquery::new()),
            Arc::new(ScalarSubqueryToJoin::new()),
            Arc::new(ExtractEquijoinPredicate::new()),
            Arc::new(EliminateFilter::new()),
            Arc::new(FilterNullJoinKeys::default()),
            Arc::new(PushDownFilter::new()),
        ]);
        let plan = optimizer.optimize(plan, &config, |_, _| {}).unwrap();

        let optimized_plan =
            optimal_left_deep_join_plan(plan, &TestCostEstimator).unwrap();

        assert!(matches!(optimized_plan, LogicalPlan::Limit(_)));

        Ok(())
    }
}
