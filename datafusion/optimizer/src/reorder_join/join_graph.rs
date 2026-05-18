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

use std::sync::Arc;

use datafusion_common::{
    DataFusionError, NullEquality, Result, plan_datafusion_err, plan_err,
};
use datafusion_expr::{
    Expr, JoinType, LogicalPlan,
    utils::{check_all_columns_from_schema, split_conjunction_owned},
};

pub type NodeId = usize;

pub struct Node {
    pub plan: Arc<LogicalPlan>,
    pub(crate) connections: Vec<EdgeId>,
}

impl Node {
    pub fn connections(&self) -> &[EdgeId] {
        &self.connections
    }

    pub(crate) fn connection_with<'graph>(
        &self,
        node_id: NodeId,
        join_graph: &'graph JoinGraph,
    ) -> Option<&'graph Edge> {
        self.connections
            .iter()
            .filter_map(|edge_id| join_graph.get_edge(*edge_id))
            .find(move |x| x.nodes.contains(&node_id))
    }

    pub fn neighbours(&self, node_id: NodeId, join_graph: &JoinGraph) -> Vec<NodeId> {
        self.connections
            .iter()
            .filter_map(|edge_id| join_graph.get_edge(*edge_id))
            .flat_map(|edge| edge.nodes)
            .filter(|&id| id != node_id)
            .collect()
    }
}

pub type EdgeId = usize;

/// An edge connecting two nodes in the join graph.
///
/// For symmetric edges (`join_type == JoinType::Inner`), the order of
/// `nodes` carries no semantic meaning; either side may end up on the
/// physical left or right after reordering.
///
/// For asymmetric edges (`LeftSemi` / `LeftAnti`, established by
/// `flatten_joins_recursive`), the order is load-bearing: `nodes[0]` is
/// always the preserved (LHS) relation that contributes rows to the
/// output, and `nodes[1]` is always the RHS blob that acts as a filter.
/// `RightSemi` / `RightAnti` joins are normalized to the `Left` variants
/// at extraction time so the optimizer interior only ever sees this
/// orientation. The reconstruction code in
/// `left_deep_join_plan::into_logical_plan` relies on this invariant to
/// orient the rebuilt `Join` correctly.
pub struct Edge {
    pub nodes: [NodeId; 2],
    pub on: Vec<(Expr, Expr)>,
    pub join_type: JoinType,
    pub null_equality: NullEquality,
}

pub struct JoinGraph {
    pub(crate) nodes: VecMap<Node>,
    edges: VecMap<Edge>,
    /// Non-equi predicates hoisted out of decomposed `Join.filter` clauses
    /// and out of `LogicalPlan::Filter` nodes that sit between joins.
    /// The enumerator must reapply these on top of the reordered plan.
    filters: Vec<Expr>,
}

impl JoinGraph {
    pub fn try_from_logical_plan(
        value: LogicalPlan,
    ) -> Result<(JoinGraph, Vec<LogicalPlan>), DataFusionError> {
        // First, extract the join subtree from any wrapper operators
        let (join_subtree, wrappers) = extract_join_subtree(value)?;

        // Now convert only the join subtree to a query graph
        let mut join_graph = JoinGraph::new();
        flatten_joins_recursive(join_subtree, &mut join_graph)?;
        Ok((join_graph, wrappers))
    }

    pub(crate) fn new() -> Self {
        Self {
            nodes: VecMap::new(),
            edges: VecMap::new(),
            filters: Vec::new(),
        }
    }
    pub fn filters(&self) -> &[Expr] {
        &self.filters
    }

    pub(crate) fn add_filter(&mut self, expr: Expr) {
        self.filters.push(expr);
    }

    pub(crate) fn add_node(&mut self, node_data: Arc<LogicalPlan>) -> NodeId {
        self.nodes.insert(Node {
            plan: node_data,
            connections: Vec::new(),
        })
    }

    pub fn add_node_with_edge(
        &mut self,
        other: NodeId,
        node_data: Arc<LogicalPlan>,
        on: Vec<(Expr, Expr)>,
        join_type: JoinType,
        null_equality: NullEquality,
    ) -> Option<NodeId> {
        if self.nodes.contains_key(other) {
            let new_id = self.nodes.insert(Node {
                plan: node_data,
                connections: Vec::new(),
            });
            self.add_edge(new_id, other, on, join_type, null_equality);
            Some(new_id)
        } else {
            None
        }
    }

    fn add_edge(
        &mut self,
        from: NodeId,
        to: NodeId,
        on: Vec<(Expr, Expr)>,
        join_type: JoinType,
        null_equality: NullEquality,
    ) -> Option<EdgeId> {
        if self.nodes.contains_key(from) && self.nodes.contains_key(to) {
            let edge_id = self.edges.insert(Edge {
                nodes: [from, to],
                on,
                join_type,
                null_equality,
            });
            if let Some(from) = self.nodes.get_mut(from) {
                from.connections.push(edge_id);
            }
            if let Some(to) = self.nodes.get_mut(to) {
                to.connections.push(edge_id);
            }
            Some(edge_id)
        } else {
            None
        }
    }

    pub fn remove_node(&mut self, node_id: NodeId) -> Option<Arc<LogicalPlan>> {
        if let Some(node) = self.nodes.remove(node_id) {
            // Remove all edges connected to this node
            for edge_id in &node.connections {
                if let Some(edge) = self.edges.remove(*edge_id) {
                    // Remove the edge from the other node's connections
                    for other_node_id in edge.nodes {
                        if other_node_id != node_id
                            && let Some(other_node) = self.nodes.get_mut(other_node_id)
                        {
                            other_node.connections.retain(|id| id != edge_id);
                        }
                    }
                }
            }
            Some(node.plan)
        } else {
            None
        }
    }

    pub fn remove_edge(&mut self, edge_id: EdgeId) -> Option<Edge> {
        if let Some(edge) = self.edges.remove(edge_id) {
            // Remove the edge from both nodes' connections
            for node_id in edge.nodes {
                if let Some(node) = self.nodes.get_mut(node_id) {
                    node.connections.retain(|id| *id != edge_id);
                }
            }
            Some(edge)
        } else {
            None
        }
    }

    pub(crate) fn nodes(&self) -> impl Iterator<Item = (NodeId, &Node)> {
        self.nodes.iter()
    }

    pub(crate) fn get_node(&self, key: NodeId) -> Option<&Node> {
        self.nodes.get(key)
    }

    pub(crate) fn get_edge(&self, key: EdgeId) -> Option<&Edge> {
        self.edges.get(key)
    }

    /// Returns the id of an edge directly connecting `a` and `b`, if one
    /// exists.
    fn find_edge_between(&self, a: NodeId, b: NodeId) -> Option<EdgeId> {
        let node_a = self.nodes.get(a)?;
        node_a.connections.iter().copied().find(|&eid| {
            self.edges
                .get(eid)
                .map(|e| e.nodes.contains(&b))
                .unwrap_or(false)
        })
    }

    /// Appends `pairs` to the given edge's `on` list.
    fn extend_edge_on(&mut self, edge_id: EdgeId, pairs: Vec<(Expr, Expr)>) {
        if let Some(edge) = self.edges.get_mut(edge_id) {
            edge.on.extend(pairs);
        }
    }

    /// Returns true if a path already connects `from` to `to`, treating
    /// edges as undirected. Used to detect cycles before adding a new
    /// edge; if a path exists, the new edge would close a cycle.
    fn path_exists(&self, from: NodeId, to: NodeId) -> bool {
        use std::collections::HashSet;
        if from == to {
            return true;
        }
        let mut visited: HashSet<NodeId> = HashSet::new();
        let mut stack: Vec<NodeId> = vec![from];
        while let Some(n) = stack.pop() {
            if !visited.insert(n) {
                continue;
            }
            if let Some(node) = self.nodes.get(n) {
                for &eid in &node.connections {
                    if let Some(edge) = self.edges.get(eid) {
                        for &neighbour in &edge.nodes {
                            if neighbour == n {
                                continue;
                            }
                            if neighbour == to {
                                return true;
                            }
                            if !visited.contains(&neighbour) {
                                stack.push(neighbour);
                            }
                        }
                    }
                }
            }
        }
        false
    }
}

/// Extracts the join subtree from a logical plan, separating it from wrapper operators.
///
/// This function traverses the plan tree from the root downward, collecting all non-join
/// operators until it finds the topmost join node. The join subtree (all consecutive joins)
/// is extracted and returned separately from the wrapper operators.
///
/// # Arguments
///
/// * `plan` - The logical plan to extract from
///
/// # Returns
///
/// Returns a tuple of (join_subtree, wrapper_operators) where:
/// - `join_subtree` is the topmost join and all joins beneath it
/// - `wrapper_operators` is a vector of non-join operators above the joins, in order from root to join
///
/// # Errors
///
/// Returns an error if the plan doesn't contain any joins.
pub(crate) fn extract_join_subtree(
    plan: LogicalPlan,
) -> Result<(LogicalPlan, Vec<LogicalPlan>)> {
    let mut wrappers = Vec::new();
    let mut current = plan;
    let original_display = current.display().to_string();

    // Descend through single-input non-join nodes until we find a join.
    // Wrappers that sit *between* joins are no longer rejected here; they are
    // handled inside `flatten_joins_recursive` (absorbed as opaque leaves or,
    // for `Filter` directly above a decomposable join, hoisted to the
    // side-channel). This pass only strips wrappers above the topmost join.
    loop {
        match current {
            LogicalPlan::Join(_) => {
                // Found the join subtree root
                return Ok((current, wrappers));
            }
            other => {
                let inputs = other.inputs();
                if inputs.is_empty() {
                    return plan_err!(
                        "Plan does not contain any join nodes: {}",
                        original_display
                    );
                }
                if inputs.len() != 1 {
                    return plan_err!(
                        "Join extraction only supports single-input operators, found {} inputs in: {}",
                        inputs.len(),
                        other.display()
                    );
                }

                let next = (*inputs[0]).clone();
                wrappers.push(other.clone());
                current = next;
            }
        }
    }
}

/// Reconstructs a logical plan by wrapping an optimized join plan with the original wrapper operators.
///
/// This function takes an optimized join plan and re-applies the wrapper operators (Filter, Sort,
/// Aggregate, etc.) that were removed during extraction. The wrappers are applied in reverse order
/// (innermost to outermost) to reconstruct the original plan structure.
///
/// # Arguments
///
/// * `join_plan` - The optimized join plan to wrap
/// * `wrappers` - Vector of wrapper operators in order from outermost to innermost (root to join)
///
/// # Returns
///
/// Returns the fully reconstructed logical plan with all wrapper operators reapplied.
///
/// # Errors
///
/// Returns an error if reconstructing any wrapper operator fails.
pub fn reconstruct_plan(
    join_plan: LogicalPlan,
    wrappers: Vec<LogicalPlan>,
) -> Result<LogicalPlan> {
    let mut current = join_plan;

    // Apply wrappers in reverse order (from innermost to outermost)
    for wrapper in wrappers.into_iter().rev() {
        // Use with_new_exprs to reconstruct the wrapper with the new input
        current = wrapper.with_new_exprs(wrapper.expressions(), vec![current])?;
    }

    Ok(current)
}

fn flatten_joins_recursive(plan: LogicalPlan, join_graph: &mut JoinGraph) -> Result<()> {
    match plan {
        // Inner joins decompose into the graph. (Cross joins are encoded as
        // Inner with an empty `on` list, which is also handled here: the
        // equi-key loop simply runs zero iterations and the children are
        // joined by absence of edges, matching cross-product connectivity.)
        // The join's `filter` clause is hoisted into the side-channel so the
        // enumerator can reapply it after reordering.
        LogicalPlan::Join(join) if join.join_type == JoinType::Inner => {
            if let Some(filter) = join.filter.clone() {
                for conj in split_conjunction_owned(filter) {
                    join_graph.add_filter(conj);
                }
            }

            flatten_joins_recursive(
                Arc::unwrap_or_clone(Arc::clone(&join.left)),
                join_graph,
            )?;
            flatten_joins_recursive(
                Arc::unwrap_or_clone(Arc::clone(&join.right)),
                join_graph,
            )?;

            // Group each equi-pair by which two nodes it connects. A
            // single `Join.on` can mix pairs that span different node-
            // pairs (e.g. an outer join in a bushy plan whose `on`
            // contains keys from disjoint sub-trees); putting all pairs
            // on every edge produces edges that reference columns
            // missing from their endpoints' schemas, and the resulting
            // multi-edge structure forms a cycle that IK84 can't
            // process.
            use std::collections::HashMap;
            let mut pairs_by_node_pair: HashMap<(NodeId, NodeId), Vec<(Expr, Expr)>> =
                HashMap::new();
            let mut insertion_order: Vec<(NodeId, NodeId)> = Vec::new();

            for (left_key, right_key) in &join.on {
                let left_columns = left_key.column_refs();
                let right_columns = right_key.column_refs();

                let matching_nodes: Vec<NodeId> = join_graph
                    .nodes()
                    .filter_map(|(node_id, node)| {
                        let schema = node.plan.schema();
                        let has_left =
                            check_all_columns_from_schema(&left_columns, schema.as_ref())
                                .unwrap_or(false);
                        let has_right = check_all_columns_from_schema(
                            &right_columns,
                            schema.as_ref(),
                        )
                        .unwrap_or(false);
                        if (has_left && !has_right) || (!has_left && has_right) {
                            Some(node_id)
                        } else {
                            None
                        }
                    })
                    .collect();

                if matching_nodes.len() != 2 {
                    return plan_err!(
                        "Could not find exactly two nodes for join predicate: {} = {} (found {} nodes)",
                        left_key,
                        right_key,
                        matching_nodes.len()
                    );
                }

                let mut endpoints = [matching_nodes[0], matching_nodes[1]];
                endpoints.sort();
                let key = (endpoints[0], endpoints[1]);
                if !pairs_by_node_pair.contains_key(&key) {
                    insertion_order.push(key);
                }
                pairs_by_node_pair
                    .entry(key)
                    .or_default()
                    .push((left_key.clone(), right_key.clone()));
            }

            for (node_a, node_b) in insertion_order {
                let pairs = pairs_by_node_pair.remove(&(node_a, node_b)).unwrap();

                // If a prior recursive call already connected these two
                // nodes by an edge, merge our pairs into it instead of
                // adding a parallel edge.
                if let Some(existing_edge_id) =
                    join_graph.find_edge_between(node_a, node_b)
                {
                    join_graph.extend_edge_on(existing_edge_id, pairs);
                    continue;
                }

                // Cycle check: adding this edge would close a cycle.
                // IK84 needs a tree, so demote the equi-pairs of this
                // group to side-channel filter conjuncts; they'll be
                // re-applied as a Filter above the reordered join.
                if join_graph.path_exists(node_a, node_b) {
                    for (l, r) in pairs {
                        join_graph.add_filter(l.eq(r));
                    }
                    continue;
                }

                join_graph.add_edge(
                    node_a,
                    node_b,
                    pairs,
                    join.join_type,
                    join.null_equality,
                );
            }

            Ok(())
        }
        // Semi/anti joins (Left and Right variants) decompose
        // asymmetrically: the preserved side participates in reordering
        // normally, while the filtering side becomes one opaque blob.
        // The semi/anti edge then connects the LHS node that owns the
        // join key(s) to the blob. This mirrors DuckDB's relation_manager
        // logic at relation_manager.cpp:334-346.
        //
        // Right{Semi,Anti} are normalized to Left{Semi,Anti} here by
        // flipping the join's children and on-keys.
        //
        // If the LHS key(s) span more than one already-extracted sub-
        // relation (multi-LHS semi), the resulting join graph would be
        // cyclic and IK84 cannot handle it. We fall back to opaque in
        // that case.
        LogicalPlan::Join(join)
            if matches!(
                join.join_type,
                JoinType::LeftSemi
                    | JoinType::LeftAnti
                    | JoinType::RightSemi
                    | JoinType::RightAnti
            ) =>
        {
            // Pre-flight multi-LHS guard. We inspect the qualifiers on
            // the LHS key columns *before* recursing into the LHS
            // subtree, because we cannot cleanly undo a recursive flatten.
            // If the LHS key columns reference more than one distinct
            // table qualifier, we treat the whole join as opaque.
            let (lhs_child, rhs_child, semi_join_type, on) = match join.join_type {
                JoinType::RightSemi | JoinType::RightAnti => (
                    Arc::clone(&join.right),
                    Arc::clone(&join.left),
                    join.join_type.swap(),
                    join.on
                        .iter()
                        .map(|(l, r)| (r.clone(), l.clone()))
                        .collect::<Vec<_>>(),
                ),
                _ => (
                    Arc::clone(&join.left),
                    Arc::clone(&join.right),
                    join.join_type,
                    join.on.clone(),
                ),
            };

            // Collect the table qualifiers of every LHS key column. We
            // require exactly one distinct qualifier: zero means at
            // least one column is unqualified (can't safely identify the
            // owner pre-flight), more than one means multi-LHS. Either
            // case falls back to opaque.
            let lhs_qualifiers: std::collections::HashSet<_> = on
                .iter()
                .flat_map(|(lhs_key, _)| lhs_key.column_refs())
                .map(|c| c.relation.clone())
                .collect();
            if lhs_qualifiers.len() != 1 || lhs_qualifiers.contains(&None) {
                join_graph.add_node(Arc::new(LogicalPlan::Join(join)));
                return Ok(());
            }

            // Recurse into the LHS subtree, then attach the blob.
            flatten_joins_recursive(Arc::unwrap_or_clone(lhs_child), join_graph)?;
            let blob_id = join_graph.add_node(rhs_child);

            // Find the single LHS-side node that owns the join key(s).
            // With the single-qualifier guard above this should be a
            // unique owner per key, with all keys agreeing. Any deviation
            // is an internal-invariant error.
            let mut lhs_owner: Option<NodeId> = None;
            for (lhs_key, _) in &on {
                let lhs_cols = lhs_key.column_refs();
                let owners: Vec<NodeId> = join_graph
                    .nodes()
                    .filter(|(id, _)| *id != blob_id)
                    .filter_map(|(id, node)| {
                        check_all_columns_from_schema(
                            &lhs_cols,
                            node.plan.schema().as_ref(),
                        )
                        .unwrap_or(false)
                        .then_some(id)
                    })
                    .collect();
                if owners.len() != 1 {
                    return plan_err!(
                        "Semi/anti join LHS key {} has {} candidate owner(s) \
                         among extracted left-side nodes; expected exactly 1",
                        lhs_key,
                        owners.len()
                    );
                }
                let owner = owners[0];
                match lhs_owner {
                    None => lhs_owner = Some(owner),
                    Some(prev) if prev != owner => {
                        return plan_err!(
                            "Semi/anti join LHS keys span multiple sub-relations after \
                             extraction; this should have been caught by the multi-LHS guard"
                        );
                    }
                    _ => {}
                }
            }
            let lhs_owner = lhs_owner.ok_or_else(|| {
                plan_datafusion_err!(
                    "Semi/anti join has no equi-keys; cannot determine LHS owner"
                )
            })?;

            // Convention: nodes[0] = LHS owner, nodes[1] = blob.
            join_graph.add_edge(
                lhs_owner,
                blob_id,
                on,
                semi_join_type,
                join.null_equality,
            );
            Ok(())
        }
        // Other non-inner joins (Left/Right/Full/Mark) are not freely
        // reorderable, so the entire join subtree becomes one opaque leaf.
        LogicalPlan::Join(join) => {
            join_graph.add_node(Arc::new(LogicalPlan::Join(join)));
            Ok(())
        }
        // A `Filter` directly above a decomposable join is part of the join
        // region: hoist its conjuncts to the side-channel and recurse into
        // the join.
        LogicalPlan::Filter(filter)
            if matches!(
                filter.input.as_ref(),
                LogicalPlan::Join(j) if j.join_type == JoinType::Inner
            ) =>
        {
            for conj in split_conjunction_owned(filter.predicate) {
                join_graph.add_filter(conj);
            }
            let inner = Arc::unwrap_or_clone(filter.input);
            flatten_joins_recursive(inner, join_graph)
        }
        // Anything else (Aggregate, Projection, Sort, Limit, Window, Filter
        // not over a decomposable join, base scans, ...) is absorbed as an
        // opaque leaf. Joins nested inside such a wrapper are intentionally
        // hidden from the enumerator (matches Databend's dphyp behavior).
        other => {
            join_graph.add_node(Arc::new(other));
            Ok(())
        }
    }
}

/// A simple Vec-based map that uses `Option<T>` for sparse storage
/// Keys are never reused once removed
pub(crate) struct VecMap<V>(Vec<Option<V>>);

impl<V> VecMap<V> {
    pub(crate) fn new() -> Self {
        Self(Vec::new())
    }

    pub(crate) fn insert(&mut self, value: V) -> usize {
        let idx = self.0.len();
        self.0.push(Some(value));
        idx
    }

    pub(crate) fn get(&self, key: usize) -> Option<&V> {
        self.0.get(key)?.as_ref()
    }

    pub(crate) fn get_mut(&mut self, key: usize) -> Option<&mut V> {
        self.0.get_mut(key)?.as_mut()
    }

    pub(crate) fn remove(&mut self, key: usize) -> Option<V> {
        self.0.get_mut(key)?.take()
    }

    pub(crate) fn contains_key(&self, key: usize) -> bool {
        self.0.get(key).and_then(|v| v.as_ref()).is_some()
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = (usize, &V)> {
        self.0
            .iter()
            .enumerate()
            .filter_map(|(idx, slot)| slot.as_ref().map(|v| (idx, v)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test::*;
    use datafusion_expr::logical_plan::JoinType;
    use datafusion_expr::{LogicalPlanBuilder, col, lit};
    use datafusion_functions_aggregate::expr_fn::sum;

    fn edge_has_keys(edge: &Edge, key_a: &str, key_b: &str) -> bool {
        edge.on.iter().any(|(l, r)| {
            let s = format!("{l}{r}");
            s.contains(key_a) && s.contains(key_b)
        })
    }

    /// Test converting a three-way join with filter into a JoinGraph
    #[test]
    fn test_try_from_three_way_join_with_filter() -> Result<(), DataFusionError> {
        // Create three-way join: customer JOIN orders JOIN lineitem
        // with a filter on the orders-lineitem join
        let customer = scan_tpch_table("customer");
        let orders = scan_tpch_table("orders");
        let lineitem = scan_tpch_table("lineitem");

        let plan = LogicalPlanBuilder::from(customer.clone())
            .join(
                orders.clone(),
                JoinType::Inner,
                (vec!["c_custkey"], vec!["o_custkey"]),
                None,
            )
            .unwrap()
            .join_with_expr_keys(
                lineitem.clone(),
                JoinType::Inner,
                (vec![col("o_orderkey")], vec![col("l_orderkey")]),
                Some(col("l_quantity").gt(lit(10.0))),
            )
            .unwrap()
            .build()
            .unwrap();

        // Convert to JoinGraph
        let join_graph = JoinGraph::try_from_logical_plan(plan)?.0;

        // Verify structure: 3 nodes, 2 edges
        assert_eq!(join_graph.nodes().count(), 3);
        assert_eq!(join_graph.edges.iter().count(), 2);

        // Verify connectivity: one node has 2 connections (orders), two nodes have 1
        let mut connections: Vec<usize> = join_graph
            .nodes()
            .map(|(_, node)| node.connections().len())
            .collect();
        connections.sort();
        assert_eq!(connections, vec![1, 1, 2]);

        // Verify edges have correct join predicates
        let edges: Vec<&Edge> = join_graph.edges.iter().map(|(_, e)| e).collect();
        assert!(
            edges
                .iter()
                .any(|e| edge_has_keys(e, "c_custkey", "o_custkey")),
            "Missing customer-orders join"
        );
        assert!(
            edges
                .iter()
                .any(|e| edge_has_keys(e, "o_orderkey", "l_orderkey")),
            "Missing orders-lineitem join"
        );

        // The non-equi join.filter should now live in the side-channel
        assert_eq!(join_graph.filters().len(), 1);
        let f = format!("{}", join_graph.filters()[0]);
        assert!(
            f.contains("l_quantity"),
            "expected l_quantity in side-channel filter, got: {f}"
        );

        Ok(())
    }

    /// A Filter sitting between two inner joins should be hoisted to the
    /// side-channel and the joins on either side should still decompose.
    #[test]
    fn test_filter_between_inner_joins() -> Result<(), DataFusionError> {
        let customer = scan_tpch_table("customer");
        let orders = scan_tpch_table("orders");
        let lineitem = scan_tpch_table("lineitem");

        let plan = LogicalPlanBuilder::from(customer)
            .join(
                orders,
                JoinType::Inner,
                (vec!["c_custkey"], vec!["o_custkey"]),
                None,
            )
            .unwrap()
            .filter(col("o_totalprice").gt(lit(100.0)))
            .unwrap()
            .join(
                lineitem,
                JoinType::Inner,
                (vec!["o_orderkey"], vec!["l_orderkey"]),
                None,
            )
            .unwrap()
            .build()
            .unwrap();

        let join_graph = JoinGraph::try_from_logical_plan(plan)?.0;

        assert_eq!(join_graph.nodes().count(), 3);
        assert_eq!(join_graph.edges.iter().count(), 2);
        assert_eq!(join_graph.filters().len(), 1);
        let f = format!("{}", join_graph.filters()[0]);
        assert!(
            f.contains("o_totalprice"),
            "expected o_totalprice in side-channel filter, got: {f}"
        );

        Ok(())
    }

    /// An Aggregate sitting between two inner joins is opaque: the entire
    /// subtree below it becomes one leaf Node and reordering only crosses
    /// the aggregate boundary, not through it.
    #[test]
    fn test_aggregate_between_inner_joins() -> Result<(), DataFusionError> {
        let customer = scan_tpch_table("customer");
        let orders = scan_tpch_table("orders");
        let lineitem = scan_tpch_table("lineitem");

        // (customer JOIN orders) -> Aggregate(group=o_orderkey) -> JOIN lineitem
        let plan = LogicalPlanBuilder::from(customer)
            .join(
                orders,
                JoinType::Inner,
                (vec!["c_custkey"], vec!["o_custkey"]),
                None,
            )
            .unwrap()
            .aggregate(vec![col("o_orderkey")], vec![sum(col("o_totalprice"))])
            .unwrap()
            .join(
                lineitem,
                JoinType::Inner,
                (vec!["o_orderkey"], vec!["l_orderkey"]),
                None,
            )
            .unwrap()
            .build()
            .unwrap();

        let join_graph = JoinGraph::try_from_logical_plan(plan)?.0;

        // Two leaves: the aggregated subtree (opaque) and lineitem.
        assert_eq!(join_graph.nodes().count(), 2);
        assert_eq!(join_graph.edges.iter().count(), 1);
        assert_eq!(join_graph.filters().len(), 0);

        // One leaf must be a LogicalPlan::Aggregate.
        let has_aggregate_leaf = join_graph
            .nodes()
            .any(|(_, n)| matches!(n.plan.as_ref(), LogicalPlan::Aggregate(_)));
        assert!(
            has_aggregate_leaf,
            "expected an opaque-leaf Node whose plan is LogicalPlan::Aggregate"
        );

        Ok(())
    }

    /// A non-inner join nested inside an inner join should be wrapped as one
    /// opaque leaf, leaving only the surrounding inner-join structure visible
    /// to the enumerator.
    #[test]
    fn test_left_join_inside_inner_chain() -> Result<(), DataFusionError> {
        let customer = scan_tpch_table("customer");
        let orders = scan_tpch_table("orders");
        let lineitem = scan_tpch_table("lineitem");

        // (customer LEFT orders) INNER lineitem
        let plan = LogicalPlanBuilder::from(customer)
            .join(
                orders,
                JoinType::Left,
                (vec!["c_custkey"], vec!["o_custkey"]),
                None,
            )
            .unwrap()
            .join(
                lineitem,
                JoinType::Inner,
                (vec!["o_orderkey"], vec!["l_orderkey"]),
                None,
            )
            .unwrap()
            .build()
            .unwrap();

        let join_graph = JoinGraph::try_from_logical_plan(plan)?.0;

        // Two leaves: the LEFT-join subtree (opaque) and lineitem.
        assert_eq!(join_graph.nodes().count(), 2);
        assert_eq!(join_graph.edges.iter().count(), 1);

        // The opaque-leaf node's plan should be a LogicalPlan::Join with Left type.
        let has_left_join_leaf = join_graph.nodes().any(|(_, n)| {
            matches!(
                n.plan.as_ref(),
                LogicalPlan::Join(j) if j.join_type == JoinType::Left
            )
        });
        assert!(
            has_left_join_leaf,
            "expected an opaque-leaf Node whose plan is a LEFT Join"
        );

        Ok(())
    }

    /// If the root of the plan is a non-inner join (and not a semi/anti),
    /// the whole plan collapses into a single opaque leaf.
    #[test]
    fn test_root_non_inner_join_is_single_node() -> Result<(), DataFusionError> {
        let customer = scan_tpch_table("customer");
        let orders = scan_tpch_table("orders");

        let plan = LogicalPlanBuilder::from(customer)
            .join(
                orders,
                JoinType::Left,
                (vec!["c_custkey"], vec!["o_custkey"]),
                None,
            )
            .unwrap()
            .build()
            .unwrap();

        let join_graph = JoinGraph::try_from_logical_plan(plan)?.0;

        assert_eq!(join_graph.nodes().count(), 1);
        assert_eq!(join_graph.edges.iter().count(), 0);
        assert_eq!(join_graph.filters().len(), 0);

        let only_node = join_graph.nodes().next().unwrap().1;
        assert!(matches!(
            only_node.plan.as_ref(),
            LogicalPlan::Join(j) if j.join_type == JoinType::Left
        ));

        Ok(())
    }

    /// LeftSemi joins decompose asymmetrically: the LHS subtree (an
    /// inner-join chain) is flattened normally, the RHS becomes one
    /// opaque blob, and a `LeftSemi` edge connects the LHS node that
    /// owns the join key to the blob.
    #[test]
    fn test_leftsemi_decomposes_asymmetrically() -> Result<(), DataFusionError> {
        let customer = scan_tpch_table("customer");
        let orders = scan_tpch_table("orders");
        let lineitem = scan_tpch_table("lineitem");

        // Subquery returning l_orderkey only.
        let subquery = LogicalPlanBuilder::from(lineitem)
            .aggregate(vec![col("l_orderkey")], vec![sum(col("l_quantity"))])
            .unwrap()
            .project(vec![col("l_orderkey")])
            .unwrap()
            .build()
            .unwrap();

        // LeftSemi(Inner(customer, orders), subquery) on o_orderkey.
        let plan = LogicalPlanBuilder::from(customer)
            .join(
                orders,
                JoinType::Inner,
                (vec!["c_custkey"], vec!["o_custkey"]),
                None,
            )
            .unwrap()
            .join(
                subquery,
                JoinType::LeftSemi,
                (vec!["o_orderkey"], vec!["l_orderkey"]),
                None,
            )
            .unwrap()
            .build()
            .unwrap();

        let join_graph = JoinGraph::try_from_logical_plan(plan)?.0;

        // 3 nodes (customer, orders, blob); 2 edges (inner + semi).
        assert_eq!(join_graph.nodes().count(), 3);
        assert_eq!(join_graph.edges.iter().count(), 2);

        // Locate the semi edge and check orientation: nodes[0] should be
        // the node owning o_orderkey (orders); nodes[1] should be the blob.
        let semi_edges: Vec<&Edge> = join_graph
            .edges
            .iter()
            .map(|(_, e)| e)
            .filter(|e| e.join_type == JoinType::LeftSemi)
            .collect();
        assert_eq!(semi_edges.len(), 1);

        let lhs_node = join_graph.get_node(semi_edges[0].nodes[0]).unwrap();
        let blob_node = join_graph.get_node(semi_edges[0].nodes[1]).unwrap();
        let lhs_field_names: Vec<&str> = lhs_node
            .plan
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .collect();
        assert!(
            lhs_field_names.contains(&"o_orderkey"),
            "semi edge nodes[0] should own o_orderkey, got fields: {lhs_field_names:?}"
        );
        let blob_field_names: Vec<&str> = blob_node
            .plan
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .collect();
        assert_eq!(blob_field_names, vec!["l_orderkey"]);

        Ok(())
    }

    /// RightSemi joins are normalized to LeftSemi by flipping the join's
    /// children at extraction time, so the optimizer interior only ever
    /// sees Left{Semi,Anti}.
    #[test]
    fn test_rightsemi_normalized_to_leftsemi() -> Result<(), DataFusionError> {
        let customer = scan_tpch_table("customer");
        let lineitem = scan_tpch_table("lineitem");
        let orders = scan_tpch_table("orders");

        let subquery = LogicalPlanBuilder::from(lineitem)
            .aggregate(vec![col("l_orderkey")], vec![sum(col("l_quantity"))])
            .unwrap()
            .project(vec![col("l_orderkey")])
            .unwrap()
            .build()
            .unwrap();

        // RightSemi: the subquery is the LHS (filtering side), the
        // inner-join chain is the RHS (preserved side).
        let plan = LogicalPlanBuilder::from(subquery)
            .join(
                LogicalPlanBuilder::from(customer)
                    .join(
                        orders,
                        JoinType::Inner,
                        (vec!["c_custkey"], vec!["o_custkey"]),
                        None,
                    )
                    .unwrap()
                    .build()
                    .unwrap(),
                JoinType::RightSemi,
                (vec!["l_orderkey"], vec!["o_orderkey"]),
                None,
            )
            .unwrap()
            .build()
            .unwrap();

        let join_graph = JoinGraph::try_from_logical_plan(plan)?.0;

        assert_eq!(join_graph.nodes().count(), 3);
        assert_eq!(join_graph.edges.iter().count(), 2);

        let semi_edges: Vec<&Edge> = join_graph
            .edges
            .iter()
            .map(|(_, e)| e)
            .filter(|e| matches!(e.join_type, JoinType::LeftSemi | JoinType::RightSemi))
            .collect();
        assert_eq!(semi_edges.len(), 1);
        assert_eq!(
            semi_edges[0].join_type,
            JoinType::LeftSemi,
            "RightSemi should have been normalized to LeftSemi"
        );

        Ok(())
    }

    /// A LeftSemi whose LHS keys span more than one already-extracted
    /// sub-relation cannot be represented as a tree-edge in the join
    /// graph (IK84 requires a tree). The multi-LHS guard falls back to
    /// treating the entire join as one opaque leaf.
    #[test]
    fn test_multi_lhs_leftsemi_falls_back_to_opaque() -> Result<(), DataFusionError> {
        let customer = scan_tpch_table("customer");
        let orders = scan_tpch_table("orders");

        // Subquery that exposes two columns (one matching customer,
        // one matching orders).
        let subquery = LogicalPlanBuilder::from(scan_tpch_table("orders"))
            .project(vec![col("o_custkey"), col("o_orderkey")])
            .unwrap()
            .build()
            .unwrap();

        // LeftSemi where LHS key spans BOTH customer AND orders:
        //   (Inner(customer, orders))  LEFTSEMI  subquery
        //     ON customer.c_custkey = subquery.o_custkey
        //    AND orders.o_orderkey  = subquery.o_orderkey
        let plan = LogicalPlanBuilder::from(customer)
            .join(
                orders,
                JoinType::Inner,
                (vec!["c_custkey"], vec!["o_custkey"]),
                None,
            )
            .unwrap()
            .join(
                subquery,
                JoinType::LeftSemi,
                (
                    vec!["c_custkey", "o_orderkey"],
                    vec!["o_custkey", "o_orderkey"],
                ),
                None,
            )
            .unwrap()
            .build()
            .unwrap();

        let join_graph = JoinGraph::try_from_logical_plan(plan)?.0;

        // Multi-LHS guard: the entire join becomes one opaque node.
        assert_eq!(
            join_graph.nodes().count(),
            1,
            "multi-LHS LeftSemi should collapse to one opaque node"
        );
        assert_eq!(join_graph.edges.iter().count(), 0);

        let only_node = join_graph.nodes().next().unwrap().1;
        assert!(matches!(
            only_node.plan.as_ref(),
            LogicalPlan::Join(j) if j.join_type == JoinType::LeftSemi
        ));

        Ok(())
    }
}
