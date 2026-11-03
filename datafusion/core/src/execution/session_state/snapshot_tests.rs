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

use super::*;
use crate::execution::context::SessionContext;
use arrow_schema::{DataType, Field, Schema};
use datafusion_catalog::empty::EmptyTable;
use datafusion_catalog::{TableFunctionArgs, TableProvider};
use datafusion_common::Result;
use datafusion_common::config::ConfigNonZeroUsize;
use datafusion_expr::execution_props::CaseEqualityPlanner;
use datafusion_expr::{Case, LogicalPlanBuilder, Volatility, create_udf};
use datafusion_functions_aggregate::count::Count;
use datafusion_functions_window::row_number::RowNumber;

const ALIAS: &str = "snapshot_extra_alias";
const TABLE: &str = "snapshot_table";

fn configured_state() -> SessionState {
    SessionStateBuilder::new()
        .with_default_features()
        .with_config(
            SessionConfig::new()
                .with_target_partitions(1)
                .with_batch_size(512)
                .with_default_catalog_and_schema("snapshot_catalog", "snapshot_schema")
                .with_create_default_catalog_and_schema(false),
        )
        .build()
}

fn scalar(aliased: bool) -> Arc<ScalarUDF> {
    let udf = create_udf(
        "snapshot_scalar",
        vec![DataType::Int64],
        DataType::Int64,
        Volatility::Immutable,
        Arc::new(|args| Ok(args[0].clone())),
    );
    Arc::new(if aliased {
        udf.with_aliases([ALIAS])
    } else {
        udf
    })
}

fn aggregate(aliased: bool) -> Arc<AggregateUDF> {
    let udf = AggregateUDF::new_from_impl(Count::new());
    Arc::new(if aliased {
        udf.with_aliases([ALIAS])
    } else {
        udf
    })
}

fn window(aliased: bool) -> Arc<WindowUDF> {
    let udf = WindowUDF::new_from_impl(RowNumber::new());
    Arc::new(if aliased {
        udf.with_aliases([ALIAS])
    } else {
        udf
    })
}

#[cfg(feature = "nested_expressions")]
fn higher_order(aliased: bool) -> Arc<HigherOrderUDF> {
    use datafusion_functions_nested::array_transform::ArrayTransform;

    let udf = HigherOrderUDF::new_from_impl(ArrayTransform::new());
    Arc::new(if aliased {
        udf.with_aliases([ALIAS])
    } else {
        udf
    })
}

#[derive(Debug)]
struct SnapshotTable;

impl TableFunctionImpl for SnapshotTable {
    fn call_with_args(&self, _: TableFunctionArgs) -> Result<Arc<dyn TableProvider>> {
        Ok(Arc::new(EmptyTable::new(Arc::new(Schema::empty()))))
    }
}

fn populated_state() -> Result<SessionState> {
    let mut state = configured_state();
    // A survives under the extra alias after B replaces its canonical binding.
    state.register_udf(scalar(true))?;
    state.register_udf(scalar(false))?;
    state.register_udaf(aggregate(true))?;
    state.register_udaf(aggregate(false))?;
    state.register_udwf(window(true))?;
    state.register_udwf(window(false))?;
    #[cfg(feature = "nested_expressions")]
    {
        state.register_higher_order_function(higher_order(true))?;
        state.register_higher_order_function(higher_order(false))?;
    }
    let table: Arc<dyn TableFunctionImpl> = Arc::new(SnapshotTable);
    state.register_udtf(TABLE, Arc::clone(&table));
    state.register_udtf(ALIAS, table);
    state.register_udtf(TABLE, Arc::new(SnapshotTable));
    Ok(state)
}

fn assert_bindings<T>(
    expected: &HashMap<String, Arc<T>>,
    actual: &HashMap<String, Arc<T>>,
) {
    assert_eq!(expected.len(), actual.len());
    for (name, function) in expected {
        assert!(
            actual
                .get(name)
                .is_some_and(|other| Arc::ptr_eq(function, other)),
            "binding changed for {name}"
        );
    }
}

fn assert_registries(expected: &SessionState, actual: &SessionState) {
    assert_bindings(expected.scalar_functions(), actual.scalar_functions());
    assert_bindings(
        expected.higher_order_functions(),
        actual.higher_order_functions(),
    );
    assert_bindings(expected.aggregate_functions(), actual.aggregate_functions());
    assert_bindings(expected.window_functions(), actual.window_functions());
    assert_bindings(expected.table_functions(), actual.table_functions());
}

#[test]
fn plain_clone_shares_inner_and_all_registry_bindings() -> Result<()> {
    let original = populated_state()?;
    let snapshot = original.clone();
    assert!(Arc::ptr_eq(&original.inner, &snapshot.inner));
    assert_eq!(snapshot.config().batch_size(), 512);
    assert_eq!(snapshot.config().target_partitions(), 1);
    assert_registries(&original, &snapshot);
    Ok(())
}

#[test]
fn config_mutation_detaches_without_replaying_contested_aliases() -> Result<()> {
    let mut original = populated_state()?;
    let mut snapshot = original.clone();
    snapshot.config_mut().options_mut().execution.batch_size =
        ConfigNonZeroUsize::try_new(1024).unwrap();
    assert!(!Arc::ptr_eq(&original.inner, &snapshot.inner));
    assert_eq!(original.config().batch_size(), 512);
    assert_eq!(snapshot.config().batch_size(), 1024);
    assert_registries(&original, &snapshot);

    original.config_mut().options_mut().catalog.default_catalog = "changed".into();
    assert_eq!(
        original.config().options().catalog.default_catalog,
        "changed"
    );
    assert_eq!(
        snapshot.config().options().catalog.default_catalog,
        "snapshot_catalog"
    );
    assert_registries(&original, &snapshot);
    Ok(())
}

macro_rules! registry_snapshot_test {
    ($test:ident, $fixture:ident, $registry:ident, $register:ident, $deregister:ident) => {
        #[test]
        fn $test() -> Result<()> {
            let mut original = configured_state();
            let a = $fixture(true);
            let b = $fixture(false);
            let name = b.name().to_owned();
            let initial = original.$registry().get(&name).cloned();
            let replaced = original.$register(Arc::clone(&a))?;
            match (initial, replaced) {
                (Some(expected), Some(actual)) => {
                    assert!(Arc::ptr_eq(&expected, &actual))
                }
                (None, None) => {}
                _ => panic!("registration returned the wrong prior binding for {name}"),
            }
            let replaced = original.$register(Arc::clone(&b))?.unwrap();
            assert!(Arc::ptr_eq(&replaced, &a));
            assert!(Arc::ptr_eq(&original.$registry()[&name], &b));
            assert!(Arc::ptr_eq(&original.$registry()[ALIAS], &a));

            let original_bindings = original.$registry().clone();
            let mut configured = original.clone();
            configured.config_mut().options_mut().execution.batch_size =
                ConfigNonZeroUsize::try_new(1024).unwrap();
            assert_bindings(&original_bindings, configured.$registry());
            assert_eq!(original.config().batch_size(), 512);

            let mut snapshot = original.clone();
            assert!(Arc::ptr_eq(&original.inner, &snapshot.inner));
            let removed = snapshot.$deregister(&name)?.unwrap();
            assert!(Arc::ptr_eq(&removed, &b));
            assert!(!Arc::ptr_eq(&original.inner, &snapshot.inner));
            assert!(!snapshot.$registry().contains_key(&name));
            assert!(Arc::ptr_eq(&snapshot.$registry()[ALIAS], &a));
            assert_bindings(&original_bindings, original.$registry());

            let c = $fixture(false);
            let before_insert = snapshot.clone();
            assert!(snapshot.$register(Arc::clone(&c))?.is_none());
            assert!(!Arc::ptr_eq(&before_insert.inner, &snapshot.inner));
            assert!(!before_insert.$registry().contains_key(&name));
            assert!(Arc::ptr_eq(&before_insert.$registry()[ALIAS], &a));
            assert!(Arc::ptr_eq(&snapshot.$registry()[&name], &c));
            let snapshot_bindings = snapshot.$registry().clone();

            let d = $fixture(true);
            let before_replace = original.clone();
            assert!(Arc::ptr_eq(
                &original.$register(Arc::clone(&d))?.unwrap(),
                &b
            ));
            assert!(!Arc::ptr_eq(&original.inner, &before_replace.inner));
            assert_bindings(&original_bindings, before_replace.$registry());
            assert_bindings(&original_bindings, configured.$registry());
            assert!(Arc::ptr_eq(&original.$registry()[ALIAS], &d));
            assert_bindings(&snapshot_bindings, snapshot.$registry());
            let before_remove = original.clone();
            assert!(Arc::ptr_eq(&original.$deregister(&name)?.unwrap(), &d));
            assert!(!Arc::ptr_eq(&original.inner, &before_remove.inner));
            assert!(Arc::ptr_eq(&before_remove.$registry()[&name], &d));
            assert!(Arc::ptr_eq(&before_remove.$registry()[ALIAS], &d));
            assert!(!original.$registry().contains_key(&name));
            assert!(!original.$registry().contains_key(ALIAS));
            assert_bindings(&snapshot_bindings, snapshot.$registry());
            assert!(!Arc::ptr_eq(&original.inner, &snapshot.inner));
            Ok(())
        }
    };
}

registry_snapshot_test!(
    scalar_registry_mutations_are_isolated,
    scalar,
    scalar_functions,
    register_udf,
    deregister_udf
);
registry_snapshot_test!(
    aggregate_registry_mutations_are_isolated,
    aggregate,
    aggregate_functions,
    register_udaf,
    deregister_udaf
);
registry_snapshot_test!(
    window_registry_mutations_are_isolated,
    window,
    window_functions,
    register_udwf,
    deregister_udwf
);
#[cfg(feature = "nested_expressions")]
registry_snapshot_test!(
    higher_order_registry_mutations_are_isolated,
    higher_order,
    higher_order_functions,
    register_higher_order_function,
    deregister_higher_order_function
);

#[test]
fn table_registry_mutations_are_isolated() -> Result<()> {
    let mut original = configured_state();
    let a: Arc<dyn TableFunctionImpl> = Arc::new(SnapshotTable);
    let b: Arc<dyn TableFunctionImpl> = Arc::new(SnapshotTable);
    original.register_udtf(TABLE, Arc::clone(&a));
    original.register_udtf(ALIAS, Arc::clone(&a));
    assert!(!Arc::ptr_eq(
        &original.table_functions()[TABLE],
        &original.table_functions()[ALIAS]
    ));
    original.register_udtf(TABLE, Arc::clone(&b));
    assert!(Arc::ptr_eq(
        original.table_functions()[TABLE].function(),
        &b
    ));
    assert!(Arc::ptr_eq(
        original.table_functions()[ALIAS].function(),
        &a
    ));
    let original_bindings = original.table_functions().clone();

    let mut configured = original.clone();
    configured.config_mut().options_mut().execution.batch_size =
        ConfigNonZeroUsize::try_new(1024).unwrap();
    assert_bindings(&original_bindings, configured.table_functions());
    let mut snapshot = original.clone();
    assert!(Arc::ptr_eq(&original.inner, &snapshot.inner));
    assert!(Arc::ptr_eq(&snapshot.deregister_udtf(TABLE)?.unwrap(), &b));
    assert!(!Arc::ptr_eq(&original.inner, &snapshot.inner));
    assert!(!snapshot.table_functions().contains_key(TABLE));
    assert!(Arc::ptr_eq(
        snapshot.table_functions()[ALIAS].function(),
        &a
    ));
    assert_bindings(&original_bindings, original.table_functions());

    let c: Arc<dyn TableFunctionImpl> = Arc::new(SnapshotTable);
    let before_insert = snapshot.clone();
    snapshot.register_udtf(TABLE, Arc::clone(&c));
    assert!(!Arc::ptr_eq(&before_insert.inner, &snapshot.inner));
    assert!(!before_insert.table_functions().contains_key(TABLE));
    assert!(Arc::ptr_eq(
        snapshot.table_functions()[TABLE].function(),
        &c
    ));
    let snapshot_bindings = snapshot.table_functions().clone();
    let before_replace = original.clone();
    let d: Arc<dyn TableFunctionImpl> = Arc::new(SnapshotTable);
    original.register_udtf(ALIAS, Arc::clone(&d));
    assert!(!Arc::ptr_eq(&original.inner, &before_replace.inner));
    assert_bindings(&original_bindings, before_replace.table_functions());
    assert!(Arc::ptr_eq(
        original.table_functions()[ALIAS].function(),
        &d
    ));
    let before_remove = original.clone();
    assert!(Arc::ptr_eq(&original.deregister_udtf(TABLE)?.unwrap(), &b));
    assert!(!Arc::ptr_eq(&original.inner, &before_remove.inner));
    assert!(Arc::ptr_eq(
        before_remove.table_functions()[TABLE].function(),
        &b
    ));
    assert!(original.deregister_udtf(TABLE)?.is_none());
    assert!(original.table_functions().contains_key(ALIAS));
    assert_bindings(&snapshot_bindings, snapshot.table_functions());
    assert!(!Arc::ptr_eq(&original.inner, &snapshot.inner));
    Ok(())
}

#[test]
fn prepared_plans_share_arcs_but_insert_remove_and_errors_are_isolated() -> Result<()> {
    let mut original = populated_state()?;
    let plan = Arc::new(LogicalPlanBuilder::empty(true).build()?);
    let fields = vec![Arc::new(Field::new("parameter", DataType::Int64, true))];
    original.store_prepared("original".into(), fields.clone(), Arc::clone(&plan))?;
    let prepared = original.get_prepared("original").unwrap();
    assert_eq!(prepared.fields, fields);
    assert!(Arc::ptr_eq(&prepared.plan, &plan));

    let mut snapshot = original.clone();
    assert!(Arc::ptr_eq(&original.inner, &snapshot.inner));
    assert!(Arc::ptr_eq(
        &snapshot.get_prepared("original").unwrap(),
        &prepared
    ));
    snapshot.store_prepared("snapshot".into(), vec![], Arc::clone(&plan))?;
    assert!(!Arc::ptr_eq(&original.inner, &snapshot.inner));
    assert!(original.get_prepared("snapshot").is_none());
    assert!(Arc::ptr_eq(
        &snapshot.get_prepared("original").unwrap(),
        &prepared
    ));
    let snapshot_plan = snapshot.get_prepared("snapshot").unwrap();
    let error = snapshot
        .store_prepared("snapshot".into(), fields, Arc::clone(&plan))
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("Prepared statement 'snapshot' already exists")
    );
    assert!(Arc::ptr_eq(
        &snapshot.get_prepared("snapshot").unwrap(),
        &snapshot_plan
    ));

    let before_remove = snapshot.clone();
    snapshot.remove_prepared("original")?;
    assert!(!Arc::ptr_eq(&snapshot.inner, &before_remove.inner));
    assert!(Arc::ptr_eq(
        &before_remove.get_prepared("original").unwrap(),
        &prepared
    ));
    assert!(snapshot.get_prepared("original").is_none());
    assert!(Arc::ptr_eq(
        &original.get_prepared("original").unwrap(),
        &prepared
    ));
    let error = original.remove_prepared("snapshot").unwrap_err();
    assert!(
        error
            .to_string()
            .contains("Prepared statement 'snapshot' does not exist")
    );
    assert!(Arc::ptr_eq(
        &snapshot.get_prepared("snapshot").unwrap(),
        &snapshot_plan
    ));
    let before_remove_original = original.clone();
    original.remove_prepared("original")?;
    assert!(!Arc::ptr_eq(&original.inner, &before_remove_original.inner));
    assert!(Arc::ptr_eq(
        &before_remove_original.get_prepared("original").unwrap(),
        &prepared
    ));
    original.store_prepared("later".into(), vec![], plan)?;
    assert!(snapshot.get_prepared("later").is_none());
    assert!(Arc::ptr_eq(
        &snapshot.get_prepared("snapshot").unwrap(),
        &snapshot_plan
    ));
    assert_registries(&original, &snapshot);
    Ok(())
}

#[derive(Debug)]
struct SnapshotCasePlanner;

impl CaseEqualityPlanner for SnapshotCasePlanner {
    fn plan_equality(&self, _: &Case, _: &DFSchema) -> Result<Option<Expr>> {
        Ok(None)
    }
}

fn seed_execution_props(state: &mut SessionState) -> Arc<dyn CaseEqualityPlanner> {
    let planner: Arc<dyn CaseEqualityPlanner> = Arc::new(SnapshotCasePlanner);
    let props = state.execution_props_mut();
    props.case_equality_planner = Some(Arc::clone(&planner));
    props.query_execution_start_time = Some(DateTime::from_timestamp(0, 0).unwrap());
    assert_eq!(props.alias_generator.next("snapshot"), "snapshot_1");
    planner
}

#[test]
fn execution_props_are_query_local_and_reset_preserves_case_planner() -> Result<()> {
    let mut original = populated_state()?;
    let planner = seed_execution_props(&mut original);
    let epoch = original.execution_props().query_execution_start_time;
    let mut snapshot = original.clone();
    assert!(Arc::ptr_eq(&original.inner, &snapshot.inner));
    assert!(Arc::ptr_eq(
        &original.execution_props().alias_generator,
        &snapshot.execution_props().alias_generator
    ));
    snapshot.execution_props_mut().query_execution_start_time = None;
    assert_eq!(original.execution_props().query_execution_start_time, epoch);
    snapshot.mark_start_execution();
    assert!(Arc::ptr_eq(&original.inner, &snapshot.inner));
    assert!(!Arc::ptr_eq(
        &original.execution_props().alias_generator,
        &snapshot.execution_props().alias_generator
    ));
    assert_eq!(
        snapshot.execution_props().alias_generator.next("snapshot"),
        "snapshot_1"
    );
    assert_eq!(
        original.execution_props().alias_generator.next("snapshot"),
        "snapshot_2"
    );
    assert_eq!(original.execution_props().query_execution_start_time, epoch);
    assert!(original.execution_props().config_options().is_none());
    assert!(
        snapshot
            .execution_props()
            .query_execution_start_time
            .is_some()
    );
    assert_ne!(snapshot.execution_props().query_execution_start_time, epoch);
    assert!(Arc::ptr_eq(
        snapshot.execution_props().config_options().unwrap(),
        snapshot.config().options()
    ));
    assert!(Arc::ptr_eq(
        snapshot
            .execution_props()
            .case_equality_planner
            .as_ref()
            .unwrap(),
        &planner
    ));
    snapshot.execution_props_mut().case_equality_planner = None;
    assert!(Arc::ptr_eq(
        original
            .execution_props()
            .case_equality_planner
            .as_ref()
            .unwrap(),
        &planner
    ));
    assert_registries(&original, &snapshot);
    Ok(())
}

#[test]
fn session_context_state_resets_only_its_query_execution_props() -> Result<()> {
    let mut original = populated_state()?;
    let planner = seed_execution_props(&mut original);
    let epoch = original.execution_props().query_execution_start_time;
    let context = SessionContext::new_with_state(original.clone());
    let mut first = context.state();
    let second = context.state();
    for state in [&first, &second] {
        assert!(Arc::ptr_eq(&original.inner, &state.inner));
        assert_registries(&original, state);
        assert!(state.execution_props().query_execution_start_time.is_some());
        assert_ne!(state.execution_props().query_execution_start_time, epoch);
        assert!(Arc::ptr_eq(
            state.execution_props().config_options().unwrap(),
            state.config().options()
        ));
        assert!(Arc::ptr_eq(
            state
                .execution_props()
                .case_equality_planner
                .as_ref()
                .unwrap(),
            &planner
        ));
        assert_eq!(
            state.execution_props().alias_generator.next("snapshot"),
            "snapshot_1"
        );
    }
    assert!(!Arc::ptr_eq(
        &first.execution_props().alias_generator,
        &second.execution_props().alias_generator
    ));
    assert!(!Arc::ptr_eq(
        &original.execution_props().alias_generator,
        &first.execution_props().alias_generator
    ));
    first.execution_props_mut().case_equality_planner = None;
    assert!(Arc::ptr_eq(
        second
            .execution_props()
            .case_equality_planner
            .as_ref()
            .unwrap(),
        &planner
    ));
    let live = context.state_ref();
    let live = live.read();
    assert_eq!(live.execution_props().query_execution_start_time, epoch);
    assert!(live.execution_props().config_options().is_none());
    assert!(Arc::ptr_eq(
        &original.execution_props().alias_generator,
        &live.execution_props().alias_generator
    ));
    assert_eq!(
        live.execution_props().alias_generator.next("snapshot"),
        "snapshot_2"
    );
    assert!(Arc::ptr_eq(
        live.execution_props()
            .case_equality_planner
            .as_ref()
            .unwrap(),
        &planner
    ));
    Ok(())
}
