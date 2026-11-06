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

//! Regression controls for replacing only the runtime of an existing state.

use super::*;
use crate::datasource::empty::EmptyTable;
use crate::execution::object_store::ObjectStoreUrl;
use arrow_schema::{DataType, Field, Schema};
use datafusion_common::{Result, ScalarValue};
use datafusion_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl, Signature, Volatility,
};
use object_store::memory::InMemory;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ConfigurableFunction {
    name: &'static str,
    aliases: Vec<String>,
    signature: Signature,
    generation: u64,
}
impl ConfigurableFunction {
    fn new(name: &'static str, aliases: &[&str], generation: u64) -> Self {
        Self {
            name,
            aliases: aliases.iter().map(|name| (*name).to_owned()).collect(),
            signature: Signature::nullary(Volatility::Immutable),
            generation,
        }
    }
}
impl ScalarUDFImpl for ConfigurableFunction {
    fn name(&self) -> &str {
        self.name
    }
    fn aliases(&self) -> &[String] {
        &self.aliases
    }
    fn signature(&self) -> &Signature {
        &self.signature
    }
    fn return_type(&self, _args: &[DataType]) -> Result<DataType> {
        Ok(DataType::Int64)
    }
    fn schema_name(&self, _args: &[Expr]) -> Result<String> {
        Ok(format!("{}_generation_{}", self.name, self.generation))
    }
    fn invoke_with_args(&self, _args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        Ok(ColumnarValue::Scalar(ScalarValue::Int64(Some(1))))
    }
    fn with_updated_config(&self, _config: &ConfigOptions) -> Option<ScalarUDF> {
        Some(ScalarUDF::new_from_impl(Self {
            generation: self.generation + 1,
            ..self.clone()
        }))
    }
}

fn assert_arc_maps_match<T: ?Sized>(
    original: &HashMap<String, Arc<T>>,
    replacement: &HashMap<String, Arc<T>>,
) {
    assert_eq!(original.len(), replacement.len());
    for (name, value) in original {
        assert!(
            replacement
                .get(name)
                .is_some_and(|other| Arc::ptr_eq(value, other)),
            "registration changed: {name}"
        );
    }
}
fn assert_arc_slices_match<T: ?Sized>(original: &[Arc<T>], replacement: &[Arc<T>]) {
    assert_eq!(original.len(), replacement.len());
    for (original, replacement) in original.iter().zip(replacement) {
        assert!(Arc::ptr_eq(original, replacement));
    }
}

#[test]
fn with_runtime_env_preserves_exact_function_aliases_and_configured_instances()
-> Result<()> {
    let original_runtime = Arc::new(RuntimeEnv::default());
    let replacement_runtime = Arc::new(RuntimeEnv::default());
    let mut state = SessionStateBuilder::new()
        .with_default_features()
        .with_runtime_env(Arc::clone(&original_runtime))
        .build();
    let first = Arc::new(ScalarUDF::new_from_impl(ConfigurableFunction::new(
        "preserved_first",
        &["shared_alias", "first_only"],
        7,
    )));
    let second = Arc::new(ScalarUDF::new_from_impl(ConfigurableFunction::new(
        "preserved_second",
        &["shared_alias", "preserved_first"],
        11,
    )));
    state.register_udf(Arc::clone(&first))?;
    state.register_udf(Arc::clone(&second))?;
    state.register_udtf("preserved_table_function", Arc::new(PreservedTableFunction));
    assert!(Arc::ptr_eq(&state.udf("shared_alias")?, &second));
    assert!(Arc::ptr_eq(&state.udf("preserved_first")?, &second));
    assert!(Arc::ptr_eq(&state.udf("first_only")?, &first));
    // The configured callback really changes the instance/schema if called again.
    let first_schema = first.schema_name(&[])?;
    let configured = first.inner().with_updated_config(state.config_options());
    assert!(configured.is_some_and(|udf| {
        udf.schema_name(&[]).is_ok_and(|name| name != first_schema)
    }));
    let original = state.clone();
    let replacement = state.with_runtime_env(Arc::clone(&replacement_runtime));
    assert_arc_maps_match(original.scalar_functions(), replacement.scalar_functions());
    assert_arc_maps_match(
        original.higher_order_functions(),
        replacement.higher_order_functions(),
    );
    assert_arc_maps_match(
        original.aggregate_functions(),
        replacement.aggregate_functions(),
    );
    assert_arc_maps_match(original.window_functions(), replacement.window_functions());
    assert_arc_maps_match(original.table_functions(), replacement.table_functions());
    for name in [
        "shared_alias",
        "preserved_first",
        "first_only",
        "preserved_second",
    ] {
        let before = original.udf(name)?;
        let after = replacement.udf(name)?;
        assert!(Arc::ptr_eq(&before, &after));
        assert_eq!(before.name(), after.name());
        assert_eq!(before.aliases(), after.aliases());
        assert_eq!(before.schema_name(&[])?, after.schema_name(&[])?);
        assert_eq!(before.return_type(&[])?, after.return_type(&[])?);
    }
    assert!(Arc::ptr_eq(original.runtime_env(), &original_runtime));
    assert!(Arc::ptr_eq(replacement.runtime_env(), &replacement_runtime));
    assert!(Arc::ptr_eq(
        original.config_options(),
        replacement.config_options()
    ));
    Ok(())
}

#[derive(Debug)]
struct PreservedTableFunction;
impl TableFunctionImpl for PreservedTableFunction {
    fn call_with_args(
        &self,
        _args: datafusion_catalog::TableFunctionArgs,
    ) -> Result<Arc<dyn crate::catalog::TableProvider>> {
        Ok(Arc::new(EmptyTable::new(Arc::new(Schema::empty()))))
    }
}

#[derive(Debug)]
struct PreservedPlanner;
impl ExprPlanner for PreservedPlanner {}
#[cfg(feature = "sql")]
impl TypePlanner for PreservedPlanner {}
#[cfg(feature = "sql")]
impl RelationPlanner for PreservedPlanner {
    fn plan_relation(
        &self,
        relation: sqlparser::ast::TableFactor,
        _context: &mut dyn datafusion_expr::planner::RelationPlannerContext,
    ) -> Result<datafusion_expr::planner::RelationPlanning> {
        Ok(datafusion_expr::planner::RelationPlanning::Original(
            Box::new(relation),
        ))
    }
}

#[derive(Debug)]
struct PreservedExtension;

#[tokio::test]
async fn with_runtime_env_preserves_session_state_and_shared_runtime_registry()
-> Result<()> {
    let original_runtime = Arc::new(RuntimeEnv::default());
    let replacement_runtime = Arc::new(RuntimeEnv::default());
    let url = ObjectStoreUrl::parse("memory://runtime-swap/")?;
    let original_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let replacement_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    original_runtime.register_object_store(url.as_ref(), Arc::clone(&original_store));
    replacement_runtime
        .register_object_store(url.as_ref(), Arc::clone(&replacement_store));
    let original_registry = Arc::clone(&original_runtime.object_store_registry);
    let replacement_registry = Arc::clone(&replacement_runtime.object_store_registry);
    let extension = Arc::new(PreservedExtension);
    let config = SessionConfig::new()
        .with_target_partitions(3)
        .with_batch_size(19)
        .with_default_catalog_and_schema("kept_catalog", "kept_schema")
        .with_information_schema(true)
        .with_extension(Arc::clone(&extension));
    let mut state = SessionStateBuilder::new()
        .with_default_features()
        .with_session_id("kept_session".to_owned())
        .with_config(config)
        .with_runtime_env(Arc::clone(&original_runtime))
        .build();
    state.expr_planners.push(Arc::new(PreservedPlanner));
    #[cfg(feature = "sql")]
    {
        state.type_planner = Some(Arc::new(PreservedPlanner));
        state.relation_planners.push(Arc::new(PreservedPlanner));
    }
    state.table_options.csv.delimiter = b'|';
    state.mark_start_execution();
    let table_schema = Arc::new(Schema::new(vec![Field::new(
        "KeptColumn",
        DataType::Int64,
        false,
    )]));
    let table: Arc<dyn crate::catalog::TableProvider> =
        Arc::new(EmptyTable::new(Arc::clone(&table_schema)));
    let schema = state.schema_for_ref("probe")?;
    schema.register_table("probe".to_owned(), Arc::clone(&table))?;
    state.prepared_plans.insert(
        "kept_plan".to_owned(),
        Arc::new(PreparedPlan {
            fields: vec![Arc::new(Field::new("Parameter", DataType::Int32, false))],
            plan: Arc::new(LogicalPlan::EmptyRelation(datafusion_expr::EmptyRelation {
                produce_one_row: false,
                schema: Arc::new(DFSchema::empty()),
            })),
        }),
    );
    let original = state.clone();
    let replacement = state.with_runtime_env(Arc::clone(&replacement_runtime));
    assert_eq!(replacement.session_id(), "kept_session");
    assert!(Arc::ptr_eq(
        original.config_options(),
        replacement.config_options()
    ));
    assert!(
        replacement
            .config()
            .get_extension::<PreservedExtension>()
            .is_some_and(|value| Arc::ptr_eq(&extension, &value))
    );
    assert_eq!(
        original.table_options().entries(),
        replacement.table_options().entries()
    );
    assert!(Arc::ptr_eq(
        original.catalog_list(),
        replacement.catalog_list()
    ));
    assert!(Arc::ptr_eq(&schema, &replacement.schema_for_ref("probe")?));
    assert!(
        replacement
            .schema_for_ref("probe")?
            .table("probe")
            .await?
            .is_some_and(
                |value| Arc::ptr_eq(&value, &table) && value.schema() == table_schema
            )
    );
    assert!(Arc::ptr_eq(
        original.query_planner(),
        replacement.query_planner()
    ));
    assert_arc_slices_match(original.expr_planners(), replacement.expr_planners());
    assert_arc_slices_match(&original.analyzer.rules, &replacement.analyzer.rules);
    assert_arc_slices_match(original.optimizers(), replacement.optimizers());
    assert_arc_slices_match(
        original.physical_optimizers(),
        replacement.physical_optimizers(),
    );
    #[cfg(feature = "sql")]
    {
        assert_arc_slices_match(
            original.relation_planners(),
            replacement.relation_planners(),
        );
        assert!(
            original
                .type_planner
                .as_ref()
                .zip(replacement.type_planner.as_ref())
                .is_some_and(|(before, after)| Arc::ptr_eq(before, after))
        );
    }
    assert_arc_maps_match(&original.table_factories, &replacement.table_factories);
    assert_arc_maps_match(&original.file_formats, &replacement.file_formats);
    assert_arc_maps_match(&original.prepared_plans, &replacement.prepared_plans);
    assert!(Arc::ptr_eq(
        &original.serializer_registry,
        &replacement.serializer_registry
    ));
    assert!(Arc::ptr_eq(
        &original.extension_types,
        &replacement.extension_types
    ));
    assert_eq!(
        original.execution_props.query_execution_start_time,
        replacement.execution_props.query_execution_start_time
    );
    assert!(Arc::ptr_eq(
        &original.execution_props.alias_generator,
        &replacement.execution_props.alias_generator
    ));
    assert!(
        original
            .execution_props
            .config_options
            .as_ref()
            .zip(replacement.execution_props.config_options.as_ref())
            .is_some_and(|(before, after)| Arc::ptr_eq(before, after))
    );
    assert!(Arc::ptr_eq(original.runtime_env(), &original_runtime));
    assert!(Arc::ptr_eq(
        &original_runtime.object_store_registry,
        &original_registry
    ));
    assert!(Arc::ptr_eq(
        &replacement_runtime.object_store_registry,
        &replacement_registry
    ));
    assert!(Arc::ptr_eq(
        &original_runtime.object_store(&url)?,
        &original_store
    ));
    assert!(Arc::ptr_eq(
        &replacement.runtime_env().object_store(&url)?,
        &replacement_store
    ));
    let private_url = ObjectStoreUrl::parse("memory://private-runtime-only/")?;
    replacement
        .runtime_env()
        .register_object_store(private_url.as_ref(), Arc::new(InMemory::new()));
    assert!(replacement.runtime_env().object_store(&private_url).is_ok());
    assert!(original.runtime_env().object_store(&private_url).is_err());
    assert!(Arc::ptr_eq(
        &original_runtime.object_store(&url)?,
        &original_store
    ));
    Ok(())
}
