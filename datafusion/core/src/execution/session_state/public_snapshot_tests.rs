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

use crate::execution::context::SessionContext;
use crate::execution::session_state::{SessionState, SessionStateBuilder};
use datafusion_common::config::{
    ConfigEntry, ConfigExtension, ConfigNonZeroUsize, ExtensionOptions,
};
use datafusion_common::{Result, config_err};
use datafusion_execution::config::SessionConfig;
use std::any::Any;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

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

#[derive(Debug)]
struct CountedTableExtension {
    clones: Arc<AtomicUsize>,
    value: String,
}

impl Clone for CountedTableExtension {
    fn clone(&self) -> Self {
        self.clones.fetch_add(1, Ordering::Relaxed);
        Self {
            // Only clone telemetry is shared; the configuration value is independent.
            clones: Arc::clone(&self.clones),
            value: self.value.clone(),
        }
    }
}

impl ConfigExtension for CountedTableExtension {
    const PREFIX: &'static str = "snapshot_counted";
}

impl ExtensionOptions for CountedTableExtension {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn cloned(&self) -> Box<dyn ExtensionOptions> {
        Box::new(self.clone())
    }

    fn set(&mut self, key: &str, value: &str) -> Result<()> {
        if key != "value" {
            return config_err!("Unknown snapshot extension key: {key}");
        }
        self.value = value.to_owned();
        Ok(())
    }

    fn entries(&self) -> Vec<ConfigEntry> {
        vec![ConfigEntry {
            key: "snapshot_counted.value".into(),
            value: Some(self.value.clone()),
            description: "Snapshot isolation test value",
        }]
    }
}

#[test]
fn public_snapshots_clone_table_extensions_only_on_first_mutation() {
    let clones = Arc::new(AtomicUsize::new(0));
    let mut original = configured_state();
    original.register_table_options_extension(CountedTableExtension {
        clones: Arc::clone(&clones),
        value: "original".into(),
    });
    clones.store(0, Ordering::Relaxed);

    let mut snapshot = original.clone();
    assert_eq!(clones.load(Ordering::Relaxed), 0, "plain state clone");
    let context = SessionContext::new_with_state(original.clone());
    let first = context.state();
    let second = context.state();
    assert_eq!(clones.load(Ordering::Relaxed), 0, "query state snapshots");

    snapshot.config_mut().options_mut().execution.batch_size =
        ConfigNonZeroUsize::try_new(1024).unwrap();
    assert_eq!(
        clones.load(Ordering::Relaxed),
        1,
        "first shared-state detachment"
    );
    for state in [&original, &snapshot, &first, &second] {
        assert_eq!(
            state
                .table_options()
                .extensions
                .get::<CountedTableExtension>()
                .unwrap()
                .value,
            "original"
        );
    }
    snapshot
        .config_mut()
        .options_mut()
        .execution
        .target_partitions = 2;
    snapshot
        .table_options_mut()
        .extensions
        .get_mut::<CountedTableExtension>()
        .unwrap()
        .set("value", "snapshot")
        .unwrap();
    assert_eq!(clones.load(Ordering::Relaxed), 1, "already detached state");
    assert_eq!(
        snapshot
            .table_options()
            .extensions
            .get::<CountedTableExtension>()
            .unwrap()
            .value,
        "snapshot"
    );
    assert_eq!(original.config().batch_size(), 512);
    assert_eq!(original.config().target_partitions(), 1);

    original
        .table_options_mut()
        .extensions
        .get_mut::<CountedTableExtension>()
        .unwrap()
        .set("value", "changed original")
        .unwrap();
    assert_eq!(
        clones.load(Ordering::Relaxed),
        2,
        "independent original detachment"
    );
    assert_eq!(
        original
            .table_options()
            .extensions
            .get::<CountedTableExtension>()
            .unwrap()
            .value,
        "changed original"
    );
    assert_eq!(
        snapshot
            .table_options()
            .extensions
            .get::<CountedTableExtension>()
            .unwrap()
            .value,
        "snapshot"
    );
    for state in [&first, &second] {
        assert_eq!(
            state
                .table_options()
                .extensions
                .get::<CountedTableExtension>()
                .unwrap()
                .value,
            "original"
        );
    }
    let live = context.state_ref();
    let live = live.read();
    assert_eq!(
        live.table_options()
            .extensions
            .get::<CountedTableExtension>()
            .unwrap()
            .value,
        "original"
    );
    assert_eq!(clones.load(Ordering::Relaxed), 2);
}
