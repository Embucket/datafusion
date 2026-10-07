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

#![cfg_attr(test, allow(clippy::needless_pass_by_value))]
// Make sure fast / cheap clones on Arc are explicit:
// https://github.com/apache/datafusion/issues/11143
#![cfg_attr(not(test), deny(clippy::clone_on_ref_ptr))]

pub mod file_format;
pub mod source;

use std::sync::Arc;

use datafusion_common::Result;
use datafusion_datasource::file_groups::FileGroup;
use datafusion_datasource::file_scan_config::FileScanConfigBuilder;
use datafusion_datasource::{file::FileSource, file_scan_config::FileScanConfig};
use datafusion_execution::object_store::ObjectStoreUrl;
pub use file_format::*;
use regex_syntax::hir::{Hir, HirKind, Look};

/// Extract exact values only when a regex matches the entire field without flags or classes.
fn exact_null_values(pattern: &str) -> Option<Vec<String>> {
    if !pattern.starts_with(r"\A") || !pattern.ends_with(r"\z") {
        return None;
    }
    let hir = regex_syntax::Parser::new().parse(pattern).ok()?;
    let HirKind::Concat(parts) = hir.kind() else {
        return None;
    };
    let [start, values, end] = parts.as_slice() else {
        return None;
    };
    if !matches!(start.kind(), HirKind::Look(Look::Start))
        || !matches!(end.kind(), HirKind::Look(Look::End))
    {
        return None;
    }
    let alternatives = match values.kind() {
        HirKind::Alternation(parts) => parts.as_slice(),
        _ => std::slice::from_ref(values),
    };
    if alternatives.len() > 64 {
        return None;
    }
    alternatives.iter().map(exact_literal).collect()
}

fn exact_literal(hir: &Hir) -> Option<String> {
    match hir.kind() {
        HirKind::Empty => Some(String::new()),
        HirKind::Literal(value) => std::str::from_utf8(&value.0).ok().map(str::to_owned),
        _ => None,
    }
}

/// Returns a [`FileScanConfig`] for given `file_groups`
pub fn partitioned_csv_config(
    file_groups: Vec<FileGroup>,
    file_source: Arc<dyn FileSource>,
) -> Result<FileScanConfig> {
    Ok(
        FileScanConfigBuilder::new(ObjectStoreUrl::local_filesystem(), file_source)
            .with_file_groups(file_groups)
            .build(),
    )
}

#[cfg(test)]
mod null_value_tests {
    use super::exact_null_values;

    #[test]
    fn extracts_only_exact_null_values() {
        assert_eq!(
            exact_null_values(r"\A(?:\\N|NA|)\z"),
            Some(vec![r"\N".to_owned(), "NA".to_owned(), String::new()])
        );
        assert_eq!(
            exact_null_values(r"\A(?:NA|)\z"),
            Some(vec!["NA".to_owned(), String::new()])
        );
        assert_eq!(exact_null_values(r"\Afoo\z"), Some(vec!["foo".to_owned()]));
        assert_eq!(exact_null_values("^foo$"), None);
        assert_eq!(exact_null_values(r"\A(?i:foo)\z"), None);
        assert_eq!(exact_null_values(r"\Afo.*\z"), None);
    }
}
