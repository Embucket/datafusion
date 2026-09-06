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
use datafusion_common::assert_batches_eq;

#[tokio::test]
async fn sql_unpivot_excludes_nulls_by_default() -> Result<()> {
    let ctx = SessionContext::new();
    let batches = plan_and_collect(
        &ctx,
        "SELECT *
         FROM (VALUES (1, 10, 20), (2, 30, NULL)) t(id, jan, feb)
         UNPIVOT(sales FOR month IN (jan, feb))
         ORDER BY id, month",
    )
    .await?;

    assert_batches_eq!(
        [
            "+----+-------+-------+",
            "| id | month | sales |",
            "+----+-------+-------+",
            "| 1  | feb   | 20    |",
            "| 1  | jan   | 10    |",
            "| 2  | jan   | 30    |",
            "+----+-------+-------+",
        ],
        &batches
    );
    Ok(())
}

#[tokio::test]
async fn sql_unpivot_includes_nulls_and_column_aliases() -> Result<()> {
    let ctx = SessionContext::new();
    let batches = plan_and_collect(
        &ctx,
        "SELECT up.*
         FROM (VALUES (1, 10, NULL)) t(id, jan, feb)
         UNPIVOT INCLUDE NULLS(sales FOR month IN (jan AS JAN, feb AS FEB)) AS up
         ORDER BY month",
    )
    .await?;

    assert_batches_eq!(
        [
            "+----+-------+-------+",
            "| id | month | sales |",
            "+----+-------+-------+",
            "| 1  | FEB   |       |",
            "| 1  | JAN   | 10    |",
            "+----+-------+-------+",
        ],
        &batches
    );
    Ok(())
}

#[tokio::test]
async fn dataframe_unpivot_uses_the_shared_logical_builder() -> Result<()> {
    let ctx = SessionContext::new();
    let df = ctx
        .sql("SELECT * FROM (VALUES (1, 10, 20)) t(id, jan, feb)")
        .await?
        .unpivot(
            "sales",
            "month",
            vec![
                (Column::from_name("jan"), Some("JAN".to_string())),
                (Column::from_name("feb"), Some("FEB".to_string())),
            ],
            false,
        )?
        .sort(vec![col("month").sort(true, true)])?;
    let batches = df.collect().await?;

    assert_batches_eq!(
        [
            "+----+-------+-------+",
            "| id | month | sales |",
            "+----+-------+-------+",
            "| 1  | FEB   | 20    |",
            "| 1  | JAN   | 10    |",
            "+----+-------+-------+",
        ],
        &batches
    );
    Ok(())
}

#[tokio::test]
async fn unpivot_coerces_value_columns_to_a_common_type() -> Result<()> {
    let ctx = SessionContext::new();
    let batches = plan_and_collect(
        &ctx,
        "SELECT *
         FROM (
           SELECT 1 AS id, CAST(10 AS SMALLINT) AS jan, CAST(20 AS BIGINT) AS feb
         ) t
         UNPIVOT(sales FOR month IN (jan, feb))
         ORDER BY month",
    )
    .await?;

    assert_batches_eq!(
        [
            "+----+-------+-------+",
            "| id | month | sales |",
            "+----+-------+-------+",
            "| 1  | feb   | 20    |",
            "| 1  | jan   | 10    |",
            "+----+-------+-------+",
        ],
        &batches
    );
    assert_eq!(batches[0].schema().field(2).data_type(), &DataType::Int64);
    Ok(())
}

#[tokio::test]
async fn unpivot_rejects_duplicate_input_columns() {
    let ctx = SessionContext::new();
    let error = ctx
        .sql(
            "SELECT *
             FROM (VALUES (1, 10)) t(id, jan)
             UNPIVOT(sales FOR month IN (jan, jan))",
        )
        .await
        .expect_err("duplicate UNPIVOT columns should fail");

    assert_contains!(error.to_string(), "UNPIVOT input columns must be unique");
}

#[tokio::test]
async fn unpivot_rejects_output_name_conflicts() {
    let ctx = SessionContext::new();
    let error = ctx
        .sql(
            "SELECT *
             FROM (VALUES (1, 10, 'existing')) t(id, jan, month)
             UNPIVOT(sales FOR month IN (jan))",
        )
        .await
        .expect_err("UNPIVOT output names should not replace preserved columns");

    assert_contains!(
        error.to_string(),
        "UNPIVOT output column conflicts with a preserved input column"
    );
}
