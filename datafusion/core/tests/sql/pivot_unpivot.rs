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

use datafusion::common::test_util::batches_to_string;
use datafusion::error::Result;
use datafusion::prelude::SessionContext;

#[tokio::test]
async fn pivot_list_is_lowered_to_filtered_aggregates() -> Result<()> {
    let batches = SessionContext::new()
        .sql(
            "SELECT *
             FROM (VALUES (1, 10, 'a'), (1, 20, 'b'), (2, 7, 'a')) t(id, amount, kind)
             PIVOT(SUM(amount) FOR kind IN ('a', 'b'))
             ORDER BY id",
        )
        .await?
        .collect()
        .await?;

    insta::assert_snapshot!(batches_to_string(&batches), @r"
    +----+-----+-----+
    | id | 'a' | 'b' |
    +----+-----+-----+
    | 1  | 10  | 20  |
    | 2  | 7   |     |
    +----+-----+-----+
    ");
    Ok(())
}

#[tokio::test]
async fn chained_pivots_apply_explicit_column_aliases() -> Result<()> {
    let batches = SessionContext::new()
        .sql(
            "SELECT SUM(q1_sales) AS q1_sales,
                    SUM(q2_sales) AS q2_sales,
                    MAX(q1_discount) AS q1_discount,
                    MAX(q2_discount) AS q2_discount
             FROM (
                 SELECT amount,
                        quarter AS sales_quarter,
                        quarter AS discount_quarter,
                        discount
                 FROM (VALUES
                     (100, 'Q1', 10),
                     (200, 'Q2', 20)
                 ) sales(amount, quarter, discount)
             )
             PIVOT(SUM(amount) FOR sales_quarter IN ('Q1', 'Q2'))
             PIVOT(MAX(discount) FOR discount_quarter IN ('Q1', 'Q2'))
             AS p(q1_sales, q2_sales, q1_discount, q2_discount)",
        )
        .await?
        .collect()
        .await?;

    insta::assert_snapshot!(batches_to_string(&batches), @r"
    +----------+----------+-------------+-------------+
    | q1_sales | q2_sales | q1_discount | q2_discount |
    +----------+----------+-------------+-------------+
    | 100      | 200      | 10          | 20          |
    +----------+----------+-------------+-------------+
    ");
    Ok(())
}

#[tokio::test]
async fn unpivot_excludes_nulls_by_default() -> Result<()> {
    let batches = SessionContext::new()
        .sql(
            "SELECT *
             FROM (VALUES (1, 10, 20), (2, 30, NULL)) t(id, jan, feb)
             UNPIVOT(sales FOR month IN (jan, feb))
             ORDER BY id, month",
        )
        .await?
        .collect()
        .await?;

    insta::assert_snapshot!(batches_to_string(&batches), @r"
    +----+-------+-------+
    | id | month | sales |
    +----+-------+-------+
    | 1  | FEB   | 20    |
    | 1  | JAN   | 10    |
    | 2  | JAN   | 30    |
    +----+-------+-------+
    ");
    Ok(())
}
