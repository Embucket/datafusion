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

use datafusion_core::prelude::*;
use datafusion_common::Result;

/// Test for PIVOT ... IN (ANY) functionality
///
/// This tests creates a table similar to the example in the user query:
/// ```sql
/// CREATE OR REPLACE TABLE quarterly_sales(
///   empid INT,
///   amount INT,
///   quarter TEXT)
///   AS SELECT * FROM VALUES
///     (1, 10000, '2023_Q1'),
///     (1, 400, '2023_Q1'),
///     (2, 4500, '2023_Q1'),
///     (2, 35000, '2023_Q1'),
///     (1, 5000, '2023_Q2'),
///     (1, 3000, '2023_Q2'),
///     (2, 200, '2023_Q2'),
///     (2, 90500, '2023_Q2'),
///     (1, 6000, '2023_Q3'),
///     (1, 5000, '2023_Q3'),
///     (2, 2500, '2023_Q3'),
///     (2, 9500, '2023_Q3'),
///     (3, 2700, '2023_Q3'),
///     (1, 8000, '2023_Q4'),
///     (1, 10000, '2023_Q4'),
///     (2, 800, '2023_Q4'),
///     (2, 4500, '2023_Q4'),
///     (3, 2700, '2023_Q4'),
///     (3, 16000, '2023_Q4'),
///     (3, 10200, '2023_Q4');
/// ```
#[tokio::test]
async fn pivot_with_any() -> Result<()> {
    let ctx = SessionContext::new();
    
    // Create a table with test data
    ctx.sql(
        "CREATE TABLE quarterly_sales AS 
         SELECT * FROM VALUES 
             (1, 10000, '2023_Q1'),
             (1, 400, '2023_Q1'),
             (2, 4500, '2023_Q1'),
             (2, 35000, '2023_Q1'),
             (1, 5000, '2023_Q2'),
             (1, 3000, '2023_Q2'),
             (2, 200, '2023_Q2'),
             (2, 90500, '2023_Q2'),
             (1, 6000, '2023_Q3'),
             (1, 5000, '2023_Q3'),
             (2, 2500, '2023_Q3'),
             (2, 9500, '2023_Q3'),
             (3, 2700, '2023_Q3'),
             (1, 8000, '2023_Q4'),
             (1, 10000, '2023_Q4'),
             (2, 800, '2023_Q4'),
             (2, 4500, '2023_Q4'),
             (3, 2700, '2023_Q4'),
             (3, 16000, '2023_Q4'),
             (3, 10200, '2023_Q4')
         AS t(empid, amount, quarter)",
    )
    .await?;

    // Execute a simple pivot query using ANY and ORDER BY
    let result = ctx
        .sql(
            "SELECT *
             FROM quarterly_sales
             PIVOT(SUM(amount) FOR quarter IN (ANY ORDER BY quarter))
             ORDER BY empid",
        )
        .await?
        .collect()
        .await?;

    // Verify the results
    assert_eq!(result.len(), 3); // 3 employees
    
    // Print the result for debugging
    for batch in &result {
        println!("{:?}", batch);
    }
    
    // Check that we have columns for empid and each quarter
    let schema = result[0].schema();
    assert!(schema.field_with_name("empid").is_ok());
    assert!(schema.field_with_name("2023_Q1").is_ok());
    assert!(schema.field_with_name("2023_Q2").is_ok());
    assert!(schema.field_with_name("2023_Q3").is_ok());
    assert!(schema.field_with_name("2023_Q4").is_ok());
    
    Ok(())
}

/// Test for PIVOT ... IN (ANY) functionality without ORDER BY clause
#[tokio::test]
async fn pivot_with_any_no_order() -> Result<()> {
    let ctx = SessionContext::new();
    
    // Create a table with test data
    ctx.sql(
        "CREATE TABLE quarterly_sales AS 
         SELECT * FROM VALUES 
             (1, 10000, '2023_Q1'),
             (1, 400, '2023_Q1'),
             (2, 4500, '2023_Q1'),
             (2, 35000, '2023_Q1'),
             (1, 5000, '2023_Q2'),
             (1, 3000, '2023_Q2'),
             (2, 200, '2023_Q2'),
             (2, 90500, '2023_Q2'),
             (1, 6000, '2023_Q3'),
             (1, 5000, '2023_Q3'),
             (2, 2500, '2023_Q3'),
             (2, 9500, '2023_Q3'),
             (3, 2700, '2023_Q3'),
             (1, 8000, '2023_Q4'),
             (1, 10000, '2023_Q4'),
             (2, 800, '2023_Q4'),
             (2, 4500, '2023_Q4'),
             (3, 2700, '2023_Q4'),
             (3, 16000, '2023_Q4'),
             (3, 10200, '2023_Q4')
         AS t(empid, amount, quarter)",
    )
    .await?;

    // Execute a simple pivot query using ANY without ORDER BY
    let result = ctx
        .sql(
            "SELECT *
             FROM quarterly_sales
             PIVOT(SUM(amount) FOR quarter IN (ANY))
             ORDER BY empid",
        )
        .await?
        .collect()
        .await?;

    // Verify the results
    assert_eq!(result.len(), 3); // 3 employees
    
    // Print the result for debugging
    for batch in &result {
        println!("{:?}", batch);
    }
    
    // Check that we have columns for empid and each quarter
    let schema = result[0].schema();
    assert!(schema.field_with_name("empid").is_ok());
    assert!(schema.field_with_name("2023_Q1").is_ok());
    assert!(schema.field_with_name("2023_Q2").is_ok());
    assert!(schema.field_with_name("2023_Q3").is_ok());
    assert!(schema.field_with_name("2023_Q4").is_ok());
    
    Ok(())
} 