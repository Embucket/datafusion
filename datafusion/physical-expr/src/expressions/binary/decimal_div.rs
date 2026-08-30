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

//! Decimal division that rounds half away from zero to match Snowflake.

use std::cmp::Ordering;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, Datum, PrimitiveArray};
use arrow::compute::kernels::arity::try_binary;
use arrow::datatypes::{
    ArrowNativeType, ArrowNativeTypeOp, DataType, Decimal32Type, Decimal64Type,
    Decimal128Type, Decimal256Type, DecimalType,
};
use arrow::error::ArrowError;

pub(super) fn is_decimal_division(lhs: &DataType, rhs: &DataType) -> bool {
    matches!(
        (lhs, rhs),
        (DataType::Decimal32(..), DataType::Decimal32(..))
            | (DataType::Decimal64(..), DataType::Decimal64(..))
            | (DataType::Decimal128(..), DataType::Decimal128(..))
            | (DataType::Decimal256(..), DataType::Decimal256(..))
    )
}

pub(super) fn div_half_away_from_zero(
    lhs: &dyn Datum,
    rhs: &dyn Datum,
) -> Result<ArrayRef, ArrowError> {
    let (left, left_scalar) = lhs.get();
    let (right, right_scalar) = rhs.get();
    match (left.data_type(), right.data_type()) {
        (DataType::Decimal32(..), DataType::Decimal32(..)) => {
            decimal_div_op::<Decimal32Type>(left, left_scalar, right, right_scalar)
        }
        (DataType::Decimal64(..), DataType::Decimal64(..)) => {
            decimal_div_op::<Decimal64Type>(left, left_scalar, right, right_scalar)
        }
        (DataType::Decimal128(..), DataType::Decimal128(..)) => {
            decimal_div_op::<Decimal128Type>(left, left_scalar, right, right_scalar)
        }
        (DataType::Decimal256(..), DataType::Decimal256(..)) => {
            decimal_div_op::<Decimal256Type>(left, left_scalar, right, right_scalar)
        }
        (left_type, right_type) => Err(ArrowError::InvalidArgumentError(format!(
            "Invalid decimal division: {left_type} / {right_type}"
        ))),
    }
}

fn decimal_div_op<T: DecimalType>(
    left: &dyn Array,
    left_scalar: bool,
    right: &dyn Array,
    right_scalar: bool,
) -> Result<ArrayRef, ArrowError> {
    let left = left.as_primitive::<T>();
    let right = right.as_primitive::<T>();
    let (left_precision, left_scale, right_scale) =
        match (left.data_type(), right.data_type()) {
            (DataType::Decimal32(p, l), DataType::Decimal32(_, r))
            | (DataType::Decimal64(p, l), DataType::Decimal64(_, r))
            | (DataType::Decimal128(p, l), DataType::Decimal128(_, r))
            | (DataType::Decimal256(p, l), DataType::Decimal256(_, r)) => (*p, *l, *r),
            _ => unreachable!("decimal types were checked before dispatch"),
        };

    let result_scale = left_scale.saturating_add(4).min(T::MAX_SCALE);
    let multiplier_power = result_scale - left_scale + right_scale;
    let result_precision = multiplier_power
        .saturating_add(left_precision as i8)
        .cast_unsigned()
        .min(T::MAX_PRECISION);
    let (left_multiplier, right_multiplier) = match multiplier_power.cmp(&0) {
        Ordering::Greater => (
            T::Native::usize_as(10).pow_checked(multiplier_power as _)?,
            T::Native::ONE,
        ),
        Ordering::Equal => (T::Native::ONE, T::Native::ONE),
        Ordering::Less => (
            T::Native::ONE,
            T::Native::usize_as(10).pow_checked(multiplier_power.neg_wrapping() as _)?,
        ),
    };

    let divide = |left: T::Native, right: T::Native| {
        let numerator = left.mul_checked(left_multiplier)?;
        let denominator = right.mul_checked(right_multiplier)?;
        let mut quotient = numerator.div_checked(denominator)?;
        let remainder = numerator.mod_wrapping(denominator);
        if !remainder.is_zero() {
            let absolute_remainder = if remainder.is_lt(T::Native::ZERO) {
                remainder.neg_wrapping()
            } else {
                remainder
            };
            let absolute_denominator = if denominator.is_lt(T::Native::ZERO) {
                denominator.neg_wrapping()
            } else {
                denominator
            };
            if absolute_remainder
                .is_ge(absolute_denominator.sub_wrapping(absolute_remainder))
            {
                quotient = if numerator.is_lt(T::Native::ZERO)
                    != denominator.is_lt(T::Native::ZERO)
                {
                    quotient.sub_checked(T::Native::ONE)?
                } else {
                    quotient.add_checked(T::Native::ONE)?
                };
            }
        }
        Ok(quotient)
    };

    let result: PrimitiveArray<T> = match (left_scalar, right_scalar) {
        (true, true) | (false, false) => try_binary(left, right, divide)?,
        (true, false) => match (left.null_count() == 0).then(|| left.value(0)) {
            Some(value) => right.try_unary(|right| divide(value, right))?,
            None => PrimitiveArray::new_null(right.len()),
        },
        (false, true) => match (right.null_count() == 0).then(|| right.value(0)) {
            Some(value) => left.try_unary(|left| divide(left, value))?,
            None => PrimitiveArray::new_null(left.len()),
        },
    };

    Ok(Arc::new(result.with_precision_and_scale(
        result_precision,
        result_scale,
    )?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Decimal128Array;

    #[test]
    fn rounds_half_away_from_zero() {
        let left = Decimal128Array::from(vec![Some(5), Some(-5), Some(1), Some(-1)])
            .with_precision_and_scale(2, 0)
            .unwrap();
        let right = Decimal128Array::from(vec![Some(3), Some(3), Some(32), Some(32)])
            .with_precision_and_scale(2, 0)
            .unwrap();
        let result = div_half_away_from_zero(&left, &right).unwrap();
        let result = result.as_primitive::<Decimal128Type>();
        assert_eq!(
            result.iter().collect::<Vec<_>>(),
            vec![Some(16667), Some(-16667), Some(313), Some(-313)]
        );
    }
}
