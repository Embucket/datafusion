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

//! Decimal division that rounds half away from zero (Snowflake semantics).
//!
//! Arrow's `div` kernel truncates the decimal quotient toward zero, while
//! Snowflake rounds the last digit half away from zero (verified live:
//! `5/3 = 1.666667`, `-5/3 = -1.666667`, `5/2000000 = 0.000003`). This kernel
//! replicates arrow's decimal division exactly — the same result
//! precision/scale derivation (`result_scale = min(s1 + 4, MAX_SCALE)`), the
//! same overflow and divide-by-zero errors — but corrects the truncated
//! quotient using the remainder.

use std::cmp::Ordering;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, Datum, PrimitiveArray};
use arrow::compute::kernels::arity::try_binary;
use arrow::datatypes::{
    ArrowNativeType, ArrowNativeTypeOp, DataType, Decimal32Type, Decimal64Type,
    Decimal128Type, Decimal256Type, DecimalType,
};
use arrow::error::ArrowError;

/// Whether `lhs / rhs` is a decimal division this kernel handles (same-width
/// decimal operands, the shape produced by DataFusion's binary type coercion).
pub(super) fn is_decimal_division(lhs: &DataType, rhs: &DataType) -> bool {
    matches!(
        (lhs, rhs),
        (DataType::Decimal32(..), DataType::Decimal32(..))
            | (DataType::Decimal64(..), DataType::Decimal64(..))
            | (DataType::Decimal128(..), DataType::Decimal128(..))
            | (DataType::Decimal256(..), DataType::Decimal256(..))
    )
}

/// Divide two decimal [`Datum`], rounding the quotient half away from zero.
///
/// Drop-in replacement for arrow's `div` on decimal inputs: identical result
/// type derivation and error behavior, only the final-digit rounding differs.
pub(super) fn div_half_away_from_zero(
    lhs: &dyn Datum,
    rhs: &dyn Datum,
) -> Result<ArrayRef, ArrowError> {
    let (l, l_s) = lhs.get();
    let (r, r_s) = rhs.get();
    match (l.data_type(), r.data_type()) {
        (DataType::Decimal32(..), DataType::Decimal32(..)) => {
            decimal_div_op::<Decimal32Type>(l, l_s, r, r_s)
        }
        (DataType::Decimal64(..), DataType::Decimal64(..)) => {
            decimal_div_op::<Decimal64Type>(l, l_s, r, r_s)
        }
        (DataType::Decimal128(..), DataType::Decimal128(..)) => {
            decimal_div_op::<Decimal128Type>(l, l_s, r, r_s)
        }
        (DataType::Decimal256(..), DataType::Decimal256(..)) => {
            decimal_div_op::<Decimal256Type>(l, l_s, r, r_s)
        }
        (l_t, r_t) => Err(ArrowError::InvalidArgumentError(format!(
            "Invalid decimal division: {l_t} / {r_t}"
        ))),
    }
}

fn decimal_div_op<T: DecimalType>(
    l: &dyn Array,
    l_s: bool,
    r: &dyn Array,
    r_s: bool,
) -> Result<ArrayRef, ArrowError> {
    let l = l.as_primitive::<T>();
    let r = r.as_primitive::<T>();

    let (p1, s1, s2) = match (l.data_type(), r.data_type()) {
        (DataType::Decimal32(p1, s1), DataType::Decimal32(_, s2))
        | (DataType::Decimal64(p1, s1), DataType::Decimal64(_, s2))
        | (DataType::Decimal128(p1, s1), DataType::Decimal128(_, s2))
        | (DataType::Decimal256(p1, s1), DataType::Decimal256(_, s2)) => (*p1, *s1, *s2),
        _ => unreachable!("callers dispatch on matching decimal types"),
    };

    // Result type derivation identical to arrow-arith's `decimal_op` Op::Div.
    let result_scale = s1.saturating_add(4).min(T::MAX_SCALE);
    let mul_pow = result_scale - s1 + s2;
    let result_precision = (mul_pow.saturating_add(p1 as i8) as u8).min(T::MAX_PRECISION);

    let (l_mul, r_mul) = match mul_pow.cmp(&0) {
        Ordering::Greater => (
            T::Native::usize_as(10).pow_checked(mul_pow as _)?,
            T::Native::ONE,
        ),
        Ordering::Equal => (T::Native::ONE, T::Native::ONE),
        Ordering::Less => (
            T::Native::ONE,
            T::Native::usize_as(10).pow_checked(mul_pow.neg_wrapping() as _)?,
        ),
    };

    let op = |l: T::Native, r: T::Native| -> Result<T::Native, ArrowError> {
        let ln = l.mul_checked(l_mul)?;
        let rn = r.mul_checked(r_mul)?;
        let mut quotient = ln.div_checked(rn)?;
        // Round half away from zero: if twice the remainder reaches the
        // divisor, bump the quotient one step away from zero.
        // `abs_rem >= abs_rn - abs_rem` avoids overflowing 2*rem.
        let rem = ln.mod_wrapping(rn);
        if !rem.is_zero() {
            let abs_rem = if rem.is_lt(T::Native::ZERO) {
                rem.neg_wrapping()
            } else {
                rem
            };
            let abs_rn = if rn.is_lt(T::Native::ZERO) {
                rn.neg_wrapping()
            } else {
                rn
            };
            if abs_rem.is_ge(abs_rn.sub_wrapping(abs_rem)) {
                let negative = ln.is_lt(T::Native::ZERO) != rn.is_lt(T::Native::ZERO);
                quotient = if negative {
                    quotient.sub_checked(T::Native::ONE)?
                } else {
                    quotient.add_checked(T::Native::ONE)?
                };
            }
        }
        Ok(quotient)
    };

    // Scalar/array dispatch mirroring arrow-arith's `try_op!` macro.
    let array: PrimitiveArray<T> = match (l_s, r_s) {
        (true, true) | (false, false) => try_binary(l, r, op)?,
        (true, false) => match (l.null_count() == 0).then(|| l.value(0)) {
            None => PrimitiveArray::new_null(r.len()),
            Some(lv) => r.try_unary(|rv| op(lv, rv))?,
        },
        (false, true) => match (r.null_count() == 0).then(|| r.value(0)) {
            None => PrimitiveArray::new_null(l.len()),
            Some(rv) => l.try_unary(|lv| op(lv, rv))?,
        },
    };

    Ok(Arc::new(array.with_precision_and_scale(
        result_precision,
        result_scale,
    )?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Decimal128Array, Scalar};

    fn dec128(values: Vec<Option<i128>>, precision: u8, scale: i8) -> Decimal128Array {
        Decimal128Array::from(values)
            .with_precision_and_scale(precision, scale)
            .unwrap()
    }

    fn values(result: &ArrayRef) -> (Vec<Option<i128>>, DataType) {
        let arr = result.as_primitive::<Decimal128Type>();
        (arr.iter().collect(), arr.data_type().clone())
    }

    #[test]
    fn rounds_half_away_from_zero_array_array() {
        // NUMBER(2,0) operands -> arrow result scale 0 + 4.
        // 5/3 = 1.666666... -> 1.6667 (16667, not the truncated 16666)
        // -5/3 -> -1.6667; 1/8 = 0.125 exact at scale 4 -> 1250;
        // 5/4 = 1.25 exact -> 12500; 1/16 = 0.0625 -> rounds to 0.0625 (exact).
        let l = dec128(vec![Some(5), Some(-5), Some(1), Some(5), None], 2, 0);
        let r = dec128(vec![Some(3), Some(3), Some(8), Some(4), Some(2)], 2, 0);
        let result = div_half_away_from_zero(&l, &r).unwrap();
        let (vals, dt) = values(&result);
        assert_eq!(dt, DataType::Decimal128(6, 4));
        assert_eq!(
            vals,
            vec![Some(16667), Some(-16667), Some(1250), Some(12500), None]
        );
    }

    #[test]
    fn rounds_exact_half_away_from_zero_not_even() {
        // 1/32 = 0.03125 exactly: the digit past result scale 4 is an exact
        // half. Half-away-from-zero (Snowflake, verified live) gives 0.0313;
        // truncation and half-even would both give 0.0312, so this pins the
        // rounding mode, both signs.
        let l = dec128(vec![Some(1), Some(-1)], 2, 0);
        let r = dec128(vec![Some(32), Some(32)], 2, 0);
        let result = div_half_away_from_zero(&l, &r).unwrap();
        let (vals, dt) = values(&result);
        assert_eq!(dt, DataType::Decimal128(6, 4));
        assert_eq!(vals, vec![Some(313), Some(-313)]);
    }

    #[test]
    fn scalar_array_combos_match() {
        let l = dec128(vec![Some(5)], 2, 0);
        let r = dec128(vec![Some(3), Some(-3), None], 2, 0);
        let scalar_l = Scalar::new(l);
        let result = div_half_away_from_zero(&scalar_l, &r).unwrap();
        let (vals, _) = values(&result);
        assert_eq!(vals, vec![Some(16667), Some(-16667), None]);

        let l = dec128(vec![Some(5), Some(-5), None], 2, 0);
        let r = dec128(vec![Some(3)], 2, 0);
        let scalar_r = Scalar::new(r);
        let result = div_half_away_from_zero(&l, &scalar_r).unwrap();
        let (vals, _) = values(&result);
        assert_eq!(vals, vec![Some(16667), Some(-16667), None]);
    }

    #[test]
    fn divide_by_zero_errors() {
        let l = dec128(vec![Some(5)], 2, 0);
        let r = dec128(vec![Some(0)], 2, 0);
        let err = div_half_away_from_zero(&l, &r).unwrap_err();
        assert!(
            err.to_string().contains("Divide by zero"),
            "expected divide-by-zero, got: {err}"
        );
    }

    #[test]
    fn scale_is_capped_like_arrow() {
        // s1 = 36 -> result scale capped at MAX_SCALE (38), matching arrow.
        let l = dec128(vec![Some(10_i128.pow(36))], 38, 36);
        let r = dec128(vec![Some(3)], 2, 0);
        let result = div_half_away_from_zero(&l, &r).unwrap();
        let (_, dt) = values(&result);
        assert_eq!(dt, DataType::Decimal128(38, 38));
    }
}
