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

use arrow_schema::DataType;
use std::mem::size_of;

use crate::joins::MapOffset;
use crate::joins::chain::traverse_chain;
use arrow::array::{Array, ArrayRef, AsArray, BooleanArray};
use arrow::buffer::BooleanBuffer;
use arrow::datatypes::ArrowNumericType;
use datafusion_common::{Result, ScalarValue, internal_err};

/// Conversion to the dense map's index domain. Native <=64-bit integers retain
/// their branch-free wrapping representation. Decimal128 values must fit i64:
/// truncating arbitrary i128 probes would alias distinct keys separated by 2^64.
trait ArrayMapKey: Copy {
    fn map_key(self) -> Option<u64>;
}

macro_rules! native_map_keys {
    ($($t:ty),*) => {$(
        impl ArrayMapKey for $t {
            #[inline]
            fn map_key(self) -> Option<u64> {
                Some(self as u64)
            }
        }
    )*};
}

native_map_keys!(i8, i16, i32, i64, u8, u16, u32, u64);

impl ArrayMapKey for i128 {
    #[inline]
    fn map_key(self) -> Option<u64> {
        i64::try_from(self).ok().map(|value| value as u64)
    }
}

/// Downcast supported native integers and Decimal128 without materializing a
/// narrowed Arrow array. Decimal range eligibility is checked from build bounds.
///
/// Usage: `downcast_supported_integer!(data_type => (Method, arg1, arg2, ...))`
///
/// The `Method` must be an associated method of [`ArrayMap`] that is generic over
/// `<T: ArrowNumericType>` and allow `T::Native: ArrayMapKey`.
macro_rules! downcast_supported_integer {
    ($DATA_TYPE:expr => ($METHOD:ident $(, $ARGS:expr)*)) => {
        match $DATA_TYPE {
            arrow::datatypes::DataType::Int8 => ArrayMap::$METHOD::<arrow::datatypes::Int8Type>($($ARGS),*),
            arrow::datatypes::DataType::Int16 => ArrayMap::$METHOD::<arrow::datatypes::Int16Type>($($ARGS),*),
            arrow::datatypes::DataType::Int32 => ArrayMap::$METHOD::<arrow::datatypes::Int32Type>($($ARGS),*),
            arrow::datatypes::DataType::Int64 => ArrayMap::$METHOD::<arrow::datatypes::Int64Type>($($ARGS),*),
            arrow::datatypes::DataType::UInt8 => ArrayMap::$METHOD::<arrow::datatypes::UInt8Type>($($ARGS),*),
            arrow::datatypes::DataType::UInt16 => ArrayMap::$METHOD::<arrow::datatypes::UInt16Type>($($ARGS),*),
            arrow::datatypes::DataType::UInt32 => ArrayMap::$METHOD::<arrow::datatypes::UInt32Type>($($ARGS),*),
            arrow::datatypes::DataType::UInt64 => ArrayMap::$METHOD::<arrow::datatypes::UInt64Type>($($ARGS),*),
            arrow::datatypes::DataType::Decimal128(..) => ArrayMap::$METHOD::<arrow::datatypes::Decimal128Type>($($ARGS),*),
            _ => {
                return internal_err!(
                    "Unsupported type for ArrayMap: {:?}",
                    $DATA_TYPE
                );
            }
        }
    };
}

/// A dense map for single-column integer join keys within a limited range.
///
/// Maps join keys to build-side indices using direct array indexing:
/// `data[val - min_val_in_build_side] -> val_idx_in_build_side + 1`.
///
/// NULL values are ignored on both the build side and the probe side.
///
/// # Handling Negative Numbers with `wrapping_sub`
///
/// This implementation supports signed integer ranges (e.g., `[-5, 5]`) efficiently by
/// treating them as `u64` (Two's Complement) and relying on the bitwise properties of
/// wrapping arithmetic (`wrapping_sub`).
///
/// In Two's Complement representation, `a_signed - b_signed` produces the same bit pattern
/// as `a_unsigned.wrapping_sub(b_unsigned)` (modulo 2^N). This allows us to perform
/// range calculations and zero-based index mapping uniformly for both signed and unsigned
/// types without branching.
///
/// ## Examples
///
/// Consider an `Int64` range `[-5, 5]`.
/// * `min_val (-5)` casts to `u64`: `...11111011` (`u64::MAX - 4`)
/// * `max_val (5)` casts to `u64`: `...00000101` (`5`)
///
/// **1. Range Calculation**
///
/// ```text
/// In modular arithmetic, this is equivalent to:
///   (5 - (2^64 - 5)) mod 2^64
/// = (5 - 2^64 + 5) mod 2^64
/// = (10 - 2^64) mod 2^64
/// = 10
///
/// ```
/// The resulting `range` (10) correctly represents the size of the interval `[-5, 5]`.
///
/// **2. Index Lookup (in `get_matched_indices_with_limit_offset`)**
///
/// For a probe value of `0` (which is stored as `0u64`):
/// ```text
/// In modular arithmetic, this is equivalent to:
///   (0 - (2^64 - 5)) mod 2^64
/// = (-2^64 + 5) mod 2^64
/// = 5
/// ```
/// This correctly maps `-5` to index `0`, `0` to index `5`, etc.
#[derive(Debug)]
pub struct ArrayMap {
    // data[probSideVal-offset] -> valIdxInBuildSide + 1; 0 for absent
    data: Vec<u32>,
    // min val in buildSide
    offset: u64,
    // next[buildSideIdx] -> next matching valIdxInBuildSide + 1; 0 for end of chain.
    // If next is empty, it means there are no duplicate keys (no conflicts).
    // It uses the same chain-based conflict resolution as [`JoinHashMapType`].
    next: Vec<u32>,
    num_of_distinct_key: usize,
}

impl ArrayMap {
    pub fn is_supported_type(data_type: &DataType) -> bool {
        matches!(
            data_type,
            DataType::Int8
                | DataType::Int16
                | DataType::Int32
                | DataType::Int64
                | DataType::UInt8
                | DataType::UInt16
                | DataType::UInt32
                | DataType::UInt64
                | DataType::Decimal128(..)
        )
    }

    pub(crate) fn key_to_u64(v: &ScalarValue) -> Option<u64> {
        match v {
            ScalarValue::Int8(Some(v)) => Some(*v as u64),
            ScalarValue::Int16(Some(v)) => Some(*v as u64),
            ScalarValue::Int32(Some(v)) => Some(*v as u64),
            ScalarValue::Int64(Some(v)) => Some(*v as u64),
            ScalarValue::UInt8(Some(v)) => Some(*v as u64),
            ScalarValue::UInt16(Some(v)) => Some(*v as u64),
            ScalarValue::UInt32(Some(v)) => Some(*v as u64),
            ScalarValue::UInt64(Some(v)) => Some(*v),
            ScalarValue::Decimal128(Some(v), _, _) => v.map_key(),
            _ => None,
        }
    }

    /// Estimates the maximum memory usage for an `ArrayMap` with the given parameters.
    ///
    pub fn estimate_memory_size(min_val: u64, max_val: u64, num_rows: usize) -> usize {
        let range = Self::calculate_range(min_val, max_val);
        if range >= usize::MAX as u64 {
            return usize::MAX;
        }
        let size = (range + 1) as usize;
        size.saturating_mul(size_of::<u32>())
            .saturating_add(num_rows.saturating_mul(size_of::<u32>()))
    }

    pub fn calculate_range(min_val: u64, max_val: u64) -> u64 {
        max_val.wrapping_sub(min_val)
    }

    #[inline]
    fn key_to_index(key: u64, offset: u64, data_len: usize) -> Option<usize> {
        let idx = key.wrapping_sub(offset);
        if idx < data_len as u64 {
            Some(idx as usize)
        } else {
            None
        }
    }

    /// Creates a new [`ArrayMap`] from the given array of join keys.
    ///
    /// Note: This function processes only the non-null values in the input `array`,
    /// ignoring any rows where the key is `NULL`.
    ///
    pub(crate) fn try_new(array: &ArrayRef, min_val: u64, max_val: u64) -> Result<Self> {
        let range = Self::calculate_range(min_val, max_val);
        if range >= usize::MAX as u64 {
            return internal_err!("ArrayMap key range is too large to be allocated.");
        }
        let size = (range + 1) as usize;

        let mut data: Vec<u32> = vec![0; size];
        let mut next: Vec<u32> = vec![];
        let mut num_of_distinct_key = 0;

        downcast_supported_integer!(
            array.data_type() => (
                fill_data,
                array,
                min_val,
                &mut data,
                &mut next,
                &mut num_of_distinct_key
            )
        )?;

        Ok(Self {
            data,
            offset: min_val,
            next,
            num_of_distinct_key,
        })
    }

    fn fill_data<T: ArrowNumericType>(
        array: &ArrayRef,
        offset_val: u64,
        data: &mut [u32],
        next: &mut Vec<u32>,
        num_of_distinct_key: &mut usize,
    ) -> Result<()>
    where
        T::Native: ArrayMapKey,
    {
        let arr = array.as_primitive::<T>();
        // Iterate in reverse to maintain FIFO order when there are duplicate keys.
        for (i, val) in arr.iter().enumerate().rev() {
            if let Some(val) = val {
                let Some(key) = val.map_key() else {
                    return internal_err!(
                        "ArrayMap build key exceeds its supported domain"
                    );
                };
                let Some(idx) = Self::key_to_index(key, offset_val, data.len()) else {
                    return internal_err!("failed build Array idx >= data.len()");
                };

                if data[idx] != 0 {
                    if next.is_empty() {
                        *next = vec![0; array.len()]
                    }
                    next[i] = data[idx]
                } else {
                    *num_of_distinct_key += 1;
                }
                data[idx] = (i) as u32 + 1;
            }
        }
        Ok(())
    }

    pub fn num_of_distinct_key(&self) -> usize {
        self.num_of_distinct_key
    }

    /// Returns the memory usage of this [`ArrayMap`] in bytes.
    pub fn size(&self) -> usize {
        self.data.capacity() * size_of::<u32>() + self.next.capacity() * size_of::<u32>()
    }

    pub fn get_matched_indices_with_limit_offset(
        &self,
        prob_side_keys: &[ArrayRef],
        limit: usize,
        current_offset: MapOffset,
        probe_indices: &mut Vec<u32>,
        build_indices: &mut Vec<u64>,
    ) -> Result<Option<MapOffset>> {
        if prob_side_keys.len() != 1 {
            return internal_err!(
                "ArrayMap expects 1 join key, but got {}",
                prob_side_keys.len()
            );
        }
        let array = &prob_side_keys[0];

        downcast_supported_integer!(
            array.data_type() => (
                lookup_and_get_indices,
                self,
                array,
                limit,
                current_offset,
                probe_indices,
                build_indices
            )
        )
    }

    /// Looks up `key` (a raw probe value cast to `u64`) in the build side,
    /// returning the 1-based build-side slot if the key maps to a non-empty
    /// bucket, or `None` otherwise.
    #[inline]
    fn get_value(&self, key: u64) -> Option<u32> {
        let idx = Self::key_to_index(key, self.offset, self.data.len())?;
        let value = self.data[idx];
        (value != 0).then_some(value)
    }

    fn lookup_and_get_indices<T: ArrowNumericType>(
        &self,
        array: &ArrayRef,
        limit: usize,
        current_offset: MapOffset,
        probe_indices: &mut Vec<u32>,
        build_indices: &mut Vec<u64>,
    ) -> Result<Option<MapOffset>>
    where
        T::Native: ArrayMapKey,
    {
        probe_indices.clear();
        build_indices.clear();

        let arr = array.as_primitive::<T>();

        let have_null = arr.null_count() > 0;

        if self.next.is_empty() {
            for prob_idx in current_offset.0..arr.len() {
                if build_indices.len() == limit {
                    return Ok(Some((prob_idx, None)));
                }

                // short circuit
                if have_null && arr.is_null(prob_idx) {
                    continue;
                }
                // SAFETY: prob_idx is guaranteed to be within bounds by the loop range.
                let Some(prob_val) = unsafe { arr.value_unchecked(prob_idx) }.map_key()
                else {
                    continue;
                };
                let Some(build_value) = self.get_value(prob_val) else {
                    continue;
                };
                build_indices.push((build_value - 1) as u64);
                probe_indices.push(prob_idx as u32);
            }
            Ok(None)
        } else {
            let mut remaining_output = limit;
            let to_skip = match current_offset {
                // None `initial_next_idx` indicates that `initial_idx` processing hasn't been started
                (idx, None) => idx,
                // Zero `initial_next_idx` indicates that `initial_idx` has been processed during
                // previous iteration, and it should be skipped
                (idx, Some(0)) => idx + 1,
                // Otherwise, process remaining `initial_idx` matches by traversing `next_chain`,
                // to start with the next index
                (idx, Some(next_idx)) => {
                    let is_last = idx == arr.len() - 1;
                    if let Some(next_offset) = traverse_chain(
                        &self.next,
                        idx,
                        next_idx as u32,
                        &mut remaining_output,
                        probe_indices,
                        build_indices,
                        is_last,
                    ) {
                        return Ok(Some(next_offset));
                    }
                    idx + 1
                }
            };

            for prob_side_idx in to_skip..arr.len() {
                if remaining_output == 0 {
                    return Ok(Some((prob_side_idx, None)));
                }

                if have_null && arr.is_null(prob_side_idx) {
                    continue;
                }

                let is_last = prob_side_idx == arr.len() - 1;

                // SAFETY: prob_idx is guaranteed to be within bounds by the loop range.
                let Some(prob_val) =
                    unsafe { arr.value_unchecked(prob_side_idx) }.map_key()
                else {
                    continue;
                };
                let Some(build_idx) = self.get_value(prob_val) else {
                    continue;
                };

                if let Some(offset) = traverse_chain(
                    &self.next,
                    prob_side_idx,
                    build_idx,
                    &mut remaining_output,
                    probe_indices,
                    build_indices,
                    is_last,
                ) {
                    return Ok(Some(offset));
                }
            }
            Ok(None)
        }
    }

    pub fn contain_keys(&self, probe_side_keys: &[ArrayRef]) -> Result<BooleanArray> {
        if probe_side_keys.len() != 1 {
            return internal_err!(
                "ArrayMap join expects 1 join key, but got {}",
                probe_side_keys.len()
            );
        }
        let array = &probe_side_keys[0];

        downcast_supported_integer!(
            array.data_type() => (
                contain_keys_helper,
                self,
                array
            )
        )
    }

    fn contain_keys_helper<T: ArrowNumericType>(
        &self,
        array: &ArrayRef,
    ) -> Result<BooleanArray>
    where
        T::Native: ArrayMapKey,
    {
        let arr = array.as_primitive::<T>();
        let buffer = BooleanBuffer::collect_bool(arr.len(), |i| {
            if arr.is_null(i) {
                return false;
            }
            // SAFETY: i is within bounds [0, arr.len())
            unsafe { arr.value_unchecked(i) }
                .map_key()
                .is_some_and(|key| self.get_value(key).is_some())
        });
        Ok(BooleanArray::new(buffer, None))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Int32Array;
    use arrow::array::Int64Array;
    use arrow::array::UInt64Array;
    use std::sync::Arc;

    #[test]
    fn decimal_bounds_require_a_lossless_index_domain() {
        for scale in [0, 2, 19] {
            for value in [i128::from(i64::MIN), -1, 0, i128::from(i64::MAX)] {
                assert_eq!(
                    ArrayMap::key_to_u64(&ScalarValue::Decimal128(
                        Some(value),
                        38,
                        scale
                    )),
                    Some(value as u64)
                );
            }
            for value in [
                i128::from(i64::MIN) - 1,
                i128::from(i64::MAX) + 1,
                1_i128 << 64,
            ] {
                assert_eq!(
                    ArrayMap::key_to_u64(&ScalarValue::Decimal128(
                        Some(value),
                        38,
                        scale
                    )),
                    None
                );
            }
        }
        assert_eq!(
            ArrayMap::key_to_u64(&ScalarValue::Decimal128(None, 38, 0)),
            None
        );
    }

    #[test]
    fn decimal_probes_do_not_alias_after_u64_truncation() -> Result<()> {
        use arrow::array::Decimal128Array;

        // Exercise both the unique-key and duplicate-chain lookup paths, and
        // scaled decimals: the map indexes unscaled values, never SQL casts.
        for duplicates in [false, true] {
            for scale in [0, 2, 19] {
                let values = if duplicates {
                    vec![Some(-1), Some(0), None, Some(1), Some(1)]
                } else {
                    vec![Some(-1), Some(0), None, Some(1)]
                };
                let build: ArrayRef = Arc::new(
                    Decimal128Array::from(values).with_precision_and_scale(38, scale)?,
                );
                let map = ArrayMap::try_new(&build, -1_i64 as u64, 1)?;
                let probe = [Arc::new(
                    Decimal128Array::from(vec![
                        Some(-1),
                        Some(0),
                        Some(1),
                        None,
                        Some((1_i128 << 64) - 1),
                        Some(1_i128 << 64),
                        Some((1_i128 << 64) + 1),
                        Some(-(1_i128 << 64)),
                        Some(i128::from(i64::MIN) - 1),
                        Some(i128::from(i64::MAX) + 1),
                    ])
                    .with_precision_and_scale(38, scale)?,
                ) as ArrayRef];
                let contains = map.contain_keys(&probe)?;
                assert_eq!(
                    contains.iter().collect::<Vec<_>>(),
                    vec![
                        Some(true),
                        Some(true),
                        Some(true),
                        Some(false),
                        Some(false),
                        Some(false),
                        Some(false),
                        Some(false),
                        Some(false),
                        Some(false)
                    ]
                );
                let mut offset = Some((0, None));
                let (mut probes, mut builds, mut matches) = (vec![], vec![], vec![]);
                while let Some(next) = offset {
                    offset = map.get_matched_indices_with_limit_offset(
                        &probe,
                        1,
                        next,
                        &mut probes,
                        &mut builds,
                    )?;
                    matches.extend(probes.iter().copied().zip(builds.iter().copied()));
                }
                let mut expected = vec![(0, 0), (1, 1), (2, 3)];
                if duplicates {
                    expected.push((2, 4));
                }
                assert_eq!(matches, expected);
            }
        }
        Ok(())
    }

    #[test]
    fn invalid_decimal_build_cannot_silently_truncate() -> Result<()> {
        use arrow::array::Decimal128Array;
        let build: ArrayRef = Arc::new(
            Decimal128Array::from(vec![(1_i128 << 64) + 1])
                .with_precision_and_scale(38, 0)?,
        );
        assert!(ArrayMap::try_new(&build, 0, 2).is_err());
        Ok(())
    }

    #[test]
    fn test_array_map_limit_offset_duplicate_elements() -> Result<()> {
        let build: ArrayRef = Arc::new(Int32Array::from(vec![1, 1, 2]));
        let map = ArrayMap::try_new(&build, 1, 2)?;
        let probe = [Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef];

        let mut prob_idx = Vec::new();
        let mut build_idx = Vec::new();
        let mut next = Some((0, None));
        let mut results = vec![];

        while let Some(o) = next {
            next = map.get_matched_indices_with_limit_offset(
                &probe,
                1,
                o,
                &mut prob_idx,
                &mut build_idx,
            )?;
            results.push((prob_idx.clone(), build_idx.clone(), next));
        }

        let expected = vec![
            (vec![0], vec![0], Some((0, Some(2)))),
            (vec![0], vec![1], Some((0, Some(0)))),
            (vec![1], vec![2], None),
        ];
        assert_eq!(results, expected);
        Ok(())
    }

    #[test]
    fn test_array_map_with_limit_and_misses() -> Result<()> {
        let build: ArrayRef = Arc::new(Int32Array::from(vec![1, 2]));
        let map = ArrayMap::try_new(&build, 1, 2)?;
        let probe = [Arc::new(Int32Array::from(vec![10, 1, 2])) as ArrayRef];

        let (mut p_idx, mut b_idx) = (vec![], vec![]);
        // Skip 10, find 1, next is 2
        let next = map.get_matched_indices_with_limit_offset(
            &probe,
            1,
            (0, None),
            &mut p_idx,
            &mut b_idx,
        )?;
        assert_eq!(p_idx, vec![1]);
        assert_eq!(b_idx, vec![0]);
        assert_eq!(next, Some((2, None)));

        // Find 2, end
        let next = map.get_matched_indices_with_limit_offset(
            &probe,
            1,
            next.unwrap(),
            &mut p_idx,
            &mut b_idx,
        )?;
        assert_eq!(p_idx, vec![2]);
        assert_eq!(b_idx, vec![1]);
        assert!(next.is_none());
        Ok(())
    }

    #[test]
    fn test_array_map_with_build_duplicates_and_misses() -> Result<()> {
        let build_array: ArrayRef = Arc::new(Int32Array::from(vec![1, 1]));
        let array_map = ArrayMap::try_new(&build_array, 1, 1)?;
        // prob: 10(m), 1(h1, h2), 20(m), 1(h1, h2)
        let probe_array: ArrayRef = Arc::new(Int32Array::from(vec![10, 1, 20, 1]));
        let prob_side_keys = [probe_array];

        let mut prob_indices = Vec::new();
        let mut build_indices = Vec::new();

        // batch_size=3, should get 2 matches from first '1' and 1 match from second '1'
        let result_offset = array_map.get_matched_indices_with_limit_offset(
            &prob_side_keys,
            3,
            (0, None),
            &mut prob_indices,
            &mut build_indices,
        )?;

        assert_eq!(prob_indices, vec![1, 1, 3]);
        assert_eq!(build_indices, vec![0, 1, 0]);
        assert_eq!(result_offset, Some((3, Some(2))));
        Ok(())
    }

    #[test]
    fn test_array_map_rejects_large_out_of_range_probe_key() -> Result<()> {
        let build: ArrayRef =
            Arc::new(UInt64Array::from(vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]));
        let map = ArrayMap::try_new(&build, 0, 10)?;

        assert_eq!(ArrayMap::key_to_index(3, 0, 11), Some(3));

        // Pick a key for which the computed bucket offset is larger than
        // u32::MAX but has low 32 bits equal to 3. It must be bounds-checked
        // before casting to usize, otherwise 32-bit targets can truncate it
        // into range.
        let out_of_range_key = (1_u64 << 32) + 3;
        assert_eq!(ArrayMap::key_to_index(out_of_range_key, 0, 11), None);

        let probe = [Arc::new(UInt64Array::from(vec![
            Some(3),
            Some(out_of_range_key),
            Some(11),
            None,
        ])) as ArrayRef];

        let mut matched_probe_indices = vec![];
        let mut matched_build_indices = vec![];
        let next = map.get_matched_indices_with_limit_offset(
            &probe,
            10,
            (0, None),
            &mut matched_probe_indices,
            &mut matched_build_indices,
        )?;
        assert_eq!(matched_probe_indices, vec![0]);
        assert_eq!(matched_build_indices, vec![3]);
        assert!(next.is_none());

        let contains = map.contain_keys(&probe)?;
        assert!(contains.value(0));
        assert!(!contains.value(1));
        assert!(!contains.value(2));
        assert!(!contains.value(3));

        Ok(())
    }

    #[test]
    fn test_array_map_i64_with_negative_and_positive_numbers() -> Result<()> {
        // Build array with a mix of negative and positive i64 values, no duplicates
        let build_array: ArrayRef = Arc::new(Int64Array::from(vec![-5, 0, 5, -2, 3, 10]));
        let min_val = -5_i128;
        let max_val = 10_i128;

        let array_map = ArrayMap::try_new(&build_array, min_val as u64, max_val as u64)?;

        // Probe array
        let probe_array: ArrayRef = Arc::new(Int64Array::from(vec![0, -5, 10, -1]));
        let prob_side_keys = [Arc::clone(&probe_array)];

        let mut prob_indices = Vec::new();
        let mut build_indices = Vec::new();

        // Call once to get all matches
        let result_offset = array_map.get_matched_indices_with_limit_offset(
            &prob_side_keys,
            10, // A batch size larger than number of probes
            (0, None),
            &mut prob_indices,
            &mut build_indices,
        )?;

        // Expected matches, in probe-side order:
        // Probe 0 (value 0) -> Build 1 (value 0)
        // Probe 1 (value -5) -> Build 0 (value -5)
        // Probe 2 (value 10) -> Build 5 (value 10)
        let expected_prob_indices = vec![0, 1, 2];
        let expected_build_indices = vec![1, 0, 5];

        assert_eq!(prob_indices, expected_prob_indices);
        assert_eq!(build_indices, expected_build_indices);
        assert!(result_offset.is_none());

        Ok(())
    }
}
