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

use std::mem::{size_of, size_of_val};
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, AsArray, BooleanArray, StructArray, UInt32Array, new_empty_array,
    new_null_array,
};
use arrow::compute::{interleave, take};
use arrow::datatypes::DataType;
use datafusion_common::{DataFusionError, Result, internal_err};
use datafusion_expr::{EmitTo, GroupsAccumulator};

/// Keeps only the first row of each group in compact Arrow arrays, avoiding
/// ScalarValue materialization for nested values such as Variant structs.
pub(super) struct UnorderedStructFirstValueGroupsAccumulator {
    data_type: DataType,
    ignore_nulls: bool,
    locations: Vec<Option<(usize, usize)>>,
    chunks: Vec<Option<ArrayRef>>,
    chunk_references: Vec<usize>,
    free_chunks: Vec<usize>,
    chunk_bytes: usize,
}

fn compact_view_children(input: ArrayRef) -> ArrayRef {
    match input.data_type().clone() {
        DataType::BinaryView => {
            let array = input.as_binary_view();
            let retained = array
                .data_buffers()
                .iter()
                .map(|buffer| buffer.capacity())
                .sum::<usize>();
            if retained > array.total_buffer_bytes_used().saturating_mul(2) {
                Arc::new(array.gc())
            } else {
                input
            }
        }
        DataType::Utf8View => {
            let array = input.as_string_view();
            let retained = array
                .data_buffers()
                .iter()
                .map(|buffer| buffer.capacity())
                .sum::<usize>();
            if retained > array.total_buffer_bytes_used().saturating_mul(2) {
                Arc::new(array.gc())
            } else {
                input
            }
        }
        DataType::Struct(fields) => {
            let array = input.as_struct();
            let columns = array
                .columns()
                .iter()
                .cloned()
                .map(compact_view_children)
                .collect::<Vec<_>>();
            if columns
                .iter()
                .zip(array.columns())
                .all(|(new, old)| Arc::ptr_eq(new, old))
            {
                input
            } else {
                Arc::new(StructArray::new(fields, columns, array.nulls().cloned()))
            }
        }
        _ => input,
    }
}

impl UnorderedStructFirstValueGroupsAccumulator {
    pub(super) fn new(data_type: DataType, ignore_nulls: bool) -> Self {
        Self {
            data_type,
            ignore_nulls,
            locations: Vec::new(),
            chunks: Vec::new(),
            chunk_references: Vec::new(),
            free_chunks: Vec::new(),
            chunk_bytes: 0,
        }
    }

    fn update(
        &mut self,
        input: &ArrayRef,
        group_indices: &[usize],
        filter: Option<&BooleanArray>,
        total_num_groups: usize,
    ) -> Result<()> {
        if input.len() != group_indices.len()
            || filter.is_some_and(|filter| filter.len() != input.len())
        {
            return internal_err!("FIRST_VALUE input and group lengths differ");
        }
        self.locations.resize(total_num_groups, None);
        let mut rows = Vec::new();
        let mut groups = Vec::new();
        for (row, &group) in group_indices.iter().enumerate() {
            if group >= total_num_groups {
                return internal_err!("FIRST_VALUE group index is out of bounds");
            }
            if filter.is_some_and(|filter| filter.is_null(row) || !filter.value(row))
                || (self.ignore_nulls && input.is_null(row))
                || self.locations[group].is_some()
            {
                continue;
            }
            let row = u32::try_from(row).map_err(|_| {
                DataFusionError::Internal(
                    "FIRST_VALUE batch exceeds u32 row indices".into(),
                )
            })?;
            self.locations[group] = Some((usize::MAX, rows.len()));
            rows.push(row);
            groups.push(group);
        }
        if rows.is_empty() {
            return Ok(());
        }

        let compact = match take(input.as_ref(), &UInt32Array::from(rows), None) {
            Ok(compact) => compact_view_children(compact),
            Err(error) => {
                for group in groups {
                    self.locations[group] = None;
                }
                return Err(error.into());
            }
        };
        let slot = self.free_chunks.pop().unwrap_or(self.chunks.len());
        self.chunk_bytes += compact.get_array_memory_size();
        if slot == self.chunks.len() {
            self.chunks.push(Some(compact));
            self.chunk_references.push(groups.len());
        } else {
            self.chunks[slot] = Some(compact);
            self.chunk_references[slot] = groups.len();
        }
        for (index, group) in groups.into_iter().enumerate() {
            self.locations[group] = Some((slot, index));
        }
        Ok(())
    }

    fn emit(&mut self, emit_to: EmitTo) -> Result<(ArrayRef, BooleanArray)> {
        let locations = emit_to.take_needed(&mut self.locations);
        let is_set =
            BooleanArray::from(locations.iter().map(Option::is_some).collect::<Vec<_>>());
        let values = if locations.is_empty() {
            new_empty_array(&self.data_type)
        } else {
            let null = new_null_array(&self.data_type, 1);
            let mut sources: Vec<&dyn Array> = Vec::with_capacity(self.chunks.len() + 1);
            let mut remap = vec![0; self.chunks.len()];
            for (slot, chunk) in self.chunks.iter().enumerate() {
                if let Some(chunk) = chunk {
                    remap[slot] = sources.len();
                    sources.push(chunk.as_ref());
                }
            }
            let null_slot = sources.len();
            sources.push(null.as_ref());
            let indices = locations
                .iter()
                .map(|location| {
                    location.map_or((null_slot, 0), |(slot, row)| (remap[slot], row))
                })
                .collect::<Vec<_>>();
            interleave(&sources, &indices)?
        };

        for (slot, _) in locations.into_iter().flatten() {
            self.chunk_references[slot] -= 1;
            if self.chunk_references[slot] == 0 {
                if let Some(chunk) = self.chunks[slot].take() {
                    self.chunk_bytes -= chunk.get_array_memory_size();
                }
                self.free_chunks.push(slot);
            }
        }
        Ok((values, is_set))
    }
}

impl GroupsAccumulator for UnorderedStructFirstValueGroupsAccumulator {
    fn update_batch(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        opt_filter: Option<&BooleanArray>,
        total_num_groups: usize,
    ) -> Result<()> {
        let Some(input) = values.first() else {
            return internal_err!("FIRST_VALUE requires one argument");
        };
        self.update(input, group_indices, opt_filter, total_num_groups)
    }

    fn evaluate(&mut self, emit_to: EmitTo) -> Result<ArrayRef> {
        self.emit(emit_to).map(|(values, _)| values)
    }

    fn state(&mut self, emit_to: EmitTo) -> Result<Vec<ArrayRef>> {
        let (values, is_set) = self.emit(emit_to)?;
        Ok(vec![values, Arc::new(is_set)])
    }

    fn merge_batch(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        total_num_groups: usize,
    ) -> Result<()> {
        if values.len() != 2 {
            return internal_err!("FIRST_VALUE requires value and is_set state");
        }
        self.update(
            &values[0],
            group_indices,
            Some(values[1].as_boolean()),
            total_num_groups,
        )
    }

    fn convert_to_state(
        &self,
        values: &[ArrayRef],
        opt_filter: Option<&BooleanArray>,
    ) -> Result<Vec<ArrayRef>> {
        let Some(input) = values.first() else {
            return internal_err!("FIRST_VALUE requires one argument");
        };
        let is_set = BooleanArray::from(
            (0..input.len())
                .map(|row| {
                    opt_filter
                        .is_none_or(|filter| filter.is_valid(row) && filter.value(row))
                        && (!self.ignore_nulls || input.is_valid(row))
                })
                .collect::<Vec<_>>(),
        );
        Ok(vec![Arc::clone(input), Arc::new(is_set)])
    }

    fn size(&self) -> usize {
        size_of_val(self)
            + self.locations.capacity() * size_of::<Option<(usize, usize)>>()
            + self.chunks.capacity() * size_of::<Option<ArrayRef>>()
            + self.chunk_references.capacity() * size_of::<usize>()
            + self.free_chunks.capacity() * size_of::<usize>()
            + self.chunk_bytes
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::builder::make_view;
    use arrow::array::{
        Array, ArrayRef, AsArray, BinaryViewArray, BooleanArray, Int64Array, StructArray,
    };
    use arrow::buffer::{Buffer, ScalarBuffer};
    use arrow::datatypes::{DataType, Field};
    use datafusion_common::Result;
    use datafusion_expr::{EmitTo, GroupsAccumulator};

    use super::UnorderedStructFirstValueGroupsAccumulator;

    fn values(numbers: Vec<i64>) -> ArrayRef {
        Arc::new(StructArray::from(vec![(
            Arc::new(Field::new("value", DataType::Int64, false)),
            Arc::new(Int64Array::from(numbers)) as ArrayRef,
        )]))
    }

    fn numbers(array: &ArrayRef) -> Vec<i64> {
        let values = array
            .as_struct()
            .column(0)
            .as_primitive::<arrow::datatypes::Int64Type>();
        (0..array.len()).map(|index| values.value(index)).collect()
    }

    #[test]
    fn first_values_survive_partial_emission_and_merge() -> Result<()> {
        let batch = values(vec![10, 20, 11, 30]);
        let mut partial = UnorderedStructFirstValueGroupsAccumulator::new(
            batch.data_type().clone(),
            false,
        );
        partial.update_batch(&[batch], &[0, 1, 0, 2], None, 3)?;
        let first = partial.state(EmitTo::First(1))?;
        assert_eq!(numbers(&first[0]), vec![10]);
        assert_eq!(first[1].as_boolean(), &BooleanArray::from(vec![true]));

        let remainder = partial.state(EmitTo::All)?;
        assert_eq!(numbers(&remainder[0]), vec![20, 30]);
        let mut final_acc = UnorderedStructFirstValueGroupsAccumulator::new(
            remainder[0].data_type().clone(),
            false,
        );
        final_acc.merge_batch(&remainder, &[0, 1], 2)?;
        assert_eq!(numbers(&final_acc.evaluate(EmitTo::All)?), vec![20, 30]);
        Ok(())
    }

    #[test]
    fn null_and_filter_state_are_preserved() -> Result<()> {
        let data_type = values(vec![1]).data_type().clone();
        let nulls = arrow::array::new_null_array(&data_type, 2);
        let mut respect =
            UnorderedStructFirstValueGroupsAccumulator::new(data_type.clone(), false);
        respect.update_batch(std::slice::from_ref(&nulls), &[0, 1], None, 2)?;
        let state = respect.state(EmitTo::All)?;
        assert!(state[0].is_null(0));
        assert_eq!(state[1].as_boolean(), &BooleanArray::from(vec![true, true]));

        let mut ignore = UnorderedStructFirstValueGroupsAccumulator::new(data_type, true);
        ignore.update_batch(
            &[nulls],
            &[0, 1],
            Some(&BooleanArray::from(vec![true, false])),
            2,
        )?;
        let state = ignore.state(EmitTo::All)?;
        assert_eq!(
            state[1].as_boolean(),
            &BooleanArray::from(vec![false, false])
        );
        Ok(())
    }

    #[test]
    fn retains_only_winners_from_wide_input_batch() -> Result<()> {
        let batch = values((0..10_000).collect());
        let input_bytes = batch.get_array_memory_size();
        let mut accumulator = UnorderedStructFirstValueGroupsAccumulator::new(
            batch.data_type().clone(),
            false,
        );
        accumulator.update_batch(&[batch], &vec![0; 10_000], None, 1)?;
        assert!(accumulator.size() < input_bytes / 10);
        assert_eq!(numbers(&accumulator.evaluate(EmitTo::All)?), vec![0]);
        Ok(())
    }

    #[test]
    fn compact_view_winner_does_not_pin_source_buffers() -> Result<()> {
        let payloads = (0..10_000)
            .map(|index| format!("variant value {index:05} with a long payload"))
            .collect::<Vec<_>>();
        let view = Arc::new(BinaryViewArray::from_iter_values(
            payloads.iter().map(String::as_bytes),
        )) as ArrayRef;
        let batch = Arc::new(StructArray::from(vec![
            (
                Arc::new(Field::new("metadata", DataType::BinaryView, false)),
                Arc::clone(&view),
            ),
            (
                Arc::new(Field::new("value", DataType::BinaryView, false)),
                view,
            ),
        ])) as ArrayRef;
        let input_bytes = batch.get_array_memory_size();
        let mut accumulator = UnorderedStructFirstValueGroupsAccumulator::new(
            batch.data_type().clone(),
            false,
        );
        accumulator.update_batch(&[batch], &vec![0; 10_000], None, 1)?;
        assert!(accumulator.size() < input_bytes / 20);
        let selected = accumulator.evaluate(EmitTo::All)?;
        assert_eq!(
            selected.as_struct().column(1).as_binary_view().value(0),
            payloads[0].as_bytes()
        );
        Ok(())
    }

    #[test]
    fn compact_view_winner_releases_unused_buffer_capacity() -> Result<()> {
        let value = b"retained payload";
        let mut backing = Vec::with_capacity(4 * 1024 * 1024);
        backing.extend_from_slice(value);
        let view = BinaryViewArray::try_new(
            ScalarBuffer::from(vec![make_view(value, 0, 0)]),
            vec![Buffer::from_vec(backing)],
            None,
        )?;
        let batch = Arc::new(StructArray::from(vec![(
            Arc::new(Field::new("value", DataType::BinaryView, false)),
            Arc::new(view) as ArrayRef,
        )])) as ArrayRef;
        let mut accumulator = UnorderedStructFirstValueGroupsAccumulator::new(
            batch.data_type().clone(),
            false,
        );
        accumulator.update_batch(&[batch], &[0], None, 1)?;
        assert!(accumulator.size() < 1024);
        let selected = accumulator.evaluate(EmitTo::All)?;
        assert_eq!(
            selected.as_struct().column(0).as_binary_view().value(0),
            value
        );
        Ok(())
    }
}
