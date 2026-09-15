<!--
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements. See the NOTICE file
distributed with this work for additional information
regarding copyright ownership. The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied. See the License for the
specific language governing permissions and limitations
under the License.
-->

# Sort-merge join spill readback

The materializing stream buffers the current group of equal join keys. Payloads
that cannot be admitted to the memory pool spill to temporary files; join keys
and per-row match state remain resident.

Readback retains the original spill file and caches decoded payloads only while
the next output operation needs them. Before restoring another working set, it
evicts decoded payloads no longer needed and releases their reservations. A
later streamed row can revisit the same spilled group without rewriting files
or retaining the whole decoded group. Removing the buffered head also removes
its cached file handle and adjusts the remaining working-set indices.

Full joins restore batches with pending null output, plus the dequeued head for
deferred filter failures; they do not eagerly reload every matched batch.
This bounds accumulation of **visited spill payloads**, not total process RSS:
join keys, match state, output batches and the current readback window still
consume memory, and readback may temporarily exceed the configured pool limit.
Repeated streamed keys can require reading a spill file again after its decoded
payload is evicted. This trades possible extra read I/O for bounded accumulation
of restored payloads; it is not a claim of faster execution for every spilled join.

`spill_restored_batches_do_not_accumulate` checks bounded reservations, exact
spill/non-spill output for inner/left/right/full joins, revisiting the group and
early cancellation. The core SMJ RSS validation tests additionally exercise a
large single-key group through SQL and real spill files.
