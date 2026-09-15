# Sort-merge join maintenance

Keep spill-backed payload readback limited to the current output working set.
Preserve backing files until their buffered batches are removed: repeated keys
on the streamed side can revisit an already-read group. Keep memory reservation
growth/shrink balanced, including eviction, dequeuing, errors and cancellation.

When changing restoration or freeze logic, test full-join null output and
deferred filter failures as well as ordinary matches. Do not remove required
restores merely because a batch has no current matched output indices.

Run the SMJ unit tests and the core SMJ memory-limit validation before accepting
spill changes. Preserve the RSS allowance; a lower accounting metric alone is
not evidence that resident memory decreased. Document changes to the readback
lifecycle and its remaining memory limits in README.md.
