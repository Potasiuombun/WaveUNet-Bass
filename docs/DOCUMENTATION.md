# Documentation Map

This file organizes the project documentation and identifies which docs to use first.

## Canonical Docs (use these first)

1. `README.md`
- Primary entrypoint.
- Current build, inspect, train, and overfit commands.
- Current split/leakage guarantee.

2. `docs/guides/SERIALIZED_QUICK_REF.md`
- Operational command reference for serialized datasets.
- CLI flags and troubleshooting.

3. `configs/baseline.yaml`
- Baseline experiment config.

4. `configs/overfit_nmse.yaml`
- Overfit-oriented config (batch size 64, NMSE-focused loss).

## Implementation-Focused Docs

- `docs/guides/SERIALIZED_DATASET_IMPL.md`
- `docs/guides/SERIALIZED_DATASET_GUIDE.md`
- `docs/guides/MIGRATION.md`

Use these when changing internals of dataset building or migrating older workflows.

## Historical / Delivery Notes

These contain milestone snapshots and may be partially outdated:

- `docs/archive/BASELINE_SUMMARY.md`
- `docs/archive/DELIVERABLES.md`
- `docs/archive/DELIVERY_CHECKLIST.md`
- `docs/archive/REQUIREMENTS_FULFILLMENT.md`
- `docs/archive/SERIALIZED_DATASET_DELIVERY.md`
- `docs/archive/LEGACY_DATASET_SUPPORT.md`
- `docs/archive/LEGACY_FORMAT_PATCH_SUMMARY.md`

Treat them as archival context, not source of truth for commands.

## Current Source of Truth for Split Behavior

- Code: `src/data/splits.py` (`split_by_track`)
- Guarantee: all frames from the same track remain in exactly one split.
- Result: no cross-split leakage from overlapping neighboring frames.

## Current Source of Truth for Metadata Consistency Checks

- Build-time checks: `scripts/prepare_dataset.py`
- Inspector checks: `scripts/inspect_dataset.py`
- Data assembly: `src/data/serialized.py`
