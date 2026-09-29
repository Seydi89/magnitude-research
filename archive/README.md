# Archived authoritative runners

This directory preserves the final code-only versions of the historical pilots. It is intentionally isolated from the supported package in `src/magnitude_research` because those projects evolved independently and contain overlapping implementations and command-line conventions.

Use the archive when an old numerical result must be reproduced exactly. Use the shared package and top-level `experiments/` for new work.

Not included:

- raw or prepared datasets;
- embedding caches;
- generated HTML, plots, JSON reports and CSV result dumps;
- superseded numbered package generations.

The mapping from archived project to research conclusion is in `docs/EXPERIMENT_LEDGER.md`.
