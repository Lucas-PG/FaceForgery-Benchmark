# Generated experiment outputs

The default output root for `explicability.py` is this directory. See [the execution guide](../docs/explicability.md).

The explicit `prepare` command creates `plan.json`, frozen cohort manifests, calibrated prediction summaries, and per-model outcomes. Explicit `run` commands populate `task1/<model>/<method>/`, `task2/<model>/<method>/`, `task3/<model>/<method>/`, and `task3_controls/<model>/<method>/` with raw attribution arrays, metadata, and images. The `compare` command produces Task 2 native-coordinate boards; `report` audits complete, unsupported, missing, and corrupt cells.

No actual experiment results are implied by this directory scaffold. Runtime outputs, dataset images, and private annotation linkages are ignored by Git. Share only permitted, reviewed research artifacts; never publish the blinded-review linkage key.
