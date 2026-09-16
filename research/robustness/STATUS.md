# Delivery scope and research status

The implementation is on `robust-generalization` in `Lucas-PG/FaceForgery-Benchmark`, proposed by PR #2 into the fork's `ICLR`. The upstream repository was read only. The immutable upstream snapshot and the preserved earlier fork history are recorded in `import.json`.

## Implemented

Canonical fake-positive label conversion, stable image/video/group identities, integrity-bound manifests and prediction exports, source-validation calibration, strict frame/video evaluation, legacy FFT compatibility, paired group-bootstrap comparisons, error complementarity, shared-error populations, per-generator reports, seed-completeness/condition checks, measured model-compute profiling, deterministic paired augmentation, bounded spatial/spectral fusion ablations, optional standard distillation, checkpointed epoch resume, and primary new-pilot explanations with frozen cohorts.

The known evaluation label mismatch, unreadable-image substitution and family-dependent robust-regime initialization semantics have explicit fixes and regression tests. Historical checkpoint reconstruction remains versioned; the changes do not retroactively certify how old local GPU runs were initialized.

## Preserved

The existing six-page primary `paper.pdf`, ten-page reference, original SIBGRAPI manuscript, pt-BR presentation PDF/PowerPoint/HTML/notebook, twelve-method XAI implementation, original dependency lock, and imported historical result tables remain available. The upstream README is archived at `docs/legacy-readme.md`; the root README and `START_HERE.md` now explain the integrated workflow.

## Verification versus experimental evidence

The complete CPU software suite is exercised through the read-only `research-checks.yml` workflow. It includes generated-data training/save/load/evaluation, interruption/resume equivalence, native-encoding compatibility, file-integrity tests and preservation checks for the paper/presentation. See `VERIFICATION.md` and the final PR checks for version-specific observed results. One-time import and maintenance workflows were removed after their changes were committed.

No full MFFI/Celeb-DF/DF40 run, GPU training matrix, SOTA reproduction, new real-image heatmap, calibrated reliability guarantee or publication acceptance is asserted. Model-only timing from synthetic resident inputs is explicitly separated from deployment cost. A file hash is an integrity check, not a signature or proof of visual ground truth.

## How to proceed

Start with `START_HERE.md`, then `docs/robustness.md` and `docs/pilot-xai.md`. The scientific decisions, limitations and prior-art rationale are in `AUDIT.md`, `RESEARCH.md` and `PUBLICATION.md`. Training and pilot XAI execution require explicit `--execute`; planning and artifact analysis do not submit work to the institution's hardware.
