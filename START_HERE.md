# Start here — audited robustness and generalization research

This is the integration branch **`robust-generalization` in `Lucas-PG/FaceForgery-Benchmark`**, proposed through **PR #2 into this fork's `ICLR`**. The source repository `lucasdocunha/FaceForgery-Benchmark` is read-only. No experiment or automatic merge is required to review the implementation.

## What this revision is for

The next research question is **when spatial and spectral evidence remains useful under image degradation and unseen forgery/source distributions**. The implementation prioritizes measurement validity, matched augmentation experiments, error complementarity and a bounded fusion pilot. A learned gate is a hypothesis about usefulness, **not calibrated uncertainty or a claim of architectural novelty**.

The full CPU software suite passed **209 tests and 20 subtests** at the implementation version recorded in [VERIFICATION.md](research/robustness/VERIFICATION.md). The existing six-page article, expanded reference and presentation remain intact. No unmeasured performance has been inserted into either manuscript.

## Where to find everything

| Location | Purpose |
|---|---|
| [Import provenance](research/robustness/import.json) | Exact upstream commit/tree and earlier fork history preserved in this branch |
| [Complete execution guide](docs/robustness.md) | Environment, conversion, audit, calibration, training, evaluation, analysis and migration commands |
| [Audit](research/robustness/AUDIT.md) | Confirmed code issues, historical evidence boundaries and required validation |
| [DF40 subset audit and guide](docs/df40-audit.md) | Distinguish the imported local subset from official DF40 protocols; safe conversion route |
| [Research rationale](research/robustness/RESEARCH.md) | Primary-source basis, related work, novelty boundaries and research priorities |
| [Publication plan](research/robustness/PUBLICATION.md) | Proposed argument, figures, measurable contributions and evidence gates |
| [Verification](research/robustness/VERIFICATION.md) | Observed software checks and exact limits of their evidence |
| `research_cli.py` | Explicit audit, planning, training, inference, analysis and export entry point |
| `src/robustness/manifests.py`, `artifacts.py`, `provenance.py` | Canonical labels, stable identities, integrity records and checkpoint-bound predictions |
| `src/robustness/imaging.py`, `legacy_encoding.py` | Matched clean/corrupted RGB views; preserved original evaluation encodings |
| `src/robustness/models.py`, `engine.py` | Spatial/spectral fusion ablations, optional standard distillation and epoch-safe resume |
| `src/robustness/statistics.py`, `analysis.py`, `commands.py`, `identity.py` | Grouped uncertainty, complementarity, shared failures, generator/seed reports and timing |
| `src/robustness/df40.py` | Reviewed local-DF40-subset adapter with explicit coverage limitations |
| `src/robustness/explain.py`, [pilot-XAI guide](docs/pilot-xai.md) | New-pilot fixed-cohort IG, occlusion and declared CNN-layer Grad-CAM |
| `configs/research/` | Bounded pilot templates and factorized ablation ladder; no automatic HPC submission |
| `scripts/prepare_celeb_df.py`, `evaluate_celeb_df.py` | Corrected, fail-fast Celeb-DF preparation/evaluation with explicit label conventions |
| `src/models/_regime.py` | Correct new initialization semantics and explicit historical constructor compatibility |
| `tests/robustness/` | Synthetic software tests, not face-forgery scientific experiments |
| `src/explicability/`, `docs/explicability.md` | Preserved three-protocol XAI workflow and twelve method adapters |
| `paper.pdf`, `paper/six-page/` | Preserved six-page IEEE/SIBGRAPI-format primary manuscript |
| `paper/explicability/`, `paper/original-sibgrapi.pdf` | Preserved ten-page expanded reference and original supplied manuscript |
| `presentation/` | Preserved 15-slide pt-BR PDF/PPTX/HTML and accompanying notebook |

## Review order

Read the audit before comparing new scores. Review the canonical manifest and prediction contracts next, then the paired augmentation/fusion ablations. The execution guide separates inexpensive artifact analysis from explicit GPU execution. The publication guide distinguishes implemented methods, historical reported values, hypotheses and the evidence still needed for empirical claims.

## Important migration changes

1. Canonical labels are **0 = real, 1 = fake**. Celeb-DF source labels are converted because their declared convention is opposite, never because AUC is low.
2. Missing, corrupt, duplicate or misaligned samples fail instead of being replaced by another image or silently omitted.
3. New model builders treat `finetune_robust` as pretrained and `scratch_robust` as random initialization. Unversioned old run configs retain `legacy-v1` construction semantics when loading weights. This does not retroactively certify their training histories.
4. `evaluate_celeb_df.py` requires a declared checkpoint, canonical manifest, explicit class-1 meaning and source-validation calibration. Old cache/queue commands need migration and reject unsupported arguments.
5. New research training writes content-identified runs under `outputs/research/`; it does not overwrite the old `models/` layout or imported results.
6. The imported DF40 preparation builds a local subset. The new adapter records that scope rather than relabeling it as the complete official benchmark. Source labels and group annotations must be independently reviewed.

## Explanation coordinates

The new-pilot explanation path produces end-to-end **RGB-input attributions**, including the differentiable FFT path. These are not native-frequency maps. The retained pure-frequency-model XAI route remains separate. Both preserve fixed targets, declared baselines, raw arrays and provenance; neither produces scientific findings without later execution and analysis.

## Boundaries

Historical files in `results/` remain intact as reports from their producing pipelines. Especially for Celeb-DF, robust initialization, incomplete seed tables, custom DF40 subsets and test-selected ensembles, their claims must pass the audit before reuse. Statistical intervals condition on the supplied groups and checkpoints; they do not establish causal explanations or across-seed uncertainty. Face images, private annotations, checkpoint caches and meeting transcripts are not published by these commands.

**No full-dataset training, external-dataset inference, SOTA reproduction or new real-image XAI result is claimed by this implementation delivery.** The PR provides the implementation and documented path to those experiments, without requiring new hardware access to review it.
