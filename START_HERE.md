# Start here — audited robustness and generalization research

This is the integration branch **`robust-generalization` in `Lucas-PG/FaceForgery-Benchmark`**, proposed through **PR #2 into this fork's `ICLR`**. The source repository `lucasdocunha/FaceForgery-Benchmark` is read-only. No experiment or automatic merge is required to review the implementation.

## What this revision is for

The next research question is **when spatial and spectral evidence remains useful under image degradation and unseen forgery/source distributions**. The implementation prioritizes measurement validity, matched augmentation experiments, error complementarity, and a bounded fusion pilot. A learned gate is a hypothesis about usefulness, **not calibrated uncertainty or a claim of architectural novelty**.

A six-page author-review paper is not equivalent to an executed research study. The current primary paper, expanded reference, and presentation are preserved. The new research direction, contribution/evidence map, literature, protocol, and writing changes are in `research/robustness/`. No unmeasured performance is inserted into either manuscript.

## Where to find everything

| Location | Purpose |
|---|---|
| `research/robustness/import.json` | Exact upstream commit/tree and earlier fork work preserved in this branch |
| `docs/robustness.md` | Full command-by-command execution and migration guide |
| `research/robustness/AUDIT.md` | Confirmed software issues, historical evidence boundaries, and required checks |
| `research/robustness/RESEARCH.md` | Primary-source rationale, related work, novelty boundaries, and publication strategy |
| `research/robustness/PUBLICATION.md` | Proposed paper argument, figures, measurable contributions, and acceptance-independent quality gates |
| `research_cli.py` | Explicit audit, planning, training, inference, analysis, and export entry point |
| `src/robustness/manifests.py`, `artifacts.py`, `provenance.py` | Canonical labels, stable identities, file integrity, and checkpoint-bound predictions |
| `src/robustness/imaging.py`, `legacy_encoding.py` | Matched clean/corrupted RGB views; preserved original evaluation encodings |
| `src/robustness/models.py`, `engine.py` | RGB/spectral fusion ablations, optional standard distillation, checkpoint resume |
| `src/robustness/statistics.py`, `analysis.py`, `commands.py` | Grouped uncertainty, error complementarity, shared failures, generator/seed tables, timing |
| `configs/research/` | Bounded pilot templates and a factorized ablation ladder; no automatic HPC submission |
| `scripts/prepare_celeb_df.py`, `evaluate_celeb_df.py` | Corrected, fail-fast Celeb-DF preparation/evaluation with explicit label conventions |
| `src/models/_regime.py` | Correct new initialization semantics; explicit historical reconstruction compatibility |
| `tests/robustness/` | Small synthetic software tests, not face-forgery scientific experiments |
| `src/explicability/`, `docs/explicability.md` | Preserved three-protocol XAI workflow and twelve method adapters |
| `paper.pdf`, `paper/six-page/` | Preserved six-page IEEE/SIBGRAPI-format primary manuscript |
| `paper/explicability/`, `paper/original-sibgrapi.pdf` | Preserved ten-page expanded reference and original supplied manuscript |
| `presentation/` | Preserved pt-BR slides in PDF/PPTX/HTML and accompanying notebook |

## Review order

Read the audit before comparing new scores. Review the canonical manifest and prediction contracts next, then the paired augmentation/fusion ablations. The execution guide separates inexpensive artifact analysis from explicit GPU execution. The publication guide distinguishes implemented methods, historical reported values, hypotheses, and the evidence still needed for empirical claims.

## Important migration changes

1. Canonical labels are **0 = real, 1 = fake**. Celeb-DF source labels are inverted **because its documented convention is opposite**, never because its AUC is low.
2. Missing, corrupt, duplicate, or misaligned samples now fail instead of being replaced by another image or silently omitted.
3. New model builders treat `finetune_robust` as pretrained and `scratch_robust` as random initialization. Unversioned old run configs retain `legacy-v1` construction semantics when loading saved weights. This does not retroactively certify their training histories.
4. `evaluate_celeb_df.py` now requires one declared checkpoint, a canonical manifest, explicit class-1 semantics, and a source-validation calibration. Old cache/queue commands need migration; they fail on unsupported arguments rather than silently producing misleading tables.
5. New research training writes content-identified runs under `outputs/research/`; it does not overwrite the old `models/` layout or imported results.

## Boundaries

Historical files in `results/` remain intact as reports from their producing pipelines. Especially for Celeb-DF, robust initialization, incomplete seed tables, and test-selected ensembles, their claims must pass the audit before reuse. Statistical intervals are conditional on supplied groups and checkpoints; they do not establish causal explanations or across-seed uncertainty. Images, annotations, cache weights, and private meeting transcripts are not published by these commands.

**No full-dataset training, external-dataset inference, SOTA reproduction, or new real-image XAI result is claimed by this implementation delivery.** Software verification is recorded separately in `research/robustness/VERIFICATION.md`.
