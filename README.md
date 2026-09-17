# FaceForgery Benchmark — robustness and generalization

Research on spatial, spectral and pretrained cues for face-forgery detection under degradation and distribution shift.

**Start with [the project map](START_HERE.md) and [the execution guide](docs/robustness.md).** This branch integrates the latest supplied upstream snapshot with the fork's previous explainability, manuscript and presentation work, then adds an audited research workflow. Source provenance is recorded in [`research/robustness/import.json`](research/robustness/import.json).

## Research question

When do spatial and spectral cues remain useful under image degradation, unseen forgery methods and source-domain changes? The next study separates these questions, audits historical evidence, and tests a bounded set of matched augmentation/fusion controls before committing to expensive replication.

## Included

Canonical fake-positive labels; collision-safe identities; fail-fast image loading; checkpoint-bound prediction exports; source-validation calibration; frame/video evaluation; grouped paired bootstrap intervals; error complementarity and shared-failure analysis; generator/seed summaries; measured model-compute profiling; paired augmentation and residual RGB/spectral fusion pilots; optional standard teacher–student distillation; and preserved XAI protocols.

The fusion gate, consistency objective and standard distillation are **testable baselines**, not claims of novelty or demonstrated superiority. Read the [research rationale](research/robustness/RESEARCH.md), [audit](research/robustness/AUDIT.md) and [publication plan](research/robustness/PUBLICATION.md).

## Quick start

```bash
# After installing the isolated environment in docs/robustness.md:
python research_cli.py --help
python -m pytest tests/robustness tests/explicability -q

# Review a plan; this does not train or submit a GPU job:
python research_cli.py train --config configs/research/pilot-rgb.yaml
```

The pilot config requires certified source train/validation manifests and real local data paths. The guide describes conversion, audit, artifact-only analysis and explicit later execution. Training requires `--execute`. Historical GPU runs are not replayed by CI.

## Manuscripts and presentation

| Artifact | Location |
|---|---|
| Six-page primary paper, original IEEE/SIBGRAPI two-column format | [`paper.pdf`](paper.pdf), editable source in [`paper/six-page/`](paper/six-page/) |
| Ten-page expanded reference | [`paper/explicability/main.pdf`](paper/explicability/main.pdf) |
| Original supplied SIBGRAPI manuscript | [`paper/original-sibgrapi.pdf`](paper/original-sibgrapi.pdf) |
| pt-BR presentation | [`presentation/apresentacao.pdf`](presentation/apresentacao.pdf), [PowerPoint](presentation/apresentacao.pptx), [notebook](presentation/apresentacao.ipynb) |
| Research-to-publication handoff | [`research/robustness/PUBLICATION.md`](research/robustness/PUBLICATION.md) |

These preserved documents are author-review/historical material, not newly executed fusion results or an acceptance claim. New results must come from the audited artifacts before entering an empirical paper. The six-page manuscript is not silently replaced by the expanded reference.

## Compatibility and evidence

The [archived upstream README](docs/legacy-readme.md) describes the previous training and evaluation workflow. Its historical badge/test assertions are preserved as source content, not a claim about this revision. The original dependency lock and reported result files remain unchanged. New initialization semantics are versioned; old checkpoint reconstruction retains its original constructor behavior. The Celeb-DF evaluator has an intentionally stricter CLI; old queue scripts should not be launched unchanged.

Only the **Lucas-PG fork** and its new working branch are modified. `lucasdocunha/FaceForgery-Benchmark` is read-only. Neither `main` nor either `ICLR` branch is directly updated. See [verification](research/robustness/VERIFICATION.md) for precisely what software was checked. No new dataset-wide metrics, real-image heatmaps, demographic findings or conference/journal acceptance are inferred from passing tests.
