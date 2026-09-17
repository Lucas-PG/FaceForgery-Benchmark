# Delivery status: implementation and manuscript revision

This supersedes the earlier handoff's framing of image/GPU access as a prerequisite for completing the requested code and writing work. The requested delivery is a research-backed implementation and pre-execution manuscript revision using the known ICLR baseline; full research execution is not part of this delivery.

## Delivered

The opt-in package implements all three protocols: a frozen common 64-image cohort; a validation-selected matched RGB/frequency comparison; and a complete six-architecture shared-error population with label-matched controls and a separate three-seed sensitivity intersection. Twelve method adapters, native-coordinate figures, numerical diagnostics, head-randomization and perturbation controls, raw artifacts, resumable shards, coverage, and prediction-table export are included. The original training, evaluation, model factories, and dependency lock are preserved.

**The primary manuscript is `paper.pdf`, now exactly six pages including references in the original IEEE/SIBGRAPI two-column conference format.** Its identical compiled copy, editable LaTeX source, and build record are in `paper/six-page/`. The shared bibliography remains `paper/explicability/references.bib`.

The ten-page expanded manuscript at `paper/explicability/main.pdf`, its source, and its operational appendices remain unchanged as a reference. `paper/original-sibgrapi.pdf` preserves the original baseline PDF byte-for-byte. `paper/six-page/build.json` binds the primary PDF to its source, records the six-page and format checks, and records hashes verifying preservation of both reference PDFs. The primary edition uses normal 10-point body text and standard IEEE conference margins, not scaled text or reduced margins. Both revised editions retain the pre-execution evidence boundary and do not claim submission or acceptance of this extension.

## What was checked

The earlier lightweight contracts workflow successfully ran 21 small synthetic CPU tests after the NumPy 2.x curve-integration fix. The checked contracts include exact deterministic sampling, rejection of incomplete populations, explicit validation curation, threshold calibration, analytic IG, frozen-parameter CAM, group coefficient conservation, stochastic reproducibility, all existing Fourier encoding modes including six/seven-channel hybrids, and numeric-map rendering. Python syntax was also checked. These are software checks, not weeks-long benchmark experiments or validation on the released trained models. The six-page formatting update did not change the XAI implementation or rerun research experiments.

Evidence: [successful contract run](https://github.com/Lucas-PG/FaceForgery-Benchmark/actions/runs/34619443119), [preserved ten-page reference build](https://github.com/Lucas-PG/FaceForgery-Benchmark/actions/runs/34621541196), [six-page IEEE build](https://github.com/Lucas-PG/FaceForgery-Benchmark/actions/runs/34695303971), and [validated primary-paper packaging](https://github.com/Lucas-PG/FaceForgery-Benchmark/actions/runs/34695579409).

The primary-paper packaging checks exactly six pages, original US Letter page dimensions, IEEE conference class, no overfull boxes or unresolved citations, nonempty pages, safe text bounds, source/build identity, and unchanged reference hashes. Page images and extracted text are included in the build artifact. The reference-only packaging workflow can no longer overwrite the primary paper. Later source changes require their own checks; workflow evidence is tied to its actual commit, not an unqualified assertion about future revisions.

## Evidence boundaries

| Claim or artifact | Status in this delivery |
| --- | --- |
| Earlier benchmark scores quoted in the revised article | Explicitly attributed to the supplied prior paper |
| Protocol definitions and executable implementation | Delivered |
| Six-page primary manuscript and preserved ten-page reference | Compiled, committed, and separately identified |
| Mathematical/code consequences, such as frequency bins not being facial coordinates | Explained with assumptions and primary-source context |
| New full-cohort heatmaps and model ranking on those maps | Await later explicit execution; not invented |
| Current-release metrics and shared-error counts | Reproducible export commands provided; no transient earlier draft counts treated as bundled evidence |
| Nuisance, skin-tone, demographic, or causal explanations of failures | Hypotheses only; require suitable annotations and analysis |
| Full six-checkpoint integration under research hardware and caches | Not claimed by the synthetic tests |
| Final author approval and venue submission | Outside this implementation delivery |

`RESEARCH.md` retains notes from the earlier evidence audit. The current manuscript deliberately does not rely on transient reanalysis counts that were not delivered with their corresponding result files. It keeps the previous study and the supplied fine-tuned weight collection separate rather than treating them as a controlled ablation.

## Next execution is a separate operation

The team can review this PR without providing images or GPU access. When experiments are scheduled, follow `docs/explicability.md`, freeze the plan from the actual prediction files, run the desired shards, inspect all coverage and controls, and replace only those manuscript hypotheses supported by observed outputs. No automatic GPU run, training matrix, or PR merge is configured.
