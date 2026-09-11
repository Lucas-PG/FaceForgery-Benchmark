# Delivery status: implementation and manuscript revision

This supersedes the earlier handoff's framing of image/GPU access as a prerequisite for completing the requested code and writing work. The requested delivery is a research-backed implementation and pre-execution manuscript revision using the known ICLR baseline; full research execution is not part of this delivery.

## Delivered

The opt-in package implements all three protocols: a frozen common 64-image cohort; a validation-selected matched RGB/frequency comparison; and a complete six-architecture shared-error population with label-matched controls and a separate three-seed sensitivity intersection. Twelve method adapters, native-coordinate figures, numerical diagnostics, head-randomization and perturbation controls, raw artifacts, resumable shards, coverage, and prediction-table export are included. The original training, evaluation, model factories, and dependency lock are preserved.

The revised manuscript is compiled at `paper.pdf` and `paper/explicability/main.pdf`, with editable LaTeX and BibTeX sources. `paper/original-sibgrapi.pdf` preserves the original PDF byte-for-byte. `paper/explicability/build.json` binds the generated PDF to its source and build run. The compiled revision has 10 total pages including bibliography and appendices, with no overfull-box warnings in the recorded build. It is an author-review/pre-execution revision, not an anonymous submission or an acceptance claim.

## What was checked

The lightweight contracts workflow successfully ran 21 small synthetic CPU tests after the NumPy 2.x curve-integration fix. The checked contracts include exact deterministic sampling, rejection of incomplete populations, explicit validation curation, threshold calibration, analytic IG, frozen-parameter CAM, group coefficient conservation, stochastic reproducibility, all existing Fourier encoding modes including six/seven-channel hybrids, and numeric-map rendering. Python syntax and LaTeX/citation compilation were also checked. These are software checks, not weeks-long benchmark experiments or validation on the released trained models.

Evidence: [successful contract run](https://github.com/Lucas-PG/FaceForgery-Benchmark/actions/runs/34619443119), [successful revised-manuscript build](https://github.com/Lucas-PG/FaceForgery-Benchmark/actions/runs/34619884676). Later changes to sources require their own checks; the workflow evidence is tied to its actual commit, not an unqualified assertion about all future revisions.

## Evidence boundaries

| Claim or artifact | Status in this delivery |
| --- | --- |
| Earlier benchmark scores quoted in the revised article | Explicitly attributed to the supplied prior paper |
| Protocol definitions and executable implementation | Delivered |
| Mathematical/code consequences, such as frequency bins not being facial coordinates | Explained with assumptions and primary-source context |
| New full-cohort heatmaps and model ranking on those maps | Await later explicit execution; not invented |
| Current-release metrics and shared-error counts | Reproducible export commands provided; no transient earlier draft counts treated as bundled evidence |
| Nuisance, skin-tone, demographic, or causal explanations of failures | Hypotheses only; require suitable annotations and analysis |
| Full six-checkpoint integration under research hardware and caches | Not claimed by the synthetic tests |
| Final author approval and venue submission | Outside this implementation delivery |

`RESEARCH.md` retains notes from the earlier evidence audit. The current manuscript deliberately does not rely on transient reanalysis counts that were not delivered with their corresponding result files. It keeps the previous study and the supplied fine-tuned weight collection separate rather than treating them as a controlled ablation.

## Next execution is a separate operation

The team can review this PR without providing images or GPU access. When experiments are scheduled, follow `docs/explicability.md`, freeze the plan from the actual prediction files, run the desired shards, inspect all coverage and controls, and replace only those manuscript hypotheses supported by observed outputs. No automatic GPU run, training matrix, or PR merge is configured.
