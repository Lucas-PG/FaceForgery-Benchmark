# What changed, and why

## Article and experimental question

The article was rewritten around one question: **does a training-regime improvement on MFFI Test-D carry over to the external evaluation?** The title, abstract, introduction, related-work structure, methods, results, limitations and conclusion were rewritten to answer that question. The result is a seven-page IEEE conference-format article, not a lightly edited version of the earlier XAI/protocol manuscript.

The explanation material remains, but a selected two-image gallery and unexecuted controls are no longer presented as evidence for a general mechanism. The preceding study is cited explicitly, and the supplied manuscript is preserved in `upstream/`.

## Primary numerical population

Primary seeds are **7, 42, 123, 2024, and 2025**. The standard table provides a complete 6-family × 7-representation × 5-seed matrix, or 210 records. The robust RGB exports provide the matching 30 records. The external RGB records are joined on the same family/seed keys, with no missing-seed imputation.

Source files:

- `tables/results_seed_level.csv`
- `tables/robust_5seeds_gpu0.csv` and `tables/robust_5seeds_gpu1.csv`
- `tables/df40_all_runs_individual.csv`

The build verifies their Git blobs against upstream commit `d156d77897288816f6b619d577cc77d374ea0013`. `paper/six-page/generated/source_ledger.json` records their hashes. The source files themselves are unchanged.

All primary means and sample standard deviations are recomputed. Paired changes are calculated within seed, then summarized with `ddof=1`; they are not inferred by subtracting rounded manuscript table cells. A separate Python `csv`/`statistics` implementation checks 126 resulting summary quantities independently of the pandas generator. Seed SD is consistently distinguished from a population confidence interval.

The resulting primary Test-D gains are positive in all 30 pairs and range from 5.53 to 12.96 percentage points across family means. The external pattern differs: CLIP gains 6.04 points and DINOv3 6.46, while ResNet loses 2.02 and Xception 6.77. The paper reports this bounded contrast without a causal or universal robustness claim.

## Comparisons that were not pooled

Seed 987 is excluded from the primary matrix. An earlier robust-training path can construct and store a run using the unsuffixed `finetune` designation, so that identifier alone does not establish a standard-versus-robust treatment. The selected primary population follows the complete standard table and its matching robust exports, not a score-based choice of convenient seeds.

`tables/ensemble_robust_canonical_5seeds.csv` supplies secondary ensemble summaries. Its constituent evaluations are not fully recoverable from the aggregate export, and self-ensemble membership strings include seed 987. The article therefore reproduces these as separate reported summaries. It does not equate them to the primary cohort or interpret a five-member self-ensemble as five independent replications.

Headline six-model values such as 0.9614/0.8845/0.8610 were not carried into the rewritten primary comparison because a corresponding complete matched-cohort source row was not established. Table V instead shows the actual CLIP–DINOv3 and all-six summary rows for mean, geometric and stacking fusion, without selecting just the best target-test score.

The corrected Celeb-DF summaries remain ancillary. The source reports 67.80% frame / 72.60% video AUC for six-model geometric fusion and 65.09% / 70.17% for the pair. The associated legacy evaluation is seed-987-based, and the corrected per-frame export and exact manifest are unavailable in the release. These numbers are not reinterpreted as five-seed replication, independently verified image-level evidence, or a matched official-protocol leaderboard. No label or score was flipped simply because a reported AUC was below chance.

## Method and scope corrections

The manuscript now distinguishes the MFFI clean test, its degraded Test-D condition, and the project's DF40-derived local subset. MFFI's original split design already varies authentic-image sources; “clean” is not described as a proof of i.i.d. testing. The Test-D design includes conventional disturbances and patch-based adversarial perturbations; missing per-image operation metadata prevents a corruption-specific analysis.

The DF40 preparation code builds a local subset from available folders and selected real-image pools. Its aggregate export does not establish full official DF40 coverage, exact external membership, or source/person independence. Those stronger claims are not made.

Model names and settings follow the actual code: the DINO configuration is **DINOv3 ConvNeXt-Base**, not a Tiny backbone or a vision transformer. The released defaults use 224×224 inputs, including Xception. Family-specific budgets and overrides are acknowledged rather than invented as a uniform resolved training history.

The robust augmentation description follows the implemented JPEG, Gaussian noise/blur, geometric and photometric transformations. H.264 and motion-blur operations were not added to the methods. Standard RGB and non-RGB paths do not use identical augmentation, so the representation matrix is not labeled a fully factorial augmentation-matched ablation. The study also distinguishes channel adaptation, initialization and pretraining differences from a causal representation effect.

## Writing research and references

`WRITING_RESEARCH.md` records direct reading of papers from Stanford, MIT, Harvard-associated interpretability research, OpenAI, Microsoft Research, Berkeley/Adobe/Michigan and other leading vision groups. It describes the applied writing principles: a testable central question, explicit evaluation units, result-driven headings, matched comparisons, bounded conclusions and explanations appropriate to their evidence.

The article has 21 references. Bibliographic metadata was checked against primary proceedings, author reports and institutional records. Long author lists are explicitly abbreviated with “et al.” rather than shortened in a way that implies an incorrect complete list. Venue names use conventional IEEE-style abbreviations. No institution name is used as a substitute for a scientific argument.

The AI-assistance disclosure identifies ChatGPT and its role in drafting and analysis-code preparation. It does not claim that human author approval, peer review or publication acceptance has already occurred.

## Verification and remaining limits

The final PDF is seven pages, in the standard IEEE double-column template, with 10-point body text, no unresolved references and no overfull boxes. Every page's text bounds and fonts were checked, and all seven page previews are included. The root `paper.pdf` and `paper/six-page/main.pdf` are identical. The supplied PDF is archived byte-for-byte.

This session verified source text, numerical aggregation, citations and PDF geometry programmatically. Direct visual inspection of the rendered previews was unavailable. Historical detector predictions, missing per-image outputs, external membership and complete run configurations were not regenerated by this editorial task. The article makes these evidence limits explicit; it does not manufacture stronger findings to compensate for them.
