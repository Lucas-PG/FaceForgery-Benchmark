# Final article rewrite — 23 September 2026

## Read the paper

**[When Robustness Does Not Transfer: Spatial and Spectral Cues for Face Forgery Detection](../six-page/main.pdf)**

The final article has **seven pages including references**, using the standard IEEEtran conference double-column template and 10-point body text. [`../../paper.pdf`](../../paper.pdf) is an identical copy. The folder remains named `six-page` for compatibility with the existing project layout; its updated [build guide](../six-page/README.md) describes the current seven-page edition.

## Review resources

| Resource | Contents |
|---|---|
| [Article source](../six-page/main.tex) and [bibliography](../six-page/references.bib) | Fully rewritten argument, methods, results, interpretation, limitations and 21 references |
| [Writing research](WRITING_RESEARCH.md) | Primary papers studied and the concrete writing decisions derived from them |
| [Changes and evidence](CHANGES_AND_EVIDENCE.md) | Corrected comparisons, excluded headline values, cohort distinctions and methodological qualifications |
| [Generated evidence](../six-page/generated/) | All primary seed records, recomputed summaries, source hashes, TeX tables and quantitative figure |
| [Build verification](../six-page/build.json) | Independent numerical checks, citation checks, every page's text bounds, fonts and SHA-256 identities |
| [Page previews](preview/) | Seven rendered pages and a contact sheet for inspection |
| [Preserved input](upstream/) | The supplied manuscript PDF, source and bibliography |

## Branches and safety

The original source was read from `lucasdocunha/FaceForgery-Benchmark:ICLR`, pinned at `d156d77897288816f6b619d577cc77d374ea0013`. Its exact snapshot was brought into this fork as `upstream-iclr-20260923`. The rewrite is on `paper-rewrite-20260923`, reviewed in [PR #3](https://github.com/Lucas-PG/FaceForgery-Benchmark/pull/3). The PR compares the rewrite against the imported snapshot, rather than mixing the upstream synchronization into the editorial diff. No automatic merge is performed.

All writes are confined to the Lucas-PG fork. Existing `main`, `ICLR`, `explicability` and `robust-generalization` branches were not overwritten. Neither new text nor commits were written to the source repository.

## Verification result

The final paper was published at commit `755ec5f3debb8ac0c32f7dd1106a34aca410009b` by successful [build 35932879499](https://github.com/Lucas-PG/FaceForgery-Benchmark/actions/runs/35932879499). Subsequent commits update documentation only.

The build checked 126 numerical summary quantities using an independent aggregation implementation, all 30 primary seed pairings, 21 bibliography entries, zero unresolved references, zero overfull boxes, standard IEEE page geometry, and exact PDF identity. The final PDF hash is `ce25b4fe90c73c0a9991e41d5956b18d01397517f559440d263cf281599056e7`.

Verification comprised source/text review and automated numerical/layout checks. Direct visual inspection of the rendered pages was unavailable in this session; the previews are included rather than claiming that review occurred. This editorial task also did not retrain detectors or regenerate missing historical per-image predictions. The article distinguishes measured records, recomputed summaries, ancillary reports and unexecuted explanation controls accordingly.
