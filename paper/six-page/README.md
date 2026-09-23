# Primary manuscript: seven-page IEEE edition

**When Robustness Does Not Transfer: Spatial and Spectral Cues for Face Forgery Detection**

The final rewritten article is [`main.pdf`](main.pdf), identical byte-for-byte to [`../../paper.pdf`](../../paper.pdf). It contains **seven total pages, including 21 references**, in the standard IEEEtran conference double-column layout, with 10-point body text and US Letter paper. No reduced margins, compressed line spacing, or scaled body text are used. The final columns are balanced typographically.

Edit [`main.tex`](main.tex) and the article-specific [`references.bib`](references.bib). The generated tables, numerical macros, quantitative figure and underlying evidence exports are in [`generated/`](generated/). Do not manually edit generated numerical values.

## Scope and provenance

The source branch was imported into the Lucas-PG fork at upstream `ICLR@d156d77897288816f6b619d577cc77d374ea0013`. The rewrite is on `paper-rewrite-20260923`; the exact imported baseline is `upstream-iclr-20260923`. All writes occurred in the fork. The supplied source/PDF/bibliography are preserved in [`../rewrite/upstream/`](../rewrite/upstream/). Earlier expanded papers and presentation assets are unchanged.

This is a complete rewrite and reanalysis of existing experimental records, not a new GPU-training or full-dataset inference run. The primary comparison uses 210 standard fine-tuning records and 30 robust RGB records, with the same five seeds. Secondary ensemble summaries and the small qualitative gallery have explicitly different evidence status. See the [changes and evidence audit](../rewrite/CHANGES_AND_EVIDENCE.md).

## Rebuild

Use the exact source history, including the pinned upstream commit. Install the versions in [`../rewrite/build-environment.txt`](../rewrite/build-environment.txt) and a TeX installation containing IEEEtran, latexmk, recommended fonts, booktabs, microtype and balance. From the repository root:

```bash
python paper/rewrite/build_evidence.py
cd paper/six-page
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
cp main.pdf ../../paper.pdf
cd ../..
python paper/rewrite/verify_paper.py
```

On Ubuntu the TeX/system packages used were `latexmk`, `texlive-latex-extra`, `texlive-publishers`, `texlive-fonts-recommended`, and `poppler-utils`. The environment is recorded for reproducibility, not presented as a recommendation to use the latest versions.

The **Build rewritten IEEE paper** workflow is restricted to `Lucas-PG/FaceForgery-Benchmark:paper-rewrite-20260923`. It regenerates the evidence, compiles the paper, validates 6–8 total pages and only publishes the verified document artifacts to that working branch. It neither merges a PR nor writes to the source repository.

## Verification

The final successful build is [run 35932879499](https://github.com/Lucas-PG/FaceForgery-Benchmark/actions/runs/35932879499). [`build.json`](build.json) records:

- Seven pages in the standard IEEE layout, with text bounds and font sizes checked on every page.
- Twenty-one cited bibliography entries, no unresolved citations/references and no overfull boxes.
- An independent `csv`/`statistics` implementation checking **126 numerical summary quantities** against the separate pandas generation path.
- Exact primary seed identities and all 30 positive paired Test-D changes.
- The final PDF, source, bibliography, source-data and preserved-input hashes.

Final PDF SHA-256: `ce25b4fe90c73c0a9991e41d5956b18d01397517f559440d263cf281599056e7`.

[Page previews and a contact sheet](../rewrite/preview/) are included. Verification in this session comprised source/text review and automated numerical/layout checks; direct visual inspection of the rendered pages was unavailable. Passing document checks does not certify the provenance of every historical detector run or establish new experimental results. Those limitations are stated in the article rather than omitted.

## Research and editorial rationale

[`../rewrite/WRITING_RESEARCH.md`](../rewrite/WRITING_RESEARCH.md) records the primary papers studied and the writing decisions derived from them. The manuscript includes an AI-assistance disclosure identifying ChatGPT and its role in the text and analysis scripts, consistent with the [IEEE author guidance](https://conferences.ieeeauthorcenter.ieee.org/author-ethics/guidelines-and-policies/submission-policies/). The disclosure does not claim that human author approval or journal review has already occurred.
