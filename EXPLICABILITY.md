# Explicability extension

Start with [the implementation and execution guide](docs/explicability.md).

## Manuscript editions

The primary article is [`paper.pdf`](paper.pdf): **six pages in the original IEEE/SIBGRAPI two-column conference format**, including references. Its editable source and build instructions are in [`paper/six-page/`](paper/six-page/).

The **ten-page expanded reference** remains unchanged at [`paper/explicability/main.pdf`](paper/explicability/main.pdf), with its LaTeX source and shared bibliography in [`paper/explicability/`](paper/explicability/). It retains the extended discussion and operational appendices; it is not the primary paper.

The original SIBGRAPI manuscript is preserved separately at [`paper/original-sibgrapi.pdf`](paper/original-sibgrapi.pdf). Both revised editions are **pre-execution manuscripts**, not claims that new heatmaps or experimental findings have already been produced. The primary edition's page count, PDF/source hashes, format checks, and preserved-reference hashes are recorded in [`paper/six-page/build.json`](paper/six-page/build.json).

## Implementation

The opt-in pipeline implements the shared 64-image protocol, validation-selected RGB–frequency comparisons, and complete-roster shared-error investigation. It includes twelve attribution methods, raw maps, declared baselines and targets, numerical controls, resumable shards, and figure/table generation. The existing training and evaluation paths remain unchanged.

```bash
uv run python explicability.py --help
```

Changes are proposed through [PR #1](https://github.com/Lucas-PG/FaceForgery-Benchmark/pull/1), from `explicability` into `ICLR`. No automatic merge is configured. Full research experiments are operator-invoked, not scheduled by this extension.
