# Explicability extension

Start with [the implementation and execution guide](docs/explicability.md).

The revised article is at [`paper.pdf`](paper.pdf); editable sources are in [`paper/explicability/`](paper/explicability/). The original SIBGRAPI manuscript is preserved separately at [`paper/original-sibgrapi.pdf`](paper/original-sibgrapi.pdf). The compiled revision is a **pre-execution manuscript**, not a claim that new heatmaps or experimental findings have already been produced.

The opt-in pipeline implements the shared 64-image protocol, validation-selected RGB–frequency comparisons, and complete-roster shared-error investigation. It includes twelve attribution methods, raw maps, declared baselines and targets, numerical controls, resumable shards, and figure/table generation. The existing training and evaluation paths remain unchanged.

```bash
uv run python explicability.py --help
```

Changes are proposed through [PR #1](https://github.com/Lucas-PG/FaceForgery-Benchmark/pull/1), from `explicability` into `ICLR`. No automatic merge is configured. Full research experiments are operator-invoked, not scheduled by this extension.
