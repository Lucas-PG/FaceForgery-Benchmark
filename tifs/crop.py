"""Measure fixed-track preprocessing on identical clips, weights and frozen thresholds."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .io import atomic_text, digest, sha, write_json
from .learning import _predict
from .models import build_model
from .stats_core import digest_key, interval, paired_difference, representatives, wide

KEYS = ["recording_id", "route"]
IDENTIFIERS = {k: str for k in (*KEYS, "cohort", "cluster_id", "family", "split")}


def _events(panel, scores, cutoff):
    prediction = pd.read_csv(
        scores, dtype={k: str for k in KEYS}, keep_default_na=False
    )
    joined = panel.merge(
        prediction[KEYS + ["score"]], on=KEYS, validate="one_to_one", how="left"
    )
    if not np.isfinite(pd.to_numeric(joined.score, errors="coerce")).all():
        raise ValueError(
            "Tracked spatial comparison requires one finite score per registered view"
        )
    for cohort, part in joined[joined.split == "test"].groupby("cohort"):
        v = representatives(wide(part))
        alerts = v.scores > cutoff
        for label, event, metric in (
            (1, alerts.all(1), "all_tpr"),
            (0, alerts.any(1), "any_fpr"),
        ):
            keep = (v.meta.label == label).to_numpy()
            if keep.any():
                yield (
                    (cohort, metric),
                    pd.Series(event[keep], index=v.meta.recording_id[keep]),
                )


def _differences(events, runs, tail, bootstrap):
    """Correspondence-preserving comparisons per independent class-specific lineage."""
    baseline, primary = runs
    keys = events[primary, "tracked"].keys()
    rows = []
    for cohort, metric in sorted(keys):
        cells = {key: value[cohort, metric] for key, value in events.items()}
        anchor = cells[primary, "tracked"]
        if any(not vec.index.equals(anchor.index) for vec in cells.values()):
            raise ValueError("Unmatched recording IDs in the crop diagnostic")
        comparisons = {
            f"tracked_minus_end_to_end:{run}": (
                cells[run, "tracked"],
                cells[run, "end_to_end"],
            )
            for run in runs
        }
        for processing in ("tracked", "end_to_end"):
            comparisons[f"primary_minus_baseline:{processing}"] = (
                cells[primary, processing],
                cells[baseline, processing],
            )
        for comparison, (a, b) in comparisons.items():
            effect, lo, hi = paired_difference(a, b, tail)
            rows.append(
                {
                    "cohort": cohort,
                    "metric": metric,
                    "comparison": comparison,
                    "n": len(a),
                    "estimate": effect,
                    "lower": lo,
                    "upper": hi,
                    "interval": "paired discordance, pointwise",
                }
            )
        # Each recording has four observed decisions. The difference in method gains
        # lies in {-2,-1,0,1,2}; bootstrap the one independent unit, preserving all cells.
        response = (
            cells[primary, "tracked"].astype(int)
            - cells[baseline, "tracked"].astype(int)
            - cells[primary, "end_to_end"].astype(int)
            + cells[baseline, "end_to_end"].astype(int)
        )
        lower = upper = None
        if bootstrap:
            values, counts = np.unique(response, return_counts=True)
            seed = int(digest_key("crop-interaction", f"{cohort}/{metric}")[:16], 16)
            rng = np.random.default_rng(seed)
            draws = rng.multinomial(
                len(response), counts / len(response), size=bootstrap
            )
            lower, upper = np.quantile(draws @ values / len(response), [tail, 1 - tail])
        rows.append(
            {
                "cohort": cohort,
                "metric": metric,
                "comparison": "tracked_minus_end_to_end:primary_gain",
                "n": len(response),
                "estimate": response.mean(),
                "lower": lower,
                "upper": upper,
                "interval": "paired bootstrap percentile, diagnostic",
            }
        )
    return pd.DataFrame(rows)


def crop_analysis(cfg, full, out, jobs, tracked):

    dest = out / "crop"
    dest.mkdir(exist_ok=True)
    tracked_path = dest / "panel/panel.csv"
    original = full.merge(tracked[KEYS], on=KEYS, validate="one_to_one")
    a = original.set_index(KEYS).sort_index()
    b = tracked.set_index(KEYS).sort_index()
    if not a.index.equals(b.index) or not set(tracked.split).issubset(
        {"diagnostic", "test"}
    ):
        raise ValueError("Tracked diagnostic includes an unregistered observation")
    for field in (
        "label",
        "cluster_id",
        "cohort",
        "family",
        "split",
        "sha256",
        "source_sha256",
        "offset_seconds",
        "window_start_seconds",
        "duration",
    ):
        if not a[field].equals(b[field]):
            raise ValueError(f"Unmatched diagnostic metadata: {field}")
    opt = cfg["analysis"]
    tail = (1 - float(opt.get("confidence", 0.95))) / 2
    policies = json.loads((out / "analysis/policies.json").read_text())
    runs = [opt["baseline_run"], opt["primary_run"]]
    events, rows = {}, []
    for run in runs:
        spec, seed = jobs[run]
        source = out / "runs" / run
        receipt = json.loads((source / "model.json").read_text())
        weights = Path(receipt["checkpoint"])
        if (
            sha(weights) != receipt["checkpoint_sha256"]
            or sha(source / "predictions.csv") != receipt["prediction_sha256"]
        ):
            raise ValueError(f"Changed model inputs for crop diagnostic: {run}")
        settings = {**cfg["training"], **spec.get("training", {})}
        device = torch.device(settings["device"])
        model = build_model(spec, settings)
        snapshot = torch.load(weights, map_location="cpu", weights_only=True, mmap=True)
        if snapshot["signature"] != receipt["train_signature"]:
            raise ValueError(f"Checkpoint signature mismatch: {run}")
        model.load_state_dict(snapshot["model"], strict=True)
        del snapshot
        model.to(device)
        folder = dest / run
        folder.mkdir(exist_ok=True)
        identity = digest(
            [receipt["checkpoint_sha256"], sha(tracked_path), spec, settings]
        )
        pred = _predict(model, tracked, settings, spec, seed, device, folder, identity)
        del model
        cutoff = policies[run]["shared"]["cutoff"]
        for processing, scores in (
            ("tracked", pred),
            ("end_to_end", source / "predictions.csv"),
        ):
            groups = dict(_events(tracked, scores, cutoff))
            if not groups:
                raise ValueError(
                    "The diagnostic crop inventory needs sealed-test recordings"
                )
            events[run, processing] = groups
            for (cohort, metric), flagged in groups.items():
                n, count = len(flagged), int(flagged.sum())
                lower, upper = interval(count, n, tail)
                rows.append(
                    {
                        "run": run,
                        "cohort": cohort,
                        "processing": processing,
                        "metric": metric,
                        "n": n,
                        "events": count,
                        "estimate": count / n,
                        "lower": lower,
                        "upper": upper,
                        "frozen_cutoff": cutoff,
                    }
                )
    result = dest / "comparison.csv"
    effects = dest / "paired_effects.csv"
    atomic_text(result, pd.DataFrame(rows).to_csv(index=False))
    frame = _differences(events, runs, tail, int(opt.get("bootstrap", 500)))
    atomic_text(effects, frame.to_csv(index=False))
    write_json(
        dest / "complete.json",
        {
            "panel": sha(tracked_path),
            "result": sha(result),
            "effects": sha(effects),
            "analysis": sha(out / "analysis/complete.json"),
        },
    )
    return result
