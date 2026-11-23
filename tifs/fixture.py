"""Generate a deterministic small, two-class video campaign for engineering verification."""

import argparse
import csv
import subprocess
from pathlib import Path

import numpy as np
import yaml

from .io import atomic_text
from .panel import SOURCE


def make(root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for split, n in (
        ("train", 5),
        ("dev", 5),
        ("cal", 5),
        ("diagnostic", 4),
        ("test", 6),
    ):
        for y in (0, 1):
            for j in range(n):
                rid = f"{split}-{y}-{j}"
                path = root / f"{rid}.mp4"
                if not path.exists():
                    rng = np.random.default_rng(len(rows) + 39)
                    # Source-specific colors, geometry and motion avoid accidental exact duplicates.
                    small = rng.integers(0, 256, (6, 6, 3), dtype=np.uint8)
                    grid = np.repeat(np.repeat(small, 8, 0), 8, 1)
                    frames = np.stack(
                        [np.roll(grid, k * (1 + y), axis=1) for k in range(12)]
                    )
                    subprocess.run(
                        [
                            "ffmpeg",
                            "-nostdin",
                            "-loglevel",
                            "error",
                            "-y",
                            "-f",
                            "rawvideo",
                            "-pixel_format",
                            "rgb24",
                            "-video_size",
                            "48x48",
                            "-framerate",
                            "10",
                            "-i",
                            "pipe:0",
                            "-threads",
                            "1",
                            "-c:v",
                            "libx264",
                            "-pix_fmt",
                            "yuv420p",
                            str(path),
                        ],
                        input=frames.tobytes(),
                        check=True,
                    )
                cohort = (
                    "fixture_external"
                    if split in {"test", "diagnostic"}
                    else "fixture_source"
                )
                rows.append(
                    {
                        "recording_id": rid,
                        "label": y,
                        "cohort": cohort,
                        "family": (
                            "real"
                            if y == 0
                            else (
                                "synthetic-target"
                                if cohort == "fixture_external"
                                else "synthetic-source"
                            )
                        ),
                        "cluster_id": rid,
                        "split": split,
                        "source_path": str(path),
                        "source_ids": rid,
                        "identity_ids": rid,
                        "session_ids": rid,
                    }
                )
    with (root / "sources.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SOURCE)
        writer.writeheader()
        writer.writerows(rows)
    # A known geometric track exercises the independent registration diagnostic
    # on held-out fixture recordings. The synthetic clips contain no real faces.
    track = root / "geometric_fixture_track.csv"
    atomic_text(
        track,
        "time_s,x0,y0,x1,y1\n0,0.125,0.125,0.875,0.875\n2,0.125,0.125,0.875,0.875\n",
    )
    tracked_sources = root / "tracked_sources.csv"
    with tracked_sources.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[*SOURCE, "track_path", "track_reviewer"])
        writer.writeheader()
        writer.writerows(
            dict(row, track_path=str(track), track_reviewer="known-fixture-geometry")
            for row in rows
            if row["split"] == "test"
        )
    names = [
        "clean",
        "augmented",
        "worst_view",
        "partial_auc",
        "constrained",
        "shuffled",
        "per_channel",
        "no_clean",
    ]
    run = lambda role: "tiny-" + role + "__s42"
    cfg = {
        "version": 1,
        "mode": "fixture",
        "output": str(root / "out"),
        "seeds": [42],
        "data": {
            "sources": str(root / "sources.csv"),
            "frames": 2,
            "size": 32,
            "crop": "full",
            "prepare_workers": 4,
            "routes": {
                "native": {"kind": "native"},
                "codec": {"kind": "encode", "crf": 31, "preset": "ultrafast"},
            },
        },
        "training": {
            "train_routes": ["native", "codec"],
            "epochs": 1,
            "batch_size": 4,
            "learning_rate": 0.005,
            "workers": 0,
            "device": "cpu",
            "alpha": 0.05,
            "clean_margin": 0.02,
            "train_frames": 2,
            "clean_reference": "../{backbone}-clean__s{seed}/predictions.csv",
        },
        "crop_diagnostic": {"sources": str(tracked_sources), "splits": ["test"]},
        "models": [
            {"name": "tiny-" + k, "backbone": "tiny", "objective": k} for k in names
        ],
        "analysis": {
            "cohorts": ["fixture_external"],
            "primary_run": run("constrained"),
            "baseline_run": run("augmented"),
            "shuffled_run": run("shuffled"),
            "clean_run": run("clean"),
            "control_runs": [run("worst_view"), run("partial_auc")],
            "bootstrap": 8,
        },
    }
    file = root / "fixture.yaml"
    atomic_text(file, yaml.safe_dump(cfg, sort_keys=False))
    return file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    print(make(parser.parse_args().directory))
