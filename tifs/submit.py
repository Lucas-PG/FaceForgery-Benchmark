"""Submit the sealed study DAG to SLURM. All workers read one shared checkout and output."""

import argparse
import csv
import json
import shlex
import subprocess
from pathlib import Path

from .campaign import ROOT, load_config, runs


def persistent_inputs(cfg):
    """Trace shared inventories to their actual media and model locations."""
    yield Path(cfg["output"])
    yield Path(cfg["data"].get("cache_dir", cfg["output"]))
    for block in (cfg["data"], cfg.get("crop_diagnostic", {})):
        if block.get("cache_dir"):
            yield Path(block["cache_dir"])
        for kind in (
            "sources",
            "renditions",
            "duplicate_clearances",
            "alignment_clearances",
        ):
            if not block.get(kind):
                continue
            file = Path(block[kind]).resolve()
            yield file
            if file.exists() and kind in {"sources", "renditions"}:
                with file.open(newline="") as f:
                    for row in csv.DictReader(f):
                        for key in (
                            "source_path",
                            "path",
                            "receipt_path",
                            "track_path",
                        ):
                            if row.get(key):
                                yield file.parent / row[key]
    if cfg.get("asset_registry"):
        registry = Path(cfg["asset_registry"]).resolve()
        yield registry
        if registry.exists():
            for entry in json.loads(registry.read_text()).values():
                for key, value in entry.items():
                    if not key.endswith("_sha256") and isinstance(value, str):
                        yield Path(value)


def graph(cfg, config_path, parallel, node=None, partition="gpu"):
    if parallel < 1:
        raise ValueError("parallel must be positive")
    if not node and any(
        p.resolve().is_relative_to("/scratch") for p in persistent_inputs(cfg)
    ):
        raise ValueError(
            "Node-local /scratch requires --node to keep all dependent jobs on the same host"
        )
    matrix = list(runs(cfg).values())
    clean = [i for i, (s, _) in enumerate(matrix) if s["objective"] == "clean"]
    others = [i for i in range(len(matrix)) if i not in clean]
    stages = [
        ("prepare", "prepare", [], []),
        ("clean", "model", clean, ["prepare"]),
        ("models", "model", others, ["clean"]),
        ("analyze", "analyze", [], ["models"]),
    ]
    if cfg.get("crop_diagnostic"):
        stages.append(("crop", "crop", [], ["analyze"]))
    stages.append(("report", "report", [], [stages[-1][0]]))
    for name, action, indices, after in stages:
        if action == "model" and not indices:
            raise ValueError("Expected clean and comparison model jobs")
        logdir = Path(cfg["output"]).resolve() / "logs"
        args = [
            "sbatch",
            "--parsable",
            "--export=ALL",
            f"--chdir={ROOT}",
            f"--partition={partition}",
            f"--job-name=tifs-{name}",
            f"--output={logdir}/%x-%A_%a.out",
            f"--error={logdir}/%x-%A_%a.err",
            "--time=48:00:00",
            "--cpus-per-task=8",
            "--mem=64G",
        ]
        if node:
            args.append("--nodelist=" + node)
        if action in {"model", "crop"}:
            args.append("--gres=gpu:1")
        if indices:
            args.append("--array=" + ",".join(map(str, indices)) + f"%{parallel}")
        argv = [
            str(ROOT / "scripts" / "tifs_job.sh"),
            str(Path(config_path).resolve()),
            action,
        ]
        yield name, after, args, argv


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--parallel", type=int, default=2)
    p.add_argument("--node", help="pin all jobs when using node-local material")
    p.add_argument("--partition", default="gpu")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    cfg = load_config(args.config)
    jobs = {}
    steps = list(graph(cfg, args.config, args.parallel, args.node, args.partition))
    if not args.dry_run:
        (Path(cfg["output"]) / "logs").mkdir(parents=True, exist_ok=True)
    for name, after, base, argv in steps:
        command = list(base)
        if after:
            command.append("--dependency=afterok:" + ":".join(jobs[k] for k in after))
            command.append("--kill-on-invalid-dep=yes")
        command.extend(argv)
        if args.dry_run:
            jobs[name] = name
            print(shlex.join(command))
        else:
            job = subprocess.check_output(command, text=True).strip().split(";")[0]
            if not job.isdecimal():
                raise ValueError(f"Unexpected SLURM job ID: {job}")
            jobs[name] = job
            print(f"{name} {job}", flush=True)


if __name__ == "__main__":
    main()
