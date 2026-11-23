"""Orchestrate a sealed recording-level research campaign with explicit stage receipts."""

import argparse
import contextlib
import fcntl
import importlib.metadata
import json
import os
import platform
import re
import subprocess
from pathlib import Path

import pandas as pd
import yaml

from .io import digest, sha, write_json

ROOT = Path(__file__).resolve().parents[1]
STAGES = (
    "plan",
    "enrollment",
    "check",
    "prepare",
    "model",
    "analyze",
    "crop",
    "report",
    "all",
)


def expand(obj):
    if isinstance(obj, dict):
        return {k: expand(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [expand(v) for v in obj]
    if isinstance(obj, str):
        value = os.path.expandvars(obj)
        if re.search(r"\$\{[^}]+\}", value):
            raise ValueError(f"Unresolved environment variable: {obj}")
        return value
    return obj


def load_config(file):
    cfg = expand(yaml.safe_load(Path(file).read_text()))
    if cfg.get("version") != 1:
        raise ValueError("Expected campaign version 1")
    return cfg


def runs(cfg):
    names = {}
    for spec in cfg["models"]:
        name = spec["name"]
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", name) or "__s" in name:
            raise ValueError(f"Invalid model name: {name}")
        for seed in spec.get("seeds", cfg["seeds"]):
            key = f"{name}__s{int(seed)}"
            if key in names:
                raise ValueError(f"Duplicate run {key}")
            names[key] = (spec, int(seed))
    if not names:
        raise ValueError("Empty model matrix")
    for ref in ("primary_run", "baseline_run", "shuffled_run", "clean_run"):
        if cfg["analysis"][ref] not in names:
            raise ValueError(f"Unscheduled run in analysis.{ref}")
    from .design import check_matrix

    check_matrix(cfg, names)
    return names


def revision():
    return subprocess.check_output(
        ["git", "-C", ROOT, "rev-parse", "HEAD"], text=True
    ).strip()


def code_id():
    paths = sorted((ROOT / "tifs").glob("*.py")) + sorted(
        (ROOT / "src/models").glob("*.py")
    )
    return digest({str(p.relative_to(ROOT)): sha(p) for p in paths})


def versions():
    env = {"python": platform.python_version()}
    for binary in ("ffmpeg", "ffprobe"):
        build = subprocess.check_output([binary, "-version"], text=True)
        env[binary] = build.splitlines()[0]
        env[binary + "_build"] = digest(build)
    for pkg in (
        "torch",
        "torchvision",
        "transformers",
        "timm",
        "numpy",
        "pandas",
        "scipy",
        "pillow",
        "opencv-python-headless",
        "scikit-learn",
        "scikit-image",
        "dlib",
        "dlib-bin",
        "safetensors",
        "matplotlib",
    ):
        try:
            env[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            pass
    return env


@contextlib.contextmanager
def lock(path, *, wait=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
        except BlockingIOError as exc:
            raise RuntimeError(
                f"Another process owns {path}. Retry after it finishes."
            ) from exc
        yield


def read_tracked(out, receipt):
    from .panel import verify_material

    directory = out / "crop/panel"
    registered = receipt.get("tracked")
    if not registered or any(sha(directory / k) != v for k, v in registered.items()):
        raise ValueError("Tracked diagnostic differs from its pre-scoring registration")
    data = pd.read_csv(
        directory / "panel.csv",
        dtype={
            key: str
            for key in (
                "recording_id",
                "route",
                "cluster_id",
                "cohort",
                "family",
                "split",
            )
        },
        keep_default_na=False,
    )
    verify_material(data)
    return data


def get_panel(out):
    file = out / "panel" / "panel.csv"
    receipt = json.loads((file.parent / "ready.json").read_text())
    if sha(file) != receipt["panel"] or any(
        sha(file.parent / k) != v for k, v in receipt["support"].items()
    ):
        raise ValueError("Panel changed after preparation")
    panel = pd.read_csv(
        file,
        dtype={
            "recording_id": str,
            "route": str,
            "cluster_id": str,
            "cohort": str,
            "family": str,
            "split": str,
        },
        keep_default_na=False,
    )
    if panel.duplicated(["recording_id", "route"]).any():
        raise ValueError("Duplicate observation in panel")
    from .panel import verify_material

    verify_material(panel)
    return panel, receipt


def manifest_identity(cfg):
    d = cfg["data"]
    files = [d["sources"]]
    if d.get("renditions"):
        files.append(d["renditions"])
    for field in ("duplicate_clearances", "alignment_clearances"):
        if d.get(field):
            files.append(d[field])
    if cfg.get("crop_diagnostic"):
        cd = cfg["crop_diagnostic"]
        files += [cd[k] for k in ("sources", "renditions") if cd.get(k)]
    return {str(Path(f).resolve()): sha(f) for f in files}


def bind_registry(cfg):
    reg = (
        json.loads(Path(cfg["asset_registry"]).read_text())
        if cfg.get("asset_registry")
        else {}
    )
    fields = {
        "checkpoint",
        "weights",
        "upstream",
        "processor",
        "detector_config",
        "backbone_config",
        "landmarks",
        "run_config",
    }
    allowed = fields | {f + "_sha256" for f in fields} | {"clip_config_sha256"}
    for spec in cfg["models"]:
        if "assets" in spec:
            entry = reg[spec["assets"]]
            if set(entry) - allowed:
                raise ValueError(
                    "Asset registry may only bind files, revisions and digests"
                )
            spec.update(entry)


def register(cfg, out):
    bind_registry(cfg)
    if cfg["mode"] == "confirmatory":
        tracked = subprocess.check_output(
            [
                "git",
                "-C",
                ROOT,
                "status",
                "--porcelain",
                "--",
                "tifs",
                "src/models",
                "scripts/tifs_job.sh",
            ],
            text=True,
        )
        if tracked.strip():
            raise ValueError(
                "Commit the research code before registering a confirmatory campaign"
            )
    base = {
        "config": cfg,
        "inputs": manifest_identity(cfg),
        "code": code_id(),
        "commit": revision(),
        "versions": versions(),
    }
    path = out / "protocol.json"
    with lock(out / ".register.lock", wait=True):
        if path.is_file():
            old = json.loads(path.read_text())
            if old != base:
                raise ValueError(
                    "Configuration, input inventory, environment or revision differs from protocol.json; choose a new output root"
                )
        else:
            write_json(path, base)
    return digest(base)


def verify_predictions(path, panel):
    frame = pd.read_csv(
        path, dtype={"recording_id": str, "route": str}, keep_default_na=False
    )
    raw = frame["score"]
    numeric = pd.to_numeric(raw, errors="coerce")
    missing = numeric.isna()
    if (missing & raw.astype(str).str.strip().ne("")).any() or not numeric[
        ~missing
    ].between(0, 1).all():
        raise ValueError(f"Invalid probability in {path}")
    if missing.any() and (
        "failure" not in frame or frame.loc[missing, "failure"].str.strip().eq("").any()
    ):
        raise ValueError(f"Missing recorded inference failure in {path}")
    keys = ["recording_id", "route"]
    expected = pd.MultiIndex.from_frame(panel.loc[panel.split != "train", keys])
    actual = pd.MultiIndex.from_frame(frame[keys])
    if (
        actual.has_duplicates
        or len(actual) != len(expected)
        or not actual.isin(expected).all()
    ):
        raise ValueError(f"Prediction keys disagree with panel: {path}")


def model_artifacts(path, spec):
    metadata = path / "model.json"
    if spec["backbone"] in {"gend", "lipforensics"}:
        return {
            "upstream": spec["upstream"],
            "checkpoint_sha256": spec["weights_sha256"],
        }
    model = json.loads(metadata.read_text())
    checkpoint = Path(model["checkpoint"])
    st = checkpoint.stat()
    return {
        "model_json": sha(metadata),
        "checkpoint": str(checkpoint),
        "sha256": model["checkpoint_sha256"],
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
        "ctime_ns": st.st_ctime_ns,
    }


def check_complete(path, identity):
    record = json.loads((path / "complete.json").read_text())
    if (
        record["identity"] != digest(identity)
        or sha(path / "predictions.csv") != record["predictions_sha256"]
    ):
        raise ValueError(f"Corrupt prediction receipt in {path}")
    expected = record["artifacts"]
    current = model_artifacts(path, identity["model"])
    if current != expected:
        stable = {k: v for k, v in current.items() if k not in {"mtime_ns", "ctime_ns"}}
        original = {
            k: v for k, v in expected.items() if k not in {"mtime_ns", "ctime_ns"}
        }
        if stable != original or sha(current["checkpoint"]) != current["sha256"]:
            raise ValueError(f"Corrupt detector provenance or checkpoint in {path}")


def model_run(key, cfg, panel, receipt, out, study):
    spec, seed = runs(cfg)[key]
    path = out / "runs" / key
    path.mkdir(parents=True, exist_ok=True)
    identity = {"study": study, "panel": receipt["panel"], "model": spec, "seed": seed}
    with lock(path / ".lock"):
        signature = path / "identity.json"
        if signature.exists() and json.loads(signature.read_text()) != identity:
            raise ValueError(f"Unsafe resume: {key} identity differs")
        write_json(signature, identity)
        done = path / "complete.json"
        if done.exists():
            check_complete(path, identity)
            print(f"Verified cached {key}", flush=True)
            return
        if spec["backbone"] in {"gend", "lipforensics"}:
            from .external import run_external as run_model
        else:
            from .learning import run_model
        print(f"Starting {key}", flush=True)
        pred = Path(run_model(cfg, panel, spec, seed, path))
        verify_predictions(pred, panel)
        write_json(
            done,
            {
                "identity": digest(identity),
                "predictions_sha256": sha(pred),
                "artifacts": model_artifacts(path, spec),
                "host": platform.node(),
                "job": os.environ.get("SLURM_JOB_ID"),
            },
        )


def execute(cfg, stage, key=None):
    matrix = runs(cfg)
    if stage == "plan":
        for i, (name, (spec, seed)) in enumerate(matrix.items()):
            print(
                f"{i:3d}  {name}  {spec['backbone']}  {spec['objective']}  seed={seed}"
            )
        return
    out = Path(cfg["output"]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if stage in {"enrollment", "all", "prepare", "check"}:
        from .design import inventory_power

        inventory_power(cfg, out / "enrollment")
        if stage == "enrollment":
            return
    study = register(cfg, out)
    if stage in {"all", "check", "prepare"}:
        from .preflight import probe

        stamp = out / "asset_probe.json"
        if not stamp.exists() or json.loads(stamp.read_text())["study"] != study:
            write_json(stamp, {"study": study, "loaded": probe(cfg)})
    if stage == "check":
        print(f"Protocol registered at {out}; assets verified")
        return
    if stage in {"prepare", "all"}:
        from .panel import prepare

        with lock(out / "panel" / ".lock"):
            # prepare validates source assets and reuses content-addressed per-view frame caches.
            directory = out / "panel"
            ready = directory / "ready.json"
            if ready.exists():
                _, previous = get_panel(out)
                if previous["study"] != study:
                    raise ValueError("Prepared data belong to a different study")
                if previous.get("tracked"):
                    read_tracked(out, previous)
                print("Verified prepared panel", flush=True)
            else:
                p = Path(prepare(cfg, directory))
                panel = pd.read_csv(
                    p,
                    dtype={
                        k: str
                        for k in (
                            "recording_id",
                            "route",
                            "cluster_id",
                            "family",
                            "cohort",
                            "split",
                        )
                    },
                    keep_default_na=False,
                )
                from .design import check_panel

                check_panel(cfg, panel, directory)
                for k in cfg.get("crop_diagnostic", {}).get("splits", []):
                    if k not in {"test", "diagnostic"}:
                        raise ValueError(
                            "Crop diagnostic must exclude training and source calibration"
                        )
                if cfg.get("crop_diagnostic"):
                    from .panel import prepare_crop_diagnostic

                    crop_dir = out / "crop" / "panel"
                    prepare_crop_diagnostic(cfg, crop_dir)
                support = {f.name: sha(f) for f in directory.glob("*.csv") if f != p}
                support["panel_audit.json"] = sha(directory / "panel_audit.json")
                write_json(
                    ready,
                    {
                        "study": study,
                        "panel": sha(p),
                        "support": support,
                        "tracked": {
                            q.name: sha(q)
                            for q in crop_dir.iterdir()
                            if q.suffix in {".csv", ".json"}
                        }
                        if cfg.get("crop_diagnostic")
                        else {},
                    },
                )
        if stage == "prepare":
            return
    panel, receipt = get_panel(out)
    if receipt["study"] != study:
        raise ValueError("Prepared panel belongs to another protocol")
    if stage in {"model", "all"}:
        if stage == "model" and key is None:
            raise ValueError("model requires --run NAME__sSEED or --index N")
        for name in [key] if key is not None else matrix:
            model_run(name, cfg, panel, receipt, out, study)
        if stage == "model":
            return
    if stage in {"analyze", "all"}:
        from .analysis import analyze

        preds = {}
        for name in matrix:
            folder = out / "runs" / name
            pred = folder / "predictions.csv"
            identity = json.loads((folder / "identity.json").read_text())
            spec, seed = matrix[name]
            if identity != {
                "study": study,
                "panel": receipt["panel"],
                "model": spec,
                "seed": seed,
            }:
                raise ValueError(f"Stale predictions for {name}")
            check_complete(folder, identity)
            preds[name] = pred
        with lock(out / ".analysis.lock"):
            work = out / "analysis"
            ready = work / "complete.json"
            proof = {"study": study, "scores": {k: sha(v) for k, v in preds.items()}}
            if ready.exists():
                previous = json.loads(ready.read_text())
                if any(previous[k] != v for k, v in proof.items()) or any(
                    sha(work / k) != v for k, v in previous["outputs"].items()
                ):
                    raise ValueError(
                        "Corrupt analysis receipt or changed score ledger; restore original artifacts"
                    )
                print("Verified cached analysis", flush=True)
            else:
                work.mkdir(parents=True, exist_ok=True)
                analyze(cfg, panel, preds, work)
                write_json(
                    ready,
                    {
                        **proof,
                        "outputs": {
                            p.name: sha(p)
                            for p in work.iterdir()
                            if p.is_file() and p != ready
                        },
                    },
                )
        if stage == "analyze":
            return
    if stage in {"crop", "all"} and cfg.get("crop_diagnostic"):
        from .crop import crop_analysis

        with lock(out / ".crop.lock"):
            crop_analysis(cfg, panel, out, matrix, read_tracked(out, receipt))
    if stage == "crop":
        return
    if stage in {"report", "all"}:
        from .report import report

        stamp = json.loads((out / "analysis/complete.json").read_text())
        if stamp["study"] != study or any(
            sha(out / "analysis" / k) != v for k, v in stamp["outputs"].items()
        ):
            raise ValueError("Analysis receipt mismatch")
        with lock(out / ".report.lock"):
            if cfg.get("crop_diagnostic"):
                crop_receipt = json.loads((out / "crop/complete.json").read_text())
                if (
                    crop_receipt["result"] != sha(out / "crop/comparison.csv")
                    or crop_receipt["effects"] != sha(out / "crop/paired_effects.csv")
                    or crop_receipt["analysis"] != sha(out / "analysis/complete.json")
                ):
                    raise ValueError(
                        "Run the crop diagnostic on this completed analysis first"
                    )
            report(cfg, out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--config", required=True, type=Path)
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--run")
    g.add_argument("--index", type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if (args.index is not None or args.run) and args.stage != "model":
        parser.error("--index/--run apply only to model")
    key = args.run
    if args.index is not None:
        matrix = list(runs(cfg))
        if not 0 <= args.index < len(matrix):
            parser.error("Run index outside model matrix")
        key = matrix[args.index]
    execute(cfg, args.stage, key)


if __name__ == "__main__":
    main()
