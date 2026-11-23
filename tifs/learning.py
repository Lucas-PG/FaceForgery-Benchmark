"""Source-only learning and resumable held-out single-copy scoring."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import importlib.metadata
import json
import os
import random
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.utils.atomic import atomic_torch_save

from .io import digest
from .io import write_json as _authority
from .models import (
    build_model,
    set_training_mode,
    sha256,
    validate_assets,
    video_logits,
)
from .training_data import RecordingEpoch, SingleCopyDataset
from .training_objectives import CONSTRAINED, OBJECTIVES, loss_for_batch


def _write_csv(df, path):
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, newline=""
    ) as tmp:
        df.to_csv(tmp, index=False)
        tmp.flush()
        os.fsync(tmp.fileno())
    os.replace(tmp.name, path)


def _frame_digest(frame: pd.DataFrame) -> str:
    return hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()


def _fingerprint():
    root = Path(__file__).resolve().parents[1]
    files = [
        Path(__file__),
        *Path(__file__).parent.glob("training*.py"),
        Path(__file__).with_name("models.py"),
    ]
    files.extend(
        root / "src" / "models" / x for x in ("clip.py", "dino.py", "_hf_layers.py")
    )
    proc = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "revision": proc.stdout.strip() if proc.returncode == 0 else "unversioned",
        "files": {p.name: sha256(p) for p in files},
        "torch": str(torch.__version__),
        "packages": {
            name: importlib.metadata.version(name) for name in ("numpy", "pandas")
        },
    }


def _floor(cfg, panel, settings, spec, seed, run_dir):
    if spec["objective"] not in CONSTRAINED - {"no_clean"}:
        return 0.0, {}
    if "clean_reference" not in settings:
        floor = float(settings["clean_floor"])
        if not 0 <= floor <= 1:
            raise ValueError("Prespecified clean_floor must lie in [0,1]")
        return floor, {"fixed_native_floor": floor}
    name = str(settings["clean_reference"]).format(
        seed=seed, backbone=spec["backbone"], out=run_dir
    )
    filename = Path(name)
    if not filename.is_absolute():
        filename = run_dir / filename
    reference_signature = None
    if cfg.get("mode") == "confirmatory":
        path = filename.resolve()
        archive = json.loads((path.parent / "model.json").read_text())
        if (
            path.parent.parent != run_dir.resolve().parent
            or archive["spec"]["objective"] != "clean"
            or archive["seed"] != seed
            or archive["training"] != settings
            or sha256(path) != archive["prediction_sha256"]
        ):
            raise ValueError(
                "Clean floor requires the matched same-seed clean run's verified predictions"
            )
        ignored = {"name", "objective", "seeds"}
        stripped = lambda arm: {k: v for k, v in arm.items() if k not in ignored}
        if stripped(spec) != stripped(archive["spec"]):
            raise ValueError("Clean floor reference differs in backbone or assets")
        reference_signature = archive["train_signature"]
    dev = panel[panel.split == "dev"]
    if dev.empty or set(dev.label) != {0, 1}:
        raise ValueError(
            "Clean reference needs genuine and fake development recordings"
        )
    scores = pd.read_csv(
        filename, dtype={"recording_id": str, "route": str}, keep_default_na=False
    )
    view = dev.merge(
        scores[["recording_id", "route", "score"]],
        on=["recording_id", "route"],
        validate="one_to_one",
        how="left",
    )
    # Derive the clean comparator target on the SAME development-fit lineages
    # and operating budget used by the analysis. The model-selection half of dev
    # is never a source of supervision, even through an aggregated clean floor.
    from .stats_core import representatives, split_dev, wide

    fit, _ = split_dev(representatives(wide(view)))
    if not np.isfinite(fit.scores).all():
        raise ValueError(
            "Incomplete or nonfinite clean reference development predictions"
        )
    options = cfg.get("analysis", {})
    source_alpha = float(
        options.get(
            "source_alpha",
            0.6 * options["alpha"] if "alpha" in options else settings["alpha"],
        )
    )
    quantile = np.quantile(
        fit.scores[fit.real].max(axis=1), 1 - source_alpha, method="higher"
    )
    recall = float(
        (fit.scores[~fit.real, fit.routes.index("native")] > quantile).mean()
    )
    margin = float(settings.get("clean_margin", 0.02))
    if not 0 <= margin < 1:
        raise ValueError("clean_margin must be a probability")
    return max(0.0, recall - margin), {
        "dev_reference_sha256": digest(
            [fit.meta.recording_id.tolist(), fit.routes, fit.scores.tolist()]
        ),
        "source_alpha": source_alpha,
        "reference_train_signature": reference_signature,
        "reference": str(filename),
        "native_recall": recall,
        "margin": margin,
    }


def _amp(device, enabled):
    if device.type == "cuda" and enabled:
        return torch.autocast(
            "cuda",
            dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        )
    return contextlib.nullcontext()


def _loader(dataset, settings, device, batch):
    workers = int(settings.get("workers", 0))
    opts = {"num_workers": workers, "pin_memory": device.type == "cuda"}
    if workers:
        opts["prefetch_factor"] = int(settings.get("prefetch_factor", 1))
    return DataLoader(dataset, shuffle=False, batch_size=batch, **opts)


def _fit(model, train_rows, seed, spec, settings, run_dir, signature, floor, device):
    epochs = int(settings["epochs"])
    if epochs < 1:
        raise ValueError("epochs must be positive")
    head = []
    body = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (body if name.startswith("backbone.") else head).append(parameter)
    groups = [{"params": head, "lr": float(settings["learning_rate"])}]
    if body:
        groups.append(
            {
                "params": body,
                "lr": float(settings.get("backbone_lr", settings["learning_rate"])),
            }
        )
    optimizer = torch.optim.AdamW(
        groups, weight_decay=float(settings.get("weight_decay", 0.01))
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            device.type == "cuda"
            and settings.get("amp", True)
            and not torch.cuda.is_bf16_supported()
        ),
    )
    objective = spec["objective"]
    nfalse = len(settings["train_routes"]) if objective == "per_channel" else 1
    ndual = nfalse + int(objective != "no_clean")
    dual = torch.full((ndual,), float(settings.get("dual_initial", 1.0)), device=device)
    path = run_dir / "checkpoint.pt"
    if objective == "shuffled":
        # A rare stratum can become too small after epoch-specific class balancing.
        # Validate every epoch before allocating hours to a doomed shuffled run.
        for epoch in range(epochs):
            RecordingEpoch(train_rows, settings, objective, seed, epoch)
    first_epoch, history = 0, []
    if path.is_file():
        # Load optimizer and RNG on CPU; optimizer.load_state_dict maps state to parameter device.
        last = torch.load(path, map_location="cpu", weights_only=True)
        if last["signature"] != signature:
            raise ValueError(
                f"Incompatible resume state in {run_dir}; select a new output directory"
            )
        model.load_state_dict(last["model"], strict=True)
        optimizer.load_state_dict(last["optimizer"])
        scheduler.load_state_dict(last["scheduler"])
        scaler.load_state_dict(last["scaler"])
        dual = last["dual"].to(device)
        first_epoch, history = last["epoch"], last["history"]
        torch.set_rng_state(last["rng_cpu"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(last["rng_cuda"], device)

    for epoch in range(first_epoch, epochs):
        started = time.monotonic()
        exposure = 0
        dataset = RecordingEpoch(train_rows, settings, objective, seed, epoch)
        loader = _loader(dataset, settings, device, int(settings["batch_size"]))
        set_training_mode(model)
        loss_sum = 0.0
        tail_counts = []
        violations = torch.zeros_like(dual)
        start_lr = optimizer.param_groups[0]["lr"]
        for frames, labels in loader:
            exposure += int(np.prod(frames.shape[:3]))
            labels = labels.to(device, non_blocking=True)
            frames = frames.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _amp(device, settings.get("amp", True)):
                logits = video_logits(model, frames, spec, settings)
                loss, violation = loss_for_batch(
                    logits, labels, objective, settings, dual, floor
                )
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            if objective == "partial_auc":
                tail_counts.append(
                    max(
                        1,
                        int(
                            np.ceil(
                                settings["alpha"] * (len(labels) // 2) * logits.shape[1]
                            )
                        ),
                    )
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(settings.get("max_grad_norm", 5)),
                error_if_nonfinite=True,
            )
            scaler.step(optimizer)
            scaler.update()
            if violation is not None:
                violations += len(labels) * violation
                dual = (dual + float(settings.get("dual_lr", 0.1)) * violation).clamp(
                    0, float(settings.get("dual_max", 50))
                )
            loss_sum += len(labels) * loss.detach().item()
        scheduler.step()
        history.append(
            {
                "epoch": epoch + 1,
                "lr": start_lr,
                "mean_loss": loss_sum / len(dataset),
                "recordings": len(dataset),
                "updates": len(loader),
                "route_views": int(dataset.paths.shape[1]),
                "clean_floor": floor,
                "shuffle": dataset.shuffle_diagnostics,
                "min_negative_tail_count": min(tail_counts) if tail_counts else None,
                "dual": dual.tolist() if objective in CONSTRAINED else [],
                "surrogate_violations": (violations / len(dataset)).tolist()
                if objective in CONSTRAINED
                else [],
            }
        )
        atomic_torch_save(
            {
                "signature": signature,
                "epoch": epoch + 1,
                "history": history,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "dual": dual,
                "rng_cpu": torch.get_rng_state(),
                "rng_cuda": (
                    torch.cuda.get_rng_state(device) if device.type == "cuda" else None
                ),
            },
            path,
        )
        _authority(run_dir / "training_history.json", history)
        elapsed = time.monotonic() - started
        print(
            f"{spec['name']} seed={seed} epoch={epoch + 1}/{epochs} "
            f"updates={len(loader)} loss={history[-1]['mean_loss']:.5f} "
            f"frame_views={exposure} seconds={elapsed:.1f} frame_views/s={exposure / elapsed:.1f}",
            flush=True,
        )
    return path


@torch.inference_mode()
def _predict(model, data, settings, spec, seed, device, run_dir, identity):
    model.eval()
    rows = data[data.split != "train"].reset_index(drop=True)
    if rows.empty:
        raise ValueError("No evaluation renditions")
    cache = run_dir / "score_shards"
    cache.mkdir(exist_ok=True)
    shard_size = int(settings.get("shard_size", 256))
    outputs = []
    for start in range(0, len(rows), shard_size):
        shard = rows.iloc[start : start + shard_size]
        key = digest([identity, _frame_digest(shard), seed])
        dest = cache / f"{key}.csv"
        receipt = dest.with_suffix(".json")
        if (
            dest.exists()
            and receipt.exists()
            and sha256(dest) == json.loads(receipt.read_text())["sha256"]
        ):
            frame = pd.read_csv(
                dest, dtype={"recording_id": str, "route": str}, keep_default_na=False
            )
        else:
            scores = []
            loader = _loader(
                SingleCopyDataset(shard),
                settings,
                device,
                int(settings.get("inference_batch", 8)),
            )
            for frames in loader:
                with _amp(device, settings.get("amp", True)):
                    logits = video_logits(
                        model, frames.to(device, non_blocking=True), spec, settings
                    )
                scores.extend(torch.sigmoid(logits.double()).cpu().tolist())
            if not np.isfinite(scores).all():
                raise ValueError("Nonfinite model scores")
            frame = shard[["recording_id", "route"]].copy()
            frame["score"] = scores
            _write_csv(frame, dest)
            _authority(receipt, {"sha256": sha256(dest), "signature": key})
        outputs.append(frame)
    result = pd.concat(outputs, ignore_index=True)
    if result.duplicated(["recording_id", "route"]).any() or len(result) != len(rows):
        raise ValueError("Incomplete single-copy inference output")
    path = run_dir / "predictions.csv"
    _write_csv(result, path)
    return path


def run_model(
    cfg: dict, panel: pd.DataFrame, spec: dict, seed: int, run_dir: Path
) -> Path:
    """Fit a fixed-budget model, then score all evaluation splits exactly once."""
    if spec["objective"] not in OBJECTIVES or seed < 0:
        raise ValueError("Unknown objective or invalid seed")
    settings = {**cfg.get("training", {}), **spec.get("training", {})}
    settings.setdefault("alpha", 0.05)
    if not 0 < settings["alpha"] < 1 or settings.get("temperature", 0.3) <= 0:
        raise ValueError("Invalid false-alert budget or surrogate temperature")
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    data = panel.copy()
    data["recording_id"] = data.recording_id.astype(str)
    data["route"] = data.route.astype(str)
    if data.duplicated(["recording_id", "route"]).any() or not set(data.label).issubset(
        {0, 1}
    ):
        raise ValueError("Invalid model panel keys/labels")
    data = data.sort_values(["recording_id", "route"]).reset_index(drop=True)
    with (run_dir / ".model.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        device = torch.device(
            settings.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        assets = validate_assets(spec)
        floor, reference = _floor(cfg, data, settings, spec, seed, run_dir)
        train = data[data.split == "train"]
        software = _fingerprint()
        for package in (
            {"timm"}
            if spec["backbone"] == "dino"
            else {"transformers"}
            if spec["backbone"] == "clip"
            else set()
        ):
            software["packages"][package] = importlib.metadata.version(package)
        signature = digest(
            [
                spec,
                settings,
                seed,
                assets,
                _frame_digest(train),
                reference,
                software,
                device.type,
                torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            ]
        )
        identity = digest([signature, _frame_digest(data)])
        done = run_dir / "model.json"
        path = run_dir / "predictions.csv"
        if done.exists():
            saved = json.loads(done.read_text())
            if saved["train_signature"] != signature:
                raise ValueError(f"Stale model artifacts in {run_dir}")
            if (
                saved["identity"] == identity
                and path.exists()
                and sha256(path) == saved["prediction_sha256"]
                and Path(saved["checkpoint"]).is_file()
                and sha256(saved["checkpoint"]) == saved["checkpoint_sha256"]
            ):
                return path
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        random.seed(seed)
        np.random.seed(seed % 2**32)
        torch.manual_seed(seed)
        if device.type == "cpu":
            torch.set_num_threads(int(settings.get("cpu_threads", 1)))
        torch.use_deterministic_algorithms(bool(settings.get("deterministic", True)))
        torch.backends.cudnn.benchmark = False
        model = build_model(spec, settings).to(device)
        if spec["objective"] == "frozen":
            checkpoint = Path(spec["checkpoint"]).resolve()
        else:
            checkpoint = _fit(
                model, train, seed, spec, settings, run_dir, signature, floor, device
            )
        ck_hash = sha256(checkpoint)
        path = _predict(
            model,
            data,
            settings,
            spec,
            seed,
            device,
            run_dir,
            digest([signature, ck_hash]),
        )
        _authority(
            done,
            {
                "identity": identity,
                "train_signature": signature,
                "spec": spec,
                "seed": seed,
                "software": software,
                "assets": assets,
                "training": settings,
                "reference": reference,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": ck_hash,
                "prediction_sha256": sha256(path),
                "device": str(device),
                "panel_sha256": _frame_digest(data),
                "score": "sigmoid(mean frame logit difference)",
            },
        )
        return path
