"""Strict archived detector identity and the necessary HF4/HF5 vision-wrapper mapping."""

import json
import os
from pathlib import Path

import pytest
import torch
from torch import nn

from tifs import models


class Vision(nn.Module):
    def __init__(self, wrapped):
        super().__init__()
        if wrapped:
            self.vision_model = nn.Module()
        impl = self.vision_model if wrapped else self
        impl.embeddings = nn.Linear(2, 3)
        impl.encoder = nn.Linear(3, 2)


class Detector(nn.Module):
    def __init__(self, wrapped):
        super().__init__()
        self.backbone = Vision(wrapped)
        self.classifier = nn.Linear(2, 2)


@pytest.mark.parametrize("detector", [False, True])
@pytest.mark.parametrize("legacy_source", [False, True])
def test_strict_hf4_hf5_prefix_and_head(detector, legacy_source):
    constructor = Detector if detector else Vision
    source = constructor(legacy_source).state_dict()
    target = constructor(not legacy_source)
    models._load_clip_state(target, source, detector=detector)
    roundtrip = constructor(legacy_source)
    models._load_clip_state(roundtrip, target.state_dict(), detector=detector)
    assert all(torch.equal(v, roundtrip.state_dict()[k]) for k, v in source.items())
    trimmed = dict(source)
    trimmed.pop(next(iter(trimmed)))
    with pytest.raises(ValueError, match="keys"):
        models._load_clip_state(target, trimmed, detector=detector)
    corrupted = dict(source)
    corrupted[next(iter(source))] = torch.zeros(17)
    with pytest.raises(ValueError, match="dimensions"):
        models._load_clip_state(target, corrupted, detector=detector)


def test_collision_and_untrusted_self_declared_archived_weights(tmp_path):
    old = Vision(True).state_dict()
    old["embeddings.weight"] = old["vision_model.embeddings.weight"]
    with pytest.raises(ValueError, match="Duplicate"):
        models._load_clip_state(Vision(False), old)
    path = tmp_path / "best.pth"
    torch.save(old, path)
    sha = models.sha256(path)
    with pytest.raises(ValueError, match="published"):
        models.validate_assets(
            {
                "backbone": "clip",
                "objective": "frozen",
                "checkpoint": str(path),
                "checkpoint_sha256": sha,
                "positive_index": 1,
            }
        )


def test_generic_finetune_metadata_only_for_exact_attested_pair(tmp_path, monkeypatch):
    weight = tmp_path / "weights"
    weight.write_bytes(b"fixture-only")
    cfg = tmp_path / "run_config.json"
    cfg.write_text(
        json.dumps(
            {
                "model_family": "dino",
                "regime": "finetune",
                "seed": 42,
                "fourier_mode": "none",
                "image_size": 224,
                "model_size": "base",
            }
        )
    )
    weight_hash, cfg_hash = models.sha256(weight), models.sha256(cfg)
    monkeypatch.setitem(models.HISTORICAL, "dino", (weight_hash, cfg_hash))
    spec = {
        "backbone": "dino",
        "objective": "frozen",
        "positive_index": 1,
        "checkpoint": weight,
        "checkpoint_sha256": weight_hash,
        "run_config": cfg,
        "run_config_sha256": cfg_hash,
    }
    assert models.validate_assets(spec)["checkpoint"] == weight_hash
    cfg.write_text(cfg.read_text() + " ")
    with pytest.raises(ValueError, match="matching released file"):
        models.validate_assets(spec | {"run_config_sha256": models.sha256(cfg)})
    with pytest.raises(ValueError, match="class 1"):
        models.validate_assets(spec | {"positive_index": 0})


@pytest.mark.skipif(
    not os.environ.get("TIFS_MFFI_RELEASE"), reason="pinned MFFI weights unavailable"
)
@pytest.mark.parametrize("family", ["clip", "dino"])
def test_pinned_real_archived_detector_and_cpu_score(family):
    root = (
        Path(os.environ["TIFS_MFFI_RELEASE"]) / family / "none/finetune_robust/seed_42"
    )
    weight = root / "weights/best.pth"
    cfg = root / "results/run_config.json"
    wh, ch = models.HISTORICAL[family]
    spec = {
        "backbone": family,
        "objective": "frozen",
        "positive_index": 1,
        "checkpoint": weight,
        "checkpoint_sha256": wh,
        "run_config": cfg,
        "run_config_sha256": ch,
    }
    assets = models.validate_assets(spec)
    assert assets == {"checkpoint": wh, "run_config": ch}
    torch.set_num_threads(2)
    detector = models.build_model(spec, {}).eval()
    x = torch.randint(
        0,
        256,
        (1, 1, 3, 224, 224),
        generator=torch.Generator().manual_seed(4),
        dtype=torch.uint8,
    )
    with torch.inference_mode():
        value = models.video_logits(detector, x, spec, {})
    assert value.shape == (1,) and torch.isfinite(value).all()
