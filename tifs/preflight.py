"""Verify local asset bytes and actual architecture construction before queuing training."""

import gc

import torch
from PIL import Image

from .external import GenD, LipForensics, external_assets
from .models import build_model, validate_assets, video_logits


def probe(cfg):
    result = []
    done = set()
    for spec in cfg["models"]:
        kind = spec["backbone"]
        historical = spec["objective"] == "frozen"
        key = kind, spec.get("assets"), historical
        if key in done:
            continue
        done.add(key)
        with torch.inference_mode():
            if kind in {"gend", "lipforensics"}:
                root = external_assets(spec)
                if kind == "gend":
                    model = GenD(spec, root, torch.device("cpu"))
                    x = model.processor(
                        images=[Image.new("RGB", (224, 224))], return_tensors="pt"
                    ).pixel_values
                    result_tensor = model.head(model.encoder(x).pooler_output)
                else:
                    model = LipForensics(spec, root, torch.device("cpu"))
                    result_tensor = model.model(
                        torch.zeros(1, 1, 25, 88, 88), lengths=[25]
                    )
            else:
                validate_assets(spec)
                settings = {**cfg["training"], **spec.get("training", {})}
                model = build_model(spec, settings).eval()
                size = cfg["data"]["size"]
                result_tensor = video_logits(
                    model,
                    torch.zeros((1, 1, 3, size, size), dtype=torch.uint8),
                    spec,
                    settings,
                )
            if not torch.isfinite(result_tensor).all():
                raise ValueError(f"Nonfinite preflight scores from {spec['name']}")
            result.append(spec["name"])
            del model
            gc.collect()
    return result
