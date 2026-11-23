"""Offline spatial detector construction. Checkpoint conventions are explicit."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .io import sha as sha256

HISTORICAL_REPO = "lucasoc/MFFI-Models"
HISTORICAL_REVISION = "b408d7572fcc04638ce20adee0546471d46dce2a"
# Official LFS object digest and SHA-256 of the published sibling run_config.json.
HISTORICAL = {
    "clip": (
        "6435d90c05d9628bd9527929404d0c3beeb241239f6d7b51bec9c72a2d1e1838",
        "e9e91312664f392922f2ed581a4b0a7dd1556245236d4f82027273b95fdc70e0",
    ),
    "dino": (
        "47e087733f1847e8fc744e63ea2efff375bb76dce2e9d404c864037853b67043",
        "4fd72c5241403b076acb3781e02d728d45cff7e9e61b37760e44f62335299696",
    ),
}


def validate_assets(spec: dict) -> dict[str, str]:
    """Inspect immutable bytes before allocating the accelerator or reusing output."""
    family = spec["backbone"]
    if family not in {"dino", "clip", "tiny"}:
        raise ValueError(f"Unsupported spatial detector {family}")
    frozen = spec["objective"] == "frozen"
    assets = {}
    if frozen or family != "tiny":
        key = "checkpoint" if frozen else "weights"
        path = Path(spec[key])
        if not frozen and family == "clip":
            assets["clip_config"] = sha256(path / "config.json")
            if (
                "clip_config_sha256" in spec
                and assets["clip_config"] != spec["clip_config_sha256"]
            ):
                raise ValueError("CLIP architecture configuration SHA256 mismatch")
            path = path / "model.safetensors"
        digest = sha256(path)
        if digest != spec[f"{key}_sha256"]:
            raise ValueError(f"{key} SHA256 mismatch: {path}")
        assets[key] = digest
    if frozen and family != "tiny":
        if spec.get("positive_index") != 1:
            raise ValueError("Released MFFI checkpoints use class 1 for fake")
        if assets["checkpoint"] != HISTORICAL[family][0]:
            raise ValueError(
                f"Expected the published seed-42 robust RGB {family} checkpoint"
            )
        config = Path(spec["run_config"])
        actual = sha256(config)
        if actual != HISTORICAL[family][1] or actual != spec.get("run_config_sha256"):
            raise ValueError(
                "Historical training configuration is not the matching released file"
            )
        old = json.loads(config.read_text())
        # Both published configs serialize the generic 'finetune' name and omit
        # 'robust'. The immutable checkpoint/config pair identifies the actual run.
        if (
            old["model_family"],
            old["fourier_mode"],
            old["seed"],
            old["image_size"],
        ) != (family, "none", 42, 224):
            raise ValueError(
                "Unsupported historical family, RGB representation, seed or image geometry"
            )
        if family == "dino" and old.get("model_size", "base") != spec.get(
            "model_size", "base"
        ):
            raise ValueError("Historical DINO size disagrees with configuration")
        assets["run_config"] = actual
    return assets


class TinyClassifier(nn.Sequential):
    """Small offline network for verifying the pipeline on CPU fixtures."""

    def __init__(self):
        super().__init__(
            nn.Conv2d(3, 8, 3, stride=2, padding=1),
            nn.GroupNorm(2, 8),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(8, 2),
        )


def _load_clip_state(model, state, *, detector=False):
    """Permit only the HF4/HF5 vision wrapper rename; every tensor and head must match."""
    namespace = "backbone." if detector else ""
    target = model.state_dict()
    if set(state) != set(target):
        legacy = namespace + "vision_model."
        target_has_wrapper = any(k.startswith(legacy) for k in target)
        roots = ("embeddings.", "encoder.", "pre_layrnorm.", "post_layernorm.")
        renamed = {}
        for name, value in state.items():
            key = name
            if target_has_wrapper and name.startswith(namespace):
                suffix = name[len(namespace) :]
                if suffix.startswith(roots):
                    key = legacy + suffix
            elif not target_has_wrapper and name.startswith(legacy):
                suffix = name[len(legacy) :]
                if suffix.startswith(roots):
                    key = namespace + suffix
            if key in renamed:
                raise ValueError(
                    f"Duplicate CLIP tensor after wrapper conversion: {key}"
                )
            renamed[key] = value
        state = renamed
    if set(state) != set(target):
        raise ValueError(
            "CLIP checkpoint keys do not match the complete vision model and expected head"
        )
    if any(state[k].shape != v.shape for k, v in target.items()):
        raise ValueError("Incompatible CLIP checkpoint tensor dimensions")
    model.load_state_dict(state, strict=True)


def build_model(spec: dict, settings: dict) -> nn.Module:
    family = spec["backbone"]
    frozen = spec["objective"] == "frozen"
    dropout = float(settings.get("dropout", 0.1))
    blocks = int(settings.get("unfreeze_blocks", 1))
    if blocks < 0:
        raise ValueError("unfreeze_blocks must be nonnegative")

    if family == "tiny":
        model = TinyClassifier()
    elif family == "dino":
        import timm

        from src.models.dino import DINOVisionClassifier, unfreeze_for_finetune

        size = spec.get("model_size", "base")
        backbone = timm.create_model(
            f"convnext_{size}.dinov3_lvd1689m", pretrained=False, num_classes=0
        )
        if not frozen:
            state = torch.load(spec["weights"], map_location="cpu", weights_only=True)
            backbone.load_state_dict(state, strict=True)
        model = DINOVisionClassifier(backbone, dropout)
        unfreeze_for_finetune(model, blocks)
    elif family == "clip":
        from transformers import CLIPVisionConfig, CLIPVisionModel

        from src.models.clip import CLIPClassifier, unfreeze_for_finetune

        if frozen:
            # The RGB historical fine-tuned OpenAI ViT-B/16 head/backbone.
            config = CLIPVisionConfig(
                patch_size=16,
                image_size=224,
                hidden_size=768,
                intermediate_size=3072,
                num_hidden_layers=12,
                num_attention_heads=12,
                hidden_act="quick_gelu",
            )
            backbone = CLIPVisionModel(config)
        else:
            from safetensors.torch import load_file

            directory = Path(spec["weights"])
            config = CLIPVisionConfig.from_pretrained(
                str(directory), local_files_only=True
            )
            if (
                config.image_size,
                config.patch_size,
                config.hidden_size,
                config.num_hidden_layers,
            ) != (224, 16, 768, 12):
                raise ValueError(
                    "Matched CLIP requires the 224px ViT-B/16 vision snapshot"
                )
            backbone = CLIPVisionModel(config)
            _load_clip_state(backbone, load_file(str(directory / "model.safetensors")))
        if hasattr(backbone, "set_attn_implementation"):
            backbone.set_attn_implementation("eager")
        model = CLIPClassifier(backbone, dropout)
        unfreeze_for_finetune(model, blocks)
    else:
        raise ValueError(f"Unsupported spatial detector {family}")
    if frozen:
        state = torch.load(spec["checkpoint"], map_location="cpu", weights_only=True)
        if family == "clip":
            _load_clip_state(model, state, detector=True)
        else:
            model.load_state_dict(state, strict=True)
        model.requires_grad_(False)
    return model


def set_training_mode(model: nn.Module):
    model.train()
    for module in model.modules():
        parameters = list(module.parameters())
        if parameters and not any(param.requires_grad for param in parameters):
            module.eval()


def video_logits(model: nn.Module, frames: torch.Tensor, spec: dict, settings: dict):
    """Map [..., T, C, H, W] uint8 to one signed logit per submitted copy."""
    shape = frames.shape
    if frames.dtype != torch.uint8 or shape[-3] != 3:
        raise ValueError("Expected uint8 RGB frames")
    frames = frames.flatten(0, -4)
    if spec["backbone"] != "tiny" and frames.shape[-2:] != (224, 224):
        raise ValueError("DINO/CLIP comparison requires 224x224 frame cache")
    # Keep TCC checkpoints on their actual historical image normalization.
    norm = (
        "clip"
        if spec["backbone"] == "clip" and spec["objective"] != "frozen"
        else "imagenet"
    )
    mean, std = (
        ((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        if norm == "clip"
        else ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    )
    mean = torch.tensor(mean, device=frames.device).view(1, 3, 1, 1)
    std = torch.tensor(std, device=frames.device).view(1, 3, 1, 1)
    positive = (
        int(spec.get("positive_index", 1)) if spec["objective"] == "frozen" else 1
    )
    means = []
    for part in frames.split(int(settings.get("frame_batch", 32))):
        images = (part.float().div_(255) - mean) / std
        if model.training and settings.get("activation_checkpoint", False):
            output = checkpoint(model, images, use_reentrant=False)
        else:
            output = model(images)
        means.append((output[:, positive] - output[:, 1 - positive]).float())
    return torch.cat(means).reshape(shape[:-3]).mean(-1)
