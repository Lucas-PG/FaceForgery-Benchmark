"""Provision pinned official DINOv3 and CLIP initialization tensors for the TIFS campaign."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
from importlib.metadata import version
from pathlib import Path

from .io import sha, write_json

# Hugging Face Git revisions and the published raw object hashes. Archive layouts
# vary between releases; conversion below explicitly verifies all vision parameters.
DINO = "timm/convnext_base.dinov3_lvd1689m", "8e953102486d832f58e073cb4080f04001bb71ca"
CLIP = "openai/clip-vit-base-patch16", "57c216476eefef5ab752ec549e440a49ae4ae5f3"
SOURCES = {
    "dino_raw": (
        *DINO,
        "model.safetensors",
        "afa69ae6094b5b04a48810de78c52c92d4cbb6d9436462f07bf36defb09e26c1",
    ),
    "clip_raw": (
        *CLIP,
        "pytorch_model.bin",
        "ec89c7b09c749a60aae3c9cd910516f24b58214a7df060b48962d14c469cfbf0",
    ),
    "clip_config": (
        *CLIP,
        "config.json",
        "eaf1c9089a8553c913d27ea66407f8bfc2be9989c80c9f331ddb3d63d4c5e8ad",
    ),
}
OUTPUTS = ("dino/base.pth", "clip/config.json", "clip/model.safetensors")
FORMAT = 1


def source_spec():
    return {
        k: dict(zip(("repository", "revision", "file", "sha256"), v))
        for k, v in SOURCES.items()
    }


def _source(key, local, incoming):
    repo, rev, name, expected = SOURCES[key]
    if local is None:
        path = incoming / name
        url = f"https://huggingface.co/{repo}/resolve/{rev}/{name}?download=true"
        for attempt in range(2):
            try:
                with (
                    urllib.request.urlopen(url, timeout=120) as response,
                    path.open("wb") as out,
                ):
                    shutil.copyfileobj(response, out, length=1024 * 1024)
                break
            except OSError:
                if attempt:
                    raise
    else:
        path = Path(local).resolve(strict=True)
    if sha(path) != expected:
        raise ValueError(f"Expected official {key} sha256 {expected}: {path}")
    return path


def tensor_digest(state, family):
    """Fingerprint actual initial tensors across supported state-dict serialization formats."""
    import torch

    tensors = {}
    for name, tensor in state.items():
        key = name.removeprefix("vision_model.") if family == "clip" else name
        if key in tensors:
            raise ValueError("Duplicate canonical tensor key")
        tensors[key] = tensor
    result = hashlib.sha256()
    for key, tensor in sorted(tensors.items()):
        descriptor = json.dumps(
            [key, str(tensor.dtype), list(tensor.shape)], separators=(",", ":")
        ).encode()
        result.update(descriptor + b"\x00")
        result.update(
            memoryview(
                tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
            )
        )
    return result.hexdigest()


CANONICAL_TENSORS = {
    "dino": "29562b2bde0cda0d635685fbfb88bbcd92a765ce062d45eb00b8717d44b8edc0",
    "clip": "eb847ed98e9cbaed19a05b0bc93b17e33137f3bf038c2248165ecf18057b7b49",
}


def verify_initial(family, path):
    import torch
    from safetensors.torch import load_file

    path = Path(path)
    state = (
        torch.load(path, map_location="cpu", weights_only=True)
        if family == "dino"
        else load_file(str(path / "model.safetensors"), device="cpu")
    )
    fingerprint = tensor_digest(state, family)
    if fingerprint != CANONICAL_TENSORS[family]:
        raise ValueError(
            f"Initialization differs from the registered official {family} tensors: {path}"
        )
    return fingerprint


def _dino(raw, dest):
    import timm
    import torch
    from safetensors.torch import load_file

    model = timm.create_model(
        "convnext_base.dinov3_lvd1689m", pretrained=False, num_classes=0
    )
    state = load_file(str(raw), device="cpu")
    model.load_state_dict(state, strict=True)
    dest.mkdir()
    torch.save(state, dest / "base.pth")
    return len(state)


def _clip(raw, config_file, dest):
    import torch
    from transformers import CLIPVisionConfig, CLIPVisionModel

    # The OpenAI release includes the text tower and a contrastive projection.
    # The spatial baseline uses the unprojected vision model, including layer norm.
    config = CLIPVisionConfig.from_pretrained(str(config_file), local_files_only=True)
    if (config.image_size, config.patch_size, config.hidden_size) != (224, 16, 768):
        raise ValueError("Unexpected official CLIP vision configuration")
    model = CLIPVisionModel(config)
    released = torch.load(raw, map_location="cpu", weights_only=True)
    state = {k: v for k, v in released.items() if k.startswith("vision_model.")}
    expected = set(model.state_dict())
    # Transformers 5 flattened the vision_model prefix on the standalone tower.
    if not any(k.startswith("vision_model.") for k in expected):
        state = {k.removeprefix("vision_model."): v for k, v in state.items()}
    for name in ("vision_model.embeddings.position_ids", "embeddings.position_ids"):
        if name in state and name not in expected:
            ids = state.pop(name)
            if not torch.equal(ids, torch.arange(197).view(1, 197)):
                raise ValueError("Unexpected official CLIP positional index buffer")
    model.load_state_dict(state, strict=True)
    model.save_pretrained(dest, safe_serialization=True)
    return len(state)


def verify(folder):
    receipt = json.loads((folder / "initial.json").read_text())
    if (
        receipt.get("format") != FORMAT
        or receipt.get("sources") != source_spec()
        or set(receipt.get("outputs", {})) != set(OUTPUTS)
    ):
        raise ValueError(f"Stale or incompatible initialization receipt in {folder}")
    for name in OUTPUTS:
        if sha(folder / name) != receipt["outputs"][name]:
            raise ValueError(f"Changed initialization material: {folder / name}")
    return receipt


def provision(out, local=None):
    out = Path(out).resolve()
    local = local or {}
    out.parent.mkdir(parents=True, exist_ok=True)
    with (out.parent / (out.name + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if out.exists():
            for key, path in local.items():
                if path is not None:
                    _source(key, path, None)
            return verify(out)
        with tempfile.TemporaryDirectory(
            dir=out.parent, prefix=out.name + "-stage-"
        ) as staging:
            stage = Path(staging)
            dest = stage / "ready"
            dest.mkdir()
            raw = {k: _source(k, local.get(k), stage) for k in SOURCES}
            counts = {
                "dino": _dino(raw["dino_raw"], dest / "dino"),
                "clip": _clip(raw["clip_raw"], raw["clip_config"], dest / "clip"),
            }
            receipt = {
                "format": FORMAT,
                "sources": source_spec(),
                "verified_tensors": counts,
                "outputs": {k: sha(dest / k) for k in OUTPUTS},
                "packages": {
                    p: version(p)
                    for p in ("torch", "timm", "safetensors", "transformers")
                },
            }
            write_json(dest / "initial.json", receipt)
            os.rename(dest, out)
            return receipt


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="new directory, or an existing verified export",
    )
    for key in SOURCES:
        p.add_argument(
            "--" + key.replace("_", "-"),
            type=Path,
            help="use this local official raw file instead of downloading it",
        )
    args = p.parse_args(argv)
    provision(args.out, {k: getattr(args, k) for k in SOURCES})
    print(args.out.resolve() / "initial.json")


if __name__ == "__main__":
    main()
