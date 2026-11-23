"""Opt-in released-weight parity with the pinned projects, offline.

TIFS_PARITY_REGISTRY is an assets.json produced by python -m tifs.assets.
TIFS_PARITY_VIDEO must be a >=1s, single-face video cleared for local testing.
"""

import gc
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from tifs.external import GenD, LipForensics, external_assets

pytestmark = pytest.mark.skipif(
    not os.environ.get("TIFS_PARITY_REGISTRY"),
    reason="released asset registry unavailable",
)


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_released_gend_official_forward(tmp_path, monkeypatch):
    import torch
    from safetensors.torch import load_file
    from transformers import CLIPConfig, CLIPImageProcessor, CLIPModel, CLIPProcessor

    entry = json.loads(Path(os.environ["TIFS_PARITY_REGISTRY"]).read_text())["gend"]
    spec = {**entry, "backbone": "gend"}
    root = external_assets(spec)
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    torch.set_num_threads(2)
    imgs = np.random.default_rng(21).integers(
        0, 256, size=(2, 224, 224, 3), dtype=np.uint8
    )
    path = tmp_path / "frames.npy"
    np.save(path, imgs)
    with torch.inference_mode():
        candidate = GenD(spec, root, torch.device("cpu"))
        value, failure = candidate.score(SimpleNamespace(frames_path=path))
        assert not failure
        del candidate
        gc.collect()
        # Run the actual upstream GenDConfig -> GenD -> CLIPEncoder -> LinearProbe.
        # Only upstream network constructors are redirected to the exact local
        # architecture/processor; load all released tensors strictly, including projection.
        api = _module("parity_gend", root / "src/hf/modeling_gend.py")
        cfg = CLIPConfig.from_json_file(spec["backbone_config"])
        monkeypatch.setattr(
            CLIPModel, "from_pretrained", lambda *a, **kw: CLIPModel(cfg)
        )
        processor = CLIPImageProcessor.from_dict(
            json.loads(Path(spec["processor"]).read_text())
        )
        monkeypatch.setattr(
            CLIPProcessor, "from_pretrained", lambda *a, **kw: processor
        )
        upstream = api.GenD(
            api.GenDConfig.from_json_file(spec["detector_config"])
        ).eval()
        upstream.load_state_dict(load_file(spec["weights"]), strict=True)
        pixels = torch.stack(
            [upstream.feature_extractor.preprocess(Image.fromarray(x)) for x in imgs]
        )
        official = upstream(pixels).softmax(1)[:, 1].mean().item()
        assert value == pytest.approx(official, abs=2e-6)


@pytest.mark.skipif(
    not os.environ.get("TIFS_PARITY_VIDEO"),
    reason="cleared single-face video unavailable",
)
def test_released_lip_official_crops_and_logits(tmp_path, monkeypatch):
    import dlib
    import torch
    from torchvision.transforms import CenterCrop

    entry = json.loads(Path(os.environ["TIFS_PARITY_REGISTRY"]).read_text())["lip"]
    spec = {**entry, "backbone": "lipforensics", "clips": 1}
    root = external_assets(spec)
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    torch.set_num_threads(2)
    path = Path(os.environ["TIFS_PARITY_VIDEO"]).resolve()
    with torch.inference_mode():
        candidate = LipForensics(spec, root, torch.device("cpu"))
        value, failure = candidate.score(
            SimpleNamespace(
                media_path=str(path),
                offset_seconds=0,
                window_start_seconds=0,
                duration=1.0,
            )
        )
        assert failure == ""
        del candidate
        gc.collect()

        # Establish the same received pixels, then hand actual upstream preprocessing
        # PNG frames and independently detected landmarks. It performs its own queued
        # smoothing, warp and 96px crop, then PIL grayscale + official evaluation transforms.
        raw = subprocess.check_output(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-threads",
                "1",
                "-i",
                str(path),
                "-t",
                "1",
                "-an",
                "-vf",
                "fps=25,scale=640:480:force_original_aspect_ratio=decrease,pad=640:480:(ow-iw)/2:(oh-ih)/2",
                "-frames:v",
                "25",
                "-pix_fmt",
                "rgb24",
                "-threads",
                "1",
                "-f",
                "rawvideo",
                "pipe:1",
            ]
        )
        rgb = np.frombuffer(raw, dtype=np.uint8).reshape(25, 480, 640, 3)
        frames, points, mouths = [
            tmp_path / x for x in ("frames", "landmarks", "mouths")
        ]
        frames.mkdir()
        points.mkdir()
        detect = dlib.get_frontal_face_detector()
        predict = dlib.shape_predictor(spec["landmarks"])
        for i, img in enumerate(rgb):
            faces = detect(img, 0)
            assert len(faces) == 1
            landmarks = np.array([(p.x, p.y) for p in predict(img, faces[0]).parts()])
            np.save(points / f"{i:04d}.npy", landmarks)
            Image.fromarray(img).save(frames / f"{i:04d}.png")
        monkeypatch.setitem(
            sys.modules,
            "utils",
            _module("parity_lip_utils", root / "preprocessing/utils.py"),
        )
        crop = _module("parity_lip_crop", root / "preprocessing/crop_mouths.py")
        crop.crop_video_and_save(
            str(frames),
            str(points),
            str(mouths),
            np.load(root / "preprocessing/20words_mean_face.npy"),
            SimpleNamespace(
                window_margin=12,
                start_idx=48,
                stop_idx=68,
                crop_height=96,
                crop_width=96,
            ),
        )
        items = np.stack(
            [
                np.asarray(Image.open(p).convert("L"))[..., None]
                for p in sorted(mouths.glob("*.png"))
            ]
        )
        transf = _module("parity_lip_transforms", root / "data/transforms.py")
        clip = transf.NormalizeVideo((0.421,), (0.165,))(
            CenterCrop((88, 88))(transf.ToTensorVideo()(torch.from_numpy(items)))
        )
        monkeypatch.chdir(root)
        monkeypatch.syspath_prepend(str(root))
        from models.spatiotemporal_net import get_model

        reference = get_model(weights_forgery_path=None, device="cpu").eval()
        reference.load_state_dict(
            torch.load(spec["weights"], map_location="cpu", weights_only=True)["model"],
            strict=True,
        )
        official = reference(clip.unsqueeze(0), lengths=[25]).sigmoid().item()
        assert value == pytest.approx(official, abs=2e-6)
