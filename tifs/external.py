"""Single-copy comparators. Official checkpoint conventions, immutable local assets."""

import contextlib
import importlib.util
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .io import atomic_text, sha

LIP_REV = "d0bf5553bfb9676f1771d590472b26a3a76de894"
GEND_REV = "387a42266dd385fbe8f7626c5d3aa03eea3bfaab"

# SHA-256 of the published releases, independent of the local asset registry.
RELEASE_HASHES = {
    "gend": {
        "weights": "d76f0bdfd74a29fe1b1c1b84a80ac92486993e426878e8c7a3944281fbb96833",
        "detector_config": "7f2761e13678191774a152121f1630d19baa33a74216dd264d54445d96b16229",
        "processor": "910e70b3956ac9879ebc90b22fb3bc8a75b6a0677814500101a4c072bd7857bd",
        "backbone_config": "8a09b467700c58138c29d53c605b34ebc69beaadd13274a8a2af8ad2c2f4032a",
    },
    "lipforensics": {
        "weights": "4b7790bc8e02d0c25ecfa0d8d6a2907123c2206cc32e2bad6044e50f013c253d",
        "landmarks": "fbdc2cb80eb9aa7a758672cbfdda32ba6300efe9b6e6c7a299ff7e736b11b92f",
    },
}


def check_repo(directory, expected):
    root = Path(directory).resolve()
    options = ["git", "-C", str(root)]
    if (
        subprocess.check_output(options + ["rev-parse", "HEAD"], text=True).strip()
        != expected
    ):
        raise ValueError(f"Unexpected upstream revision in {root}, expected {expected}")
    if subprocess.check_output(
        options + ["status", "--porcelain", "--untracked-files=all"], text=True
    ).strip():
        raise ValueError(f"Upstream baseline source must be clean: {root}")
    return root


def load_module(name, path, package=False):
    locations = [str(path.parent)] if package else None
    spec = importlib.util.spec_from_file_location(
        name, path, submodule_search_locations=locations
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    before = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = before
    return module


def asset(path, expected):
    if sha(path) != expected:
        raise ValueError(f"Asset SHA-256 mismatch: {path}")
    return Path(path).resolve()


def external_assets(spec):
    """Verify before expensive training. The caller stores these identities in the run receipt."""
    kind = spec["backbone"]
    root = check_repo(spec["upstream"], GEND_REV if kind == "gend" else LIP_REV)
    for key, expected in RELEASE_HASHES[kind].items():
        if spec[key + "_sha256"] != expected:
            raise ValueError(f"Registry does not describe the released {kind} {key}")
        asset(spec[key], expected)
    return root


class GenD:
    """Load the released GenD CLIP-L/14 vision encoder and its original normalized/linear probe."""

    def __init__(self, spec, root, device):
        from safetensors.torch import load_file
        from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel

        module = load_module("tifs_gend_official", root / "src/hf/modeling_gend.py")
        cfg = json.loads(Path(spec["detector_config"]).read_text())
        if cfg.get("backbone") != "openai/clip-vit-large-patch14" or cfg[
            "head"
        ] not in {"linear", "LinearNorm"}:
            raise ValueError("Expected the released GenD_CLIP_L_14 configuration")
        vision = CLIPVisionConfig.from_pretrained(
            str(spec["backbone_config"]), local_files_only=True
        )
        wrapper = CLIPVisionModel(vision)
        self.encoder = getattr(wrapper, "vision_model", wrapper)
        self.head = module.LinearProbe(
            vision.hidden_size, 2, cfg["head"] == "LinearNorm"
        )
        # The upstream forward uses pooled vision states, never visual_projection.
        state = load_file(spec["weights"], device="cpu")
        prefix = "feature_extractor.vision_model."
        self.encoder.load_state_dict(
            {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)},
            strict=True,
        )
        self.head.load_state_dict(
            {k[6:]: v for k, v in state.items() if k.startswith("model.")}, strict=True
        )
        extra = [
            k
            for k in state
            if not k.startswith(
                (prefix, "model.", "feature_extractor.visual_projection.")
            )
        ]
        if extra:
            raise ValueError(f"Unexpected GenD checkpoint parameters: {extra[:5]}")
        self.processor = CLIPImageProcessor.from_dict(
            json.loads(Path(spec["processor"]).read_text())
        )
        self.encoder.to(device).eval()
        self.head.to(device).eval()
        self.device = device
        self.batch = int(spec.get("frame_batch", 32))
        if self.batch < 1:
            raise ValueError("frame_batch must be positive")

    def score(self, row):
        frames = np.load(row.frames_path, mmap_mode="r", allow_pickle=False)
        probs = []
        for start in range(0, len(frames), self.batch):
            images = [
                Image.fromarray(frame) for frame in frames[start : start + self.batch]
            ]
            pixels = self.processor(images=images, return_tensors="pt").pixel_values.to(
                self.device
            )
            logits = self.head(self.encoder(pixel_values=pixels).pooler_output)
            probs.extend(logits.float().softmax(1)[:, 1].tolist())
        return float(np.mean(probs)), ""


class LipForensics:
    """Official MS-TCN + mouth warp. Use contiguous frames at 25 Hz, never sparse panel samples."""

    def __init__(self, spec, root, device):
        import types

        import dlib
        import torch

        namespace = "tifs_lip_official"
        pkg = types.ModuleType(namespace)
        pkg.__path__ = [str(root / "models")]
        sys.modules[namespace] = pkg
        api = load_module(
            namespace + ".spatiotemporal_net", root / "models/spatiotemporal_net.py"
        )
        self.warp = load_module("tifs_lip_warp", root / "preprocessing/utils.py")
        params = json.loads(
            (root / "models/configs/lrw_resnet18_mstcn.json").read_text()
        )
        keys = ("num_layers", "kernel_size", "dropout", "dwpw", "width_mult")
        self.model = api.Lipreading(
            num_classes=1,
            relu_type=params["relu_type"],
            tcn_options={k: params["tcn_" + k] for k in keys},
        )
        weights = torch.load(spec["weights"], map_location="cpu", weights_only=True)
        self.model.load_state_dict(weights["model"], strict=True)
        self.model.to(device).eval()
        self.meanface = np.load(root / "preprocessing/20words_mean_face.npy")
        self.detect = dlib.get_frontal_face_detector()
        self.predict = dlib.shape_predictor(str(spec["landmarks"]))
        self.device = device
        self.frames = 25 * int(spec.get("clips", 4))
        if not 25 <= self.frames <= 200:
            raise ValueError("Expected 1–8 contiguous 25-frame clips")

    def score(self, row):
        import torch

        # Align native-time origins using the measured receipt; use a bounded-size input for landmarks.
        offset = float(row.offset_seconds) + float(row.window_start_seconds)
        if offset < 0:
            return np.nan, "unavailable_aligned_start"
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-threads",
            "1",
            "-ss",
            str(offset),
            "-i",
            row.media_path,
            "-t",
            str(float(row.duration)),
            "-an",
            "-vf",
            "fps=25,scale=640:480:force_original_aspect_ratio=decrease,pad=640:480:(ow-iw)/2:(oh-ih)/2",
            "-frames:v",
            str(self.frames),
            "-pix_fmt",
            "rgb24",
            "-threads",
            "1",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
        raw = subprocess.check_output(cmd)
        frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 480, 640, 3)
        frames = frames[: (len(frames) // 25) * 25]
        if len(frames) == 0:
            return np.nan, "shorter_than_25_aligned_frames"
        landmarks = []
        for frame in frames:
            faces = self.detect(frame, 0)
            if len(faces) != 1:
                return np.nan, "face_missing_or_ambiguous"
            shape = self.predict(frame, faces[0])
            landmarks.append(np.asarray([(p.x, p.y) for p in shape.parts()]))
        stable = [33, 36, 39, 42, 45]
        mouths = []
        for i, (frame, points) in enumerate(zip(frames, landmarks)):
            if i + 12 <= len(frames):
                smoothed = np.mean(landmarks[i : i + 12], axis=0)
                warped, transform = self.warp.warp_img(
                    smoothed[stable], self.meanface[stable], frame, (256, 256)
                )
            else:
                warped = self.warp.apply_transform(transform, frame, (256, 256))
            try:
                patch = self.warp.cut_patch(warped, transform(points)[48:68], 48, 48)
            except Exception as exc:
                if "too much bias" not in str(exc):
                    raise
                return np.nan, "mouth_outside_warp"
            if patch.shape[:2] != (96, 96):
                return np.nan, "incomplete_mouth_crop"
            mouths.append(np.asarray(Image.fromarray(patch).convert("L"))[4:92, 4:92])
        clips = (
            torch.from_numpy(np.stack(mouths))
            .reshape(-1, 1, 25, 88, 88)
            .to(self.device)
            .float()
        )
        clips = (clips / 255 - 0.421) / 0.165
        logits = self.model(clips, lengths=[25] * clips.shape[0])
        return torch.sigmoid(logits.float().mean()).item(), ""


def run_external(cfg, panel, spec, seed, run_dir):
    import torch

    root = external_assets(spec)
    device = torch.device(cfg["training"].get("device", "cuda"))
    implementation = GenD if spec["backbone"] == "gend" else LipForensics
    model = implementation(spec, root, device)
    rows = panel.loc[panel.split != "train"]
    run_dir.mkdir(parents=True, exist_ok=True)
    # Runner holds exclusive per-run lock and binds this journal to panel, code, model and config.
    with (
        contextlib.closing(sqlite3.connect(run_dir / "scoring.sqlite")) as db,
        torch.inference_mode(),
    ):
        db.execute(
            "CREATE TABLE IF NOT EXISTS scores (recording TEXT, route TEXT, score REAL, failure TEXT, PRIMARY KEY(recording, route))"
        )
        seen = set(db.execute("SELECT recording,route FROM scores"))
        for i, row in enumerate(rows.itertuples(index=False), 1):
            if (row.recording_id, row.route) in seen:
                continue
            score, failure = model.score(row)
            if not failure and not np.isfinite(score):
                raise ValueError(
                    f"Nonfinite model score: {row.recording_id}/{row.route}"
                )
            db.execute(
                "INSERT INTO scores VALUES(?,?,?,?)",
                (row.recording_id, row.route, None if failure else score, failure),
            )
            if i % 25 == 0:
                db.commit()
            if i % 100 == 0:
                print(f"{spec['name']}: {i}/{len(rows)} scored", flush=True)
        db.commit()
        df = pd.read_sql_query(
            "SELECT recording AS recording_id,route,score,failure FROM scores ORDER BY recording,route",
            db,
        )
    path = run_dir / "predictions.csv"
    atomic_text(path, df.to_csv(index=False))
    return path
