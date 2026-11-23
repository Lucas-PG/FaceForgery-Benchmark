"""Stage public reference assets and bind the lab's archived checkpoints to content hashes."""

import argparse
import subprocess
from pathlib import Path

from .external import GEND_REV, LIP_REV, RELEASE_HASHES, check_repo
from .initial import verify_initial
from .io import sha, write_json
from .models import HISTORICAL_REPO, HISTORICAL_REVISION, validate_assets

GEND_WEIGHTS = "891ce014a0308386c4d7d25b3dcf436a22db5504"
CLIP_L = "32bd64288804d66eefd0ccbe215aa642df71cc41"


def checkout(root, name, remote, rev):
    path = root / name
    if not path.exists():
        subprocess.run(["git", "clone", "--no-checkout", remote, path], check=True)
        subprocess.run(["git", "-C", path, "checkout", "--detach", rev], check=True)
    return str(check_repo(path, rev))


def bind(item, key, file, checksum_file=None):
    p = Path(file).resolve(strict=True)
    item[key] = str(p)
    item[key + "_sha256"] = sha(checksum_file if checksum_file else p)


def staged_file(repo, name, revision, directory):
    existing = directory / name
    if existing.is_file():
        return existing
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo, name, revision=revision, local_dir=directory))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in (
        "directory",
        "dino-initial",
        "clip-initial",
        "lip-weights",
        "landmarks",
    ):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument(
        "--dino-tcc",
        type=Path,
        help="Optional exact local copy of the released detector",
    )
    p.add_argument(
        "--clip-tcc",
        type=Path,
        help="Optional exact local copy of the released detector",
    )
    args = p.parse_args()
    root = args.directory.resolve()
    root.mkdir(parents=True, exist_ok=True)
    registry = {}
    for name, key, file, checksum in [
        ("dino-init", "weights", args.dino_initial, None),
        (
            "clip-init",
            "weights",
            args.clip_initial,
            args.clip_initial / "model.safetensors",
        ),
    ]:
        registry[name] = {}
        bind(registry[name], key, file, checksum)
        verify_initial(name.split("-")[0], file)
    for family, file in (("dino", args.dino_tcc), ("clip", args.clip_tcc)):
        if file is None:
            subdir = f"{family}/none/finetune_robust/seed_42"

            file, run_config = (
                staged_file(
                    HISTORICAL_REPO,
                    f"{subdir}/{suffix}",
                    HISTORICAL_REVISION,
                    root / "mffi-models",
                )
                for suffix in ("weights/best.pth", "results/run_config.json")
            )
        else:
            run_config = file.resolve().parents[1] / "results" / "run_config.json"
        entry = registry[family + "-tcc"] = {}
        bind(entry, "checkpoint", file)
        bind(entry, "run_config", run_config)
        validate_assets(
            dict(entry, backbone=family, objective="frozen", positive_index=1)
        )
    registry["clip-init"]["clip_config_sha256"] = sha(args.clip_initial / "config.json")
    gen = registry["gend"] = {
        "upstream": checkout(
            root, "GenD", "https://github.com/yermandy/GenD.git", GEND_REV
        )
    }
    for key, repo, rev, name, folder in (
        (
            "weights",
            "yermandy/GenD_CLIP_L_14",
            GEND_WEIGHTS,
            "model.safetensors",
            "detector",
        ),
        (
            "detector_config",
            "yermandy/GenD_CLIP_L_14",
            GEND_WEIGHTS,
            "config.json",
            "detector",
        ),
        (
            "backbone_config",
            "openai/clip-vit-large-patch14",
            CLIP_L,
            "config.json",
            "clip-large",
        ),
        (
            "processor",
            "openai/clip-vit-large-patch14",
            CLIP_L,
            "preprocessor_config.json",
            "clip-large",
        ),
    ):
        bind(gen, key, staged_file(repo, name, rev, root / folder))
    lip = registry["lip"] = {
        "upstream": checkout(
            root,
            "LipForensics",
            "https://github.com/ahaliassos/LipForensics.git",
            LIP_REV,
        )
    }
    bind(lip, "weights", args.lip_weights)
    bind(lip, "landmarks", args.landmarks)
    for kind, spec in (("gend", gen), ("lipforensics", lip)):
        for key, expected in RELEASE_HASHES[kind].items():
            if spec[key + "_sha256"] != expected:
                raise ValueError(f"Wrong public release for {kind}/{key}")
    write_json(root / "assets.json", registry)
    print(root / "assets.json")


if __name__ == "__main__":
    main()
