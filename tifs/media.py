"""Content-addressed generation of synchronized, losslessly cached RGB frame stacks."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
import tempfile
import threading
from datetime import UTC
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def key(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False, indent=2)
        stream.write("\n")
        tmp = Path(stream.name)
    os.replace(tmp, path)


def _cache_record(path):
    try:
        item = json.loads(path.read_text())
        if isinstance(item, dict) and item.get("payload_sha256") == key(
            item.get("payload")
        ):
            return item["payload"]
    except (OSError, ValueError, TypeError):
        pass
    return {}


def _cache_save(path, item):
    atomic_json(path, {"payload": item, "payload_sha256": key(item)})


def _run(args: list[str]) -> bytes:
    p = subprocess.run(args, capture_output=True, check=False)
    if p.returncode:
        raise ValueError(
            f"{Path(args[0]).name} failed: {p.stderr.decode(errors='replace')[-1200:]}"
        )
    return p.stdout


@lru_cache(maxsize=1)
def media_stack() -> dict:
    import PIL

    versions = {}
    for program in ("ffmpeg", "ffprobe"):
        versions[program] = _run([program, "-version"]).decode().splitlines()[0]
    return {
        **versions,
        "numpy": np.__version__,
        "Pillow": PIL.__version__,
        "implementation": digest(Path(__file__)),
    }


def probe(path: Path) -> dict:
    info = json.loads(
        _run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(path),
            ]
        )
    )
    if not info["streams"]:
        raise ValueError("no video stream")
    stream = info["streams"][0]
    duration = stream.get("duration")
    if duration in (None, "N/A"):
        duration = info["format"].get("duration", 0)
    duration = float(duration)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("video needs a finite positive presentation duration")
    width, height = int(stream["width"]), int(stream["height"])
    sar = stream.get("sample_aspect_ratio", "1:1")
    if sar not in ("N/A", "0:1", None):
        num, den = map(int, sar.split(":"))
        width = max(1, round(width * num / den))
    rotation = next(
        (
            abs(int(i["rotation"])) % 360
            for i in stream.get("side_data_list", [])
            if "rotation" in i
        ),
        abs(int(stream.get("tags", {}).get("rotate", 0))) % 360,
    )
    if rotation in (90, 270):
        width, height = height, width
    if min(width, height) < 1 or rotation not in (0, 90, 180, 270):
        raise ValueError("unsupported presentation geometry")
    return {
        "width": width,
        "height": height,
        "duration": duration,
        "codec": stream["codec_name"],
    }


def encode(source: Path, source_hash: str, rule: dict, root: Path) -> Path:
    """Produce a reproducible *simulated* route. It carries no transmission receipt."""
    allowed = {"kind", "crf", "preset", "height"}
    if set(rule) - allowed:
        raise ValueError(f"unknown encode settings: {sorted(set(rule) - allowed)}")
    crf, height = int(rule.get("crf", 28)), int(rule.get("height", 0))
    if not 0 <= crf <= 51 or height < 0 or height % 2:
        raise ValueError("H.264 needs crf 0..51 and zero or positive even height")
    preset = str(rule.get("preset", "medium"))
    if preset not in {
        "ultrafast",
        "superfast",
        "veryfast",
        "faster",
        "fast",
        "medium",
        "slow",
        "slower",
        "veryslow",
    }:
        raise ValueError("invalid x264 preset")
    dest = root / (key([source_hash, rule, media_stack()]) + ".mp4")
    receipt = dest.with_suffix(".json")
    if (
        dest.exists()
        and receipt.exists()
        and _cache_record(receipt).get("sha256") == digest(dest)
    ):
        return dest
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as tmp:
        target = Path(tmp) / "encoded.mp4"
        vf = f"scale=-2:{height}" if height else "pad=ceil(iw/2)*2:ceil(ih/2)*2"
        _run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-threads",
                "1",
                "-i",
                str(source),
                "-map",
                "0:v:0",
                "-an",
                "-filter_threads",
                "1",
                "-vf",
                vf,
                "-c:v",
                "libx264",
                "-threads",
                "1",
                "-preset",
                preset,
                "-crf",
                str(crf),
                "-pix_fmt",
                "yuv420p",
                "-map_metadata",
                "-1",
                str(target),
            ]
        )
        output_hash = digest(target)
        os.replace(target, dest)
        _cache_save(
            receipt, {"sha256": output_hash, "input_sha256": source_hash, "rule": rule}
        )
    return dest


def _track(path: Path, times: np.ndarray) -> np.ndarray:
    fields = ["time_s", "x0", "y0", "x1", "y1"]
    with path.open(newline="") as stream:
        points = np.asarray(
            [[float(row[f]) for f in fields] for row in csv.DictReader(stream)]
        )
    if points.ndim != 2 or not len(points) or not np.isfinite(points).all():
        raise ValueError("empty or nonfinite face track")
    if (
        np.any(np.diff(points[:, 0]) <= 0)
        or points[0, 0] > min(times) + 1e-8
        or points[-1, 0] < max(times) - 1e-8
    ):
        raise ValueError(
            "track must cover the entire sampled interval without duplicate timestamps"
        )
    boxes = points[:, 1:]
    if np.any(boxes < 0) or np.any(boxes > 1) or np.any(boxes[:, 2:] <= boxes[:, :2]):
        raise ValueError("face track requires normalized x0,y0,x1,y1")
    return np.column_stack([np.interp(times, points[:, 0], col) for col in boxes.T])


_detectors = threading.local()


def _detector():
    import cv2

    if not hasattr(_detectors, "model"):
        # Panel parallelism is across recordings. Avoid nested OpenCV thread pools.
        cv2.setNumThreads(1)
        detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        if detector.empty():
            raise ValueError("OpenCV frontal face cascade unavailable")
        _detectors.model = detector
    return cv2, _detectors.model


def frames(
    path: Path,
    video_hash: str,
    duration: float,
    offset: float,
    cfg: dict,
    track: Path | None,
    root: Path,
) -> dict:
    """Sample the same source-time bins on every version. Decode once per cached view."""
    n, size, mode = cfg["frames"], cfg["size"], cfg.get("crop", "full")
    cap = cfg.get("decode_max_side", 1024)
    if (
        type(n) is not int
        or type(size) is not int
        or type(cap) is not int
        or not (n >= 1 and cap >= size >= 8)
    ):
        raise ValueError("frames must be positive, decode_max_side >= size >= 8")
    if mode not in {"full", "fixed_track", "view_track", "detected"}:
        raise ValueError("unknown crop mode")
    if mode in {"fixed_track", "view_track"} and track is None:
        raise ValueError(f"{mode} requires a track file")
    face_failure = cfg.get("on_face_failure", "full")
    if face_failure not in {"full", "error"}:
        raise ValueError("on_face_failure must be full or error")
    tools = media_stack().copy()
    if mode == "detected":
        import cv2

        tools["opencv"] = cv2.__version__
        tools["cascade"] = digest(
            Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        )
    track_hash = digest(track) if track is not None else ""
    cache = key(
        [
            video_hash,
            duration,
            offset,
            n,
            size,
            cap,
            mode,
            face_failure,
            track_hash,
            tools,
        ]
    )
    dest = root / cache[:2] / (cache + ".npy")
    meta = dest.with_suffix(".json")
    if dest.exists() and meta.exists():
        info = _cache_record(meta)
        if digest(dest) == info.get("frame_sha256"):
            return {**info, "frames_path": str(dest.resolve())}
    geom = probe(path)
    # FPS chooses the frame covering the center of each equal-width time bin.
    # A timestamp offset maps source-relative time to the delivered presentation.
    if (
        not math.isfinite(offset)
        or offset < 0
        or geom["duration"] + 0.001 < offset + duration
    ):
        raise ValueError("received video does not cover the registered source interval")
    scale = min(1.0, cap / max(geom["width"], geom["height"]))
    w, h = max(1, round(geom["width"] * scale)), max(1, round(geom["height"] * scale))
    fps = n / duration
    vf = (
        # FPS rounds an input PTS exactly on a half-bin into the next bin.
        # Shift one microsecond so a frame beginning at the requested center is included.
        f"settb=expr=1/1000000,setpts=PTS-STARTPTS-{offset:.9f}/TB-1,"
        f"fps=fps={fps:.12f}:start_time=0:round=near,scale={w}:{h}:flags=bicubic,setsar=1"
    )
    raw = _run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-threads",
            "1",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-an",
            "-filter_threads",
            "1",
            "-vf",
            vf,
            "-frames:v",
            str(n),
            "-pix_fmt",
            "rgb24",
            "-threads",
            "1",
            "-f",
            "rawvideo",
            "-",
        ]
    )
    if len(raw) != n * h * w * 3:
        raise ValueError(
            f"expected {n} complete frames, decoded {len(raw) / (3 * h * w):g}"
        )
    images = np.frombuffer(raw, dtype=np.uint8).reshape(n, h, w, 3)
    times = (np.arange(n) + 0.5) / fps
    if mode == "view_track":
        times += offset
    elif mode == "fixed_track":
        times += float(cfg.get("window_start_seconds", 0))
    boxes = _track(track, times) if track is not None else None
    detector = _detector() if mode == "detected" else None
    result = np.empty((n, size, size, 3), dtype=np.uint8)
    fallback = []
    face_area = []
    for i, frame in enumerate(images):
        image = Image.fromarray(frame)
        if boxes is not None:
            box = np.rint(boxes[i] * [w, h, w, h]).astype(int)
            if np.any(box[2:] <= box[:2]):
                raise ValueError("face box smaller than one decoded pixel")
            face_area.append((box[2] - box[0]) * (box[3] - box[1]) / (w * h))
            image = image.crop(tuple(box))
        if detector is not None:
            cv2, model = detector
            found = model.detectMultiScale(
                cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), 1.1, 5
            )
            if len(found) != 1:
                if face_failure == "error":
                    raise ValueError(
                        f"expected one detected face at sample {i}, found {len(found)}"
                    )
                fallback.append(i)
            else:
                x, y, bw, bh = found[0]
                face_area.append(float(bw * bh) / (w * h))
                image = image.crop((x, y, x + bw, y + bh))
        result[i] = np.asarray(
            ImageOps.pad(image, (size, size), method=Image.Resampling.BICUBIC)
        )
    # One preregistered no-reference visual-quality predictor on the received pixels.
    gray = np.mean(result.astype(np.float32), axis=3)
    quality = float(
        np.log1p(
            np.mean(np.diff(gray, axis=1) ** 2) + np.mean(np.diff(gray, axis=2) ** 2)
        )
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as f:
        np.save(f, result, allow_pickle=False)
        tmp = Path(f.name)
    hashed = digest(tmp)
    os.replace(tmp, dest)
    info = {
        "frame_sha256": hashed,
        "quality": quality,
        "duration": duration,
        "offset_s": offset,
        "track_sha256": track_hash,
        "face_area_fraction": float(np.median(face_area)) if face_area else "",
        "face_observed_frames": len(face_area),
        "crop_fallback_frames": len(fallback),
        "crop_fallback_indices": ",".join(map(str, fallback)),
    }
    _cache_save(meta, info)
    return {**info, "frames_path": str(dest.resolve())}


def _host_port(host: str, port: int) -> str:
    # Numeric addresses only; no arbitrary FFmpeg protocol options in network endpoints.
    import ipaddress

    ip = ipaddress.ip_address(host)
    if not 1024 <= port <= 65535:
        raise ValueError("port must be 1024..65535")
    return f"[{ip}]:{port}" if ip.version == 6 else f"{ip}:{port}"


def _capture(argv=None):
    """Controlled H.264/TCP handoff. Every accepted capture must carry the source digest in-band."""
    import argparse
    import socket
    from datetime import datetime

    from .panel import MEASURED, SOURCE, read_csv, validate_sources, write_csv

    p = argparse.ArgumentParser(description=_capture.__doc__)
    p.add_argument("mode", choices=["send", "receive"])
    p.add_argument("--sources", type=Path, required=True)
    p.add_argument("--route", required=True)
    p.add_argument("--session", required=True)
    p.add_argument(
        "--host", required=True, help="sender: receiver IP; receiver: bind IP"
    )
    p.add_argument("--port", type=int, default=29418)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--splits", nargs="+", required=True)
    p.add_argument(
        "--timeout", type=int, default=120, help="seconds without incoming TCP bytes"
    )
    p.add_argument("--crf", type=int, default=26)
    opt = p.parse_args(argv)
    if (
        not opt.route
        or opt.route == "native"
        or not opt.session
        or not 0 <= opt.crf <= 51
        or opt.timeout <= 0
    ):
        p.error("invalid capture parameters")
    rows = read_csv(opt.sources, SOURCE)
    validate_sources(rows)
    selected = [r for r in rows if r["split"] in opt.splits]
    if not selected:
        p.error("no enrolled clips in these splits")
    opt.out.mkdir(parents=True, exist_ok=True)
    endpoint = _host_port(opt.host, opt.port)
    listener = None
    if opt.mode == "receive":
        listener = socket.create_server(
            (opt.host, opt.port),
            family=socket.AF_INET6 if ":" in opt.host else socket.AF_INET,
        )
        listener.settimeout(opt.timeout)
    manifest_hash = digest(opt.sources)
    inventory = []
    try:
        for row in selected:
            source = (opt.sources.parent / row["source_path"]).resolve()
            rid = row["recording_id"]
            src_hash = digest(source)
            tag = key([rid, opt.route, opt.session, src_hash, opt.crf])
            target = opt.out / (tag + ".ts")
            receipt = opt.out / (tag + ".json")
            if opt.mode == "send":
                with socket.create_connection(
                    (opt.host, opt.port), timeout=opt.timeout
                ) as sock:
                    sock.settimeout(opt.timeout)
                    sock.sendall(tag.encode("ascii"))
                    response = sock.recv(1)
                    if response == b"S":
                        continue
                    if response != b"G":
                        raise ValueError(f"receiver rejected {rid}")
                    cmd = [
                        "ffmpeg",
                        "-nostdin",
                        "-v",
                        "error",
                        "-re",
                        "-threads",
                        "1",
                        "-i",
                        str(source),
                        "-map",
                        "0:v:0",
                        "-an",
                        "-filter_threads",
                        "1",
                        "-vf",
                        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                        "-c:v",
                        "libx264",
                        "-threads",
                        "1",
                        "-preset",
                        "veryfast",
                        "-crf",
                        str(opt.crf),
                        "-pix_fmt",
                        "yuv420p",
                        "-metadata",
                        "service_name=" + tag,
                        "-muxdelay",
                        "0",
                        "-f",
                        "mpegts",
                        "pipe:1",
                    ]
                    with subprocess.Popen(cmd, stdout=subprocess.PIPE) as proc:
                        try:
                            for block in iter(lambda: proc.stdout.read(65536), b""):
                                sock.sendall(block)
                        except BaseException:
                            proc.kill()
                            raise
                        if proc.wait() != 0:
                            raise ValueError(f"transmitter failed for {rid}")
                    sock.shutdown(socket.SHUT_WR)
                    if sock.recv(1) != b"O":
                        raise ValueError(f"receiver did not verify {rid}")
                continue
            # Fixed registered run order and idempotent handshake avoid mis-pairing or unsafe path inputs.
            with listener.accept()[0] as peer:
                peer.settimeout(opt.timeout)
                header = b""
                while len(header) < 64:
                    more = peer.recv(64 - len(header))
                    if not more:
                        break
                    header += more
                if header != tag.encode("ascii"):
                    raise ValueError(
                        f"endpoints disagree on input, clip, route, session, order or codec: {rid}"
                    )
                valid = (
                    target.exists()
                    and receipt.exists()
                    and json.loads(receipt.read_text()).get("received_sha256")
                    == digest(target)
                )
                if valid:
                    peer.sendall(b"S")
                else:
                    peer.sendall(b"G")
                    with tempfile.TemporaryDirectory(dir=opt.out) as tmp:
                        ts = Path(tmp) / "received.ts"
                        total, limit = 0, max(32 * 2**20, source.stat().st_size * 32)
                        with ts.open("wb") as stream:
                            while block := peer.recv(65536):
                                total += len(block)
                                if total > limit:
                                    raise ValueError(
                                        "received transport exceeds registered size limit"
                                    )
                                stream.write(block)
                        programs = json.loads(
                            _run(
                                [
                                    "ffprobe",
                                    "-v",
                                    "error",
                                    "-show_programs",
                                    "-of",
                                    "json",
                                    str(ts),
                                ]
                            )
                        )["programs"]
                        if (
                            len(programs) != 1
                            or programs[0].get("tags", {}).get("service_name") != tag
                        ):
                            raise ValueError("in-band acquisition tag mismatch")
                        probe(ts)
                        received = digest(ts)
                        os.replace(ts, target)
                        note = {
                            "recording_id": rid,
                            "route": opt.route,
                            "source_sha256": src_hash,
                            "received_sha256": received,
                            "session": opt.session,
                            "method": "network_capture",
                            "acquired_utc": datetime.now(UTC).isoformat(),
                            "channel": "controlled H.264 MPEG-TS over TCP",
                            "settings": {
                                "codec": "libx264",
                                "preset": "veryfast",
                                "crf": opt.crf,
                                "transport": "tcp:mpegts",
                                "pix_fmt": "yuv420p",
                            },
                            "alignment": "first decoded content timestamp on both streams",
                            "offset_seconds": 0,
                            "geometry_preserved": True,
                            "endpoint": endpoint,
                            "crf": opt.crf,
                            "source_manifest_sha256": manifest_hash,
                            "decoder": media_stack(),
                        }
                        atomic_json(receipt, note)
                        peer.sendall(b"O")
            inventory.append(
                {
                    "recording_id": rid,
                    "route": opt.route,
                    "path": str(target.resolve()),
                    "receipt_path": str(receipt.resolve()),
                }
            )
    finally:
        if listener is not None:
            listener.close()
    if opt.mode == "receive":
        write_csv(opt.out / "renditions.csv", inventory, MEASURED)


if __name__ == "__main__":
    _capture()
