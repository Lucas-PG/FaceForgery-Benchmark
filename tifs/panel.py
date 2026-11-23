"""Enroll recording lineages and materialize a complete crossed panel of received videos."""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import media

SPLITS = {"train", "dev", "cal", "diagnostic", "test"}
SOURCE = (
    "recording_id",
    "label",
    "cohort",
    "family",
    "cluster_id",
    "split",
    "source_path",
    "source_ids",
    "identity_ids",
    "session_ids",
)
MEASURED = ("recording_id", "route", "path", "receipt_path")
FIELDS = (
    *SOURCE,
    "matched_genuine_id",
    "route",
    "kind",
    "media_path",
    "sha256",
    "source_sha256",
    "receipt_path",
    "receipt_sha256",
    "receipt_size",
    "receipt_mtime_ns",
    "receipt_ctime_ns",
    "offset_seconds",
    "frames_path",
    "frame_sha256",
    "frames_size",
    "frames_mtime_ns",
    "frames_ctime_ns",
    "media_size",
    "media_mtime_ns",
    "media_ctime_ns",
    "track_path",
    "track_sha256",
    "track_size",
    "track_mtime_ns",
    "track_ctime_ns",
    "quality",
    "duration",
    "native_duration_seconds",
    "crop_fallback_frames",
    "crop_fallback_indices",
    "window_start_seconds",
    "crop_reference",
    "track_reviewer",
    "face_area_fraction",
    "face_observed_frames",
    "capture_method",
    "capture_channel",
    "capture_session",
    "capture_settings",
    "acquired_utc",
)
MISSING = {"", "unknown", "na", "n/a", "none", "null", "?"}


def read_csv(path: Path, required: tuple) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        names = reader.fieldnames or []
        if set(required) - set(names) or len(names) != len(set(names)):
            raise ValueError(f"{path}: expected unique headers including {required}")
        rows = list(reader)
    if any(None in row or None in row.values() for row in rows):
        raise ValueError(f"{path}: ragged row")
    return rows


def write_csv(path: Path, rows, fields) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, newline="", delete=False
    ) as stream:
        w = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
        tmp = Path(stream.name)
    os.replace(tmp, path)


def tokens(text: str) -> list[str]:
    if not text:
        return []
    parts = text.split("|")
    if any(t.strip() != t or t.lower() in MISSING for t in parts) or len(
        set(parts)
    ) != len(parts):
        raise ValueError(f"invalid lineage tokens {text!r}")
    return parts


def validate_sources(rows: list[dict], required_ids=()) -> None:
    identifiers, declared, ownership, cohorts = set(), {}, {}, {}
    for row in rows:
        rid = row["recording_id"]
        mandatory = ("recording_id", "cohort", "family", "cluster_id", "source_path")
        if any(
            row[k].strip() != row[k] or row[k].lower() in MISSING for k in mandatory
        ):
            raise ValueError(f"blank, padded or unknown recording metadata: {rid}")
        if rid in identifiers:
            raise ValueError(f"duplicate recording_id {rid}")
        identifiers.add(rid)
        if row["label"] not in {"0", "1"} or row["split"] not in SPLITS:
            raise ValueError(f"{rid}: expected binary label and registered split")
        if (
            row["family"].lower() in {"real", "genuine", "authentic"}
            and row["label"] != "0"
        ):
            raise ValueError(f"{rid}: forged recording labeled as genuine family")
        if row["label"] == "0" and row["family"].lower() not in {
            "real",
            "genuine",
            "authentic",
        }:
            raise ValueError(
                f"{rid}: genuine recording needs real/genuine/authentic family"
            )
        cluster = row["cluster_id"]
        if declared.setdefault(cluster, row["split"]) != row["split"]:
            raise ValueError(f"cluster crosses splits: {cluster}")
        if cohorts.setdefault(cluster, row["cohort"]) != row["cohort"]:
            raise ValueError(f"cluster crosses cohorts: {cluster}")
        for k in ("source_ids", "identity_ids", "session_ids"):
            values = tokens(row[k])
            if not values and (
                k == "source_ids"
                or (k == "identity_ids" and row["split"] in required_ids)
            ):
                raise ValueError(f"{rid}: missing {k}")
            for val in values:
                if ownership.setdefault((k, val), cluster) != cluster:
                    raise ValueError(f"shared {k} {val} assigned to different clusters")
    catalog = {r["recording_id"]: r for r in rows}
    for row in rows:
        target = row.get("matched_genuine_id", "")
        if target:
            parent = catalog.get(target)
            if (
                row["label"] != "1"
                or not parent
                or parent["label"] != "0"
                or parent["cluster_id"] != row["cluster_id"]
                or not set(tokens(row["source_ids"]))
                & set(tokens(parent["source_ids"]))
            ):
                raise ValueError(
                    f"invalid authentic parent pointer: {row['recording_id']} -> {target}"
                )
    if not rows:
        raise ValueError("source inventory empty")


def _file_stat(path: Path | None, prefix: str) -> dict:
    st = path.stat() if path is not None else None
    return {
        prefix + "_size": st.st_size if st else 0,
        prefix + "_mtime_ns": st.st_mtime_ns if st else 0,
        prefix + "_ctime_ns": st.st_ctime_ns if st else 0,
    }


def verify_material(panel, *, full_hash=False) -> None:
    """Verify model inputs, acquisition receipts and source/received tracking annotations.

    A prepared file with unchanged size/mtime/ctime uses a cheap stat check. Relocated
    or touched files need the original SHA-256. full_hash rechecks every byte.
    """
    seen = {}
    for columns in (
        ("frames_path", "frame_sha256", "frames"),
        ("media_path", "sha256", "media"),
        ("receipt_path", "receipt_sha256", "receipt"),
        ("track_path", "track_sha256", "track"),
    ):
        field, sha, prefix = columns
        keys = [
            field,
            sha,
            prefix + "_size",
            prefix + "_mtime_ns",
            prefix + "_ctime_ns",
        ]
        for raw, expected, size, mtime, ctime in panel[keys].itertuples(
            index=False, name=None
        ):
            if prefix in {"receipt", "track"} and (not isinstance(raw, str) or not raw):
                if isinstance(expected, str) and expected:
                    raise ValueError(f"missing {prefix} path for registered digest")
                continue
            path = Path(raw)
            if path in seen:
                if seen[path] != expected:
                    raise ValueError(f"conflicting material digests: {path}")
                continue
            seen[path] = expected
            st = path.stat()
            changed = (st.st_size, st.st_mtime_ns, st.st_ctime_ns) != (
                int(size),
                int(mtime),
                int(ctime),
            )
            if (full_hash or changed) and media.digest(path) != expected:
                raise ValueError(
                    f"material no longer matches the registered panel: {path}"
                )


def _resolve(base: Path, raw: str) -> Path:
    return (base / raw).resolve()


class ProvenanceError(ValueError):
    """An inconsistent acquisition cannot be reclassified as missing data."""


def _receipt(
    path: Path, rid: str, route: str, source_hash: str, received_hash: str
) -> dict:
    with path.open() as stream:
        note = json.load(stream)
    expected = {
        "recording_id": rid,
        "route": route,
        "source_sha256": source_hash,
        "received_sha256": received_hash,
    }
    if not isinstance(note, dict) or any(note.get(k) != v for k, v in expected.items()):
        raise ValueError(f"receipt identity or media hash mismatch: {rid}/{route}")
    for name in ("session", "method", "acquired_utc", "channel", "alignment"):
        if not isinstance(note.get(name), str) or note[name].strip().lower() in MISSING:
            raise ValueError(f"missing acquisition {name}: {rid}/{route}")
    from datetime import datetime

    if datetime.fromisoformat(note["acquired_utc"]).tzinfo is None:
        raise ValueError("acquisition date must include a UTC offset")
    if note["method"] not in {
        "platform_download",
        "network_capture",
        "optical_recapture",
        "archive",
    }:
        raise ValueError("unrecognized measured acquisition method")
    if note["method"] == "archive" and not note.get("archive_locator"):
        raise ValueError("archived rendition needs an archive_locator")
    if note.get("time_scale", 1) != 1:
        raise ValueError(
            "time-varying alignment requires ingesting a synchronized registered excerpt"
        )
    offset = float(note["offset_seconds"])
    if not math.isfinite(offset) or offset < 0:
        raise ValueError("invalid receipt offset")
    return note


def _screen_duplicates(views, out: Path, clearances=None) -> int:
    """Conservative perceptual screen on three normalized native timestamps, across declared groups."""
    import numpy as np
    from PIL import Image

    approved = {}
    if clearances:
        for r in read_csv(
            Path(clearances), ("left_sha256", "right_sha256", "reviewer", "reason")
        ):
            if not r["reviewer"].strip() or not r["reason"].strip():
                raise ValueError(
                    "a duplicate clearance needs an adjudicator and an explanation"
                )
            approved[tuple(sorted((r["left_sha256"], r["right_sha256"])))] = r
    signatures, buckets, flagged = [], {}, []
    for row in views:
        if row["route"] != "native":
            continue
        stack = np.load(row["frames_path"], mmap_mode="r", allow_pickle=False)
        sample = [
            Image.fromarray(stack[j]) for j in (0, len(stack) // 2, len(stack) - 1)
        ]
        thumb = np.stack(
            [
                np.asarray(im.resize((8, 8), Image.Resampling.BOX), dtype=np.int16)
                for im in sample
            ]
        )
        gray = np.asarray(sample[1].convert("L").resize((9, 8), Image.Resampling.BOX))
        bits = int.from_bytes(np.packbits(gray[:, 1:] > gray[:, :-1]).tobytes(), "big")
        # If <= 3 bits differ, at least one of four disjoint 16-bit bands matches.
        band_keys = [(k, (bits >> (16 * k)) & 65535) for k in range(4)]
        candidates = set().union(*(buckets.get(k, ()) for k in band_keys))
        for idx in candidates:
            other, othbits, oththumb = signatures[idx]
            if (
                other["cluster_id"] == row["cluster_id"]
                or (bits ^ othbits).bit_count() > 3
            ):
                continue
            distance = float(np.abs(thumb - oththumb).mean())
            if distance <= 2.0:
                pair = tuple(sorted((other["source_sha256"], row["source_sha256"])))
                clearance = approved.get(pair, {})
                flagged.append(
                    {
                        "left_recording": other["recording_id"],
                        "right_recording": row["recording_id"],
                        "left_sha256": other["source_sha256"],
                        "right_sha256": row["source_sha256"],
                        "mean_abs_rgb": distance,
                        "reviewer": clearance.get("reviewer", ""),
                        "reason": clearance.get("reason", ""),
                    }
                )
        idx = len(signatures)
        signatures.append((row, bits, thumb))
        for k in band_keys:
            buckets.setdefault(k, []).append(idx)
    fields = (
        "left_recording",
        "right_recording",
        "left_sha256",
        "right_sha256",
        "mean_abs_rgb",
        "reviewer",
        "reason",
    )
    write_csv(out / "near_duplicates.csv", flagged, fields)
    return sum(not f["reviewer"] for f in flagged)


def _screen_alignment(views, out, clearances=None):
    """Flag gross within-recording content mismatches, even when an erroneous receipt is internally consistent."""
    import numpy as np
    from PIL import Image

    approvals = {}
    if clearances:
        for r in read_csv(
            Path(clearances), ("source_sha256", "received_sha256", "reviewer", "reason")
        ):
            if not r["reviewer"].strip() or not r["reason"].strip():
                raise ValueError("alignment clearance requires a reviewer and reason")
            approvals[(r["source_sha256"], r["received_sha256"])] = r

    def thumbnail(row):
        stack = np.load(row["frames_path"], mmap_mode="r", allow_pickle=False)
        return (
            np.stack(
                [
                    np.asarray(
                        Image.fromarray(stack[j]).resize(
                            (16, 16), Image.Resampling.BOX
                        ),
                        dtype=float,
                    )
                    for j in [0, len(stack) // 2, len(stack) - 1]
                ]
            )
            / 255
        )

    natives = {r["recording_id"]: r for r in views if r["kind"] == "native"}
    results = []
    for r in views:
        if r["kind"] == "native":
            continue
        v = thumbnail(r)
        ref = thumbnail(natives[r["recording_id"]])
        distance = np.abs(ref - v).mean()
        rr, vv = ref - ref.mean(), v - v.mean()
        denom = np.linalg.norm(rr) * np.linalg.norm(vv)
        corr = float((rr * vv).sum() / denom) if denom > 1e-6 else 0.0
        suspicious = distance > 0.18 and corr < 0.25
        approval = approvals.get((r["source_sha256"], r["sha256"]), {})
        results.append(
            {
                "recording_id": r["recording_id"],
                "route": r["route"],
                "source_sha256": r["source_sha256"],
                "received_sha256": r["sha256"],
                "mean_abs_rgb": distance,
                "correlation": corr,
                "suspicious": int(suspicious),
                "reviewer": approval.get("reviewer", ""),
                "reason": approval.get("reason", ""),
            }
        )
    write_csv(
        out / "alignment_audit.csv",
        results,
        (
            "recording_id",
            "route",
            "source_sha256",
            "received_sha256",
            "mean_abs_rgb",
            "correlation",
            "suspicious",
            "reviewer",
            "reason",
        ),
    )
    return sum(r["suspicious"] and not r["reviewer"] for r in results)


def delivery_coverage(views, out, *, confirmatory):
    """Detect acquisition confounding before inference. Each receiver session must see both tested classes."""
    cells = {}
    labels = {}
    fields = (
        "cohort",
        "split",
        "route",
        "capture_method",
        "capture_channel",
        "capture_session",
        "capture_settings",
    )
    for r in views:
        if r["kind"] != "measured":
            continue
        key = tuple(r[c] for c in fields)
        cells.setdefault(key, {"0": set(), "1": set()})[r["label"]].add(
            r["recording_id"]
        )
        labels.setdefault((r["cohort"], r["split"]), set()).add(r["label"])
    rows = []
    profile_routes, route_profiles = {}, {}
    for key, classes in sorted(cells.items()):
        r = dict(zip(fields, key))
        profile = tuple(
            r[k].strip().casefold()
            for k in ("capture_method", "capture_channel", "capture_settings")
        )
        profile_routes.setdefault(profile, set()).add(r["route"])
        route_profiles.setdefault(r["route"], set()).add(profile)
        # A diagnostic fold enrolled exclusively for authentic quantiles has no fake target.
        both_required = r["split"] == "test" or labels[(r["cohort"], r["split"])] == {
            "0",
            "1",
        }
        counts = {k: len(v) for k, v in classes.items()}
        imbalance = both_required and min(counts.values()) == 0
        rows.append(
            dict(
                **r,
                genuine=counts["0"],
                fake=counts["1"],
                class_confounded=int(imbalance),
            )
        )
    write_csv(
        out / "delivery_coverage.csv",
        rows,
        (*fields, "genuine", "fake", "class_confounded"),
    )
    if confirmatory and any(
        r["class_confounded"] or r["capture_settings"].strip().lower() in MISSING
        for r in rows
    ):
        raise ValueError(
            "incomplete recorded acquisition settings or class-confounded delivery cells; see delivery_coverage.csv"
        )
    aliases = [sorted(names) for names in profile_routes.values() if len(names) > 1]
    if confirmatory and aliases:
        raise ValueError(
            f"distinct measured route labels share an acquisition profile: {aliases}"
        )
    changed = sorted(
        name for name, profiles in route_profiles.items() if len(profiles) > 1
    )
    if confirmatory and changed:
        raise ValueError(
            f"measured route changed its registered acquisition profile: {changed}"
        )
    return len(rows)


def prepare(cfg: dict, out: Path) -> Path:
    """Validate a registered panel and create atomic, reusable frame/media caches."""
    data = cfg["data"]
    out = Path(out).resolve()
    src = Path(data["sources"]).resolve()
    sources = read_csv(src, SOURCE)
    validate_sources(sources, data.get("require_identity_splits", []))
    rules = data["routes"]
    if not isinstance(rules, dict) or rules.get("native") != {"kind": "native"}:
        raise ValueError("routes.native must be {kind: native}")
    for name, rule in rules.items():
        if (
            not name
            or not isinstance(rule, dict)
            or rule.get("kind") not in {"native", "encode", "measured"}
        ):
            raise ValueError(f"invalid route {name}")
        if name != "native" and rule["kind"] == "native":
            raise ValueError("exactly one native route is required")
        if rule["kind"] in {"native", "measured"} and set(rule) != {"kind"}:
            raise ValueError(f"unexpected fields in {name}")
    by_split = data.get("routes_by_split", {})
    if set(by_split) - SPLITS:
        raise ValueError("unknown split in routes_by_split")
    for split in {r["split"] for r in sources}:
        names = by_split.get(split, list(rules))
        if (
            not isinstance(names, list)
            or len(set(names)) != len(names)
            or "native" not in names
            or set(names) - set(rules)
        ):
            raise ValueError(f"invalid route panel for {split}")
    rendfile = Path(data["renditions"]).resolve() if data.get("renditions") else None
    rend = read_csv(rendfile, MEASURED) if rendfile else []
    source_map = {s["recording_id"]: s for s in sources}
    recs = {}
    for r in rend:
        ident = r["recording_id"], r["route"]
        if (
            ident in recs
            or ident[0] not in source_map
            or r["route"] not in rules
            or rules[r["route"]]["kind"] != "measured"
        ):
            raise ValueError(f"duplicate, orphan or unregistered measured row {ident}")
        recs[ident] = r

    cache = Path(data.get("cache_dir", out / "cache")).resolve()

    def collect(s):
        rid = s["recording_id"]
        names = by_split.get(s["split"], list(rules))
        source = _resolve(src.parent, s["source_path"])
        start = float(data.get("window_start_seconds", 0))
        guard = float(data.get("tail_guard_seconds", 0.05))
        limit = float(data.get("window_seconds", 1e8))
        if (
            not all(map(math.isfinite, (start, guard, limit)))
            or min(start, guard) < 0
            or limit <= 0
        ):
            raise ValueError("invalid registered time window")
        statuses, views = [], []
        try:
            native_hash = media.digest(source)
            native_duration = media.probe(source)["duration"]
            duration = min(limit, native_duration - start - guard)
            if duration <= 0:
                raise ValueError(
                    "native recording shorter than registered start/tail guard"
                )
        except (OSError, ValueError, KeyError) as exc:
            return [], [(name, str(exc)) for name in names]
        for route in names:
            kind = rules[route]["kind"]
            crop_mode = data.get("crop", "full")
            track_path = s.get("track_path")
            track_reviewer = s.get("track_reviewer", "")
            mixed_tracks = crop_mode == "registered_track"
            if mixed_tracks:
                crop_mode = "fixed_track"
            track_base = src.parent
            offset = 0.0
            receipt = None
            receipt_hash = ""
            note = {}
            try:
                if kind == "native":
                    video, sha = source, native_hash
                elif kind == "encode":
                    video = media.encode(
                        source, native_hash, rules[route], cache / "encodes"
                    )
                    sha = media.digest(video)
                    if data.get("crop") == "view_track":
                        raise ValueError(
                            "view_track requires per-rendition annotations; use measured files or detected"
                        )
                else:
                    if (rid, route) not in recs:
                        raise FileNotFoundError(
                            f"measured rendition not enrolled: {rid}/{route}"
                        )
                    r = recs[(rid, route)]
                    video = _resolve(rendfile.parent, r["path"])
                    receipt = _resolve(rendfile.parent, r["receipt_path"])
                    sha = media.digest(video)
                    # A cryptographically mismatched receipt is invalid evidence, never ordinary attrition.
                    try:
                        note = _receipt(receipt, rid, route, native_hash, sha)
                    except (ValueError, KeyError, TypeError) as exc:
                        raise ProvenanceError(str(exc)) from exc
                    receipt_hash = media.digest(receipt)
                    offset = float(note["offset_seconds"])
                    if mixed_tracks and note.get("geometry_preserved") is not True:
                        crop_mode = "view_track"
                    if crop_mode == "view_track":
                        track_path, track_base = r.get("track_path"), rendfile.parent
                        track_reviewer = r.get("track_reviewer", "")
                    if (
                        crop_mode == "fixed_track"
                        and note.get("geometry_preserved") is not True
                    ):
                        raise ValueError(
                            "fixed_track requires attested normalized geometry preservation"
                        )
                track = None
                if crop_mode in {"fixed_track", "view_track"} and track_path:
                    track = _resolve(track_base, track_path)
                if mixed_tracks and not track_reviewer.strip():
                    raise ProvenanceError(
                        f"{rid}/{route}: missing reviewer for diagnostic face track"
                    )
                view_config = {**data, "crop": crop_mode}
                info = media.frames(
                    video,
                    sha,
                    duration,
                    offset + start,
                    view_config,
                    track,
                    cache / "frames",
                )
                views.append(
                    {
                        **s,
                        **info,
                        **_file_stat(video, "media"),
                        **_file_stat(receipt, "receipt"),
                        **_file_stat(track, "track"),
                        "receipt_path": str(receipt) if receipt else "",
                        "track_path": str(track) if track else "",
                        **_file_stat(Path(info["frames_path"]), "frames"),
                        "route": route,
                        "kind": kind,
                        "media_path": str(video),
                        "source_path": str(source),
                        "sha256": sha,
                        "source_sha256": native_hash,
                        "receipt_sha256": receipt_hash,
                        "capture_method": note.get("method", ""),
                        "capture_session": note.get("session", ""),
                        "capture_channel": note.get("channel", ""),
                        "capture_settings": json.dumps(note["settings"], sort_keys=True)
                        if isinstance(note.get("settings"), dict) and note["settings"]
                        else str(note.get("settings") or "").strip(),
                        "acquired_utc": note.get("acquired_utc", ""),
                        "offset_seconds": offset,
                        "native_duration_seconds": native_duration,
                        "window_start_seconds": start,
                        "crop_reference": crop_mode,
                        "track_reviewer": track_reviewer,
                    }
                )
                statuses.append((route, ""))
            except (ValueError, KeyError, OSError) as exc:
                if isinstance(exc, ProvenanceError):
                    raise
                statuses.append((route, str(exc)))
        return views, statuses

    workers = data.get("prepare_workers", 1)
    if type(workers) is not int or not 1 <= workers <= 64:
        raise ValueError("prepare_workers must be an integer in [1,64]")
    panel, ledger = [], []
    hashes, frame_hashes, native_hashes = {}, {}, {}
    collisions = []
    incomplete = Counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for source, (views, statuses) in zip(sources, pool.map(collect, sources)):
            failed = any(reason for _, reason in statuses)
            if failed:
                incomplete[(source["cohort"], source["split"], source["label"])] += 1
            else:
                panel.extend(views)
            for v in views:
                if v["kind"] == "native":
                    prior = native_hashes.setdefault(v["sha256"], v["recording_id"])
                    if prior != v["recording_id"]:
                        raise ValueError(
                            f"duplicate sha256 across native recordings: {prior} and {v['recording_id']}"
                        )
                for index, col in ((hashes, "sha256"), (frame_hashes, "frame_sha256")):
                    previous = index.setdefault(v[col], v)
                    if previous["recording_id"] == v["recording_id"]:
                        continue
                    # Harsh processing may genuinely erase differences between
                    # linked real/fake sources. Preserve the resulting hard cases.
                    if previous["split"] != v["split"]:
                        raise ValueError(
                            f"duplicate {col} crosses splits: {previous['recording_id']} and {v['recording_id']}"
                        )
                    if (
                        col == "sha256"
                        and previous["kind"] == "measured"
                        and v["kind"] == "measured"
                        and previous["cluster_id"] != v["cluster_id"]
                    ):
                        raise ValueError(
                            "one measured file/hash assigned to distinct unrelated recording clusters"
                        )
                    collisions.append(
                        {
                            "digest_kind": col,
                            "digest": v[col],
                            "left_recording": previous["recording_id"],
                            "right_recording": v["recording_id"],
                            "left_route": previous["route"],
                            "right_route": v["route"],
                        }
                    )
            ledger.extend(
                {**source, "route": name, "included": int(not failed), "reason": reason}
                for name, reason in statuses
            )
    out.mkdir(parents=True, exist_ok=True)
    write_csv(
        out / "attrition.csv",
        ledger,
        [
            "recording_id",
            "cohort",
            "family",
            "cluster_id",
            "split",
            "label",
            "route",
            "included",
            "reason",
        ],
    )
    write_csv(
        out / "content_collisions.csv",
        collisions,
        (
            "digest_kind",
            "digest",
            "left_recording",
            "right_recording",
            "left_route",
            "right_route",
        ),
    )
    unresolved = _screen_duplicates(panel, out, data.get("duplicate_clearances"))
    misaligned = _screen_alignment(panel, out, data.get("alignment_clearances"))
    delivery_cells = delivery_coverage(
        panel,
        out,
        confirmatory=cfg.get("mode") == "confirmatory"
        and data.get("crop") != "registered_track",
    )
    audit = {
        "delivery_cells": delivery_cells,
        "observed_content_collisions": len(collisions),
        "unresolved_alignment_candidates": misaligned,
        "unresolved_near_duplicate_pairs": unresolved,
        "enrolled_recordings": len(sources),
        "included_recordings": len({v["recording_id"] for v in panel}),
        "unique_clusters": len({s["cluster_id"] for s in sources}),
        "frame_rows": len(panel),
        "missing_recordings": [
            {"cohort": c, "split": s, "label": int(y), "count": n}
            for (c, s, y), n in sorted(incomplete.items())
        ],
        "records_without_identity_ids": sum(not s["identity_ids"] for s in sources),
        "route_definitions": rules,
        "source_manifest_sha256": media.digest(src),
        "rendition_manifest_sha256": media.digest(rendfile) if rendfile else None,
        "input_config_sha256": media.key(data),
        "decoder": media.media_stack(),
        "fallback_frames": sum(v["crop_fallback_frames"] for v in panel),
    }
    media.atomic_json(out / "panel_audit.json", audit)
    if misaligned:
        raise ValueError(
            f"unresolved recording/rendition alignment candidates: {out / 'alignment_audit.csv'}"
        )
    if unresolved:
        raise ValueError(
            f"unresolved near-duplicate recording groups: {out / 'near_duplicates.csv'}"
        )
    if (incomplete and not data.get("allow_incomplete", False)) or not panel:
        raise ValueError(
            f"incomplete panel; inspect {out / 'attrition.csv'} before proceeding"
        )
    dest = out / "panel.csv"
    write_csv(dest, panel, FIELDS)
    return dest


def prepare_crop_diagnostic(cfg: dict, out: Path) -> Path:
    """On a preregistered main-panel subset, extract reviewer-audited fixed-geometry face crops."""
    spec = cfg["crop_diagnostic"]
    main = cfg["data"]
    accepted = {"sources", "renditions", "splits", "cache_dir"}
    if set(spec) - accepted:
        raise ValueError(
            f"unknown crop diagnostic options: {sorted(set(spec) - accepted)}"
        )
    data = {
        **main,
        **{k: v for k, v in spec.items() if k != "splits"},
        "crop": "registered_track",
        "allow_incomplete": False,
    }
    permitted = spec.get("splits", ["diagnostic", "test"])
    if not permitted or set(permitted) - {"diagnostic", "test"}:
        raise ValueError("crop diagnostic restricted to diagnostic/test splits")
    primary_src, subset_src = (
        Path(main["sources"]).resolve(),
        Path(spec["sources"]).resolve(),
    )
    originals = {s["recording_id"]: s for s in read_csv(primary_src, SOURCE)}
    selected = read_csv(subset_src, (*SOURCE, "track_path", "track_reviewer"))
    if not selected:
        raise ValueError("empty crop diagnostic subset")
    for s in selected:
        native = originals.get(s["recording_id"])
        if (
            not native
            or s["split"] not in permitted
            or any(native[k] != s[k] for k in SOURCE if k != "source_path")
        ):
            raise ValueError(
                f"diagnostic differs from original registration: {s['recording_id']}"
            )
        if (primary_src.parent / native["source_path"]).resolve() != (
            subset_src.parent / s["source_path"]
        ).resolve():
            raise ValueError("diagnostic source path differs from main panel")
    if spec.get("renditions"):
        if not main.get("renditions"):
            raise ValueError("no measured routes in primary inventory")
        primary_r, subset_r = (
            Path(main["renditions"]).resolve(),
            Path(spec["renditions"]).resolve(),
        )
        captures = {
            (r["recording_id"], r["route"]): r for r in read_csv(primary_r, MEASURED)
        }
        for r in read_csv(subset_r, MEASURED):
            native = captures.get((r["recording_id"], r["route"]))
            if native is None or any(
                (primary_r.parent / native[k]).resolve()
                != (subset_r.parent / r[k]).resolve()
                for k in ("path", "receipt_path")
            ):
                raise ValueError(
                    "diagnostic capture/receipt differs from main inventory"
                )
    elif any(r["kind"] == "measured" for r in main["routes"].values()):
        raise ValueError(
            "supply the prespecified subset's measured rendition inventory"
        )
    out = Path(out)
    result = prepare({**cfg, "data": data}, out)
    media.atomic_json(
        out / "crop_diagnostic.json",
        {
            "primary_sources_sha256": media.digest(primary_src),
            "subset_sources_sha256": media.digest(subset_src),
            "rows": len(selected),
            "panel_sha256": media.digest(result),
            "scope": "fixed geometry, frozen model/threshold diagnostic",
        },
    )
    return result
