"""Import official FF++ and audited external media, split connected lineages, register captures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import media
from .panel import (
    MEASURED,
    SOURCE,
    SPLITS,
    read_csv,
    tokens,
    validate_sources,
    write_csv,
)


def components(rows, conflicts="error", weights=None, seed=42):
    parents = list(range(len(rows)))

    def find(i):
        while i != parents[i]:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    seen = {}
    for i, r in enumerate(rows):
        for kind in ("source_ids", "identity_ids", "session_ids", "cluster_id"):
            for value in tokens(r.get(kind, "")):
                j = seen.setdefault((kind, value), i)
                parents[find(i)] = find(j)
    groups = {}
    for i, r in enumerate(rows):
        groups.setdefault(find(i), []).append(r)
    result, excluded = [], []
    if weights and (
        set(weights) - SPLITS
        or any(
            isinstance(w, bool) or not isinstance(w, (float, int)) or not 0 < w <= 1
            for w in weights.values()
        )
        or abs(sum(weights.values()) - 1) > 1e-8
    ):
        raise ValueError(
            "split weights must be positive fractions named for valid splits and sum to 1"
        )
    for group in groups.values():
        if len({r["cohort"] for r in group}) != 1:
            raise ValueError(
                "cross-cohort lineage detected; resolve documented duplicate or choose distinct cohorts"
            )
        members = sorted(r["recording_id"] for r in group)
        cluster = "component-" + media.key(members)[:24]
        roles = {r["split"] for r in group}
        if weights:
            u = int(media.key([seed, members])[:16], 16) / 2**64
            cumulative = 0
            for role, weight in sorted(weights.items()):
                cumulative += weight
                if u < cumulative:
                    break
        elif len(roles) == 1:
            role = roles.pop()
        elif conflicts == "exclude":
            excluded.extend(
                {**r, "reason": "connected component crosses registered splits"}
                for r in group
            )
            continue
        else:
            raise ValueError(
                f"connected component crosses registered splits: {members}"
            )
        result.extend({**r, "cluster_id": cluster, "split": role} for r in group)
    validate_sources(result)
    return sorted(result, key=lambda r: r["recording_id"]), excluded


def ffpp(root, splits, conversion, codec, families, identities=None):
    """Official pairs and shared original YouTube recordings define indivisible components."""
    mapping = json.loads(conversion.read_text())
    roles, partners = {}, {}
    for name, role in (("train", "train"), ("val", "dev"), ("test", "cal")):
        for pair in json.loads((splits / f"{name}.json").read_text()):
            if len(pair) != 2:
                raise ValueError(f"unexpected FF++ split record {pair}")
            a, b = pair
            if a in roles or b in roles or a == b:
                raise ValueError(f"duplicate original in FF++ split {pair}")
            roles.update({a: role, b: role})
            partners.update({a: b, b: a})
    people = {}
    if identities:
        for r in read_csv(identities, ("original_id", "identity_ids", "session_ids")):
            k = r["original_id"]
            if k in people or k not in roles:
                raise ValueError(f"duplicate or unknown original {k}")
            people[k] = r
    rows = []

    def add(family, ids, path):
        if not path.is_file():
            raise FileNotFoundError(f"missing registered FF++ input {path}")
        links = sorted(
            {"ffpp:original:" + i for i in ids}
            | {"youtube:" + mapping[i].split()[0] for i in ids}
        )

        def all_ids(key):
            return "|".join(
                sorted({v for i in ids for v in tokens(people.get(i, {}).get(key, ""))})
            )

        rows.append(
            {
                "recording_id": f"ffpp:{family}:{path.stem}",
                "label": str(int(family != "real")),
                "family": family,
                "cohort": "ffpp",
                "split": roles[ids[0]],
                "cluster_id": "",
                "source_ids": "|".join(links),
                "identity_ids": all_ids("identity_ids"),
                "session_ids": all_ids("session_ids"),
                "source_path": str(path.resolve()),
            }
        )

    for i in sorted(roles):
        add(
            "real",
            [i],
            root / "original_sequences" / "youtube" / codec / "videos" / f"{i}.mp4",
        )
        for family in families:
            j = partners[i]
            add(
                family,
                [i, j],
                root
                / "manipulated_sequences"
                / family
                / codec
                / "videos"
                / f"{i}_{j}.mp4",
            )
    return rows


def dfdc(root, decisions, *, candidate_pool=False):
    """Enroll exactly the preregistered visual subset; official metadata alone omits actor/donor links."""
    rows = []
    catalogs = {}
    for r in read_csv(
        decisions,
        (
            "relative_path",
            "visual_label",
            "family",
            "identity_ids",
            "session_ids",
            "split",
            "source_ids",
        ),
    ):
        path = (root / r["relative_path"]).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError(f"unavailable DFDC enrollment {r['relative_path']}")
        metadata = path.parent / "metadata.json"
        if metadata not in catalogs:
            catalogs[metadata] = json.loads(metadata.read_text())
        item = catalogs[metadata][path.name]
        y = {"REAL": "0", "FAKE": "1"}[item["label"]]
        if (
            y != r["visual_label"]
            or not tokens(r["identity_ids"])
            or not tokens(r["source_ids"])
        ):
            raise ValueError(
                "requires verified visual class, both-actor provenance and parent source IDs"
            )
        parent = item.get("original") if y == "1" else path.name
        if not parent:
            raise ValueError(f"missing official original for {path}")
        # DFDC filenames can be shared between shards/re-releases, so use the global name.
        links = set(tokens(r["source_ids"])) | {"dfdc:original:" + parent}
        rows.append(
            {
                **r,
                "recording_id": "dfdc:" + r["relative_path"],
                "label": y,
                "cohort": "dfdc",
                "cluster_id": "",
                "source_path": str(path),
                "official_original": parent,
                "source_ids": "|".join(sorted(links)),
            }
        )
    if candidate_pool:
        authentic = {}
        for r in rows:
            if r["label"] == "0":
                name = Path(r["source_path"]).name
                if name in authentic:
                    raise ValueError(f"ambiguous authentic DFDC filename: {name}")
                authentic[name] = r["recording_id"]
        for r in rows:
            if r["label"] == "1" and r["official_original"] not in authentic:
                raise ValueError(
                    f"include the authentic visual parent in the candidate pool: {r['official_original']}"
                )
            r["matched_genuine_id"] = (
                authentic[r["official_original"]] if r["label"] == "1" else ""
            )
    return rows


def _propose_track(video, duration, offset, data, folder, manual):
    import numpy as np
    from PIL import Image

    folder.mkdir(parents=True, exist_ok=True)
    timepoints = (np.arange(data["frames"]) + 0.5) * duration / data["frames"] + offset
    # Broad RGB context for annotation. Convert letterboxed coordinates back to display-relative boxes.
    side = 480
    info = media.frames(
        video,
        media.digest(video),
        duration,
        offset,
        {
            **data,
            "size": side,
            "decode_max_side": max(side, data.get("decode_max_side", 1024)),
            "crop": "full",
        },
        None,
        folder.parent / "proposal_cache",
    )
    stack = np.load(info["frames_path"], mmap_mode="r", allow_pickle=False)
    dims = media.probe(video)
    sx, sy = dims["width"], dims["height"]
    scale = side / max(sx, sy)
    sw, sh = round(scale * sx), round(scale * sy)
    ox, oy = (side - sw) // 2, (side - sh) // 2
    detector = None if manual else media._detector()
    rows = []
    for idx, (rgb, t) in enumerate(zip(stack, timepoints)):
        pic = Image.fromarray(rgb).crop((ox, oy, ox + sw, oy + sh))
        coords = [""] * 4
        if detector is not None:
            cv2, model = detector
            found = model.detectMultiScale(
                cv2.cvtColor(np.asarray(pic), cv2.COLOR_RGB2GRAY), 1.1, 5
            )
            if len(found) == 1:
                x, y, w, h = found[0]
                coords = [x / sw, y / sh, (x + w) / sw, (y + h) / sh]
        # Clean preview is in display coordinates and has no detector scores.
        pic.save(folder / f"sample-{idx:03d}.png")
        rows.append(dict(zip(("time_s", "x0", "y0", "x1", "y1"), (t, *coords))))
    fields = ("time_s", "x0", "y0", "x1", "y1")
    write_csv(folder / "boxes.csv", rows, fields)


def propose_tracks(opt):
    from .campaign import load_config
    from .panel import _receipt

    data = load_config(opt.config)["data"]
    src = Path(data["sources"]).resolve()
    raws = read_csv(src, SOURCE)
    validate_sources(raws, data.get("require_identity_splits", []))
    if set(opt.splits) - {"diagnostic", "test"} or opt.per_class < 1:
        raise ValueError(
            "track proposal requires diagnostic/test, positive per-class enrollment"
        )
    if (opt.out / "review.csv").exists():
        raise FileExistsError(
            "track review already exists; preserve adjudications, choose a new draft folder"
        )
    opt.out.mkdir(parents=True, exist_ok=True)
    rfile = Path(data["renditions"]).resolve() if data.get("renditions") else None
    incoming = read_csv(rfile, MEASURED) if rfile else []
    route_rows = {(r["recording_id"], r["route"]): r for r in incoming}
    strata = {}
    for s in raws:
        if s["split"] in opt.splits:
            strata.setdefault((s["cohort"], s["split"], s["label"]), []).append(s)
    enrolled = []
    for rows in strata.values():
        priority = lambda r: media.key([opt.seed, r["recording_id"]])
        distinct = {}
        for s in sorted(rows, key=priority):
            distinct.setdefault(s["cluster_id"], s)
        enrolled.extend(list(distinct.values())[: opt.per_class])
    if not enrolled:
        raise ValueError("no records in requested diagnostic/test splits")
    proposed, received, review = [], [], []

    def make(video, route, offset, rid, duration):
        folder = (opt.out / "annotations" / media.key([rid, route])[:24]).resolve()
        _propose_track(video, duration, offset, data, folder, opt.manual)
        track = str(folder / "boxes.csv")
        review.append(
            {
                "recording_id": rid,
                "route": route,
                "media_sha256": media.digest(video),
                "track_path": track,
                "reviewer": "",
                "decision": "",
            }
        )
        return track

    for s in sorted(enrolled, key=lambda r: r["recording_id"]):
        src_video = (src.parent / s["source_path"]).resolve()
        start = float(data.get("window_start_seconds", 0))
        duration = min(
            float(data.get("window_seconds", 1e8)),
            media.probe(src_video)["duration"]
            - start
            - float(data.get("tail_guard_seconds", 0.05)),
        )
        if duration <= 0:
            raise ValueError(f"source interval unavailable: {s['recording_id']}")

        native = make(src_video, "native", start, s["recording_id"], duration)
        proposed.append(
            {
                **s,
                "source_path": str(src_video),
                "track_path": native,
                "track_reviewer": "",
            }
        )
        for route in data.get("routes_by_split", {}).get(
            s["split"], list(data["routes"])
        ):
            if data["routes"][route]["kind"] != "measured":
                continue
            r = route_rows[s["recording_id"], route]
            video = (rfile.parent / r["path"]).resolve()
            rc = (rfile.parent / r["receipt_path"]).resolve()
            note = _receipt(
                rc,
                s["recording_id"],
                route,
                media.digest(src_video),
                media.digest(video),
            )
            track = (
                ""
                if note.get("geometry_preserved") is True
                else make(
                    video,
                    route,
                    start + float(note["offset_seconds"]),
                    s["recording_id"],
                    duration,
                )
            )
            received.append(
                {
                    **r,
                    "path": str(video),
                    "receipt_path": str(rc),
                    "track_path": track,
                    "track_reviewer": "",
                }
            )
    write_csv(
        opt.out / "selected_sources.csv",
        proposed,
        (*SOURCE, "track_path", "track_reviewer"),
    )
    write_csv(
        opt.out / "selected_renditions.csv",
        received,
        (*MEASURED, "track_path", "track_reviewer"),
    )
    write_csv(
        opt.out / "review.csv",
        review,
        ("recording_id", "route", "media_sha256", "track_path", "reviewer", "decision"),
    )
    media.atomic_json(
        opt.out / "selection.json",
        {
            "seed": opt.seed,
            "per_class_per_cohort_split": opt.per_class,
            "splits": opt.splits,
            "source_sha256": media.digest(src),
            "config_sha256": media.key(data),
            "selected_recordings": [r["recording_id"] for r in proposed],
        },
    )


def finalize_tracks(folder):
    import numpy as np

    fields = (
        "recording_id",
        "route",
        "media_sha256",
        "track_path",
        "reviewer",
        "decision",
    )
    audit = read_csv(folder / "review.csv", fields)
    decisions = {}
    for a in audit:
        k = a["recording_id"], a["route"]
        if k in decisions or not a["reviewer"].strip() or a["decision"] != "approved":
            raise ValueError(
                "all selected tracks require explicit reviewer and approved decision"
            )
        decisions[k] = a

    def attach(row, route, field):
        approval = decisions.pop((row["recording_id"], route))
        if (
            approval["track_path"] != row["track_path"]
            or media.digest(Path(row[field])) != approval["media_sha256"]
        ):
            raise ValueError("material changed between annotation and review")
        points = read_csv(Path(row["track_path"]), ("time_s", "x0", "y0", "x1", "y1"))
        media._track(
            Path(row["track_path"]), np.array([float(r["time_s"]) for r in points])
        )
        row["track_reviewer"] = approval["reviewer"]

    sources = read_csv(
        folder / "selected_sources.csv", (*SOURCE, "track_path", "track_reviewer")
    )
    renditions = read_csv(
        folder / "selected_renditions.csv", (*MEASURED, "track_path", "track_reviewer")
    )
    for s in sources:
        attach(s, "native", "source_path")
    for r in renditions:
        if r["track_path"]:
            attach(r, r["route"], "path")
    if decisions:
        raise ValueError("unmatched track reviews")
    write_csv(
        folder / "sources.csv", sources, (*SOURCE, "track_path", "track_reviewer")
    )
    write_csv(
        folder / "renditions.csv",
        renditions,
        (*MEASURED, "track_path", "track_reviewer"),
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    cmds = p.add_subparsers(dest="cmd", required=True)
    f = cmds.add_parser("ffpp")
    for k in ("root", "splits", "conversion"):
        f.add_argument("--" + k, type=Path, required=True)
    f.add_argument("--identities", type=Path)
    f.add_argument("--codec", choices=["c0", "raw", "c23", "c40"], default="c23")
    f.add_argument(
        "--families",
        nargs="+",
        default=["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"],
    )
    d = cmds.add_parser("dfdc")
    d.add_argument("--root", type=Path, required=True)
    d.add_argument("--decisions", type=Path, required=True)
    d.add_argument(
        "--candidate-pool",
        action="store_true",
        help="export reviewed candidates and official parent pointers before connected-component splitting",
    )
    e = cmds.add_parser(
        "enroll-pairs",
        help="select metadata-disjoint authentic/forged pairs under registered quotas",
    )
    e.add_argument("inputs", type=Path, nargs="+")
    e.add_argument("--quotas", type=Path, required=True)
    e.add_argument("--exclude-sources", nargs="+", type=Path, default=[])
    e.add_argument("--seed", type=int, default=42)
    e.add_argument("--attempts", type=int, default=16)
    e.add_argument("--out", type=Path, required=True, help="fresh output directory")
    g = cmds.add_parser(
        "inventory",
        help="join audited KoDF/licensed/generic inventories; paths are resolved per source",
    )
    g.add_argument("inputs", type=Path, nargs="+")
    g.add_argument(
        "--weights",
        help='Override splits by deterministic connected component, e.g. {"diagnostic":0.25,"test":0.75}',
    )
    g.add_argument("--seed", type=int, default=42)
    m = cmds.add_parser(
        "register",
        help="hash independently recovered captures and register signed-off acquisition metadata",
    )
    m.add_argument("--sources", type=Path, required=True)
    m.add_argument("--acquisitions", type=Path, required=True)
    m.add_argument("--receipts", type=Path, required=True)
    for c in (f, d, g, m):
        c.add_argument("--out", type=Path, required=True)
    for c in (f, d, g):
        c.add_argument("--conflicts", choices=["error", "exclude"], default="error")
    t = cmds.add_parser(
        "propose-tracks",
        help="choose independent records and create editable face boxes and visual previews",
    )
    t.add_argument("--config", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--splits", nargs="+", default=["diagnostic", "test"])
    t.add_argument("--seed", type=int, default=20261123)
    t.add_argument("--per-class", type=int, default=40)
    t.add_argument(
        "--manual",
        action="store_true",
        help="write blank boxes and previews without an automatic detector",
    )
    u = cmds.add_parser(
        "finalize-tracks", help="validate human-reviewed diagnostic annotations"
    )
    u.add_argument("--folder", type=Path, required=True)
    opt = p.parse_args(argv)
    if opt.cmd == "enroll-pairs":
        from .enrollment import pack

        return pack(
            opt.inputs, opt.quotas, opt.exclude_sources, opt.seed, opt.attempts, opt.out
        )
    if opt.cmd == "propose-tracks":
        return propose_tracks(opt)
    if opt.cmd == "finalize-tracks":
        return finalize_tracks(opt.folder)
    if opt.cmd == "register":
        from .panel import _receipt

        sources = read_csv(opt.sources, SOURCE)
        validate_sources(sources)
        source_map = {r["recording_id"]: r for r in sources}
        rows, seen, hashes = [], set(), {}
        for r in read_csv(
            opt.acquisitions,
            (
                "recording_id",
                "route",
                "path",
                "session",
                "method",
                "channel",
                "acquired_utc",
                "alignment",
                "offset_seconds",
            ),
        ):
            rid, route = r["recording_id"], r["route"]
            if (rid, route) in seen or not route or route == "native":
                raise ValueError("invalid or duplicate registered capture")
            seen.add((rid, route))
            src = opt.sources.parent / source_map[rid]["source_path"]
            if rid not in hashes:
                hashes[rid] = media.digest(src)
            received = (opt.acquisitions.parent / r["path"]).resolve()
            note = {
                k: v
                for k, v in r.items()
                if k
                not in {"path", "track_path", "geometry_preserved", "offset_seconds"}
            }
            note.update(
                source_sha256=hashes[rid],
                received_sha256=media.digest(received),
                offset_seconds=float(r["offset_seconds"]),
                geometry_preserved=r.get("geometry_preserved", "").lower() == "true",
            )
            receipt = opt.receipts / (media.key(note) + ".json")
            media.atomic_json(receipt, note)
            _receipt(receipt, rid, route, hashes[rid], note["received_sha256"])
            rows.append(
                {
                    "recording_id": rid,
                    "route": route,
                    "path": str(received),
                    "receipt_path": str(receipt.resolve()),
                    "track_path": str(
                        (opt.acquisitions.parent / r["track_path"]).resolve()
                    )
                    if r.get("track_path")
                    else "",
                }
            )
        write_csv(opt.out, rows, [*MEASURED, "track_path"])
        return
    if opt.cmd == "ffpp":
        raw = ffpp(
            opt.root,
            opt.splits,
            opt.conversion,
            opt.codec,
            opt.families,
            opt.identities,
        )
    elif opt.cmd == "dfdc":
        raw = dfdc(opt.root, opt.decisions, candidate_pool=opt.candidate_pool)
        if opt.candidate_pool:
            from .enrollment import CANDIDATE

            write_csv(opt.out, raw, CANDIDATE)
            return
    else:
        raw = []
        for file in opt.inputs:
            for r in read_csv(file, SOURCE):
                r["source_path"] = str((file.parent / r["source_path"]).resolve())
                if r.get("track_path"):
                    r["track_path"] = str((file.parent / r["track_path"]).resolve())
                raw.append(r)
    weights = json.loads(opt.weights) if getattr(opt, "weights", None) else None
    rows, exclusions = components(raw, opt.conflicts, weights, getattr(opt, "seed", 42))
    write_csv(opt.out, rows, [*SOURCE, "matched_genuine_id", "track_path"])
    write_csv(
        opt.out.with_suffix(".exclusions.csv"),
        exclusions,
        ["recording_id", "split", "reason"],
    )


if __name__ == "__main__":
    main()
