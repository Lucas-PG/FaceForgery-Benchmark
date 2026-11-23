"""Describe a registered FF++ Social source subset with two frozen campaign detectors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import campaign
from .analysis import _metric_rows
from .crop import IDENTIFIERS, KEYS
from .io import atomic_text, digest, sha, write_json
from .learning import _predict
from .models import build_model
from .panel import SOURCE, prepare, read_csv, verify_material, write_csv
from .stats_core import representatives, wide


def _json(path):
    return json.loads(Path(path).read_text())


def _read_panel(path):
    return pd.read_csv(path, dtype=IDENTIFIERS, keep_default_na=False)


def _outside(path, primary):
    if path.is_relative_to(primary) or primary.is_relative_to(path):
        raise ValueError(
            "Choose an independent auxiliary output outside the sealed campaign"
        )


def _sources(cfg, source_file, families=None):
    path = Path(cfg["data"]["sources"]).resolve()
    expected = {r["recording_id"]: r for r in read_csv(path, SOURCE)}
    cohort = "fixture_source" if cfg["mode"] == "fixture" else "ffpp"
    if source_file is None:
        rows = [
            r.copy()
            for r in expected.values()
            if r["cohort"] == cohort
            and r["split"] in {"dev", "cal"}
            and (not families or r["family"] in families)
        ]
        if families and set(families) - {r["family"] for r in rows}:
            raise ValueError("Requested one or more unregistered FF++ source families")
        for r in rows:
            for key in ("source_path", "track_path"):
                if r.get(key):
                    r[key] = str((path.parent / r[key]).resolve())
    else:
        rows = read_csv(source_file, SOURCE)
        for row in rows:
            original = expected.get(row["recording_id"])
            if (
                not original
                or row["cohort"] != cohort
                or row["split"] not in {"dev", "cal"}
                or any(row[k] != original[k] for k in SOURCE if k != "source_path")
            ):
                raise ValueError(
                    f"Archive recording is outside its registered source lineage: {row['recording_id']}"
                )
    if {r["label"] for r in rows} != {"0", "1"}:
        raise ValueError("Register both genuine and forged archive source recordings")
    return rows


def _validate_panel(primary, panel):
    native = panel[panel.route == "native"].sort_values("recording_id")
    origin = (
        primary[primary.route == "native"]
        .set_index("recording_id")
        .reindex(native.recording_id)
    )
    for field in (
        "cluster_id",
        "split",
        "label",
        "family",
        "cohort",
        "source_sha256",
        "sha256",
        "frame_sha256",
        "crop_reference",
        "track_sha256",
        "duration",
        "window_start_seconds",
        "offset_seconds",
    ):
        if not np.array_equal(native[field].to_numpy(), origin[field].to_numpy()):
            raise ValueError(
                f"Archive native differs from the registered media/preprocessing: {field}"
            )
    measured = panel[panel.route != "native"]
    if measured.empty or not measured.capture_method.eq("archive").all():
        raise ValueError(
            "Every social rendition must have a validated archive receipt with its release locator"
        )


def _sealed(cfg, main):
    campaign.bind_registry(cfg)
    protocol = _json(main / "protocol.json")
    if (
        cfg != protocol["config"]
        or campaign.code_id() != protocol["code"]
        or campaign.revision() != protocol["commit"]
        or campaign.versions() != protocol["versions"]
        or campaign.manifest_identity(cfg) != protocol["inputs"]
    ):
        raise ValueError(
            "Use the sealed campaign configuration, checkout, inventories and environment"
        )
    source, receipt = campaign.get_panel(main)
    analysis = _json(main / "analysis/complete.json")
    study = digest(protocol)
    if (
        analysis["study"] != study
        or receipt["study"] != study
        or sha(main / "analysis/policies.json") != analysis["outputs"]["policies.json"]
    ):
        raise ValueError("Frozen campaign panel or policy receipt changed")
    policies = _json(main / "analysis/policies.json")
    jobs = campaign.runs(cfg)
    models = {}
    for run in (cfg["analysis"]["baseline_run"], cfg["analysis"]["primary_run"]):
        folder = main / "runs" / run
        spec, seed = jobs[run]
        identity = {
            "study": study,
            "model": spec,
            "seed": seed,
            "panel": receipt["panel"],
        }
        if (
            _json(folder / "identity.json") != identity
            or sha(folder / "predictions.csv") != analysis["scores"][run]
        ):
            raise ValueError(f"Changed source run: {run}")
        campaign.check_complete(folder, identity)
        model = _json(folder / "model.json")
        if sha(model["checkpoint"]) != model["checkpoint_sha256"]:
            raise ValueError(f"Changed weights: {run}")
        cutoff = float(policies[run]["shared"]["cutoff"])
        if not np.isfinite(cutoff):
            raise ValueError("Nonfinite sealed cutoff")
        models[run] = {
            "seed": seed,
            "spec": spec,
            "weights": model["checkpoint"],
            "weight_digest": model["checkpoint_sha256"],
            "signature": model["train_signature"],
            "settings": model["training"],
            "common_cutoff": cutoff,
        }
    return source, models, study


def _score(cases, run, info, main, out):
    folder = out / run
    folder.mkdir(exist_ok=True)
    indexes = cases.loc[cases.route == "native", KEYS]
    original = pd.read_csv(
        main / "runs" / run / "predictions.csv",
        dtype={k: str for k in KEYS},
        keep_default_na=False,
    )
    native = indexes.merge(
        original[KEYS + ["score"]], on=KEYS, how="left", validate="one_to_one"
    )
    spec, settings = info["spec"], info["settings"]
    device = torch.device(settings["device"])
    model = build_model(spec, settings)
    snapshot = torch.load(
        info["weights"], map_location="cpu", weights_only=True, mmap=True
    )
    if snapshot["signature"] != info["signature"]:
        raise ValueError(f"Mismatch in trained checkpoint signature: {run}")
    model.load_state_dict(snapshot["model"], strict=True)
    del snapshot
    with torch.inference_mode():
        new = _predict(
            model.to(device),
            cases[cases.route != "native"],
            settings,
            spec,
            info["seed"],
            device,
            folder,
            digest(info),
        )
    archived = pd.read_csv(new, dtype={k: str for k in KEYS}, keep_default_na=False)
    result = cases.merge(
        pd.concat([native, archived]), on=KEYS, how="left", validate="one_to_one"
    )
    if not np.isfinite(pd.to_numeric(result.score, errors="coerce")).all():
        raise ValueError("One or more enrolled native/archive views lacks a score")
    atomic_text(folder / "all_scores.csv", result[KEYS + ["score"]].to_csv(index=False))
    return result


def analyze_archive(cfg, sources, renditions, routes, out):
    main, out = Path(cfg["output"]).resolve(), Path(out).resolve()
    _outside(out, main)
    sources, renditions = Path(sources).resolve(), Path(renditions).resolve()
    if not routes or len(routes) != len(set(routes)) or "native" in routes:
        raise ValueError(
            "Supply distinct archived route names in addition to implicit native"
        )
    _sources(cfg, sources)
    full, models, study = _sealed(cfg, main)
    settings = {
        **cfg["data"],
        "sources": str(sources),
        "renditions": str(renditions),
        "routes": {
            r: {"kind": "native" if r == "native" else "measured"}
            for r in ["native", *routes]
        },
        "routes_by_split": {},
        "allow_incomplete": False,
        "cache_dir": str(out / "media_cache"),
    }
    registration = {
        "campaign": study,
        "sources": sha(sources),
        "renditions": sha(renditions),
        "routes": routes,
        "analysis": sha(main / "analysis/complete.json"),
        "models": models,
        "data": settings,
    }
    with campaign.lock(out / ".archive.lock"):
        identity = out / "registration.json"
        if identity.exists() and _json(identity) != registration:
            raise ValueError(
                "Archive enrollment, derivation or frozen study changed; use a fresh output"
            )
        complete = out / "complete.json"
        panel_file = out / "panel" / "panel.csv"
        if complete.exists():
            previous = _json(complete)
            proof = {"registration": sha(identity), "panel": sha(panel_file)}
            if previous["inputs"] != proof or any(
                sha(out / k) != v for k, v in previous["outputs"].items()
            ):
                raise ValueError(
                    "Archive receipts or results changed; restore the original or use a fresh output"
                )
            cases = _read_panel(panel_file)
            verify_material(cases)
            _validate_panel(full, cases)
            return out / "metrics.csv"
        if not identity.exists() and panel_file.exists():
            raise ValueError("Refusing to overwrite an unregistered auxiliary panel")
        write_json(identity, registration)
        panel_file = prepare({**cfg, "data": settings}, out / "panel")
        cases = _read_panel(panel_file)
        verify_material(cases)
        _validate_panel(full, cases)
        if cfg["mode"] != "fixture":
            simulated = set(full.loc[full.kind == "encode", "sha256"])
            if any(cases.loc[cases.route != "native", "sha256"].isin(simulated)):
                raise ValueError(
                    "A claimed archive rendition exactly matches a locally simulated encode"
                )
        proof = {"registration": sha(identity), "panel": sha(panel_file)}
        rows, coverage = [], []
        tail = (1 - float(cfg["analysis"].get("confidence", 0.95))) / 2
        for run, model in models.items():
            scores = _score(cases, run, model, main, out)
            for split, subset in [
                ("pooled", scores),
                *list(scores.groupby("split", sort=True)),
            ]:
                all_cases = wide(subset)
                units = representatives(all_cases)
                flags = units.scores > model["common_cutoff"]
                rates = _metric_rows(
                    run, "shared", str(units.meta.cohort.iloc[0]), units, flags, tail
                )
                rows.extend(
                    dict(row, source_split=split, frozen_cutoff=model["common_cutoff"])
                    for row in rates
                )
                for label in (0, 1):
                    coverage.append(
                        {
                            "run": run,
                            "source_split": split,
                            "label": label,
                            "enrolled_recordings": int(
                                (all_cases.meta.label == label).sum()
                            ),
                            "independent_units": int((units.meta.label == label).sum()),
                            "scored_views": int(
                                np.isfinite(
                                    all_cases.scores[all_cases.meta.label == label]
                                ).sum()
                            ),
                        }
                    )
        for name, table in (("metrics", rows), ("coverage", coverage)):
            atomic_text(out / f"{name}.csv", pd.DataFrame(table).to_csv(index=False))
        scope = (
            "Synthetic engineering fixture."
            if cfg["mode"] == "fixture"
            else "FF++ Social source-development/calibration archive."
        )
        lines = [
            "# Archived delivery",
            "",
            scope,
            (
                f"The frozen detectors scored {cases.recording_id.nunique()} enrolled recordings on native and "
                f"{', '.join(routes)}."
            ),
            "",
        ]
        for row in rows:
            if row["source_split"] == "pooled" and row["route"] == "*":
                lines.append(
                    f"- {row['run']} {row['metric']}: {row['events']}/{row['n']} "
                    f"({row['estimate']:.1%}, pointwise {row['lower']:.1%} to {row['upper']:.1%})."
                )
        lines.extend(
            [
                "",
                (
                    "metrics.csv reports each registered route and the paired all-copy and any-copy outcomes. "
                    "Pointwise binomial intervals describe count precision under independent-lineage sampling. "
                    "Source data reuse in model development and cutoff fitting precludes a prospective external "
                    "performance or coverage claim."
                ),
            ]
        )
        atomic_text(out / "SUMMARY.md", "\n".join(lines) + "\n")
        outputs = [
            "metrics.csv",
            "coverage.csv",
            "SUMMARY.md",
            *(f"{run}/all_scores.csv" for run in models),
        ]
        outputs += [
            str(p.relative_to(out))
            for p in sorted(panel_file.parent.glob("*.csv"))
            if p != panel_file
        ]
        outputs.append("panel/panel_audit.json")
        write_json(
            complete, {"inputs": proof, "outputs": {k: sha(out / k) for k in outputs}}
        )
        return out / "metrics.csv"


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    listing = bool(argv and argv[0] == "sources")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    if listing:
        parser.add_argument(
            "--families",
            nargs="+",
            help="Predeclared enrolled FF++ families; include real",
        )
        args = parser.parse_args(argv[1:])
        cfg = campaign.load_config(args.config)
        _outside(args.out.resolve(), Path(cfg["output"]).resolve())
        rows = _sources(cfg, None, args.families)
        if args.out.exists():
            if read_csv(args.out, SOURCE) != rows:
                raise FileExistsError(
                    "Archive source snapshot already exists with different enrollment"
                )
        else:
            write_csv(args.out, rows, list(rows[0]))
    else:
        parser.add_argument("--sources", required=True, type=Path)
        parser.add_argument("--renditions", required=True, type=Path)
        parser.add_argument("--routes", required=True, nargs="+")
        args = parser.parse_args(argv)
        print(
            analyze_archive(
                campaign.load_config(args.config),
                args.sources,
                args.renditions,
                args.routes,
                args.out,
            )
        )


if __name__ == "__main__":
    main()
