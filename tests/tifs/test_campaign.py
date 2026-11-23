"""The same interfaces used on SLURM must work offline and reject stale artifacts."""

import json
from pathlib import Path

import pytest

from tifs.campaign import (
    check_complete,
    execute,
    load_config,
    model_artifacts,
    register,
    runs,
)
from tifs.io import digest, sha, write_json
from tifs.submit import graph


def production(monkeypatch, tmp_path):
    for key in ("OUTPUT", "ASSETS", "INVENTORY", "CACHE"):
        monkeypatch.setenv("TIFS_" + key, str(tmp_path / key))
    return load_config(
        Path(__file__).resolve().parents[2] / "configs/tifs/campaign.yaml"
    )


def test_production_matrix_and_resources(monkeypatch, tmp_path):
    cfg = production(monkeypatch, tmp_path)
    jobs = runs(cfg)
    assert len(jobs) == 43
    stages = {
        name: (deps, base, tail)
        for name, deps, base, tail in graph(cfg, "campaign.yaml", 3)
    }
    assert stages["models"][0] == ["clean"]
    assert stages["crop"][0] == ["analyze"] and stages["report"][0] == ["crop"]
    assert (
        "--gres=gpu:1" in stages["models"][1]
        and "--gres=gpu:1" not in stages["prepare"][1]
    )
    assert sum(s["objective"] == "clean" for s, _ in jobs.values()) == 6
    routes = cfg["data"]["routes_by_split"]
    assert "network" in routes["cal"] and "network" not in routes["train"]
    assert "redistribution" not in routes["cal"] and "redistribution" in routes["test"]
    assert cfg["analysis"]["heldout_families"] == {"kodf": ["fsgan"]}
    cfg["models"][4]["training"] = {"epochs": 99}
    with pytest.raises(ValueError, match="Confounded"):
        runs(cfg)


def test_checkpoint_receipt_detects_changed_bytes(tmp_path):
    ck = tmp_path / "checkpoint.pt"
    ck.write_bytes(b"known-weights")
    write_json(
        tmp_path / "model.json", {"checkpoint": str(ck), "checkpoint_sha256": sha(ck)}
    )
    scores = tmp_path / "predictions.csv"
    scores.write_text("recording_id,route,score\n1,native,0.7\n")
    identity = {"model": {"backbone": "tiny"}, "seed": 42}
    write_json(
        tmp_path / "complete.json",
        {
            "identity": digest(identity),
            "predictions_sha256": sha(scores),
            "artifacts": model_artifacts(tmp_path, identity["model"]),
        },
    )
    check_complete(tmp_path, identity)
    ck.touch()
    check_complete(tmp_path, identity)
    ck.write_bytes(b"other-weights")
    with pytest.raises(ValueError, match="Corrupt detector"):
        check_complete(tmp_path, identity)


def test_plan_needs_no_private_assets(monkeypatch, tmp_path, capsys):
    cfg = production(monkeypatch, tmp_path)
    execute(cfg, "plan")
    assert len(capsys.readouterr().out.splitlines()) == 43
    assert not Path(cfg["output"]).exists()
    cfg["mode"] = "fixture"
    assets = tmp_path / "assets.json"
    assets.write_text(json.dumps({"dino-init": {"objective": "clean"}}))
    cfg["asset_registry"] = str(assets)
    with pytest.raises(ValueError, match="Asset registry"):
        register(cfg, tmp_path)


def test_route_merge_resolves_child_paths_and_prevents_double_count(tmp_path):
    from tifs.combine import combine
    from tifs.panel import MEASURED, read_csv, write_csv

    sources = []
    for route in ("network", "redistribution"):
        parent = tmp_path / route
        parent.mkdir()
        for f in ("received.ts", "receipt.json"):
            (parent / f).touch()
        file = parent / "rows.csv"
        write_csv(
            file,
            [dict(zip(MEASURED, ("NA", route, "received.ts", "receipt.json")))],
            MEASURED,
        )
        sources.append(file)
    output = tmp_path / "combined.csv"
    combine(sources, output)
    rows = read_csv(output, MEASURED)
    assert [r["recording_id"] for r in rows] == ["NA", "NA"]
    assert all(Path(r["path"]).parent.name == r["route"] for r in rows)
    with pytest.raises(ValueError, match="Duplicate"):
        combine([*sources, sources[0]], output)


def test_crop_subpanel_is_bound_before_model_scoring(tmp_path):
    from tifs.campaign import read_tracked

    auxiliary = tmp_path / "crop/panel"
    auxiliary.mkdir(parents=True)
    file = auxiliary / "panel.csv"
    file.write_text("recording_id,route\noriginal,native\n")
    stamp = {"tracked": {"panel.csv": sha(file)}}
    file.write_text("recording_id,route\nreplacement,native\n")
    with pytest.raises(ValueError, match="pre-scoring registration"):
        read_tracked(tmp_path, stamp)
