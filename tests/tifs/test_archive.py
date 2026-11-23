"""Verify archived source evidence against a complete sealed, genuinely trained tiny campaign."""

import json
from pathlib import Path

import pandas as pd
import pytest

from tifs import archive, campaign, fixture, importers
from tifs.io import sha
from tifs.panel import SOURCE, read_csv, write_csv


def test_archive_cli_real_checkpoint_resume_and_negative_controls(tmp_path):
    conf = fixture.make(tmp_path / "fixture")
    cfg = campaign.load_config(conf)
    campaign.execute(cfg, "all")
    main = Path(cfg["output"])
    subset = tmp_path / "source_subset.csv"
    archive.main(["sources", "--config", str(conf), "--out", str(subset)])
    # Predeclared four-case enrollment; re-encoded fixture bytes are only an engineering surrogate.
    chosen = [r for r in read_csv(subset, SOURCE) if r["recording_id"].endswith("-0")]
    write_csv(subset, chosen, SOURCE)
    panel = pd.read_csv(main / "panel/panel.csv", keep_default_na=False)
    indexed = panel[panel.route == "codec"].set_index("recording_id")
    inventory = []
    for row in chosen:
        rid = row["recording_id"]
        inventory.append(
            {
                "recording_id": rid,
                "route": "historical",
                "path": indexed.loc[rid, "media_path"],
                "session": "fixture-both-classes",
                "method": "archive",
                "acquired_utc": "2026-01-01T00:00:00Z",
                "alignment": "engineering fixture index",
                "offset_seconds": "0",
                "channel": "fixture",
                "settings": "generated fixture",
                "archive_locator": "synthetic/" + rid,
            }
        )
    acq = tmp_path / "acquisitions.csv"
    write_csv(acq, inventory, list(inventory[0]))
    rend = tmp_path / "renditions.csv"

    def register_acquisition():
        importers.main(
            [
                "register",
                "--sources",
                str(subset),
                "--acquisitions",
                str(acq),
                "--out",
                str(rend),
                "--receipts",
                str(tmp_path / "receipts"),
            ]
        )

    register_acquisition()
    out = tmp_path / "archive-study"
    cmd = [
        "--config",
        str(conf),
        "--sources",
        str(subset),
        "--renditions",
        str(rend),
        "--routes",
        "historical",
        "--out",
        str(out),
    ]
    archive.main(cmd)
    results = pd.read_csv(out / "metrics.csv")
    assert set(results.metric) == {"tpr", "fpr", "all_tpr", "any_fpr"}
    assert set(results.source_split) == {"pooled", "dev", "cal"}
    assert set(results.route) == {"*", "historical", "native"}
    assert results.query("source_split == 'pooled'").n.eq(2).all()
    policies = json.loads((main / "analysis/policies.json").read_text())
    for run, rows in results.groupby("run"):
        assert all(
            abs(x - policies[run]["shared"]["cutoff"]) < 1e-12
            for x in rows.frozen_cutoff
        )
        a = (
            pd.read_csv(out / run / "all_scores.csv")
            .query("route=='native'")
            .set_index("recording_id")
        )
        b = (
            pd.read_csv(main / "runs" / run / "predictions.csv")
            .query("route=='native'")
            .set_index("recording_id")
        )
        assert (a.score - b.loc[a.index].score).abs().max() < 1e-12
    stamp = sha(out / "complete.json")
    archive.main(cmd)
    assert sha(out / "complete.json") == stamp
    assert "%" in (out / "SUMMARY.md").read_text()
    clean = (out / "SUMMARY.md").read_bytes()
    (out / "SUMMARY.md").write_text("corrupt")
    with pytest.raises(ValueError, match="restore the original"):
        archive.main(cmd)
    assert (out / "SUMMARY.md").read_text() == "corrupt"
    (out / "SUMMARY.md").write_bytes(clean)
    changed = [dict(chosen[0], split="train"), *chosen[1:]]
    write_csv(subset, changed, SOURCE)
    with pytest.raises(ValueError, match="registered source lineage"):
        archive.main(cmd)
    write_csv(subset, chosen, SOURCE)
    inventory[0]["method"] = "network_capture"
    write_csv(acq, inventory, list(inventory[0]))
    register_acquisition()
    with pytest.raises(ValueError, match="archive receipt"):
        archive.main([*cmd[:-1], str(tmp_path / "wrong-method")])
