import json

import pytest

from tifs import enrollment, importers, panel


def pool():
    result = []
    for i, split in enumerate(("test", "diagnostic", "")):
        base = dict(
            zip(
                panel.SOURCE,
                [
                    f"r{i}",
                    "0",
                    "A",
                    "real",
                    "",
                    split,
                    f"r{i}.mp4",
                    f"origin-{i}",
                    f"actor-{i}",
                    f"session-{i}",
                ],
            )
        )
        result.append({**base, "matched_genuine_id": ""})
        result.append(
            {
                **base,
                "recording_id": f"f{i}",
                "source_path": f"f{i}.mp4",
                "family": "swap",
                "label": "1",
                "identity_ids": f"actor-{i}|donor-{i}",
                "matched_genuine_id": f"r{i}",
            }
        )
    # This otherwise valid forgery bridges the two authentic subject groups.
    result.append(
        {
            **result[1],
            "recording_id": "bridge",
            "source_path": "bridge.mp4",
            "identity_ids": "actor-0|actor-1",
        }
    )
    return result


def test_preregistered_packing_without_bridge_or_reserved_content(tmp_path):
    rows = pool()
    file, quota, reserved = [
        tmp_path / name for name in ("pool.csv", "quota.csv", "ffpp.csv")
    ]
    panel.write_csv(
        quota,
        [
            {"cohort": "A", "split": s, "family": "swap", "pairs": 1}
            for s in ("diagnostic", "test")
        ],
        enrollment.QUOTA,
    )
    panel.write_csv(
        reserved,
        [{**rows[4], "split": "train", "cluster_id": "source-component"}],
        panel.SOURCE,
    )

    def run(src, dest):
        panel.write_csv(file, src, enrollment.CANDIDATE)
        return importers.main(
            [
                "enroll-pairs",
                str(file),
                "--quotas",
                str(quota),
                "--exclude-sources",
                str(reserved),
                "--attempts",
                "16",
                "--seed",
                "19",
                "--out",
                str(tmp_path / dest),
            ]
        )

    selected = run(rows, "a")
    assert selected == run(list(reversed(rows)), "b")
    assert {r["recording_id"] for r in selected} == {"f0", "r0", "f1", "r1"}
    assert {r["split"] for r in selected if r["matched_genuine_id"] == "r0"} == {"test"}
    groups = {r["cluster_id"] for r in selected}
    assert len(groups) == 2
    panel.validate_sources(selected)
    merged = tmp_path / "combined.csv"
    importers.main(["inventory", str(tmp_path / "a/sources.csv"), "--out", str(merged)])
    frozen = panel.read_csv(merged, enrollment.CANDIDATE)
    assert {
        (r["recording_id"], r["matched_genuine_id"])
        for r in frozen
        if r["label"] == "1"
    } == {("f0", "r0"), ("f1", "r1")}
    ledger = {
        r["recording_id"]: r
        for r in panel.read_csv(
            tmp_path / "a/candidate_ledger.csv", enrollment.CANDIDATE
        )
    }
    assert len(ledger) == len(rows)
    assert ledger["f2"]["reason"] == "overlaps reserved material"
    assert ledger["bridge"]["reason"] == "overlaps another selected pair"
    assert len(ledger["bridge"]["conflicting_selected_clusters"].split("|")) == 2


def test_insufficient_quota_preserves_audit_without_enrollment(tmp_path):
    file, quota, out = (
        tmp_path / "pool.csv",
        tmp_path / "quota.csv",
        tmp_path / "design",
    )
    rows = pool()
    panel.write_csv(file, rows, enrollment.CANDIDATE)
    panel.write_csv(
        quota,
        [{"cohort": "A", "split": "test", "family": "swap", "pairs": 9}],
        enrollment.QUOTA,
    )
    with pytest.raises(ValueError, match="unfilled"):
        enrollment.pack([file], quota, [], 42, 2, out)
    assert not (out / "sources.csv").exists()
    assert (
        int(panel.read_csv(out / "quota_coverage.csv", ("shortfall",))[0]["shortfall"])
        > 0
    )
    rows[1]["session_ids"] = ""
    panel.write_csv(file, rows, enrollment.CANDIDATE)
    with pytest.raises(ValueError, match="incomplete session_ids"):
        enrollment.pack([file], quota, [], 42, 2, tmp_path / "missing")


def test_dfdc_candidate_parent_from_official_metadata(tmp_path):
    file = tmp_path / "decisions.csv"
    for name in ("a.mp4", "b.mp4"):
        (tmp_path / name).touch()
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "a.mp4": {"label": "REAL"},
                "b.mp4": {"label": "FAKE", "original": "a.mp4"},
            }
        )
    )
    rows = [
        {
            "relative_path": name,
            "visual_label": y,
            "family": family,
            "identity_ids": "actor",
            "source_ids": "footage",
            "session_ids": "shoot",
            "split": "",
        }
        for name, y, family in (("a.mp4", "0", "real"), ("b.mp4", "1", "swap"))
    ]
    panel.write_csv(file, rows, list(rows[0]))
    output = tmp_path / "pool.csv"
    importers.main(
        [
            "dfdc",
            "--root",
            str(tmp_path),
            "--decisions",
            str(file),
            "--candidate-pool",
            "--out",
            str(output),
        ]
    )
    registered = panel.read_csv(output, enrollment.CANDIDATE)
    assert {r["recording_id"]: r["matched_genuine_id"] for r in registered} == {
        "dfdc:a.mp4": "",
        "dfdc:b.mp4": "dfdc:a.mp4",
    }
