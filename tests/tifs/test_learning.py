import numpy as np
import pandas as pd
import pytest
import torch

from tifs.learning import run_model
from tifs.models import build_model, sha256, validate_assets
from tifs.training_data import RecordingEpoch
from tifs.training_objectives import loss_for_batch


@pytest.fixture()
def panel(tmp_path):
    rows = []
    rng = np.random.default_rng(7)
    for split in ["train", "dev", "cal", "diagnostic", "test"]:
        for label in (0, 1):
            for i in range(4):
                rid = f"{split}-{label}-{i}"
                for r in ["native", "sent"]:
                    path = tmp_path / f"{rid}-{r}.npy"
                    sample = rng.integers(0, 100, size=(3, 12, 12, 3), dtype=np.uint8)
                    np.save(path, sample + 50 * label)
                    rows.append(
                        (
                            rid,
                            label,
                            "cohort",
                            "fake" if label else "real",
                            rid,
                            split,
                            r,
                            str(path),
                            0.50,
                            sha256(path),
                        )
                    )
    return pd.DataFrame(
        rows,
        columns=[
            "recording_id",
            "label",
            "cohort",
            "family",
            "cluster_id",
            "split",
            "route",
            "frames_path",
            "quality",
            "frame_sha256",
        ],
    )


def settings():
    return {
        "train_routes": ["native", "sent"],
        "epochs": 2,
        "batch_size": 4,
        "learning_rate": 0.002,
        "alpha": 0.05,
        "clean_floor": 0.45,
        "workers": 0,
        "device": "cpu",
        "train_frames": 2,
        "shard_size": 23,
    }


def test_shuffle_preserves_exact_inputs_and_frame_draws(panel):
    cfg = settings()
    actual = RecordingEpoch(panel, cfg, "constrained", 7, 1)
    shuffled = RecordingEpoch(panel, cfg, "shuffled", 7, 1)
    assert np.array_equal(actual.paths, shuffled.paths)
    assert (np.arange(len(shuffled)) != shuffled.link[:, 1]).all()
    assert sorted(shuffled.link[:, 1]) == list(range(len(shuffled)))
    for i, parent in enumerate(shuffled.link[:, 1]):
        assert actual.labels[parent] == shuffled.labels[i]
        assert torch.equal(actual[parent][0][1], shuffled[i][0][1])
        assert torch.equal(actual[i][0][0], shuffled[i][0][0])


def test_rate_controls_and_maxup():
    x = torch.tensor(
        [[1.0, -1.0], [3.0, 2.0], [-0.5, 2.0], [-2.0, -1.0]], requires_grad=True
    )
    y = torch.tensor([1, 1, 0, 0])
    cfg = {"alpha": 0.05, "temperature": 0.3, "bce_anchor": 0.2}
    hardest = torch.tensor([-1.0, 2.0, 2.0, -1.0])
    base, _ = loss_for_batch(x, y, "worst_view", cfg, torch.tensor([]), 0.8)
    assert torch.allclose(
        base, torch.nn.functional.binary_cross_entropy_with_logits(hardest, y.float())
    )
    for obj, count in [("per_channel", 3), ("constrained", 2), ("no_clean", 1)]:
        loss, v = loss_for_batch(x, y, obj, cfg, torch.ones(count), 0.8)
        assert torch.isfinite(loss) and v.shape == (count,)
        loss.backward(retain_graph=True)
        assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize(
    "objective",
    [
        "clean",
        "augmented",
        "worst_view",
        "partial_auc",
        "constrained",
        "shuffled",
        "per_channel",
        "no_clean",
    ],
)
def test_training_single_copy_and_resume(panel, tmp_path, objective):
    config = {"training": settings()}
    model = {"backbone": "tiny", "name": objective, "objective": objective}
    directory = tmp_path / "out"
    predictions = run_model(config, panel, model, 123, directory)
    first = pd.read_csv(predictions)
    assert set(first.recording_id) == set(
        panel.loc[panel.split != "train", "recording_id"]
    )
    assert first.score.between(0, 1).all()
    assert not first.duplicated(["recording_id", "route"]).any()
    epoch_path = directory / "checkpoint.pt"
    modified = epoch_path.stat().st_mtime_ns
    # Truncate output; scoring can resume from verified shards without another update.
    predictions.write_text("damaged")
    recovered = pd.read_csv(run_model(config, panel, model, 123, directory))
    assert np.allclose(recovered.score, first.score, atol=1e-13)
    assert epoch_path.stat().st_mtime_ns == modified
    with pytest.raises(ValueError, match="Stale"):
        run_model(config, panel, model, 124, directory)


def test_reference_is_source_only(panel, tmp_path):
    cfg = {"training": settings()}
    initial = run_model(
        cfg,
        panel,
        {"backbone": "tiny", "name": "clean", "objective": "clean"},
        1,
        tmp_path / "clean",
    )
    reference = pd.read_csv(initial)
    ids = panel.loc[panel.split == "dev", "recording_id"]
    reference = reference[reference.recording_id.isin(ids)]
    reference.to_csv(tmp_path / "reference.csv", index=False)
    cfg["training"]["clean_reference"] = str(tmp_path / "reference.csv")
    spec = {"backbone": "tiny", "name": "paired", "objective": "constrained"}
    path = run_model(cfg, panel, spec, 1, tmp_path / "paired")
    assert path.is_file()
    reference.iloc[1:].to_csv(tmp_path / "reference.csv", index=False)
    with pytest.raises(ValueError, match="Incomplete"):
        run_model(cfg, panel, spec, 1, tmp_path / "bad")


def test_frozen_checkpoint_strict(panel, tmp_path):
    base_spec = {"backbone": "tiny", "objective": "augmented"}
    model = build_model(base_spec, settings())
    path = tmp_path / "frozen.pth"
    torch.save(model.state_dict(), path)
    spec = {
        "name": "historical",
        "backbone": "tiny",
        "objective": "frozen",
        "checkpoint": str(path),
        "checkpoint_sha256": sha256(path),
        "positive_index": 1,
    }
    assert run_model(
        {"training": settings()}, panel, spec, 1, tmp_path / "frozen"
    ).is_file()
    spec["checkpoint_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA256"):
        validate_assets(spec)


def test_interrupted_epoch_matches_uninterrupted(panel, tmp_path, monkeypatch):
    from tifs import learning

    cfg = {"training": settings()}
    spec = {"backbone": "tiny", "objective": "constrained"}
    full = run_model(cfg, panel, spec, 7, tmp_path / "full")
    write = learning._authority

    def interrupt_after_first_epoch(path, data):
        if path.name == "training_history.json" and len(data) == 1:
            raise InterruptedError("simulated preemption after durable epoch")
        write(path, data)

    with monkeypatch.context() as m:
        m.setattr(learning, "_authority", interrupt_after_first_epoch)
        with pytest.raises(InterruptedError):
            run_model(cfg, panel, spec, 7, tmp_path / "resumed")
    resumed = run_model(cfg, panel, spec, 7, tmp_path / "resumed")
    assert full.read_bytes() == resumed.read_bytes()


def test_shuffled_three_views_are_cross_cluster_including_non_native_pairs(panel):
    extra = panel[panel.route == "sent"].copy()
    extra["route"] = "second"
    triple = pd.concat([panel, extra], ignore_index=True)
    cfg = {**settings(), "train_routes": ["native", "sent", "second"]}
    epoch = RecordingEpoch(triple, cfg, "shuffled", 1, 3)
    assert all(len(set(row)) == 3 for row in epoch.link)
    # Make each class depend on a single original; no legal derangement exists.
    triple["cluster_id"] = triple.label.astype(str)
    with pytest.raises(ValueError, match="cross-cluster"):
        RecordingEpoch(triple, cfg, "shuffled", 1, 3)


def test_adaptive_quality_strata_preserve_margin_and_log_balance(panel):
    cfg = dict(settings(), shuffle_quantiles=["duration", "quality"])
    panel = panel.assign(
        duration=4.0,
        quality=panel.recording_id.str.rsplit("-", n=1).str[-1].astype(float),
    )
    epoch = RecordingEpoch(panel, cfg, "shuffled", 42, 0)
    assert epoch.shuffle_diagnostics["mandatory_groups"] == 2
    assert epoch.shuffle_diagnostics["refined_groups"] == 4
    assert epoch.shuffle_diagnostics["smallest_group"] == 2
    assert epoch.shuffle_diagnostics["mean_absolute_duration_difference"] == 0
    assert epoch.shuffle_diagnostics["mean_absolute_quality_difference"] == 1


def test_crop_interaction_preserves_all_four_decisions():
    from tifs.crop import _differences

    ids = pd.Index(["a", "b", "c", "d"])

    def make(values):
        return {("target", "all_tpr"): pd.Series(values, index=ids).astype(bool)}

    cells = {
        ("b", "end_to_end"): make([1, 0, 0, 0]),
        ("m", "end_to_end"): make([1, 1, 0, 0]),
        ("b", "tracked"): make([0, 0, 1, 0]),
        ("m", "tracked"): make([1, 1, 1, 0]),
    }
    deltas = _differences(cells, ["b", "m"], 0.025, 1000).set_index("comparison")
    assert deltas.loc["tracked_minus_end_to_end:primary_gain", "estimate"] == 0.25
    assert deltas.loc["tracked_minus_end_to_end:primary_gain", "n"] == 4
    assert deltas.loc["primary_minus_baseline:tracked", "estimate"] == 0.5
    cells["b", "tracked"]["target", "all_tpr"].index = ["a", "b", "d", "c"]
    with pytest.raises(ValueError, match="Unmatched"):
        _differences(cells, ["b", "m"], 0.025, 10)


def test_confirmatory_clean_reference_rejects_other_seed(panel, tmp_path):
    cfg = {
        "mode": "confirmatory",
        "training": dict(settings(), clean_reference="../clean/predictions.csv"),
    }
    base = {"backbone": "tiny", "name": "clean", "objective": "clean"}
    run_model(cfg, panel, base, 1, tmp_path / "clean")
    spec = dict(base, name="paired", objective="constrained")
    with pytest.raises(ValueError, match="same-seed"):
        run_model(cfg, panel, spec, 2, tmp_path / "paired")
    assert run_model(cfg, panel, spec, 1, tmp_path / "paired").exists()


def test_clean_floor_includes_route_seen_only_in_source_development(panel, tmp_path):
    from tifs.learning import _floor

    native = panel[(panel.split == "dev") & (panel.route == "native")]
    measured = native.assign(route="measured-dev-only")
    enriched = pd.concat([panel, measured], ignore_index=True)
    dev = enriched[enriched.split == "dev"]
    reference = dev[["recording_id", "route", "label"]].copy()
    reference["score"] = np.where(reference.label == 1, 0.8, 0.1)
    reference.loc[
        (reference.route == "measured-dev-only") & (reference.label == 0), "score"
    ] = 0.95
    file = tmp_path / "dev.csv"
    reference.drop(columns="label").to_csv(file, index=False)
    settings_ = dict(settings(), clean_reference=str(file))
    spec = {"backbone": "tiny", "objective": "constrained"}
    assert _floor({}, enriched, settings_, spec, 13, tmp_path)[0] == 0.0
    assert _floor({}, panel, settings_, spec, 13, tmp_path)[0] == pytest.approx(0.98)
