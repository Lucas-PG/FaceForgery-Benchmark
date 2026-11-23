import pytest

from tifs import initial
from tifs.io import sha, write_json


def test_existing_export_reverified_and_changed_tensors_rejected(tmp_path):
    export = tmp_path / "weights"
    published = {}
    for name in initial.OUTPUTS:
        p = export / name
        p.parent.mkdir(exist_ok=True, parents=True)
        p.write_bytes(name.encode())
        published[name] = sha(p)
    receipt = {
        "format": initial.FORMAT,
        "outputs": published,
        "sources": initial.source_spec(),
    }
    write_json(export / "initial.json", receipt)
    assert initial.provision(export) == receipt
    (export / initial.OUTPUTS[0]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="Changed initialization"):
        initial.provision(export)


def test_wrong_offline_release_does_not_publish_partial_directory(tmp_path):
    raw = tmp_path / "unverified.bin"
    raw.write_bytes(b"unverified")
    out = tmp_path / "export"
    with pytest.raises(ValueError, match="Expected official"):
        initial.provision(out, {"dino_raw": raw})
    assert not out.exists()
    assert not list(tmp_path.glob("export-stage-*"))


def test_initial_identity_is_invariant_to_prefix_and_detects_modified_weights():
    import torch

    value = torch.tensor([[0.25, 0.5]])
    current = initial.tensor_digest({"encoder.weights": value}, "clip")
    assert current == initial.tensor_digest(
        {"vision_model.encoder.weights": value.clone()}, "clip"
    )
    assert current != initial.tensor_digest({"encoder.weights": value + 1}, "clip")
    with pytest.raises(ValueError, match="Duplicate"):
        initial.tensor_digest(
            {"encoder.weights": value, "vision_model.encoder.weights": value}, "clip"
        )
