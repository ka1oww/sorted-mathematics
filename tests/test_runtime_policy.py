"""Tests for exporting local model-bound runtime policy sidecars."""

import json

import pytest

from tools.export_runtime_policy import make_policy, write_policy


def test_policy_binds_cutoff_model_and_validation_split(tmp_path):
    model = tmp_path / "model.joblib"
    split = tmp_path / "validation.jsonl"
    report = tmp_path / "abstention.json"
    model.write_bytes(b"synthetic model")
    split.write_bytes(b"synthetic split")
    report.write_text(json.dumps({"cutoff": 0.5}))
    policy = make_policy(model, split, report)
    assert policy["confidence_threshold"] == 0.5
    destination = tmp_path / "model.policy.json"
    write_policy(destination, policy)
    assert json.loads(destination.read_text()) == policy


def test_policy_refuses_to_overwrite_a_different_model(tmp_path):
    first = tmp_path / "first.joblib"
    second = tmp_path / "second.joblib"
    split = tmp_path / "validation.jsonl"
    report = tmp_path / "abstention.json"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    split.write_bytes(b"split")
    report.write_text(json.dumps({"cutoff": 0.0}))
    destination = tmp_path / "policy.json"
    write_policy(destination, make_policy(first, split, report))
    with pytest.raises(ValueError, match="different model"):
        write_policy(destination, make_policy(second, split, report))


def test_policy_refuses_a_repository_output_path(tmp_path, monkeypatch):
    import tools.export_runtime_policy as exporter

    model = tmp_path / "model.joblib"
    split = tmp_path / "validation.jsonl"
    report = tmp_path / "abstention.json"
    model.write_bytes(b"model")
    split.write_bytes(b"split")
    report.write_text(json.dumps({"cutoff": 0.0}))
    monkeypatch.setattr(exporter, "PROJECT_ROOT", tmp_path)
    with pytest.raises(ValueError, match="outside the repository"):
        exporter.write_policy(
            tmp_path / "policy.json", exporter.make_policy(model, split, report)
        )
