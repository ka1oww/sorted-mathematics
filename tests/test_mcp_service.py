"""Unit coverage for the MCP service boundary and its privacy properties."""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import joblib
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from chapters import CHAPTER_SLUGS


def _write_model(path: Path) -> Path:
    vectoriser = TfidfVectorizer().fit(["synthetic unit fixture"])
    classifier = DummyClassifier(strategy="prior")
    classifier.fit([[index] for index in range(21)], list(CHAPTER_SLUGS))
    joblib.dump((vectoriser, classifier), path)
    return path


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_policy(path: Path, model: Path, threshold: float = 0.0) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "confidence_threshold": threshold,
                "selection_rule": "lowest validation threshold reaching 0.95 selective accuracy, else 0.00",
                "model_sha256": _hash(model),
                "validation_split_sha256": "0" * 64,
            }
        )
    )
    return path


def _load_service(monkeypatch, model: Path | None, policy: Path | None):
    if model is None:
        monkeypatch.delenv("SORTED_MATH_MODEL_PATH", raising=False)
    else:
        monkeypatch.setenv("SORTED_MATH_MODEL_PATH", str(model))
    if policy is None:
        monkeypatch.delenv("SORTED_MATH_POLICY_PATH", raising=False)
    else:
        monkeypatch.setenv("SORTED_MATH_POLICY_PATH", str(policy))
    sys.modules.pop("mcp_service", None)
    return importlib.import_module("mcp_service")


def test_model_is_loaded_once_at_import_and_cached(tmp_path, monkeypatch):
    model = _write_model(tmp_path / "model.joblib")
    policy = _write_policy(tmp_path / "policy.json", model)
    with patch("joblib.load", wraps=joblib.load) as load:
        service = _load_service(monkeypatch, model, policy)
        service.classify_question_payload("synthetic classifier input")
        service.classify_question_payload("another synthetic classifier input")
    assert load.call_count == 1


@pytest.mark.parametrize(
    ("threshold", "answered"), [(0.0, True), (1 / 21, True), (0.1, False)]
)
def test_classification_applies_threshold_at_its_boundary(
    tmp_path, monkeypatch, threshold, answered
):
    model = _write_model(tmp_path / "model.joblib")
    policy = _write_policy(tmp_path / "policy.json", model, threshold)
    service = _load_service(monkeypatch, model, policy)
    result = service.classify_question_payload("synthetic classifier input", 3)
    assert result["answered"] is answered
    if answered:
        assert result["refusal"] is None
    else:
        assert result["refusal"] is not None
    assert result["threshold"] == threshold
    assert len(result["candidates"]) == 3
    assert all(isinstance(row["confidence"], float) for row in result["candidates"])


def test_classification_validates_public_inputs_and_model_state(tmp_path, monkeypatch):
    service = _load_service(monkeypatch, None, None)
    with pytest.raises(service.ServiceError) as error:
        service.classify_question_payload("  ")
    assert error.value.code == "invalid_input"
    with pytest.raises(service.ServiceError):
        service.classify_question_payload("synthetic classifier input", 22)
    with pytest.raises(service.ServiceError) as error:
        service.classify_question_payload("synthetic classifier input")
    assert error.value.code == "model_unavailable"


def test_mismatched_or_corrupt_policy_never_bootstraps_a_model(tmp_path, monkeypatch):
    model = _write_model(tmp_path / "model.joblib")
    policy = _write_policy(tmp_path / "policy.json", model)
    raw = json.loads(policy.read_text())
    raw["model_sha256"] = "f" * 64
    policy.write_text(json.dumps(raw))
    service = _load_service(monkeypatch, model, policy)
    with pytest.raises(service.ServiceError) as error:
        service.classify_question_payload("synthetic classifier input")
    assert error.value.code == "model_unavailable"


def test_model_and_policy_paths_must_be_absolute(tmp_path, monkeypatch):
    model = _write_model(tmp_path / "model.joblib")
    policy = _write_policy(tmp_path / "policy.json", model)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SORTED_MATH_MODEL_PATH", model.name)
    monkeypatch.setenv("SORTED_MATH_POLICY_PATH", policy.name)
    sys.modules.pop("mcp_service", None)
    service = importlib.import_module("mcp_service")
    with pytest.raises(service.ServiceError) as error:
        service.classify_question_payload("synthetic classifier input")
    assert error.value.code == "model_unavailable"


def test_reader_returns_aggregate_pointers_only(tmp_path, monkeypatch):
    from test_reader import write_paper

    service = _load_service(monkeypatch, None, None)
    result = service.read_paper_payload(str(write_paper(tmp_path / "paper.pdf")))
    assert result["disclosure_mode"] == "pointers"
    assert set(result["questions"][0]) == {"number", "start_page", "pages"}
    assert set(result["reader"]) == {
        "rung",
        "questions",
        "pages_read",
        "pages_without_text_layer",
        "problems",
        "trustworthy",
    }


def test_reader_sanitizes_invalid_and_untrusted_inputs(tmp_path, monkeypatch):
    from test_reader import write_prose

    service = _load_service(monkeypatch, None, None)
    with pytest.raises(service.ServiceError) as missing:
        service.read_paper_payload(str(tmp_path / "missing.pdf"))
    assert missing.value.code == "paper_unreadable"
    with pytest.raises(service.ServiceError) as wrong_type:
        service.read_paper_payload(str(tmp_path / "not-a-paper.txt"))
    assert wrong_type.value.code == "invalid_input"
    with pytest.raises(service.ServiceError) as untrusted:
        service.read_paper_payload(str(write_prose(tmp_path / "prose.pdf")))
    assert untrusted.value.code == "paper_untrusted"
    assert str(tmp_path) not in untrusted.value.safe_message


def test_chapters_are_canonical_unique_and_ordered(monkeypatch):
    service = _load_service(monkeypatch, None, None)
    chapters = service.list_chapters_payload()
    assert [row["slug"] for row in chapters] == list(CHAPTER_SLUGS)
    assert len(chapters) == len({row["slug"] for row in chapters}) == 21
