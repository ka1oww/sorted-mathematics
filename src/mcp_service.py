"""Local-only service boundary for the sorted-mathematics MCP transport.

This module deliberately owns model bootstrap, runtime-policy validation, and
copyright-safe projection of reader output.  The transport in ``mcp_server``
only adapts these JSON-native payloads to MCP.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Literal, TypedDict

import joblib
import pymupdf

from chapters import CHAPTERS, CHAPTER_SLUGS
from features import normalise_question
from page_ocr import OcrUnavailable
from reader import PaperReadError, read_paper_with_report


POLICY_SCHEMA_VERSION = 1
SELECTION_RULE = (
    "lowest validation threshold reaching 0.95 selective accuracy, else 0.00"
)


class ChapterItem(TypedDict):
    slug: str
    name: str


class Candidate(TypedDict):
    slug: str
    name: str
    confidence: float


class Refusal(TypedDict):
    code: Literal["below_calibrated_threshold"]
    best_confidence: float


class ClassificationResult(TypedDict):
    answered: bool
    threshold: float
    candidates: list[Candidate]
    refusal: Refusal | None


class PaperQuestion(TypedDict):
    number: int
    start_page: int
    pages: list[int]


class PaperResult(TypedDict):
    disclosure_mode: Literal["pointers"]
    questions: list[PaperQuestion]
    reader: dict[str, object]


class ServiceError(RuntimeError):
    """A failure whose public fields are safe to send across MCP."""

    def __init__(
        self,
        code: Literal[
            "invalid_input", "model_unavailable", "paper_unreadable",
            "paper_untrusted", "ocr_unavailable", "internal_error",
        ],
        safe_message: str,
        data: dict[str, object] | None = None,
    ) -> None:
        super().__init__(safe_message)
        self.code = code
        self.safe_message = safe_message
        self.data = data or {}


CLASSIFY_QUESTION_DOC = (
    "Route one H2 Mathematics exam item to the 21 Ember chapters; this tool "
    "does not solve the item. It returns ranked chapter candidates with model "
    "confidences. When the leading confidence is below the calibrated cutoff, "
    "the response explicitly refuses while still returning its ranked "
    "candidates. Request more ranked candidates with how_many."
)


READ_PAPER_DOC = (
    "Discover numbered questions and zero-based page spans in one local PDF. "
    "It automatically uses the PDF text layer or optional OCR and fails "
    "rather than emitting an untrusted cut. It does not classify chapters. "
    "Returned items contain pointers only (number and zero-based page span), "
    "not question text."
)


def _environment_path(name: str) -> Path | None:
    value = os.environ.get(name)
    path = Path(value) if value else None
    return path if path is not None and path.is_absolute() else None


MODEL_PATH = _environment_path("SORTED_MATH_MODEL_PATH")
POLICY_PATH = _environment_path("SORTED_MATH_POLICY_PATH")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_policy(path: Path) -> dict[str, object]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("runtime policy is unavailable") from error
    if not isinstance(raw, dict) or set(raw) != {
        "schema_version", "confidence_threshold", "selection_rule",
        "model_sha256", "validation_split_sha256",
    }:
        raise ValueError("runtime policy has an unsupported schema")
    threshold = raw["confidence_threshold"]
    if (
        raw["schema_version"] != POLICY_SCHEMA_VERSION
        or isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not 0.0 <= float(threshold) <= 1.0
        or raw["selection_rule"] != SELECTION_RULE
        or not all(
            isinstance(raw[key], str)
            and len(raw[key]) == 64
            and all(character in "0123456789abcdef" for character in raw[key])
            for key in ("model_sha256", "validation_split_sha256")
        )
    ):
        raise ValueError("runtime policy is invalid")
    return raw


def _bootstrap_model() -> tuple[object, object, float] | ServiceError:
    """Try exactly once at process startup; never expose local artifact paths."""
    if MODEL_PATH is None or POLICY_PATH is None:
        return ServiceError(
            "model_unavailable",
            "Classifier is unavailable; configure its local model and policy.",
        )
    try:
        # Keep this load before policy validation: a configured model gets one
        # and only one load attempt even when its adjacent policy is malformed.
        vectoriser, classifier = joblib.load(MODEL_PATH)
        policy = _load_policy(POLICY_PATH)
        if _sha256(MODEL_PATH) != policy["model_sha256"]:
            raise ValueError("runtime policy does not match the local model")
        return vectoriser, classifier, float(policy["confidence_threshold"])
    except Exception:
        return ServiceError(
            "model_unavailable",
            "Classifier is unavailable; check its local model and matching policy.",
        )


_MODEL_BOOTSTRAP = _bootstrap_model()


def _model_or_error() -> tuple[object, object, float]:
    if isinstance(_MODEL_BOOTSTRAP, ServiceError):
        raise _MODEL_BOOTSTRAP
    return _MODEL_BOOTSTRAP


def _validate_how_many(how_many: int) -> int:
    if isinstance(how_many, bool) or not isinstance(how_many, int) or not 1 <= how_many <= 21:
        raise ServiceError("invalid_input", "how_many must be an integer from 1 through 21.")
    return how_many


def classify_question_payload(
    question_text: str, how_many: int = 3
) -> ClassificationResult:
    """Classify with the cached local model and its calibrated policy."""
    if not isinstance(question_text, str) or not question_text.strip():
        raise ServiceError("invalid_input", "question_text must be nonblank.")
    count = _validate_how_many(how_many)
    vectoriser, classifier, threshold = _model_or_error()
    probabilities = classifier.predict_proba(
        vectoriser.transform([normalise_question(question_text)])
    )[0]
    ranked = probabilities.argsort()[::-1][:count]
    candidates: list[Candidate] = []
    for index in ranked:
        confidence = float(probabilities[index])
        if not 0.0 <= confidence <= 1.0:
            raise ServiceError("internal_error", "Classifier returned an invalid confidence.")
        slug = str(classifier.classes_[index])
        if slug not in CHAPTERS:
            raise ServiceError("internal_error", "Classifier returned an unknown chapter.")
        candidates.append(
            {"slug": slug, "name": CHAPTERS[slug], "confidence": confidence}
        )
    if not candidates:
        raise ServiceError("internal_error", "Classifier returned no candidates.")
    answered = candidates[0]["confidence"] >= threshold
    refusal: Refusal | None = None
    if not answered:
        refusal = {
            "code": "below_calibrated_threshold",
            "best_confidence": candidates[0]["confidence"],
        }
    return {
        "answered": answered,
        "threshold": threshold,
        "candidates": candidates,
        "refusal": refusal,
    }


def _safe_reader_report(report: dict[str, object]) -> dict[str, object]:
    return {
        key: report.get(key)
        for key in (
            "rung", "questions", "pages_read", "pages_without_text_layer",
            "problems", "trustworthy",
        )
    }


def _project_question(question: dict[str, object]) -> PaperQuestion:
    return {
        "number": int(question["number"]),
        "start_page": int(question["start_page"]),
        "pages": [int(page) for page in question["pages"]],
    }


def _validate_pdf_path(pdf_path: str) -> Path:
    if not isinstance(pdf_path, str) or not pdf_path.strip():
        raise ServiceError("invalid_input", "pdf_path must be a nonblank local PDF path.")
    path = Path(pdf_path).expanduser()
    try:
        if path.suffix.lower() != ".pdf":
            raise ServiceError("invalid_input", "pdf_path must name a PDF file.")
        if not path.is_file() or not os.access(path, os.R_OK):
            raise ServiceError("paper_unreadable", "The requested PDF is unavailable.")
    except OSError as error:
        raise ServiceError("paper_unreadable", "The requested PDF is unavailable.") from error
    return path


def read_paper_payload(pdf_path: str) -> PaperResult:
    """Read a local PDF without placing extracted question text on the wire."""
    path = _validate_pdf_path(pdf_path)
    try:
        questions, report = read_paper_with_report(path, strict=True)
    except PaperReadError as error:
        raise ServiceError(
            "paper_untrusted",
            "The reader refused an untrusted paper segmentation.",
            _safe_reader_report(error.report),
        ) from error
    except OcrUnavailable as error:
        raise ServiceError(
            "ocr_unavailable",
            "OCR is unavailable for this paper; install the optional OCR requirements.",
        ) from error
    except (OSError, pymupdf.FileDataError, RuntimeError) as error:
        raise ServiceError("paper_unreadable", "The requested PDF could not be read.") from error
    return {
        "disclosure_mode": "pointers",
        "questions": [_project_question(question) for question in questions],
        "reader": _safe_reader_report(report),
    }


def list_chapters_payload() -> list[ChapterItem]:
    """Return the canonical ordered chapter map."""
    if len(CHAPTER_SLUGS) != 21 or len(set(CHAPTER_SLUGS)) != 21:
        raise ServiceError("internal_error", "The chapter map is unavailable.")
    return [{"slug": slug, "name": CHAPTERS[slug]} for slug in CHAPTER_SLUGS]
