"""Export a local, model-bound MCP runtime policy from an abstention report.

The resulting sidecar is intentionally local-only.  It contains hashes and a
threshold, never questions, labels, predictions, or split contents.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


SCHEMA_VERSION = 1
SELECTION_RULE = (
    "lowest validation threshold reaching 0.95 selective accuracy, else 0.00"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cutoff(report_path: Path) -> float:
    """Read the selected cutoff from a trusted aggregate abstention report."""
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        cutoff = report["cutoff"]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError) as error:
        raise ValueError("abstention report must be JSON with a cutoff") from error
    if (
        isinstance(cutoff, bool)
        or not isinstance(cutoff, (int, float))
        or not 0 <= cutoff <= 1
    ):
        raise ValueError("abstention report cutoff must be a number from 0 through 1")
    return float(cutoff)


def make_policy(model: Path, validation_split: Path, report: Path) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "confidence_threshold": load_cutoff(report),
        "selection_rule": SELECTION_RULE,
        "model_sha256": sha256(model),
        "validation_split_sha256": sha256(validation_split),
    }


def write_policy(destination: Path, policy: dict[str, object]) -> None:
    """Atomically write a sidecar, refusing to replace another model's policy."""
    if destination.resolve().is_relative_to(PROJECT_ROOT):
        raise ValueError("runtime policy output must be outside the repository")
    if destination.exists():
        try:
            existing = json.loads(destination.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(
                "existing policy is unreadable; refusing to overwrite it"
            ) from error
        if existing.get("model_sha256") != policy["model_sha256"]:
            raise ValueError("refusing to overwrite a policy for a different model")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=destination.parent, delete=False
    ) as handle:
        json.dump(policy, handle, sort_keys=True, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    try:
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--validation-split", required=True, type=Path)
    parser.add_argument("--abstention-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    policy = make_policy(
        arguments.model, arguments.validation_split, arguments.abstention_report
    )
    write_policy(arguments.output, policy)


if __name__ == "__main__":
    main()
