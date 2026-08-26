"""Where things live, and the one guard that stops a write landing outside them.

DATA-PLAN.md states the rule once ("every output is written inside this project
folder"), so it is implemented once here rather than in each writer.

The two source trees are my machine's layout. Nothing else in the repo depends
on them existing, and either can be pointed somewhere else with an env var.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRIVATE_DATA = PROJECT_ROOT / "data" / "private"
MODELS_DIR = PROJECT_ROOT / "models"

# override with EMBER_SOURCE_ROOT / TRIPLEMATH_PDF
DEFAULT_EMBER_SOURCE_ROOT = Path.home() / "Downloads" / "TUITION" / "H2 MATH EMBER"
DEFAULT_TRIPLEMATH_PDF = (
    Path.home() / "Documents" / "GitHub" / "Untitled" / "triple-math-exercises.pdf"
)


def ember_source_root():
    """Root of the revision packages and tutorials, for extract_latex/extract_pdf."""
    return Path(os.environ.get("EMBER_SOURCE_ROOT")
                or DEFAULT_EMBER_SOURCE_ROOT).expanduser()


def triplemath_pdf():
    """The Triple Math exercises PDF, for extract_triplemath."""
    return Path(os.environ.get("TRIPLEMATH_PDF")
                or DEFAULT_TRIPLEMATH_PDF).expanduser()


def assert_inside_project(path, project_root=PROJECT_ROOT):
    # every writer goes through here; a path bug should fail before it writes
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(Path(project_root).resolve()):
        raise SystemExit(f"refusing to write outside the project folder: {resolved}")
    return resolved
