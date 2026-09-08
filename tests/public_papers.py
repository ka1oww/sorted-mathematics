"""Where the public exam papers live, for the tests that need them.

The PDFs are not committed. They are third-party copyright (`(c) UCLES & MOE`
on the SEAB specimen) and this repository is public, and the standing rule for
this project is that no question text lands in a committed file. What is
committed is the labelling - question numbers and page indices, in
`public_papers_truth.json` - which is all the regression tests compare against.

Put the three papers in `tests/fixtures/papers/`, or point
`SORTED_PUBLIC_PAPERS` at wherever they already are. Without them the
paper-backed tests skip and say so; the rule's own unit tests need no PDF at
all.
"""

import json
import os
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_PAPERS_DIR = HERE / "fixtures" / "papers"
TRUTH_PATH = HERE / "fixtures" / "public_papers_truth.json"

MISSING_PAPERS = (
    f"the public exam papers are not in {DEFAULT_PAPERS_DIR} and "
    "SORTED_PUBLIC_PAPERS is unset. They are not committed: see "
    "tests/public_papers.py."
)


def papers_dir():
    return pathlib.Path(os.environ.get("SORTED_PUBLIC_PAPERS") or DEFAULT_PAPERS_DIR)


def truth():
    return json.loads(TRUTH_PATH.read_text())


def paper_path(spec):
    from tools.score_reader import resolve

    return resolve(spec, papers_dir())


def available(spec):
    return paper_path(spec).exists()


def any_available():
    return any(available(spec) for spec in truth().values())
