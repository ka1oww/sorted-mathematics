"""Emit a labelling skeleton for a paper, in the shape `score_reader` reads.

The yardstick is the specification: every later decision about the reader is
scored against these labels and nothing else, so they have to be a human's
opinion of where the questions are, not the reader's.

That is why this tool deliberately does **not** pre-fill the reader's own
answer. A skeleton seeded with predictions gets corrected where it looks wrong
and waved through where it looks plausible, which measures how convincing the
reader is rather than how right it is. What the tool supplies instead is the
context a labeller needs - the page indices, how much text each page carries,
which pages have no text layer - and an empty `questions` list per case.

    python3 tools/label_paper.py paper.pdf > yardstick.json
    python3 tools/label_paper.py paper.pdf --case my-prelim --pages 1-12

Then fill in, for each question, its number and every page it spans, following
the boundary rule the skeleton carries: a question owns every page from the
page its number appears on up to the page before the next number appears.
"""

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import page_lines

BOUNDARY_RULE = (
    "a question owns every page from the page its number appears "
    "on up to the page before the next number appears"
)


def parse_pages(text, page_count):
    """ "1-12" or "1,3,5" or both, zero-based, as PyMuPDF counts pages."""
    if not text:
        return list(range(page_count))
    pages = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            first, last = part.split("-", 1)
            pages.extend(range(int(first), int(last) + 1))
        else:
            pages.append(int(part))
    return pages


def skeleton(path, case=None, pages=None):
    """One truth-file case, with the questions left for a human to write."""
    document = page_lines.open_document(path)
    try:
        page_indices = parse_pages(pages, document.page_count)
        counts = page_lines.page_character_counts(document, page_indices)
    finally:
        document.close()
    scanned = page_lines.scanned_pages(counts)
    return {
        case or pathlib.Path(path).stem: {
            "pdf": pathlib.Path(path).name,
            "label": "TODO: board, paper and year, in a human's words",
            "pages": page_indices,
            "note": f"TODO. Boundary rule: {BOUNDARY_RULE}.",
            "characters_per_page": {str(k): v for k, v in counts.items()},
            "pages_without_text_layer": scanned,
            "questions": [],
            "_how_to_fill_in": [
                'One entry per question: {"n": 1, "pages": [1]}.',
                f"Page indices are zero-based. {BOUNDARY_RULE.capitalize()}.",
                "A question spanning a break lists every page: [11, 12].",
                "Delete this key when the labelling is done.",
            ],
        }
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf")
    parser.add_argument("--case", help="name for this paper in the truth file")
    parser.add_argument("--pages", help='e.g. "1-12" or "1,3,5"')
    arguments = parser.parse_args()
    print(
        json.dumps(skeleton(arguments.pdf, arguments.case, arguments.pages), indent=1)
    )


if __name__ == "__main__":
    main()
