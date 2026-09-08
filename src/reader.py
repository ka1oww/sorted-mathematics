"""`read_paper()`: a PDF goes in, questions with their exact page spans come out.

Both rungs sit behind this one entry point.

  rung 1  the PDF text layer plus the plain rule in `question_rule`. Free,
          about 5 ms per page, and the rule that scored 43/43 questions with
          43/43 exact page sets over four papers from two boards.
  rung 2  the same rule over lines OCR'd from the pixels, for papers mostly
          without text layers, at about half a second per page.

The rung is chosen automatically from how much text the pages carry, and it is
chosen per *paper* rather than per page: rung 1 works in points and rung 2 in
pixels, so a paper whose lines came from both sources would have no single
question column and the rule would find nothing.

A mis-cut paper that is emitted quietly poisons the corpus downstream, so the
confidence signals here raise rather than warn. `read_paper` fails loudly by
default; a caller that wants to inspect a doubtful paper asks for the report.
"""

import page_lines
import question_rule

# A question spanning more than this many pages is not a question, it is a
# missed boundary swallowing the rest of the paper.
MAXIMUM_PAGES_PER_QUESTION = 4

# A paper takes the scan path when more than this fraction of its pages have no
# text layer. Deciding per paper keeps every line in one coordinate system; the
# threshold is a majority because a digital paper may legitimately carry one
# blank or answer-space page.
SCAN_PAPER_FRACTION = 0.5


class PaperReadError(RuntimeError):
    """The paper could not be cut into questions we are willing to stand behind.

    Carries the `report` so a caller can see which signal failed.
    """

    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


class Question(dict):
    """A question and the pages it spans.

    A dict subclass so it stays JSON-serialisable and every existing caller can
    keep subscripting it, with the four fields named as attributes for reading
    ease: `number`, `start_page`, `pages`, `text`.
    """

    @property
    def number(self):
        return self["number"]

    @property
    def start_page(self):
        return self["start_page"]

    @property
    def pages(self):
        return self["pages"]

    @property
    def text(self):
        return self["text"]

    def __repr__(self):
        return (
            f"Question(number={self.number}, pages={self.pages}, "
            f"chars={len(self.text)})"
        )


def choose_rung(character_counts):
    """1 for the text layer, 2 for OCR. See SCAN_PAPER_FRACTION."""
    if not character_counts:
        return 1
    scanned = page_lines.scanned_pages(character_counts)
    return 2 if len(scanned) > SCAN_PAPER_FRACTION * len(character_counts) else 1


def confidence_report(questions, rung, page_indices, column, character_counts):
    """The two signals that catch a mis-cut paper, plus what produced it.

    sequence completeness  are all of 1..N present, with no gap and no repeat?
    page-set sanity        does any question claim an implausible span?
    """
    numbers = [question["number"] for question in questions]
    highest = max(numbers) if numbers else 0
    missing = [n for n in range(1, highest + 1) if n not in numbers]
    duplicated = sorted({n for n in numbers if numbers.count(n) > 1})
    oversized = [
        {"number": question["number"], "pages": question["pages"]}
        for question in questions
        if len(question["pages"]) > MAXIMUM_PAGES_PER_QUESTION
    ]
    empty = [
        question["number"] for question in questions if not question["text"].strip()
    ]

    problems = []
    if not questions:
        problems.append("no questions found")
    if missing:
        problems.append(f"sequence incomplete: 1..{highest} is missing {missing}")
    if duplicated:
        problems.append(f"question numbers repeat: {duplicated}")
    for question in oversized:
        problems.append(
            f"question {question['number']} claims "
            f"{len(question['pages'])} pages {question['pages']}, "
            f"more than the {MAXIMUM_PAGES_PER_QUESTION} allowed"
        )
    if empty:
        problems.append(f"questions with no text: {empty}")

    scanned = page_lines.scanned_pages(character_counts)
    return {
        "rung": rung,
        "questions": len(questions),
        "highest_number": highest,
        "missing_numbers": missing,
        "duplicated_numbers": duplicated,
        "oversized_questions": oversized,
        "empty_questions": empty,
        "pages_read": list(page_indices),
        "pages_without_text_layer": scanned,
        "question_column": column,
        "problems": problems,
        "trustworthy": not problems,
    }


def read_paper(path, pages=None, rung=None, strict=True):
    """Cut one exam paper into questions.

    Returns a list of `Question`, each carrying its question number and the
    exact page indices it spans. Page indices are zero-based, as PyMuPDF counts
    them.

    `pages` limits the read to part of the document, for a PDF holding several
    papers back to back. `rung` forces 1 or 2 instead of choosing. `strict`
    left true raises `PaperReadError` when a confidence signal fails; set it
    false only to inspect a paper that is already known to be doubtful.
    """
    questions, _report = read_paper_with_report(path, pages, rung, strict)
    return questions


def read_paper_with_report(path, pages=None, rung=None, strict=True):
    """`read_paper`, and the confidence report beside the questions."""
    document = page_lines.open_document(path)
    try:
        page_indices = (
            list(pages) if pages is not None else page_lines.all_pages(document)
        )
        counts = page_lines.page_character_counts(document, page_indices)
        chosen = rung if rung is not None else choose_rung(counts)

        if chosen == 1:
            rows = page_lines.text_layer_lines(document, page_indices)
            top_margin = page_lines.TEXT_LAYER_TOP_MARGIN
            tolerance = question_rule.TEXT_LAYER_TOLERANCE
            skew_aware = False
        elif chosen == 2:
            import page_ocr

            rows = page_ocr.ocr_lines(document, page_indices)
            heights = {row["page_height"] for row in rows}
            top_margin = page_ocr.ocr_top_margin(max(heights)) if heights else 0.0
            tolerance = question_rule.OCR_TOLERANCE
            # free on a straight page, and the difference between 3/11 and
            # 11/11 on a scan a fraction of a degree off square
            skew_aware = True
        else:
            raise ValueError(f"there is no rung {chosen}; the rungs are 1 and 2")
    finally:
        document.close()

    found, column = question_rule.segment_lines(
        rows, len(page_indices), top_margin, tolerance, skew_aware
    )
    questions = [Question(question) for question in found]

    report = confidence_report(questions, chosen, page_indices, column, counts)
    if strict and not report["trustworthy"]:
        raise PaperReadError(
            f"{path}: refusing to emit a paper this reader does not trust - "
            + "; ".join(report["problems"]),
            report,
        )
    return questions, report
