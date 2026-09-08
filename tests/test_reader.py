"""Tests for the entry point: which rung runs, and what stops a bad read.

The PDFs here are built on the spot from lorem-style text, so these run
anywhere and quote nothing. A mis-cut paper emitted quietly poisons the corpus,
so the confidence signals are the most important behaviour in this file.
"""

import unittest

import pymupdf

import page_lines
from reader import (
    MAXIMUM_PAGES_PER_QUESTION,
    PaperReadError,
    Question,
    choose_rung,
    confidence_report,
    read_paper,
    read_paper_with_report,
)

# Deliberately not exam phrasing. These build a synthetic paper whose shape the
# rule can read; nothing here is, or resembles, a real question.
QUESTIONS = [
    "Alpha bravo charlie delta echo foxtrot golf hotel india juliet.",
    "Kilo lima mike november oscar papa quebec romeo sierra tango.",
    "Uniform victor whiskey xray yankee zulu alpha bravo charlie.",
]


def write_paper(path, questions=QUESTIONS, per_page=1):
    """A synthetic exam paper: numbers in the left column, text indented."""
    document = pymupdf.open()
    page = None
    for index, text in enumerate(questions):
        if index % per_page == 0:
            page = document.new_page()
            y = 120.0
        page.insert_text((50.0, y), f"{index + 1}", fontsize=11)
        page.insert_text((90.0, y), text, fontsize=11)
        page.insert_text(
            (90.0, y + 20.0), "Delta echo foxtrot golf hotel.", fontsize=11
        )
        y += 120.0
    document.save(str(path))
    document.close()
    return path


def write_image_only(source, destination):
    """The same paper with its text layer thrown away: what a scan looks like."""
    original = pymupdf.open(source)
    scanned = pymupdf.open()
    for index in range(original.page_count):
        pixmap = original[index].get_pixmap(dpi=120)
        page = scanned.new_page(
            width=original[index].rect.width, height=original[index].rect.height
        )
        page.insert_image(page.rect, pixmap=pixmap)
    scanned.save(str(destination))
    scanned.close()
    original.close()
    return destination


def fake_questions(*specs):
    return [
        Question(
            {
                "number": number,
                "start_page": pages[0],
                "pages": list(pages),
                "text": "some text",
            }
        )
        for number, pages in specs
    ]


class RungSelection(unittest.TestCase):
    """Rung 1 is inert on an image-only PDF, so the choice is what makes the
    reader work at all on a scan."""

    def test_a_typeset_paper_takes_the_text_layer(self):
        self.assertEqual(choose_rung({0: 2091, 1: 1451, 2: 1441}), 1)

    def test_a_paper_with_no_text_anywhere_takes_the_scan_path(self):
        self.assertEqual(choose_rung({0: 0, 1: 0, 2: 0}), 2)

    def test_one_blank_page_does_not_send_a_digital_paper_to_ocr(self):
        # A digital paper may legitimately carry a blank or answer-space page.
        # OCRing the whole paper because of it would be a waste, and mixing
        # points with pixels in one document would find no column at all.
        self.assertEqual(choose_rung({0: 2091, 1: 0, 2: 1441}), 1)

    def test_a_mostly_scanned_paper_takes_the_scan_path(self):
        self.assertEqual(choose_rung({0: 0, 1: 0, 2: 1441}), 2)

    def test_the_floor_sits_far_below_a_real_page(self):
        # Measured on the three public papers: the emptiest real page carries
        # 344 characters and a rasterised page carries 0, so the classification
        # of any individual page is never in doubt.
        self.assertLess(page_lines.SCAN_CHARACTER_FLOOR, 344)
        self.assertGreater(page_lines.SCAN_CHARACTER_FLOOR, 0)


class ScanDetection(unittest.TestCase):
    def setUp(self):
        import pathlib
        import tempfile

        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = pathlib.Path(self.directory.name)

    def test_a_typeset_paper_is_read_by_the_text_layer(self):
        path = write_paper(self.root / "paper.pdf")
        questions, report = read_paper_with_report(path)
        self.assertEqual(report["rung"], 1)
        self.assertEqual([q.number for q in questions], [1, 2, 3])
        self.assertEqual(report["pages_without_text_layer"], [])

    def test_an_image_only_paper_is_recognised_as_a_scan(self):
        path = write_paper(self.root / "paper.pdf")
        scanned = write_image_only(path, self.root / "scanned.pdf")
        document = page_lines.open_document(scanned)
        counts = page_lines.page_character_counts(
            document, page_lines.all_pages(document)
        )
        document.close()
        self.assertEqual(page_lines.scanned_pages(counts), sorted(counts))
        self.assertEqual(choose_rung(counts), 2)

    def test_rung_one_forced_onto_a_scan_finds_nothing_and_says_so(self):
        # The whole reason rung 2 exists: 0 characters, 0 questions.
        path = write_paper(self.root / "paper.pdf")
        scanned = write_image_only(path, self.root / "scanned.pdf")
        with self.assertRaises(PaperReadError) as raised:
            read_paper(scanned, rung=1)
        self.assertIn("no questions found", raised.exception.report["problems"])

    def test_a_missing_ocr_stack_names_what_to_install(self):
        import page_ocr

        try:
            page_ocr._import_doctr()
        except page_ocr.OcrUnavailable as error:
            self.assertIn("python-doctr", str(error))
        # installed here: nothing to assert, the import simply worked

    def test_an_unknown_rung_is_refused(self):
        path = write_paper(self.root / "paper.pdf")
        with self.assertRaises(ValueError):
            read_paper(path, rung=3)


class ConfidenceSignals(unittest.TestCase):
    """Fail loudly rather than quietly emit a mis-cut paper."""

    def test_a_complete_ascending_sequence_is_trusted(self):
        report = confidence_report(
            fake_questions((1, [0]), (2, [1]), (3, [2])),
            1,
            [0, 1, 2],
            49.6,
            {0: 900, 1: 900, 2: 900},
        )
        self.assertTrue(report["trustworthy"])
        self.assertEqual(report["problems"], [])

    def test_a_gap_in_the_sequence_is_a_problem(self):
        report = confidence_report(
            fake_questions((1, [0]), (2, [1]), (4, [3])), 1, [0, 1, 2, 3], 49.6, {}
        )
        self.assertFalse(report["trustworthy"])
        self.assertEqual(report["missing_numbers"], [3])

    def test_a_question_claiming_too_many_pages_is_a_problem(self):
        swallowed = list(range(MAXIMUM_PAGES_PER_QUESTION + 1))
        report = confidence_report(
            fake_questions((1, swallowed)), 1, swallowed, 49.6, {}
        )
        self.assertFalse(report["trustworthy"])
        self.assertEqual(report["oversized_questions"][0]["number"], 1)

    def test_a_question_at_the_page_limit_is_allowed(self):
        allowed = list(range(MAXIMUM_PAGES_PER_QUESTION))
        report = confidence_report(fake_questions((1, allowed)), 1, allowed, 49.6, {})
        self.assertTrue(report["trustworthy"])

    def test_a_paper_with_no_questions_is_a_problem(self):
        report = confidence_report([], 1, [0], None, {0: 0})
        self.assertFalse(report["trustworthy"])
        self.assertIn("no questions found", report["problems"])

    def test_a_question_with_no_text_is_a_problem(self):
        empty = [Question({"number": 1, "start_page": 0, "pages": [0], "text": "   "})]
        report = confidence_report(empty, 1, [0], 49.6, {})
        self.assertFalse(report["trustworthy"])
        self.assertEqual(report["empty_questions"], [1])

    def test_the_report_names_which_signal_failed(self):
        report = confidence_report(
            fake_questions((1, [0]), (3, [2])), 1, [0, 1, 2], 49.6, {}
        )
        self.assertEqual(len(report["problems"]), 1)
        self.assertIn("sequence incomplete", report["problems"][0])


def write_prose(path):
    """A page carrying a text layer but no question numbers: readable, and
    nothing a reader should be willing to cut into questions."""
    document = pymupdf.open()
    page = document.new_page()
    for offset in range(12):
        page.insert_text(
            (50.0, 120.0 + offset * 20.0),
            "Lorem ipsum with no numbering anywhere.",
            fontsize=11,
        )
    document.save(str(path))
    document.close()
    return path


class LoudFailure(unittest.TestCase):
    def setUp(self):
        import pathlib
        import tempfile

        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = pathlib.Path(self.directory.name)

    def test_an_unreadable_paper_raises_by_default(self):
        path = write_prose(self.root / "prose.pdf")
        with self.assertRaises(PaperReadError):
            read_paper(path)

    def test_the_exception_carries_the_report(self):
        path = write_prose(self.root / "prose.pdf")
        with self.assertRaises(PaperReadError) as raised:
            read_paper(path)
        self.assertFalse(raised.exception.report["trustworthy"])
        self.assertEqual(raised.exception.report["questions"], 0)

    def test_strict_off_returns_the_doubtful_read_for_inspection(self):
        path = write_prose(self.root / "prose.pdf")
        questions, report = read_paper_with_report(path, strict=False)
        self.assertEqual(questions, [])
        self.assertFalse(report["trustworthy"])

    def test_a_paper_that_only_half_reads_still_raises(self):
        # The dangerous case: enough questions found to look like a success.
        path = write_paper(self.root / "paper.pdf", QUESTIONS, per_page=1)
        _questions, report = read_paper_with_report(path, strict=False)
        self.assertTrue(report["trustworthy"])
        with self.assertRaises(PaperReadError):
            read_paper(path, pages=[1, 2])  # starts at question 2, so 1 is gone


class TheQuestionShape(unittest.TestCase):
    def test_a_question_reads_as_attributes_and_as_a_dict(self):
        question = Question(
            {"number": 7, "start_page": 3, "pages": [3, 4], "text": "some text"}
        )
        self.assertEqual(question.number, 7)
        self.assertEqual(question.pages, [3, 4])
        self.assertEqual(question["pages"], [3, 4])
        self.assertEqual(question.start_page, 3)

    def test_a_question_survives_json(self):
        import json

        question = Question(
            {"number": 7, "start_page": 3, "pages": [3, 4], "text": "some text"}
        )
        self.assertEqual(json.loads(json.dumps(question))["number"], 7)


if __name__ == "__main__":
    unittest.main()
