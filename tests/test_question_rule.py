"""Tests for the one rule both rungs run, on synthetic lines only.

Nothing here opens a PDF or quotes an exam question. The point is to pin the
behaviours the rule was chosen for, so a later change that breaks one of them
fails here rather than in a silently mis-cut corpus.
"""

import unittest

from question_rule import (OCR_TOLERANCE, TEXT_LAYER_TOLERANCE, is_furniture,
                           question_column, repeated_texts, segment_lines,
                           skewed_column)

PAGE_HEIGHT = 842.0
COLUMN = 50.0
INDENT = 90.0


def row(text, y0, x0=COLUMN, page=0, size=11.0, page_height=PAGE_HEIGHT):
    return {"x0": x0, "y0": y0, "x1": x0 + 300.0, "y1": y0 + 12.0,
            "size": size, "text": text, "page": page, "page_height": page_height}


def segment(rows, page_count=1, tolerance=TEXT_LAYER_TOLERANCE, skew_aware=False):
    return segment_lines(rows, page_count, top_margin=50.0,
                         tolerance=tolerance, skew_aware=skew_aware)


class TheColumnGuard(unittest.TestCase):
    """A permissive marker pattern is made safe by where the line starts."""

    def test_a_measurement_inside_a_question_does_not_open_one(self):
        # "12 cm" matches the marker pattern as readily as question 12 does.
        # It is indented, so the column guard is the only thing between it and
        # a spurious boundary.
        rows = [row("1 A circle has radius r.", 100),
                row("12 cm is the radius here.", 130, x0=INDENT),
                row("2 A second question.", 200),
                row("3 A third question.", 300)]
        questions, column = segment(rows)
        self.assertEqual(column, COLUMN)
        self.assertEqual([q["number"] for q in questions], [1, 2, 3])

    def test_the_column_is_learnt_not_assumed(self):
        # A school with its own margins is handled without configuration: the
        # column is the smallest x0 at least three body lines share.
        rows = [row("1 First.", 100, x0=123.0),
                row("2 Second.", 200, x0=123.0),
                row("3 Third.", 300, x0=123.0),
                row("continuation", 320, x0=160.0)]
        questions, column = segment(rows)
        self.assertEqual(column, 123.0)
        self.assertEqual([q["number"] for q in questions], [1, 2, 3])

    def test_a_column_needs_three_lines_to_be_learnt(self):
        rows = [row("only line", 100, x0=10.0), row("another", 200, x0=20.0)]
        questions, column = segment(rows)
        self.assertIsNone(column)
        self.assertEqual(questions, [])


class TheSequenceGuard(unittest.TestCase):
    """The numbers must ascend, one at a time, from one."""

    def test_a_stray_number_in_the_column_is_refused(self):
        rows = [row("1 First question.", 100),
                row("7 stray number in the margin", 150),
                row("2 Second question.", 200)]
        questions, _column = segment(rows)
        self.assertEqual([q["number"] for q in questions], [1, 2])

    def test_a_paper_that_does_not_start_at_one_finds_nothing(self):
        rows = [row("3 Third question.", 100), row("4 Fourth.", 200),
                row("5 Fifth.", 300)]
        questions, _column = segment(rows)
        self.assertEqual(questions, [])

    def test_separators_may_be_a_dot_a_bracket_a_space_or_a_tab(self):
        # SEAB separates the number from the text with a tab and Cambridge with
        # a single space. A rule insisting on a full stop finds nothing on
        # either board.
        for separator in [". ", ") ", " ", "\t"]:
            with self.subTest(separator=separator):
                rows = [row(f"1{separator}First.", 100),
                        row(f"2{separator}Second.", 200),
                        row(f"3{separator}Third.", 300)]
                questions, _column = segment(rows)
                self.assertEqual([q["number"] for q in questions], [1, 2, 3])


class StopLines(unittest.TestCase):
    """Some lines belong to no question at all."""

    def test_a_section_header_ends_a_question_and_opens_none(self):
        rows = [row("1 First question.", 100),
                row("Section B: Probability and Statistics", 200),
                row("2 Second question.", 300)]
        questions, _column = segment(rows)
        self.assertEqual([q["number"] for q in questions], [1, 2])
        self.assertNotIn("Section B", questions[0]["text"])

    def test_end_matter_is_not_the_tail_of_the_last_question(self):
        rows = [row("1 First question.", 100),
                row("2 Second question.", 200),
                row("3 Third question.", 300),
                row("BLANK PAGE", 400, page=1),
                row("trailing furniture on the blank page", 450, page=1)]
        questions, _column = segment(rows, page_count=2)
        self.assertEqual(questions[-1]["pages"], [0])
        self.assertNotIn("trailing furniture", questions[-1]["text"])


class PageSpans(unittest.TestCase):
    """The cross-page-break test, which is the one that fails quietly."""

    def test_a_question_continuing_over_a_break_claims_both_pages(self):
        rows = [row("1 First question.", 100),
                row("2 Second question opens here.", 400),
                row("and continues onto the next page", 100, page=1),
                row("3 Third question.", 400, page=1)]
        questions, _column = segment(rows, page_count=2)
        self.assertEqual([q["pages"] for q in questions], [[0], [0, 1], [1]])

    def test_a_page_opening_no_question_continues_the_running_one(self):
        rows = [row("1 First question.", 100),
                row("more of question one", 100, page=1),
                row("still more", 100, page=2)]
        questions, _column = segment(rows, page_count=3)
        self.assertEqual(questions[0]["pages"], [0, 1, 2])

    def test_a_marker_at_the_top_of_a_page_takes_the_whole_page_top(self):
        # A tall display integral belonging to question 2 has glyphs whose top
        # edge sits above the baseline of question 2's own number. Sorted by y
        # alone they would be filed under question 1 on the previous page.
        # Cutting at the page top instead of the marker fixes it, and that one
        # line took the held-out paper from 9/11 to 11/11 exact.
        integral_top = 60.0                 # below the 50 pt header band,
        marker = 95.0                       # both inside PAGE_HEIGHT * 0.12
        rows = [row("1 First question.", 200),
                row("2 Second question.", 400),
                row("tall integral glyphs", integral_top, x0=INDENT, page=1),
                row("3 Third question.", marker, page=1)]
        questions, _column = segment(rows, page_count=2)
        self.assertEqual(questions[1]["pages"], [0])
        self.assertIn("tall integral", questions[2]["text"])


class Furniture(unittest.TestCase):
    """Anything printed by the page rather than by a question."""

    def test_the_header_and_footer_bands_are_not_question_text(self):
        header = row("9758/01/SP/25", 20.0)
        footer = row("[Turn over", PAGE_HEIGHT * 0.95)
        body = row("1 First question.", 400)
        self.assertTrue(is_furniture(header, set(), 50.0))
        self.assertTrue(is_furniture(footer, set(), 50.0))
        self.assertFalse(is_furniture(body, set(), 50.0))

    def test_dotted_answer_leaders_and_barcodes_are_dropped(self):
        leader = row("." * 40, 300)
        barcode = row("*0123456789*", 300, size=4.0)
        self.assertTrue(is_furniture(leader, set(), 50.0))
        self.assertTrue(is_furniture(barcode, set(), 50.0))

    def test_a_missing_font_size_is_not_treated_as_a_barcode(self):
        # OCR reports no font size. Guessing one would delete its every line.
        line = row("1 First question.", 300)
        line["size"] = None
        self.assertFalse(is_furniture(line, set(), 50.0))

    def test_text_on_half_the_pages_belongs_to_the_page(self):
        rows = [row("A MARGIN WARNING PRINTED ON EVERY PAGE", 300, page=page)
                for page in range(4)]
        self.assertIn("A MARGIN WARNING PRINTED ON EVERY PAGE",
                      repeated_texts(rows, 4))

    def test_a_short_repeated_string_is_exempt(self):
        # A bare "1" is a question marker on one page and a fraction numerator
        # on another, so short strings are never repeated furniture.
        rows = [row("1", 300, page=page) for page in range(4)]
        self.assertEqual(repeated_texts(rows, 4), set())


class SkewedColumn(unittest.TestCase):
    """A scanned page has no question column; it has a question diagonal."""

    def test_a_fixed_column_cannot_even_be_learnt_on_a_skewed_page(self):
        # 0.7 degrees over a 2339 px page drifts the left edge about 28 px, so
        # no three lines share an x0 and there is no column to find.
        rows = [row(f"{n} Question.", y, x0=130.0 + 0.0122 * y,
                    page_height=2339.0)
                for n, y in enumerate([200, 700, 1200, 1700], start=1)]
        self.assertIsNone(question_column(rows))

    def test_the_fit_recovers_every_line_of_a_skewed_column(self):
        rows = [row(f"{n} Question.", y, x0=130.0 + 0.0122 * y,
                    page_height=2339.0)
                for n, y in enumerate([200, 700, 1200, 1700], start=1)]
        in_column, slope = skewed_column(rows, OCR_TOLERANCE)
        self.assertAlmostEqual(slope, 0.0122, places=3)
        self.assertTrue(all(in_column(r) for r in rows))

    def test_the_fit_costs_nothing_on_a_straight_page(self):
        rows = [row(f"{n} Question.", y, x0=130.0, page_height=2339.0)
                for n, y in enumerate([200, 700, 1200, 1700], start=1)]
        in_column, slope = skewed_column(rows, OCR_TOLERANCE)
        self.assertAlmostEqual(slope, 0.0, places=4)
        self.assertTrue(all(in_column(r) for r in rows))

    def test_a_skewed_paper_segments_when_the_fit_is_used(self):
        rows = []
        for number, y in enumerate([200, 700, 1200, 1700], start=1):
            rows.append(row(f"{number} Question {number}.", y,
                            x0=130.0 + 0.0122 * y, page_height=2339.0))
        straight, _column = segment_lines(rows, 1, top_margin=140.0,
                                          tolerance=OCR_TOLERANCE,
                                          skew_aware=False)
        skewed, _column = segment_lines(rows, 1, top_margin=140.0,
                                        tolerance=OCR_TOLERANCE,
                                        skew_aware=True)
        self.assertLess(len(straight), 4)
        self.assertEqual([q["number"] for q in skewed], [1, 2, 3, 4])


class EmptyInput(unittest.TestCase):
    """Rung 1 on an image-only PDF is this case, and it must not throw."""

    def test_no_lines_gives_no_questions(self):
        self.assertEqual(segment([]), ([], None))

    def test_only_furniture_gives_no_questions(self):
        rows = [row("9758/01/SP/25", 10.0), row("[Turn over", 830.0)]
        self.assertEqual(segment(rows), ([], None))


if __name__ == "__main__":
    unittest.main()
