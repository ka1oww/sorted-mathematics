"""Regression against the three public exam papers and their 43 labelled
questions, including the six that run across a page break.

The measured baseline this locks in is 43/43 questions found, 0 spurious
boundaries and 43/43 exact page sets, over four papers from two examination
boards. The three numbers are asserted separately and never averaged: a reader
that finds every question but mis-sets the pages looks excellent and cuts the
corpus wrongly on every page break.

The PDFs are not committed - see `public_papers.py` - so these skip when they
are absent.
"""

import pathlib
import unittest

import public_papers
from reader import read_paper_with_report
from tools.score_reader import resolve, score

EXPECTED_TOTAL = 43
EXPECTED_PAGE_SPANNING = 6


class TheTruthSet(unittest.TestCase):
    """The labels are the specification, so their shape is worth pinning."""

    def setUp(self):
        self.truth = public_papers.truth()

    def test_it_holds_the_labelled_questions_the_baseline_was_measured_on(self):
        total = sum(len(spec["questions"]) for spec in self.truth.values())
        self.assertEqual(total, EXPECTED_TOTAL)

    def test_it_holds_the_page_spanning_questions(self):
        spanning = [question for spec in self.truth.values()
                    for question in spec["questions"]
                    if len(question["pages"]) > 1]
        self.assertEqual(len(spanning), EXPECTED_PAGE_SPANNING)

    def test_every_page_a_question_claims_was_actually_read(self):
        for case, spec in self.truth.items():
            with self.subTest(case=case):
                read = set(spec["pages"])
                for question in spec["questions"]:
                    self.assertTrue(set(question["pages"]) <= read)

    def test_it_carries_no_question_text(self):
        # The standing rule for this repository: no question text in a
        # committed file. The labels are numbers and page indices only.
        for spec in self.truth.values():
            for question in spec["questions"]:
                self.assertEqual(set(question), {"n", "pages"})


class TheScorer(unittest.TestCase):
    """The measuring stick itself. Its three numbers fail differently, and a
    scorer that quietly conflates them would hide exactly what it exists to
    show."""

    @staticmethod
    def predicted(*specs):
        return [{"number": number, "start_page": pages[0], "pages": list(pages)}
                for number, pages in specs]

    @staticmethod
    def expected(*specs):
        return [{"n": number, "pages": list(pages)} for number, pages in specs]

    def test_a_perfect_read_scores_full_marks_with_nothing_spurious(self):
        self.assertEqual(
            score(self.predicted((1, [0]), (2, [1, 2])),
                  self.expected((1, [0]), (2, [1, 2]))),
            (2, 0, 2))

    def test_a_question_found_on_the_wrong_start_page_is_not_found(self):
        self.assertEqual(
            score(self.predicted((1, [5])), self.expected((1, [0]))),
            (0, 1, 0))

    def test_a_missed_page_span_still_counts_as_found(self):
        # The failure a single accuracy number hides: every question found,
        # every page break cut wrongly.
        found, spurious, exact = score(self.predicted((1, [0])),
                                       self.expected((1, [0, 1])))
        self.assertEqual((found, exact), (1, 0))
        self.assertEqual(spurious, 0)

    def test_an_invented_boundary_is_spurious(self):
        # The labelled question is still found and still exact; the invented
        # one is counted separately, which is why spurious is its own number
        # rather than a deduction from the other two.
        self.assertEqual(
            score(self.predicted((1, [0]), (2, [1])), self.expected((1, [0]))),
            (1, 1, 1))

    def test_a_truth_file_may_not_name_a_path_outside_the_papers_directory(self):
        for escape in ["/etc/passwd", "../../secret.pdf"]:
            with self.subTest(path=escape):
                with self.assertRaises(ValueError):
                    resolve({"pdf": escape}, "tests/fixtures/papers")

    def test_a_truth_file_may_name_a_subdirectory(self):
        # The scan proxies live in one.
        self.assertEqual(resolve({"pdf": "proxies/a.pdf"}, "papers"),
                         pathlib.Path("papers/proxies/a.pdf"))


@unittest.skipUnless(public_papers.any_available(), public_papers.MISSING_PAPERS)
class TheMeasuredBaseline(unittest.TestCase):

    def setUp(self):
        self.truth = public_papers.truth()

    def read(self, spec):
        return read_paper_with_report(public_papers.paper_path(spec),
                                      spec["pages"], strict=False)

    def test_every_paper_scores_exactly_what_it_was_measured_at(self):
        for case, spec in self.truth.items():
            if not public_papers.available(spec):
                continue
            with self.subTest(case=case):
                predicted, _report = self.read(spec)
                found, spurious, exact = score(predicted, spec["questions"])
                total = len(spec["questions"])
                self.assertEqual(found, total, "found the wrong questions")
                self.assertEqual(spurious, 0, "invented a boundary")
                self.assertEqual(exact, total, "mis-set a page span")

    def test_the_six_page_spanning_questions_claim_both_their_pages(self):
        # The failure that hides behind a single accuracy number.
        checked = 0
        expected = 0
        for case, spec in self.truth.items():
            if not public_papers.available(spec):
                continue
            expected += sum(len(want["pages"]) > 1
                            for want in spec["questions"])
            predicted, _report = self.read(spec)
            by_number = {q["number"]: q for q in predicted}
            for want in spec["questions"]:
                if len(want["pages"]) == 1:
                    continue
                with self.subTest(case=case, question=want["n"]):
                    self.assertEqual(by_number[want["n"]]["pages"],
                                     want["pages"])
                    checked += 1
        self.assertEqual(checked, expected)

    def test_every_paper_takes_the_text_layer(self):
        for case, spec in self.truth.items():
            if not public_papers.available(spec):
                continue
            with self.subTest(case=case):
                _predicted, report = self.read(spec)
                self.assertEqual(report["rung"], 1)
                self.assertEqual(report["pages_without_text_layer"], [])

    def test_the_question_column_is_learnt_at_the_same_place_everywhere(self):
        # Two boards, four papers, one learnt column. If this moves, the
        # document the rule learnt from is not the document it was measured on.
        columns = set()
        for spec in self.truth.values():
            if not public_papers.available(spec):
                continue
            _predicted, report = self.read(spec)
            columns.add(report["question_column"])
        self.assertEqual(columns, {49.6})

    def test_every_paper_is_trusted_by_the_confidence_signals(self):
        for case, spec in self.truth.items():
            if not public_papers.available(spec):
                continue
            with self.subTest(case=case):
                _predicted, report = self.read(spec)
                self.assertEqual(report["problems"], [])
                self.assertTrue(report["trustworthy"])


@unittest.skipUnless(public_papers.any_available(), public_papers.MISSING_PAPERS)
class TheHeldOutPaper(unittest.TestCase):
    """CIE 9709/32 was labelled only after the rule was frozen. It is scored,
    never tuned against."""

    def setUp(self):
        self.truth = public_papers.truth()
        self.held_out = {case: spec for case, spec in self.truth.items()
                         if spec.get("held_out")}

    def test_exactly_one_paper_is_held_out(self):
        self.assertEqual(len(self.held_out), 1)

    def test_it_scores_full_marks_on_all_three_numbers(self):
        for case, spec in self.held_out.items():
            if not public_papers.available(spec):
                self.skipTest(public_papers.MISSING_PAPERS)
            with self.subTest(case=case):
                predicted, _report = read_paper_with_report(
                    public_papers.paper_path(spec), spec["pages"], strict=False)
                found, spurious, exact = score(predicted, spec["questions"])
                self.assertEqual((found, spurious, exact),
                                 (len(spec["questions"]), 0,
                                  len(spec["questions"])))


if __name__ == "__main__":
    unittest.main()
