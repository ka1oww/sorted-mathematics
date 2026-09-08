"""Rung 2: the same rule, over lines read off the pixels.

No public exam paper is a scan, so these run against proxies manufactured from
the SEAB specimen by `tools/make_scan_proxy.py`: pages rasterised at 200 dpi and
re-wrapped as an image-only PDF, with a harsher variant rotated 0.7 degrees,
speckled and re-encoded at JPEG 60. The build recipe for each proxy lives beside
its labels in `scan_proxies_truth.json`, so a missing proxy is rebuilt rather
than guessed at.

These skip twice over: without the source paper, and without the OCR stack,
which is deliberately not a dependency of the product.
"""

import json
import pathlib
import unittest

import public_papers

from reader import read_paper_with_report
from tools.make_scan_proxy import make_proxy
from tools.score_reader import score

TRUTH_PATH = (
    pathlib.Path(__file__).resolve().parent / "fixtures" / "scan_proxies_truth.json"
)


def ocr_available():
    import page_ocr

    try:
        page_ocr._import_doctr()
    except page_ocr.OcrUnavailable:
        return False
    return True


def truth():
    return json.loads(TRUTH_PATH.read_text())


def source_available(spec):
    return (public_papers.papers_dir() / spec["build"]["source"]).exists()


def ensure_proxy(spec):
    """Build the proxy from its recipe if it is not already there."""
    destination = public_papers.papers_dir() / spec["pdf"]
    if destination.exists():
        return destination
    build = spec["build"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    make_proxy(
        public_papers.papers_dir() / build["source"],
        destination,
        pages=build["pages"],
        dpi=build["dpi"],
        skew=build["skew"],
        noise=build["noise"],
        quality=build["quality"],
    )
    return destination


class TheProxyRecipes(unittest.TestCase):
    """These need neither the papers nor the OCR stack."""

    def test_every_proxy_records_how_to_rebuild_it(self):
        for case, spec in truth().items():
            with self.subTest(case=case):
                self.assertEqual(
                    set(spec["build"]),
                    {"source", "pages", "dpi", "skew", "noise", "quality"},
                )

    def test_the_labels_are_the_source_labels_renumbered_from_zero(self):
        # A proxy holds only the pages its recipe names, so proxy page i is
        # source page build.pages[i]. Getting this wrong would score the reader
        # against the wrong pages and look like a page-span bug.
        source_truth = public_papers.truth()
        for case, spec in truth().items():
            base = case.replace("-raster", "").replace("-skewed", "")
            if base not in source_truth:
                continue
            with self.subTest(case=case):
                mapping = spec["build"]["pages"]
                expected = [
                    {"n": q["n"], "pages": [mapping.index(p) for p in q["pages"]]}
                    for q in source_truth[base]["questions"]
                ]
                self.assertEqual(spec["questions"], expected)

    def test_the_labels_carry_no_question_text(self):
        for spec in truth().values():
            for question in spec["questions"]:
                self.assertEqual(set(question), {"n", "pages"})


@unittest.skipUnless(public_papers.any_available(), public_papers.MISSING_PAPERS)
@unittest.skipUnless(ocr_available(), "the OCR stack is not installed; see CLAUDE.md")
class TheScanPath(unittest.TestCase):
    def read(self, spec):
        return read_paper_with_report(ensure_proxy(spec), strict=False)

    def test_a_rasterised_paper_is_read_from_pixels_alone(self):
        for case, spec in truth().items():
            if not source_available(spec):
                continue
            with self.subTest(case=case):
                predicted, report = self.read(spec)
                self.assertEqual(report["rung"], 2)
                found, spurious, exact = score(predicted, spec["questions"])
                total = len(spec["questions"])
                self.assertEqual(found, total)
                self.assertEqual(spurious, 0)
                self.assertEqual(exact, total)

    def test_a_paper_with_no_text_layer_chooses_the_scan_path_itself(self):
        spec = truth()["seab-p1-raster"]
        if not source_available(spec):
            self.skipTest(public_papers.MISSING_PAPERS)
        _predicted, report = self.read(spec)
        self.assertEqual(report["pages_without_text_layer"], report["pages_read"])

    def test_rung_one_on_the_same_paper_finds_nothing(self):
        # 0 characters, 0 questions. This is the whole reason rung 2 exists.
        spec = truth()["seab-p1-raster"]
        if not source_available(spec):
            self.skipTest(public_papers.MISSING_PAPERS)
        predicted, report = read_paper_with_report(
            ensure_proxy(spec), rung=1, strict=False
        )
        self.assertEqual(predicted, [])
        self.assertFalse(report["trustworthy"])

    def test_the_skew_fit_measures_the_rotation_that_was_injected(self):
        spec = truth()["seab-p1-skewed"]
        if not source_available(spec):
            self.skipTest(public_papers.MISSING_PAPERS)
        _predicted, report = self.read(spec)
        # 0.7 degrees is a slope of tan(0.7) = 0.0122 px per px
        slope = float(report["question_column"].split()[-3])
        self.assertAlmostEqual(slope, 0.0122, delta=0.004)

    def test_the_fit_costs_nothing_on_a_straight_page(self):
        spec = truth()["seab-p1-raster"]
        if not source_available(spec):
            self.skipTest(public_papers.MISSING_PAPERS)
        _predicted, report = self.read(spec)
        slope = float(report["question_column"].split()[-3])
        self.assertAlmostEqual(slope, 0.0, delta=0.001)

    def test_a_scanned_paper_is_trusted_by_the_confidence_signals(self):
        for case, spec in truth().items():
            if not source_available(spec):
                continue
            with self.subTest(case=case):
                _predicted, report = self.read(spec)
                self.assertEqual(report["problems"], [])


if __name__ == "__main__":
    unittest.main()
