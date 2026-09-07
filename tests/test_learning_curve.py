"""Tests for the learning curve's subsampler in src/learning_curve.py.

Two properties carry the whole measurement, and both fail silently if broken.
A subsample that splits a paper leaves the model that paper's house style at a
fraction of the row count, so the curve measures something other than how much
data was given. A subsample that touches the test split makes the points
incomparable to each other and to RESULTS.md, and the accuracy moves for a
reason the report cannot see.

Fixtures only, in the shape DATA-PLAN.md documents, and no scikit-learn: these
run on a machine with nothing but pytest, as the GitHub Actions workflow has.
"""

import random
import tempfile
import unittest
from pathlib import Path

import learning_curve
import runs
from chapters import CHAPTER_SLUGS
from paths import PROJECT_ROOT
from split import leakage_group_key

try:
    from tools import report_learning_curve
except ModuleNotFoundError:
    report_learning_curve = None

FIXTURE_SEED = 20260907
SCHOOLS = ("ABCJC", "BCDJC", "CDEJC", "DEFJC", "EFGJC", "FGHJC", "GHIJC")
YEARS = (2021, 2022, 2023, 2024)


def a_row(row_id, chapter, school=None, year=None, paper=None, source_file=None):
    return {"id": row_id, "chapter": chapter,
            "text": f"synthetic question about {chapter}",
            "school": school, "year": year, "paper": paper,
            "source_file": source_file}


def papers(count, rows_each, prefix):
    """Whole papers, each mixing several chapters, as the corpus's do."""
    generator = random.Random(FIXTURE_SEED)
    rows = []
    for index in range(count):
        school = SCHOOLS[index % len(SCHOOLS)]
        year = YEARS[index % len(YEARS)]
        paper = f"{prefix}{index}"
        for question in range(rows_each):
            rows.append(a_row(f"{prefix}:{index}:{question}",
                              generator.choice(CHAPTER_SLUGS),
                              school=school, year=year, paper=paper))
    return rows


def a_split():
    return {"train": papers(30, 8, "train"),
            "val": papers(10, 6, "val"),
            "test": papers(10, 6, "test")}


def group_sizes(rows):
    sizes = {}
    for row in rows:
        key = leakage_group_key(row)
        sizes[key] = sizes.get(key, 0) + 1
    return sizes


class SubsampleTests(unittest.TestCase):

    def setUp(self):
        self.split_rows = a_split()

    def reduce(self, fraction, seed=42):
        return learning_curve.subsample_training_pool(
            self.split_rows, fraction, seed)

    def test_a_paper_is_kept_whole_or_dropped_whole(self):
        for fraction in (0.25, 0.5, 0.75):
            reduced = self.reduce(fraction)
            for split_name in ("train", "val"):
                whole = group_sizes(self.split_rows[split_name])
                kept = group_sizes(reduced[split_name])
                for key, count in kept.items():
                    self.assertEqual(count, whole[key],
                                     f"group {key} was split at {fraction}")

    def test_the_test_split_never_moves(self):
        for fraction in (0.25, 0.5, 0.75, 1.0):
            reduced = self.reduce(fraction)
            self.assertEqual([row["id"] for row in reduced["test"]],
                             [row["id"] for row in self.split_rows["test"]])

    def test_the_kept_rows_are_a_subset_of_the_split_they_came_from(self):
        reduced = self.reduce(0.5)
        for split_name in ("train", "val"):
            original = {row["id"] for row in self.split_rows[split_name]}
            kept = [row["id"] for row in reduced[split_name]]
            self.assertEqual(len(kept), len(set(kept)))
            self.assertTrue(set(kept) <= original)

    def test_row_order_within_a_split_is_preserved(self):
        # The transformer pairs questions with labels positionally, and the
        # report reads the rows back in file order.
        reduced = self.reduce(0.5)
        for split_name in ("train", "val"):
            kept = [row["id"] for row in reduced[split_name]]
            order = [row["id"] for row in self.split_rows[split_name]
                     if row["id"] in set(kept)]
            self.assertEqual(kept, order)

    def test_the_fraction_is_met_without_overshooting_by_more_than_one_group(self):
        for fraction in (0.25, 0.5, 0.75):
            reduced = self.reduce(fraction)
            for split_name in ("train", "val"):
                total = len(self.split_rows[split_name])
                kept = len(reduced[split_name])
                target = fraction * total
                largest = max(group_sizes(self.split_rows[split_name]).values())
                self.assertGreaterEqual(kept, target - 1)
                self.assertLessEqual(kept, target + largest)

    def test_more_data_asked_for_is_more_data_given(self):
        counts = [learning_curve.pool_rows(self.reduce(fraction))
                  for fraction in (0.25, 0.5, 0.75, 1.0)]
        self.assertEqual(counts, sorted(counts))
        self.assertEqual(counts[-1],
                         len(self.split_rows["train"]) + len(self.split_rows["val"]))

    def test_the_same_seed_gives_the_same_subsample(self):
        first = self.reduce(0.5, seed=7)
        second = self.reduce(0.5, seed=7)
        self.assertEqual([row["id"] for row in first["train"]],
                         [row["id"] for row in second["train"]])

    def test_the_order_the_rows_arrive_in_does_not_change_the_subsample(self):
        # Groups are sorted before the shuffle, so a differently ordered file
        # of the same rows gives the same subsample at the same seed.
        shuffled = dict(self.split_rows)
        shuffled["train"] = list(self.split_rows["train"])
        random.Random(1).shuffle(shuffled["train"])
        expected = {row["id"] for row in self.reduce(0.5)["train"]}
        reduced = learning_curve.subsample_training_pool(shuffled, 0.5, 42)
        self.assertEqual({row["id"] for row in reduced["train"]}, expected)

    def test_another_seed_gives_another_subsample(self):
        first = {row["id"] for row in self.reduce(0.5, seed=42)["train"]}
        second = {row["id"] for row in self.reduce(0.5, seed=43)["train"]}
        self.assertNotEqual(first, second)

    def test_the_full_fraction_is_the_whole_split(self):
        reduced = self.reduce(1.0)
        for split_name in ("train", "val", "test"):
            self.assertEqual([row["id"] for row in reduced[split_name]],
                             [row["id"] for row in self.split_rows[split_name]])

    def test_a_fraction_outside_zero_to_one_is_refused(self):
        for fraction in (0.0, -0.5, 1.5):
            with self.assertRaises(learning_curve.SubsampleError):
                self.reduce(fraction)

    def test_a_fraction_that_empties_the_validation_split_is_refused(self):
        starved = dict(self.split_rows, val=[])
        with self.assertRaises(learning_curve.SubsampleError):
            learning_curve.subsample_training_pool(starved, 0.5, 42)


class CheckTests(unittest.TestCase):
    """The checks fire on what they are for, rather than being assumed to."""

    def setUp(self):
        self.split_rows = a_split()

    def test_a_moved_test_row_is_caught(self):
        reduced = learning_curve.subsample_training_pool(self.split_rows, 0.5, 42)
        reduced["test"] = reduced["test"][1:]
        with self.assertRaises(learning_curve.SubsampleError):
            learning_curve.check_subsample(self.split_rows, reduced, 0.5)

    def test_a_half_kept_paper_is_caught(self):
        reduced = learning_curve.subsample_training_pool(self.split_rows, 0.5, 42)
        reduced["train"] = reduced["train"][1:]
        with self.assertRaises(learning_curve.SubsampleError):
            learning_curve.check_subsample(self.split_rows, reduced, 0.5)

    def test_an_invented_row_is_caught(self):
        reduced = learning_curve.subsample_training_pool(self.split_rows, 0.5, 42)
        reduced["train"] = reduced["train"] + [a_row("invented", CHAPTER_SLUGS[0],
                                                     source_file="nowhere")]
        with self.assertRaises(learning_curve.SubsampleError):
            learning_curve.check_subsample(self.split_rows, reduced, 0.5)


class LedgerFractionTests(unittest.TestCase):
    """A fractional run must not be able to reach the headline comparison."""

    CURRENT = "cccccccccccccccc"

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=PROJECT_ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.runs_dir = Path(self.temporary.name) / "runs"

    def a_run(self, fraction=learning_curve.FULL, seed=42):
        return runs.Run(
            approach=runs.APPROACH_ONE, seed=seed, predicted=["apgp"],
            test_hash=self.CURRENT, test_rows=1, train_seconds=1.0,
            parameters=10, hardware="a laptop",
            recorded="2026-09-07T00:00:00+00:00", fraction=fraction,
            train_rows=int(1205 * fraction),
        )

    def load(self, **kwargs):
        return runs.load_runs(runs.APPROACH_ONE, test_hash=self.CURRENT,
                              runs_dir=self.runs_dir, **kwargs)

    def test_a_fractional_run_is_left_out_by_default(self):
        runs.record_run(self.a_run(fraction=0.25), runs_dir=self.runs_dir)
        runs.record_run(self.a_run(), runs_dir=self.runs_dir)
        self.assertEqual([run.fraction for run in self.load()], [1.0])

    def test_a_fraction_can_be_asked_for(self):
        runs.record_run(self.a_run(fraction=0.25), runs_dir=self.runs_dir)
        self.assertEqual([run.fraction for run in self.load(fraction=0.25)], [0.25])
        self.assertEqual(self.load(), [])

    def test_every_fraction_can_be_asked_for_at_once(self):
        for fraction in (0.25, 0.5, 1.0):
            runs.record_run(self.a_run(fraction=fraction), runs_dir=self.runs_dir)
        self.assertEqual(sorted(run.fraction for run in self.load(fraction=None)),
                         [0.25, 0.5, 1.0])

    def test_a_fractional_run_does_not_overwrite_the_whole_data_run(self):
        whole = runs.record_run(self.a_run(), runs_dir=self.runs_dir)
        quarter = runs.record_run(self.a_run(fraction=0.25), runs_dir=self.runs_dir)
        self.assertNotEqual(whole, quarter)
        self.assertEqual(whole.name, f"tfidf-seed42-{self.CURRENT}.json")
        self.assertEqual(quarter.name, f"tfidf-seed42-f25-{self.CURRENT}.json")

    def test_a_record_written_before_the_fraction_existed_is_a_whole_data_run(self):
        record = self.a_run()
        without = {key: value for key, value in vars(record).items()
                   if key not in ("fraction", "train_rows")}
        path = self.runs_dir / f"tfidf-seed42-{self.CURRENT}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        import json
        path.write_text(json.dumps(without), encoding="utf-8")
        loaded = self.load()
        self.assertEqual([run.fraction for run in loaded], [learning_curve.FULL])
        self.assertIsNone(loaded[0].train_rows)

    def test_runs_dir_keeps_its_historical_positional_slot(self):
        self.assertEqual(
            runs.load_runs(runs.APPROACH_ONE, self.CURRENT,
                           runs.default_conditions(), self.runs_dir), [])


@unittest.skipUnless(report_learning_curve, "report dependencies unavailable")
class ReportSeedCoverageTests(unittest.TestCase):

    @staticmethod
    def point(fraction, seeds, mean=0.5):
        return {"fraction": fraction, "seeds": list(seeds),
                "runs": {seed: object() for seed in seeds}, "mean": mean}

    def test_shape_reports_no_common_seed_as_incomplete(self):
        measured = [self.point(0.5, (42,)), self.point(1.0, (43,))]
        verdict, sentence = report_learning_curve.shape(measured, [])
        self.assertEqual(verdict, "incomplete")
        self.assertIn("no common seed", sentence)

    def test_shape_reports_mismatched_seed_sets_as_incomplete(self):
        measured = [self.point(0.25, (42, 43)),
                    self.point(0.5, (42, 43)),
                    self.point(1.0, (42,))]
        verdict, sentence = report_learning_curve.shape(
            measured, [{"p": 1.0, "difference": 0.0}])
        self.assertEqual(verdict, "incomplete")
        self.assertIn("one common seed set", sentence)
        self.assertIn("100%: 1 run at seeds 42", sentence)

    def test_curve_comparison_reports_mismatched_seed_sets_as_incomplete(self):
        one = [self.point(0.5, (42, 43)), self.point(1.0, (42, 43))]
        two = [self.point(0.5, (42,)), self.point(1.0, (42,))]
        self.assertIn("incomplete", report_learning_curve.curves_differ(one, two))


if __name__ == "__main__":
    unittest.main()
