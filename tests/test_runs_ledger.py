"""Tests for the run ledger's leak guard in src/runs.py.

The guard is one comparison: a record whose test_hash is not the current
split's is not a measurement of the current split, and counting it would put a
score back into RESULTS.md that was earned on rows the model had trained on.
Nothing else in the repository would notice that comparison being inverted, so
it is tested here, against records written by hand.

Fixtures only, and no scikit-learn: importing src/runs.py must stay cheap
enough to run on a machine with nothing but pytest installed, which is what the
GitHub Actions workflow has.
"""

import json
import tempfile
import unittest
from pathlib import Path

import runs
from paths import PROJECT_ROOT

CURRENT = "aaaaaaaaaaaaaaaa"
EARLIER = "bbbbbbbbbbbbbbbb"


def a_run(approach=runs.APPROACH_ONE, seed=42, test_hash=CURRENT,
          predicted=("apgp", "vectors-1"), conditions=None):
    return runs.Run(
        approach=approach, seed=seed, predicted=list(predicted),
        test_hash=test_hash, test_rows=len(predicted), train_seconds=1.5,
        parameters=10, hardware="a laptop", recorded="2026-09-06T00:00:00+00:00",
        conditions=runs.default_conditions() if conditions is None else conditions,
    )


class LedgerTests(unittest.TestCase):

    def setUp(self):
        # Inside the project, because record_run refuses to write anywhere else.
        self.temporary = tempfile.TemporaryDirectory(dir=PROJECT_ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.runs_dir = Path(self.temporary.name) / "runs"

    def load(self, approach=runs.APPROACH_ONE, **kwargs):
        return runs.load_runs(approach, test_hash=CURRENT,
                              runs_dir=self.runs_dir, **kwargs)

    def test_a_recorded_run_reads_back_unchanged(self):
        runs.record_run(a_run(), runs_dir=self.runs_dir)
        loaded = self.load()
        self.assertEqual([run.seed for run in loaded], [42])
        self.assertEqual(loaded[0].predicted, ["apgp", "vectors-1"])
        self.assertEqual(loaded[0].method, runs.METHOD)

    def test_runs_come_back_in_seed_order(self):
        for seed in (44, 42, 43):
            runs.record_run(a_run(seed=seed), runs_dir=self.runs_dir)
        self.assertEqual([run.seed for run in self.load()], [42, 43, 44])

    def test_a_run_against_an_earlier_split_is_ignored(self):
        runs.record_run(a_run(seed=42, test_hash=EARLIER), runs_dir=self.runs_dir)
        runs.record_run(a_run(seed=43), runs_dir=self.runs_dir)
        self.assertEqual([run.seed for run in self.load()], [43])

    def test_a_run_of_the_other_approach_is_ignored(self):
        runs.record_run(a_run(approach=runs.APPROACH_TWO), runs_dir=self.runs_dir)
        self.assertEqual(self.load(), [])
        self.assertEqual(len(self.load(approach=runs.APPROACH_TWO)), 1)

    def test_the_same_seed_and_split_is_replaced_not_duplicated(self):
        runs.record_run(a_run(predicted=("apgp", "apgp")), runs_dir=self.runs_dir)
        runs.record_run(a_run(predicted=("apgp", "vectors-1")),
                        runs_dir=self.runs_dir)
        loaded = self.load()
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].predicted, ["apgp", "vectors-1"])

    def test_a_run_under_other_conditions_is_ignored(self):
        experiment = dict(runs.default_conditions(), class_weights=False)
        runs.record_run(a_run(seed=42, conditions=experiment),
                        runs_dir=self.runs_dir)
        runs.record_run(a_run(seed=43), runs_dir=self.runs_dir)
        self.assertEqual([run.seed for run in self.load()], [43])
        self.assertEqual([run.seed for run in self.load(conditions=None)],
                         [42, 43])

    def test_the_truncation_limit_is_read_rather_than_repeated(self):
        # The filter must follow train_transformer's limit, not a copy of it,
        # so that changing it there does not silently empty the headline.
        limit = runs.default_conditions()["max_tokens"]
        self.assertEqual(limit, runs.default_max_tokens())
        runs.record_run(a_run(), runs_dir=self.runs_dir)
        self.assertEqual(self.load()[0].conditions["max_tokens"], limit)


class UnreadableRecordTests(unittest.TestCase):

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=PROJECT_ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.runs_dir = Path(self.temporary.name) / "runs"
        self.runs_dir.mkdir(parents=True)

    def write(self, name, text):
        (self.runs_dir / name).write_text(text, encoding="utf-8")

    def load(self):
        return runs.load_runs(runs.APPROACH_ONE, test_hash=CURRENT,
                              runs_dir=self.runs_dir)

    def test_a_record_from_an_older_schema_is_skipped_not_raised(self):
        record = json.loads(json.dumps(a_run().__dict__))
        record["retired_field"] = "written by an earlier version"
        self.write(f"tfidf-seed42-{CURRENT}.json", json.dumps(record))
        runs.record_run(a_run(seed=43), runs_dir=self.runs_dir)
        self.assertEqual([run.seed for run in self.load()], [43])

    def test_a_truncated_record_is_skipped_not_raised(self):
        self.write(f"tfidf-seed42-{CURRENT}.json", '{"approach": "tfidf",')
        runs.record_run(a_run(seed=43), runs_dir=self.runs_dir)
        self.assertEqual([run.seed for run in self.load()], [43])

    def test_a_record_that_is_not_an_object_is_skipped(self):
        self.write(f"tfidf-seed42-{CURRENT}.json", "[1, 2, 3]")
        self.assertEqual(self.load(), [])

    def test_an_absent_ledger_is_empty_rather_than_an_error(self):
        self.assertEqual(
            runs.load_runs(runs.APPROACH_ONE, test_hash=CURRENT,
                           runs_dir=self.runs_dir / "not-here"),
            [])


if __name__ == "__main__":
    unittest.main()
