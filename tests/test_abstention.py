"""Tests for the abstention option in src/abstention.py and src/predict.py.

Two behaviours carry the whole measurement, and both fail silently if broken.
A threshold that does not answer everything at zero leaves questions declined
that the default promises to answer. A cutoff chosen with any help from the
test split makes the reported gain circular. The first is tested at the
boundaries below. The second holds structurally, because select_cutoff takes
only validation points and the tests never hand it test ones.

Fixtures only, and no scikit-learn except where the predict wiring is under
test: importing src/abstention.py must stay cheap enough to run on a machine
with nothing but pytest installed, which is what the GitHub Actions workflow
has.
"""

import unittest

import abstention

try:
    import predict
except ModuleNotFoundError:
    predict = None

# Four questions, hand worked. Three are answered right at zero, the two most
# confident split one right and one wrong, and only the surest stands past 0.6.
Y_TRUE = ["a", "b", "a", "b"]
PREDICTED = ["a", "a", "a", "b"]
CONFIDENCES = [0.9, 0.6, 0.4, 0.3]


def point_at(threshold):
    return abstention.score_at_cutoff(Y_TRUE, PREDICTED, CONFIDENCES, threshold)


class SweepTests(unittest.TestCase):
    def test_zero_answers_everything(self):
        point = point_at(0.0)
        self.assertEqual(point["answered"], 4)
        self.assertEqual(point["coverage"], 1.0)
        self.assertEqual(point["selective_accuracy"], 0.75)
        self.assertEqual(point["full_accuracy"], 0.75)

    def test_a_threshold_past_every_confidence_answers_nothing(self):
        point = point_at(2.0)
        self.assertEqual(point["answered"], 0)
        self.assertEqual(point["coverage"], 0.0)
        self.assertIsNone(point["selective_accuracy"])
        self.assertEqual(point["full_accuracy"], 0.0)

    def test_a_confidence_on_the_cutoff_counts_as_answered(self):
        self.assertEqual(point_at(0.6)["answered"], 2)
        self.assertEqual(point_at(0.61)["answered"], 1)

    def test_a_middle_threshold_reports_both_accuracies(self):
        point = point_at(0.5)
        self.assertEqual(point["answered"], 2)
        self.assertEqual(point["selective_accuracy"], 0.5)
        self.assertEqual(point["full_accuracy"], 0.25)

    def test_coverage_and_full_accuracy_only_fall_as_the_cutoff_rises(self):
        points = abstention.sweep_points(
            Y_TRUE, PREDICTED, CONFIDENCES, [0.0, 0.3, 0.5, 0.9, 2.0])
        coverages = [point["coverage"] for point in points]
        fulls = [point["full_accuracy"] for point in points]
        self.assertEqual(coverages, sorted(coverages, reverse=True))
        self.assertEqual(fulls, sorted(fulls, reverse=True))

    def test_the_sweep_walks_the_full_range_in_twenty_steps(self):
        self.assertEqual(len(abstention.SWEEP_THRESHOLDS), 21)
        self.assertEqual(abstention.SWEEP_THRESHOLDS[0], 0.0)
        self.assertEqual(abstention.SWEEP_THRESHOLDS[-1], 1.0)
        steps = [round(b - a, 10) for a, b in
                 zip(abstention.SWEEP_THRESHOLDS, abstention.SWEEP_THRESHOLDS[1:])]
        self.assertTrue(all(step == 0.05 for step in steps))

    def test_an_empty_sweep_is_defined_rather_than_a_traceback(self):
        point = abstention.score_at_cutoff([], [], [], 0.5)
        self.assertEqual(point["answered"], 0)
        self.assertEqual(point["coverage"], 0.0)
        self.assertIsNone(point["selective_accuracy"])


class SelectCutoffTests(unittest.TestCase):
    def test_the_lowest_threshold_reaching_the_target_wins(self):
        points = abstention.sweep_points(
            Y_TRUE, PREDICTED, CONFIDENCES, [0.0, 0.5, 0.7, 0.95])
        self.assertEqual(abstention.select_cutoff(points, target=0.8), 0.7)

    def test_a_target_nothing_reaches_answers_everything(self):
        points = abstention.sweep_points(Y_TRUE, PREDICTED, CONFIDENCES,
                                         [0.0, 0.5])
        self.assertEqual(abstention.select_cutoff(points, target=0.9), 0.0)

    def test_no_points_answers_everything(self):
        self.assertEqual(abstention.select_cutoff([]), 0.0)

    def test_the_default_target_is_ninety_five_percent(self):
        self.assertEqual(abstention.TARGET_SELECTIVE_ACCURACY, 0.95)


class ReliabilityTests(unittest.TestCase):
    def test_bins_report_confidence_beside_accuracy(self):
        bins = abstention.reliability_bins(Y_TRUE, PREDICTED, CONFIDENCES,
                                           bin_count=2)
        lower, upper = bins
        self.assertEqual((lower["questions"], upper["questions"]), (2, 2))
        self.assertAlmostEqual(lower["mean_confidence"], 0.35)
        self.assertEqual(lower["accuracy"], 1.0)
        self.assertAlmostEqual(upper["mean_confidence"], 0.75)
        self.assertEqual(upper["accuracy"], 0.5)

    def test_an_empty_bin_is_marked_not_measured(self):
        bins = abstention.reliability_bins(["a"], ["a"], [0.95], bin_count=10)
        empty = [entry for entry in bins if entry["questions"] == 0]
        self.assertTrue(empty)
        for entry in empty:
            self.assertIsNone(entry["mean_confidence"])
            self.assertIsNone(entry["accuracy"])

    def test_every_question_lands_in_exactly_one_bin(self):
        bins = abstention.reliability_bins(Y_TRUE, PREDICTED, CONFIDENCES,
                                           bin_count=10)
        self.assertEqual(sum(entry["questions"] for entry in bins), 4)

    def test_the_selective_interval_covers_the_observed_accuracy(self):
        low, high = abstention.selective_interval(4, 3)
        self.assertLessEqual(low, 0.75)
        self.assertGreaterEqual(high, 0.75)
        self.assertTrue(0.0 <= low <= high <= 1.0)


class OptionTests(unittest.TestCase):
    def test_the_default_answers_everything(self):
        cutoff, rest = abstention.split_options(["a question"])
        self.assertEqual(cutoff, 0.0)
        self.assertEqual(rest, ["a question"])

    def test_the_flag_is_read_out_and_leaves_the_question(self):
        cutoff, rest = abstention.split_options(
            ["--min-confidence", "0.5", "a", "question"])
        self.assertEqual(cutoff, 0.5)
        self.assertEqual(rest, ["a", "question"])

    def test_a_missing_or_vague_value_is_refused(self):
        with self.assertRaises(ValueError):
            abstention.split_options(["--min-confidence"])
        with self.assertRaises(ValueError):
            abstention.split_options(["--min-confidence", "surely"])

    def test_a_negative_cutoff_is_refused(self):
        with self.assertRaises(ValueError):
            abstention.split_options(["--min-confidence", "-0.1"])

    def test_a_cutoff_past_one_is_allowed_to_decline_everything(self):
        cutoff, _ = abstention.split_options(["--min-confidence", "2.0"])
        self.assertEqual(cutoff, 2.0)


class MessageTests(unittest.TestCase):
    def test_declining_names_the_best_guess_and_the_cutoff(self):
        ranked = [("NORMAL DISTRIBUTION", 0.342), ("PROBABILITY", 0.101)]
        line = abstention.format_decline(ranked, 0.5)
        self.assertIn("not sure", line)
        self.assertIn("NORMAL DISTRIBUTION", line)
        self.assertIn("34.2%", line)
        self.assertIn("50%", line)

    def test_the_trade_sentence_with_declines_states_the_trade(self):
        sentence = abstention.trade_sentence(0.45, 18, 246, 0.931, 0.961, 0.886)
        self.assertIn("declines 18 of 246", sentence)
        self.assertIn("93.1%", sentence)
        self.assertIn("96.1%", sentence)
        self.assertIn("88.6%", sentence)

    def test_the_trade_sentence_without_declines_claims_no_gain(self):
        sentence = abstention.trade_sentence(0.0, 0, 246, 0.931, 0.931, 0.931)
        self.assertIn("declines nothing", sentence)

    def test_small_gaps_mean_what_they_say(self):
        sentence = abstention.calibration_sentence(0.03, 0.01)
        self.assertIn("means what it says", sentence)
        self.assertIn("3.0 points", sentence)

    def test_large_gaps_name_the_direction(self):
        under = abstention.calibration_sentence(0.24, -0.24)
        self.assertIn("does not track accuracy closely", under)
        self.assertIn("understating", under)
        over = abstention.calibration_sentence(0.10, 0.10)
        self.assertIn("overstating", over)

    def test_the_sweep_table_marks_an_unanswered_row_plainly(self):
        table = abstention.sweep_table(
            abstention.sweep_points(Y_TRUE, PREDICTED, CONFIDENCES, [0.0, 2.0]))
        self.assertIn("Accuracy on answered", table)
        self.assertIn("no questions answered", table)

    def test_a_refit_mismatch_is_carried_verbatim_when_not_measurable(self):
        reason = "the refit model does not reproduce the ledger runs' predictions"
        section = abstention.abstention_not_measurable_section(reason)
        self.assertIn("**Abstention is not measurable.**", section)
        self.assertIn(reason, section)


@unittest.skipUnless(predict, "predict dependencies unavailable")
class PredictWiringTests(unittest.TestCase):
    """The predict.py option against a stubbed model, so no corpus is needed."""

    class StubVectoriser:
        def transform(self, questions):
            return questions

    class StubClassifier:
        CLASSES = ("apgp", "vectors-1", "probability")

        def __init__(self):
            import numpy

            self.classes_ = numpy.array(list(self.CLASSES))

        def predict_proba(self, features):
            import numpy

            return numpy.array([[0.6, 0.3, 0.1]])

    class StubJoblib:
        def __init__(self, test):
            self.test = test

        def load(self, path):
            return (self.test.StubVectoriser(), self.test.StubClassifier())

    class StubPath:
        def exists(self):
            return True

    def setUp(self):
        import io
        from contextlib import redirect_stdout

        self.io = io
        self.redirect_stdout = redirect_stdout
        self._joblib = predict.joblib
        self._model_path = predict.MODEL_PATH
        predict.joblib = self.StubJoblib(self)
        predict.MODEL_PATH = self.StubPath()

    def tearDown(self):
        predict.joblib = self._joblib
        predict.MODEL_PATH = self._model_path

    def classify(self, **kwargs):
        return predict.classify_with_abstention("a synthetic question", **kwargs)

    def test_off_by_default_the_best_guess_is_answered(self):
        answered, ranked = self.classify()
        self.assertTrue(answered)
        self.assertEqual(ranked[0][0], "apgp")
        self.assertAlmostEqual(ranked[0][1], 0.6)

    def test_a_cutoff_below_the_best_guess_still_answers(self):
        answered, _ = self.classify(min_confidence=0.5)
        self.assertTrue(answered)

    def test_a_cutoff_on_the_best_guess_still_answers(self):
        answered, _ = self.classify(min_confidence=0.6)
        self.assertTrue(answered)

    def test_a_cutoff_above_the_best_guess_declines(self):
        answered, ranked = self.classify(min_confidence=0.7)
        self.assertFalse(answered)
        self.assertEqual(len(ranked), 3)

    def test_a_negative_cutoff_is_refused(self):
        with self.assertRaises(ValueError):
            self.classify(min_confidence=-0.1)

    def run_main(self, argv):
        import sys

        old_argv = sys.argv
        sys.argv = argv
        try:
            buffer = self.io.StringIO()
            with self.redirect_stdout(buffer):
                predict.main()
        finally:
            sys.argv = old_argv
        return buffer.getvalue()

    def test_the_command_line_declines_plainly_with_candidates(self):
        output = self.run_main(
            ["predict.py", "--min-confidence", "0.7", "a synthetic question"])
        self.assertIn("not sure", output)
        self.assertIn("APGP", output)
        self.assertIn("60.0%", output)

    def test_the_command_line_default_has_no_refusal_line(self):
        output = self.run_main(["predict.py", "a synthetic question"])
        self.assertNotIn("not sure", output)
        self.assertIn("APGP", output)

    def test_a_missing_flag_value_exits_nonzero(self):
        import sys

        old_argv = sys.argv
        sys.argv = ["predict.py", "--min-confidence"]
        try:
            with self.assertRaises(SystemExit) as outcome:
                predict.main()
        finally:
            sys.argv = old_argv
        self.assertNotEqual(outcome.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
