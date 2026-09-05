"""Tests for the interval and paired-test arithmetic in src/intervals.py.

Reference values are worked by hand from the formulas, so a test failing here
means the code moved away from the textbook, not that a dataset changed.
Nothing here touches the corpus.
"""

import math
import unittest

import intervals


class WilsonIntervalTests(unittest.TestCase):

    def test_matches_a_hand_worked_value(self):
        # 229 of 246, the baseline's frozen-split score. By hand:
        # centre 0.92427, half-width 0.03214.
        low, high = intervals.wilson_interval(229, 246)
        self.assertAlmostEqual(low, 0.8921, places=4)
        self.assertAlmostEqual(high, 0.9564, places=4)

    def test_brackets_the_point_estimate(self):
        for correct, total in ((0, 10), (5, 10), (10, 10), (229, 246), (1, 1000)):
            low, high = intervals.wilson_interval(correct, total)
            self.assertLessEqual(0.0, low)
            self.assertLessEqual(low, correct / total + 1e-12)
            self.assertLessEqual(correct / total, high + 1e-12)
            self.assertLessEqual(high, 1.0)

    def test_perfect_score_still_has_width(self):
        # The Wald interval collapses to zero here; Wilson must not.
        low, high = intervals.wilson_interval(246, 246)
        self.assertEqual(high, 1.0)
        self.assertLess(low, 1.0)
        self.assertGreater(low, 0.98)

    def test_narrows_with_more_trials(self):
        narrow = intervals.wilson_interval(930, 1000)
        wide = intervals.wilson_interval(93, 100)
        self.assertLess(narrow[1] - narrow[0], wide[1] - wide[0])

    def test_rejects_impossible_counts(self):
        with self.assertRaises(ValueError):
            intervals.wilson_interval(5, 0)
        with self.assertRaises(ValueError):
            intervals.wilson_interval(11, 10)


class SpreadTests(unittest.TestCase):

    def test_identical_runs_have_zero_spread(self):
        summary = intervals.spread([0.931, 0.931, 0.931])
        self.assertEqual(summary["stdev"], 0.0)
        self.assertEqual(summary["low"], summary["high"])
        self.assertEqual(summary["n"], 3)

    def test_uses_the_sample_standard_deviation(self):
        summary = intervals.spread([0.90, 0.92, 0.94])
        self.assertAlmostEqual(summary["mean"], 0.92)
        self.assertAlmostEqual(summary["stdev"], 0.02)
        self.assertEqual((summary["low"], summary["high"]), (0.90, 0.94))

    def test_single_run_is_allowed(self):
        summary = intervals.spread([0.5])
        self.assertEqual(summary["stdev"], 0.0)
        self.assertEqual(summary["n"], 1)

    def test_no_runs_is_an_error(self):
        with self.assertRaises(ValueError):
            intervals.spread([])


class PairedOutcomeTests(unittest.TestCase):

    def test_counts_each_quadrant(self):
        truth = ["a", "a", "a", "a", "a", "b"]
        first = ["a", "a", "a", "x", "x", "b"]
        second = ["a", "a", "x", "a", "x", "x"]
        self.assertEqual(intervals.paired_outcomes(truth, first, second),
                         (2, 2, 1, 1))

    def test_quadrants_sum_to_the_row_count(self):
        truth = list("abcabcabc")
        first = list("abcxxxabc")
        second = list("xbcabcxxc")
        self.assertEqual(sum(intervals.paired_outcomes(truth, first, second)),
                         len(truth))

    def test_mismatched_lengths_are_an_error(self):
        with self.assertRaises(ValueError):
            intervals.paired_outcomes(["a", "b"], ["a"], ["a", "b"])


class McNemarTests(unittest.TestCase):

    def test_even_split_is_no_evidence(self):
        self.assertEqual(intervals.mcnemar_exact_p(4, 4), 1.0)

    def test_no_discordant_pairs_is_no_evidence(self):
        self.assertEqual(intervals.mcnemar_exact_p(0, 0), 1.0)

    def test_hand_worked_small_case(self):
        # Six discordant questions, five to one: 2 * (1 + 6) / 64.
        self.assertAlmostEqual(intervals.mcnemar_exact_p(5, 1), 14 / 64)
        self.assertAlmostEqual(intervals.mcnemar_exact_p(1, 5), 14 / 64)

    def test_lopsided_split_is_significant(self):
        # Ten to nothing: 2 * 1 / 1024.
        self.assertAlmostEqual(intervals.mcnemar_exact_p(10, 0), 2 / 1024)
        self.assertLess(intervals.mcnemar_exact_p(10, 0), 0.05)

    def test_never_exceeds_one(self):
        for a, b in ((1, 1), (2, 3), (3, 2), (7, 7)):
            self.assertLessEqual(intervals.mcnemar_exact_p(a, b), 1.0)


class PairedDifferenceTests(unittest.TestCase):

    def test_centred_on_the_observed_difference(self):
        low, high = intervals.paired_difference_interval(8, 4, 246)
        self.assertAlmostEqual((low + high) / 2, 4 / 246)

    def test_equal_discordants_bracket_zero(self):
        low, high = intervals.paired_difference_interval(5, 5, 246)
        self.assertLess(low, 0.0)
        self.assertGreater(high, 0.0)

    def test_hand_worked_width(self):
        # b=8, c=4, n=246: variance = (12 - 16/246) / 246^2.
        low, high = intervals.paired_difference_interval(8, 4, 246)
        variance = (12 - 16 / 246) / 246 ** 2
        self.assertAlmostEqual(high - low, 2 * intervals.Z_95 * math.sqrt(variance))

    def test_no_trials_is_an_error(self):
        with self.assertRaises(ValueError):
            intervals.paired_difference_interval(1, 1, 0)


class OverlapTests(unittest.TestCase):

    def test_overlapping_and_disjoint(self):
        self.assertTrue(intervals.intervals_overlap((0.89, 0.96), (0.87, 0.94)))
        self.assertTrue(intervals.intervals_overlap((0.89, 0.96), (0.96, 0.99)))
        self.assertFalse(intervals.intervals_overlap((0.89, 0.92), (0.93, 0.99)))


if __name__ == "__main__":
    unittest.main()
