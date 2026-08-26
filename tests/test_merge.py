"""Tests for the merge step's deduplication."""

import unittest

from merge import collapse_duplicates, normalise_question_number


class QuestionNumberNormalisation(unittest.TestCase):

    def test_accepts_q_prefix_and_existing_forms(self):
        self.assertEqual(normalise_question_number("Q11"), "11")
        self.assertEqual(normalise_question_number("11"), "11")
        self.assertEqual(normalise_question_number("11(a)"), "11a")
        self.assertEqual(normalise_question_number("Q11(a)"), "11a")
        self.assertEqual(normalise_question_number("  Q 11 ( a)"), "11a")

    def test_q_prefix_collapses_duplicate_provenance(self):
        rows = [
            {"id": "q-prefixed", "school": "ABCJC", "year": 2013,
             "paper": "I", "question_no": "Q9", "text": "first"},
            {"id": "bare-number", "school": "ABCJC", "year": 2013,
             "paper": "I", "question_no": "9", "text": "second"},
        ]
        collapsed, provenance_count, text_count = collapse_duplicates(rows)
        self.assertEqual([row["id"] for row in collapsed], ["q-prefixed"])
        self.assertEqual(provenance_count, 1)
        self.assertEqual(text_count, 0)


if __name__ == "__main__":
    unittest.main()
