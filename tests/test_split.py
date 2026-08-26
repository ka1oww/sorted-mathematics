"""Tests proving the train/validation/test split cannot leak.

The corpus is private, so these tests run on synthetic rows built to its
documented shape (DATA-PLAN.md) instead: about 1,450 rows across the 21
chapters, about 807 carrying (school, year, paper) provenance and clustered
into papers of 2-12 questions that each mix several chapters, the rest
provenance-free and grouped by their source file, plus a handful of rows with
neither. One chapter is deliberately starved of rows in a second
fixture variant, so the minimum-per-chapter check can be shown firing rather
than assumed to.

Run from the project root:  python3 tests/test_split.py
"""

import collections
import json
import random
import tempfile
import unittest
from pathlib import Path

import split
from chapters import CHAPTER_SLUGS

# ---------------------------------------------------------------------------
# Fixtures: synthetic rows in the real corpus's shape
# ---------------------------------------------------------------------------

SCARCE_CHAPTER = "sampling"
HEALTHY_SCARCE_ROWS = 40   # enough rows that the test floor is reachable
STARVED_SCARCE_ROWS = 4    # fewer rows than the floor: no assignment can pass

# Placeholder codes. The real corpus names real schools; the fixtures only
# need 17 distinct provenance strings, so they do not name any.
SCHOOLS = ("ABCJC", "BCDJC", "CDEJC", "DEFJC", "EFGJC", "FGHJC", "GHIJC",
           "HIJJC", "IJKJC", "JKLJC", "KLMJC", "LMNJC", "MNOJC", "NOPJC",
           "OPQJC", "PQRJC", "QRSJC")
YEARS = (2019, 2020, 2021, 2022, 2023, 2024)
PROVENANCED_ROW_TARGET = 807
TUTORIAL_SIZE_RANGE = (14, 20)
ORPHAN_ROW_COUNT = 8
FIXTURE_SEED = 20260814


def question_row(row_id, chapter, school=None, year=None, paper=None,
                 question_no=None, source="tutorial_latex"):
    """One row in the exact shape the extractors emit (see DATA-PLAN.md)."""
    return {
        "id": row_id,
        "chapter": chapter,
        "text": f"synthetic question about {chapter}",
        "raw_text": f"\\Q synthetic question about {chapter}",
        "source": source,
        "year": year,
        "school": school,
        "level": None,
        "exam": "Prelim" if school else None,
        "paper": paper,
        "question_no": question_no,
        "modified": False,
        "marks": None,
    }


def weighted_distinct_chapters(rng, chapters, weights, count):
    """Draw `count` distinct chapters, likelier chapters more often."""
    pool = list(chapters)
    chosen = []
    for _ in range(count):
        pick = rng.choices(pool, weights=[weights[c] for c in pool])[0]
        pool.remove(pick)
        chosen.append(pick)
    return chosen


def build_question_rows(seed=FIXTURE_SEED, scarce_rows=HEALTHY_SCARCE_ROWS):
    """Synthetic dataset mirroring the real corpus's difficult properties.

    * Provenanced rows come in whole papers of 2-12 questions, each paper
      mixing several chapters - the entire reason a row-level split leaks.
    * A paper's rows are split between two extraction sources (revision
      package and topical bank), so grouping by file instead of provenance
      would divide a paper.
    * Provenance-free rows come as one single-chapter tutorial file per
      chapter - large indivisible blocks, the hard case for stratifying.
    * Chapters are unevenly weighted, so the lightest have few enough rows
      that 15% of them is below the 5-row test floor.
    * `scarce_rows` controls one chapter's total row count; set it below the
      floor and no valid assignment exists at all.
    * A few orphan rows have no provenance and no recognisable source file.
    """
    rng = random.Random(seed)
    mixing_chapters = [c for c in CHAPTER_SLUGS if c != SCARCE_CHAPTER]
    chapter_weights = {c: rng.uniform(1.5, 6.0) for c in mixing_chapters}
    rows = []
    serial = 0

    # Provenanced rows, dealt into papers.
    paper_labels = [(school, year, paper_no) for school in SCHOOLS
                    for year in YEARS for paper_no in ("P1", "P2")]
    rng.shuffle(paper_labels)
    scarce_in_papers = scarce_rows // 2
    labels_used = []
    remaining = PROVENANCED_ROW_TARGET - scarce_in_papers
    while remaining > 0:
        school, year, paper_no = paper_labels[len(labels_used)]
        labels_used.append((school, year, paper_no))
        size = min(rng.randint(2, 10), remaining)
        if remaining - size == 1:
            size += 1  # never leave a final one-question paper
        palette = weighted_distinct_chapters(rng, mixing_chapters,
                                             chapter_weights,
                                             rng.randint(2, 5))
        chapters_in_paper = rng.choices(
            palette, weights=[chapter_weights[c] for c in palette], k=size)
        for ordinal, chapter in enumerate(chapters_in_paper, start=1):
            serial += 1
            if rng.random() < 0.5:
                row_id = f"rp:{chapter}:{school}-{year}-{paper_no}-{serial}"
                source = "revision_package"
            else:
                row_id = f"topical:{chapter}:q{serial:04d}"
                source = "topical_pdf"
            rows.append(question_row(row_id, chapter, school=school,
                                     year=year, paper=paper_no,
                                     question_no=str(ordinal), source=source))
        remaining -= size

    # The scarce chapter's provenanced rows, one per paper so they are as
    # spread out (and as annoying to collect into test) as in real data.
    for school, year, paper_no in labels_used[:scarce_in_papers]:
        serial += 1
        rows.append(question_row(f"topical:{SCARCE_CHAPTER}:q{serial:04d}",
                                 SCARCE_CHAPTER, school=school, year=year,
                                 paper=paper_no, question_no="last",
                                 source="topical_pdf"))

    # Tutorials: one file per chapter, so each is a single-chapter group.
    for chapter in mixing_chapters:
        for index in range(1, rng.randint(*TUTORIAL_SIZE_RANGE) + 1):
            rows.append(question_row(f"tut:{chapter}:q{index:02d}", chapter))
    for index in range(1, scarce_rows - scarce_in_papers + 1):
        rows.append(question_row(f"tut:{SCARCE_CHAPTER}:q{index:02d}",
                                 SCARCE_CHAPTER))

    # Orphans: no provenance, and an id with no container structure.
    for index in range(1, ORPHAN_ROW_COUNT + 1):
        rows.append(question_row(f"orphan{index:02d}",
                                 rng.choice(mixing_chapters)))
    return rows


# ---------------------------------------------------------------------------
# Naive baselines the group split must beat
# ---------------------------------------------------------------------------

def naive_row_level_split(rows, seed):
    """70/15/15 by shuffling rows - the split this project must never use."""
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    train_end = int(len(shuffled) * 0.70)
    val_end = int(len(shuffled) * 0.85)
    return {"train": shuffled[:train_end],
            "val": shuffled[train_end:val_end],
            "test": shuffled[val_end:]}


def naive_random_group_split(rows, seed):
    """Whole groups assigned by a weighted coin flip, ignoring chapters.

    This baseline does not leak, so beating it takes stratification quality:
    proportions closer to target and no chapter starved of test rows.
    """
    rng = random.Random(seed)
    groups = split.collect_rows_by_group(rows)
    weights = [split.SPLIT_FRACTIONS[name] for name in split.SPLIT_NAMES]
    split_rows = {name: [] for name in split.SPLIT_NAMES}
    for group_rows in groups.values():
        chosen = rng.choices(split.SPLIT_NAMES, weights=weights)[0]
        split_rows[chosen].extend(group_rows)
    return split_rows


def count_groups_straddling_splits(split_rows):
    splits_seen = collections.defaultdict(set)
    for split_name, assigned in split_rows.items():
        for row in assigned:
            splits_seen[split.leakage_group_key(row)].add(split_name)
    return sum(1 for names in splits_seen.values() if len(names) > 1)


def chapter_balance_error(split_rows):
    """Total row distance from perfectly stratified 70/15/15, over all
    (split, chapter) cells. Lower is better; 0 is unattainable with whole
    groups."""
    totals = collections.Counter(row["chapter"] for assigned
                                 in split_rows.values() for row in assigned)
    error = 0.0
    for split_name, assigned in split_rows.items():
        counts = collections.Counter(row["chapter"] for row in assigned)
        for chapter, total in totals.items():
            target = split.SPLIT_FRACTIONS[split_name] * total
            error += abs(counts[chapter] - target)
    return error


def chapters_below_test_floor(split_rows, floor=split.DEFAULT_MIN_TEST_ROWS):
    test_counts = collections.Counter(row["chapter"]
                                      for row in split_rows["test"])
    return [chapter for chapter in CHAPTER_SLUGS
            if test_counts[chapter] < floor]


# ---------------------------------------------------------------------------
# The grouping rule
# ---------------------------------------------------------------------------

class GroupKeyRules(unittest.TestCase):

    def test_rows_from_one_paper_share_a_group(self):
        first = question_row("rp:apgp:x-1", "apgp",
                             school="ABCJC", year=2023, paper="P1")
        second = question_row("rp:functions:x-2", "functions",
                              school="ABCJC", year=2023, paper="P1")
        self.assertEqual(split.leakage_group_key(first),
                         split.leakage_group_key(second))

    def test_same_school_and_year_but_different_paper_is_a_different_group(self):
        paper_one = question_row("a:b:1", "apgp",
                                 school="ABCJC", year=2023, paper="P1")
        paper_two = question_row("a:b:2", "apgp",
                                 school="ABCJC", year=2023, paper="P2")
        self.assertNotEqual(split.leakage_group_key(paper_one),
                            split.leakage_group_key(paper_two))

    def test_paper_house_style_does_not_divide_a_paper(self):
        # One school's cover prints `II` where another prints `P2`, and the
        # two spellings reach the corpus from different extractions of the
        # same paper. Grouping on the raw field makes that one paper into two
        # groups, which every check below would then pass happily, because
        # each half really does sit in one split.
        roman = question_row("topical:apgp:q0001", "apgp",
                             school="BCDJC", year=2022, paper="II")
        p_form = question_row("rp:functions:BCDJC-2022-P2-3", "functions",
                              school="BCDJC", year=2022, paper="P2")
        spaced = question_row("topical:curve-sketching:q0002", "curve-sketching",
                              school="BCDJC", year=2022, paper="p 2")
        self.assertEqual(split.leakage_group_key(roman),
                         split.leakage_group_key(p_form))
        self.assertEqual(split.leakage_group_key(roman),
                         split.leakage_group_key(spaced))
        # Folding house style must not fold two genuinely different papers.
        other_paper = question_row("topical:apgp:q0003", "apgp",
                                   school="BCDJC", year=2022, paper="I")
        self.assertNotEqual(split.leakage_group_key(roman),
                            split.leakage_group_key(other_paper))

    def test_provenance_outranks_the_source_file(self):
        # The same paper surfacing in two extraction sources must still be
        # one group, or its two copies could land on opposite sides.
        from_package = question_row("rp:vectors-1:BCDJC-2022-P1-9", "vectors-1",
                                    school="BCDJC", year=2022, paper="P1",
                                    source="revision_package")
        from_topical = question_row("topical:vectors-1:q0472", "vectors-1",
                                    school="BCDJC", year=2022, paper="P1",
                                    source="topical_pdf")
        self.assertEqual(split.leakage_group_key(from_package),
                         split.leakage_group_key(from_topical))

    def test_provenance_free_rows_group_by_source_file(self):
        first = question_row("tut:apgp:q01", "apgp")
        second = question_row("tut:apgp:q07", "apgp")
        other_file = question_row("tut:functions:q01", "functions")
        self.assertEqual(split.leakage_group_key(first),
                         split.leakage_group_key(second))
        self.assertNotEqual(split.leakage_group_key(first),
                            split.leakage_group_key(other_file))

    def test_explicit_source_file_field_is_honoured(self):
        with_field = question_row("pdf:apgp:q01", "apgp")
        with_field["source_file"] = "tutorials/apgp/APGP Tutorial.pdf"
        same_file = question_row("pdf:apgp:q02", "apgp")
        same_file["source_file"] = "tutorials/apgp/APGP Tutorial.pdf"
        self.assertEqual(split.leakage_group_key(with_field),
                         split.leakage_group_key(same_file))

    def test_row_with_no_provenance_and_no_container_stands_alone(self):
        first = question_row("orphan01", "apgp")
        second = question_row("orphan02", "apgp")
        self.assertEqual(split.leakage_group_key(first)[0], "row")
        self.assertNotEqual(split.leakage_group_key(first),
                            split.leakage_group_key(second))


# ---------------------------------------------------------------------------
# The fixtures really have the promised shape
# ---------------------------------------------------------------------------

class FixtureShape(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.rows = build_question_rows()

    def test_row_counts_match_the_data_plan(self):
        provenanced = [row for row in self.rows if row["school"]]
        self.assertAlmostEqual(len(self.rows), 1170, delta=120)
        self.assertAlmostEqual(len(provenanced), 807,
                               delta=HEALTHY_SCARCE_ROWS)
        self.assertAlmostEqual(len(self.rows) - len(provenanced), 365,
                               delta=100)

    def test_every_chapter_is_present(self):
        chapters_present = {row["chapter"] for row in self.rows}
        self.assertEqual(chapters_present, set(CHAPTER_SLUGS))

    def test_papers_hold_2_to_12_questions_and_mix_chapters(self):
        papers = collections.defaultdict(list)
        for row in self.rows:
            if row["school"]:
                papers[(row["school"], row["year"], row["paper"])].append(row)
        sizes = [len(rows) for rows in papers.values()]
        self.assertTrue(all(2 <= size <= 12 for size in sizes))
        mixed = [rows for rows in papers.values()
                 if len({row["chapter"] for row in rows}) > 1]
        # Nearly every paper mixes chapters; that is the source of difficulty.
        self.assertGreater(len(mixed), len(papers) * 0.8)

    def test_the_scarce_chapter_is_scarce(self):
        scarce = [row for row in self.rows
                  if row["chapter"] == SCARCE_CHAPTER]
        self.assertEqual(len(scarce), HEALTHY_SCARCE_ROWS)
        starved_rows = build_question_rows(scarce_rows=STARVED_SCARCE_ROWS)
        starved = [row for row in starved_rows
                   if row["chapter"] == SCARCE_CHAPTER]
        self.assertEqual(len(starved), STARVED_SCARCE_ROWS)

    def test_some_chapters_sit_below_the_15_percent_floor_threshold(self):
        # The interesting tension: chapters where 15% of the rows is fewer
        # than the 5-row test floor, so the two targets genuinely conflict.
        totals = collections.Counter(row["chapter"] for row in self.rows)
        tense = [chapter for chapter, count in totals.items()
                 if count * split.SPLIT_FRACTIONS["test"]
                 < split.DEFAULT_MIN_TEST_ROWS]
        self.assertTrue(tense)


# ---------------------------------------------------------------------------
# The assignment satisfies every rule on realistic data
# ---------------------------------------------------------------------------

class AssignmentOnRealisticRows(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.rows = build_question_rows()
        cls.split_rows = split.assign_groups_to_splits(cls.rows, seed=42)

    def test_no_group_straddles_splits(self):
        self.assertEqual(count_groups_straddling_splits(self.split_rows), 0)
        split.check_no_group_straddles_splits(self.split_rows)

    def test_every_row_lands_in_exactly_one_split(self):
        input_ids = sorted(row["id"] for row in self.rows)
        output_ids = sorted(row["id"] for assigned
                            in self.split_rows.values() for row in assigned)
        self.assertEqual(input_ids, output_ids)
        split.check_rows_conserved(self.rows, self.split_rows)

    def test_proportions_land_near_70_15_15(self):
        total = len(self.rows)
        for split_name, target in split.SPLIT_FRACTIONS.items():
            achieved = len(self.split_rows[split_name]) / total
            self.assertAlmostEqual(achieved, target, delta=0.05,
                                   msg=f"{split_name} landed at {achieved:.1%}"
                                       f" against a {target:.0%} target")

    def test_every_chapter_reaches_the_test_floor(self):
        self.assertEqual(chapters_below_test_floor(self.split_rows), [])
        split.check_every_chapter_has_test_rows(self.split_rows)

    def test_every_chapter_appears_in_every_split(self):
        for split_name, assigned in self.split_rows.items():
            chapters_present = {row["chapter"] for row in assigned}
            self.assertEqual(chapters_present, set(CHAPTER_SLUGS),
                             msg=f"chapters missing from {split_name}")

    def test_same_seed_reproduces_the_split_exactly(self):
        rerun = split.assign_groups_to_splits(self.rows, seed=42)
        for split_name in split.SPLIT_NAMES:
            self.assertEqual([row["id"] for row in rerun[split_name]],
                             [row["id"] for row in
                              self.split_rows[split_name]])
        split.check_split_is_reproducible(
            self.rows, self.split_rows, 42, split.DEFAULT_MIN_TEST_ROWS)

    def test_a_different_seed_moves_at_least_one_group(self):
        other = split.assign_groups_to_splits(self.rows, seed=43)
        same = all([row["id"] for row in other[name]]
                   == [row["id"] for row in self.split_rows[name]]
                   for name in split.SPLIT_NAMES)
        self.assertFalse(same, "the seed had no effect on the assignment")

    def test_summary_reports_what_a_reviewer_needs(self):
        summary = split.summarise_assignment(self.split_rows)
        self.assertIn("smallest test chapter", summary)
        self.assertIn(f"its own group: {ORPHAN_ROW_COUNT}", summary)
        for chapter in CHAPTER_SLUGS:
            self.assertIn(chapter, summary)
        # Coverage warnings appear only when a chapter is actually missing.
        self.assertNotIn("no train rows", summary)
        self.assertNotIn("no val rows", summary)


class ConcentratedChapters(unittest.TestCase):
    """A chapter living almost entirely in one indivisible block.

    The block cannot be divided, so the only assignment that covers all
    three splits sends the block to test (the largest fixed bill) and deals
    the scattered rows to train and validation. This shape exists in the
    main fixture too; this micro-fixture isolates it so the behaviour is
    pinned down rather than incidental.
    """

    def build_concentrated_rows(self):
        rows = []
        # A healthy companion chapter spread over twelve papers ...
        for paper_index in range(12):
            school = SCHOOLS[paper_index]
            for question in range(4):
                rows.append(question_row(
                    f"rp:apgp:{school}-2020-P1-{question}", "apgp",
                    school=school, year=2020, paper="P1"))
        # ... six of which also carry one scattered row of the concentrated
        # chapter, whose remaining twenty rows sit in a single tutorial.
        for paper_index in range(6):
            school = SCHOOLS[paper_index]
            rows.append(question_row(
                f"rp:functions:{school}-2020-P1-x", "functions",
                school=school, year=2020, paper="P1"))
        for question in range(1, 21):
            rows.append(question_row(f"tut:functions:q{question:02d}",
                                     "functions"))
        return rows

    def test_the_block_is_pinned_to_test(self):
        rows = self.build_concentrated_rows()
        groups = split.collect_rows_by_group(rows)
        pinned = split._pin_concentrated_blocks_to_test(
            groups, split.DEFAULT_MIN_TEST_ROWS)
        self.assertEqual(pinned, {("file", "tut:functions"): "test"})

    def test_the_scattered_rows_cover_train_and_validation(self):
        rows = self.build_concentrated_rows()
        split_rows = split.assign_groups_to_splits(rows, seed=42)
        counts = {
            split_name: collections.Counter(row["chapter"] for row in assigned)
            for split_name, assigned in split_rows.items()
        }
        self.assertGreaterEqual(counts["test"]["functions"], 20)
        self.assertGreaterEqual(counts["train"]["functions"], 1)
        self.assertGreaterEqual(counts["val"]["functions"], 1)
        self.assertEqual(count_groups_straddling_splits(split_rows), 0)


# ---------------------------------------------------------------------------
# The hard checks fail loudly when the rules are broken
# ---------------------------------------------------------------------------

class ChecksFailLoudly(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.rows = build_question_rows()
        cls.split_rows = split.assign_groups_to_splits(cls.rows, seed=42)

    def corrupted_copy(self):
        return {name: list(assigned)
                for name, assigned in self.split_rows.items()}

    def test_a_divided_group_is_detected(self):
        # Move one row of some multi-row test group into train, exactly the
        # corruption the whole module exists to prevent.
        corrupted = self.corrupted_copy()
        test_keys = collections.Counter(split.leakage_group_key(row)
                                        for row in corrupted["test"])
        divisible = next(key for key, count in test_keys.items() if count > 1)
        leaked = next(row for row in corrupted["test"]
                      if split.leakage_group_key(row) == divisible)
        corrupted["test"].remove(leaked)
        corrupted["train"].append(leaked)
        with self.assertRaises(split.GroupLeakageError):
            split.check_no_group_straddles_splits(corrupted)

    def test_a_lost_row_is_detected(self):
        corrupted = self.corrupted_copy()
        corrupted["val"] = corrupted["val"][1:]
        with self.assertRaises(split.RowConservationError):
            split.check_rows_conserved(self.rows, corrupted)

    def test_a_duplicated_row_is_detected(self):
        corrupted = self.corrupted_copy()
        corrupted["train"] = corrupted["train"] + [corrupted["train"][0]]
        with self.assertRaises(split.RowConservationError):
            split.check_rows_conserved(self.rows, corrupted)

    def test_duplicate_input_ids_are_refused_up_front(self):
        rows = self.rows + [dict(self.rows[0])]
        with self.assertRaises(split.RowConservationError):
            split.assign_groups_to_splits(rows, seed=42)

    def test_a_starved_chapter_fails_by_name_with_its_count(self):
        starved_rows = build_question_rows(scarce_rows=STARVED_SCARCE_ROWS)
        split_rows = split.assign_groups_to_splits(starved_rows, seed=42)
        with self.assertRaisesRegex(split.ChapterTestShortfallError,
                                    rf"'{SCARCE_CHAPTER}' has [0-4]\b"):
            split.check_every_chapter_has_test_rows(split_rows)

    def test_an_unreachable_floor_fails_even_on_healthy_data(self):
        split_rows = split.assign_groups_to_splits(self.rows, seed=42,
                                                   min_test_rows=500)
        with self.assertRaises(split.ChapterTestShortfallError):
            split.check_every_chapter_has_test_rows(split_rows,
                                                    min_test_rows=500)


# ---------------------------------------------------------------------------
# The group split beats both naive baselines
# ---------------------------------------------------------------------------

class NaiveBaselinesLoseOnTheseFixtures(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.rows = build_question_rows()
        cls.split_rows = split.assign_groups_to_splits(cls.rows, seed=42)

    def test_a_row_level_split_divides_many_groups_and_ours_divides_none(self):
        leaky = naive_row_level_split(self.rows, seed=0)
        divided = count_groups_straddling_splits(leaky)
        multi_row_groups = sum(
            1 for group_rows
            in split.collect_rows_by_group(self.rows).values()
            if len(group_rows) > 1)
        # A shuffled row split divides nearly every multi-row group: each of
        # its rows lands independently, so siblings scatter across splits.
        self.assertGreater(divided, multi_row_groups * 0.7)
        self.assertEqual(count_groups_straddling_splits(self.split_rows), 0)

    def test_greedy_stratifies_better_than_random_group_assignment(self):
        # Same leak-safety, so the greedy walk must win on balance instead:
        # against twenty random whole-group splits it must show a smaller
        # deviation from 70/15/15-per-chapter than the best of them, and
        # never starve a chapter of test rows the way most of them do.
        greedy_error = chapter_balance_error(self.split_rows)
        random_errors = []
        floor_violations = 0
        for seed in range(20):
            random_split = naive_random_group_split(self.rows, seed)
            random_errors.append(chapter_balance_error(random_split))
            if chapters_below_test_floor(random_split):
                floor_violations += 1
        self.assertLess(greedy_error, min(random_errors))
        self.assertGreater(floor_violations, 10)
        self.assertEqual(chapters_below_test_floor(self.split_rows), [])


# ---------------------------------------------------------------------------
# Reading, writing and the path guard
# ---------------------------------------------------------------------------

class WritingAndPathGuard(unittest.TestCase):

    def write_input_jsonl(self, directory, rows):
        input_path = Path(directory) / "questions.jsonl"
        with input_path.open("w", encoding="utf-8") as out:
            for row in rows:
                out.write(json.dumps(row) + "\n")
        return input_path

    def test_refuses_to_write_outside_the_project_folder(self):
        rows = build_question_rows()
        with tempfile.TemporaryDirectory() as scratch:
            input_path = self.write_input_jsonl(scratch, rows)
            # Default project root, input in a temp dir: must refuse.
            with self.assertRaises(SystemExit):
                split.write_split_files(input_path)
            self.assertEqual(list(Path(scratch).glob("*.jsonl")),
                             [input_path])

    def test_writes_three_files_that_partition_the_input(self):
        rows = build_question_rows()
        with tempfile.TemporaryDirectory() as scratch:
            input_path = self.write_input_jsonl(scratch, rows)
            summary = split.write_split_files(input_path,
                                              project_root=scratch)
            written = {}
            for split_name in split.SPLIT_NAMES:
                written[split_name] = split.read_rows(
                    Path(scratch) / f"{split_name}.jsonl")
            output_ids = sorted(row["id"] for assigned in written.values()
                                for row in assigned)
            self.assertEqual(output_ids, sorted(row["id"] for row in rows))
            self.assertEqual(count_groups_straddling_splits(written), 0)
            self.assertIn("seed", summary)
            self.assertIn("smallest test chapter", summary)

    def test_a_failed_check_leaves_no_output_files_behind(self):
        starved_rows = build_question_rows(scarce_rows=STARVED_SCARCE_ROWS)
        with tempfile.TemporaryDirectory() as scratch:
            input_path = self.write_input_jsonl(scratch, starved_rows)
            with self.assertRaises(split.ChapterTestShortfallError):
                split.write_split_files(input_path, project_root=scratch)
            self.assertEqual(list(Path(scratch).glob("*.jsonl")),
                             [input_path])


if __name__ == "__main__":
    unittest.main(verbosity=2)
