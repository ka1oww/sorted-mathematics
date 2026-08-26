"""Assign whole papers to train, validation and test, never single questions.

Questions from one exam paper are not independent samples. They share a setter,
a house style, and often the same context across parts. Split rows at random and
questions from one paper end up on both sides of the fence: the accuracy goes up
and nothing reports it.

A group is one paper, or one tutorial for questions carrying no provenance. Four
checks run before a single file is written, and each raises rather than warns.

Run: python3 src/split.py data/private/questions.jsonl
"""

import argparse
import collections
import json
import random
from pathlib import Path

from chapters import CHAPTER_SLUGS
from merge import normalise_paper
from paths import PROJECT_ROOT, assert_inside_project

SPLIT_NAMES = ("train", "val", "test")
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
DEFAULT_SEED = 42
DEFAULT_MIN_TEST_ROWS = 5
MIN_REPRESENTATION_ROWS = 1


class SplitIntegrityError(Exception): pass
class GroupLeakageError(SplitIntegrityError): pass
class ChapterTestShortfallError(SplitIntegrityError): pass
class RowConservationError(SplitIntegrityError): pass
class SplitNotReproducibleError(SplitIntegrityError): pass


# ---------------------------------------------------------------------------
# One paper is one group. Rows with no provenance fall back to their source file.
def leakage_group_key(row):
    school = row.get("school")
    year = row.get("year")
    paper = row.get("paper")
    if school is not None or year is not None:
        return ("paper", school, year, normalise_paper(paper))

    source_file = row.get("source_file")
    if source_file:
        return ("file", source_file)

    row_id = row.get("id", "")
    if ":" in row_id:
        container = row_id.rsplit(":", 1)[0]
        return ("file", container)
    return ("row", row_id)


def collect_rows_by_group(rows):
    # A dict, because insertion order decides the tie-break below.
    groups = {}
    for row in rows:
        groups.setdefault(leakage_group_key(row), []).append(row)
    return groups


# ---------------------------------------------------------------------------
# Assigning the papers

def per_chapter_row_targets(rows, min_test_rows):
    chapter_totals = collections.Counter(row["chapter"] for row in rows)
    targets = {split_name: {} for split_name in SPLIT_NAMES}
    for chapter, total in chapter_totals.items():
        test_share = max(SPLIT_FRACTIONS["test"] * total, min(min_test_rows, total))
        remainder = total - test_share
        train_ratio = SPLIT_FRACTIONS["train"] / (SPLIT_FRACTIONS["train"] + SPLIT_FRACTIONS["val"])
        targets["train"][chapter] = remainder * train_ratio
        targets["val"][chapter] = remainder * (1 - train_ratio)
        targets["test"][chapter] = test_share
    return targets


def assign_groups_to_splits(rows, seed=DEFAULT_SEED,
                            min_test_rows=DEFAULT_MIN_TEST_ROWS):
    _require_known_chapters_and_unique_ids(rows)
    groups = collect_rows_by_group(rows)
    targets = per_chapter_row_targets(rows, min_test_rows)
    placed = {split_name: collections.Counter() for split_name in SPLIT_NAMES}
    assignment = _pin_concentrated_blocks_to_test(groups, min_test_rows)
    for group_key in assignment:
        for row in groups[group_key]:
            placed["test"][row["chapter"]] += 1
    ordering = [group_key for group_key in groups
                if group_key not in assignment]
    random.Random(seed).shuffle(ordering)
    ordering.sort(key=lambda group_key: -len(groups[group_key]))

    for group_key in ordering:
        group_rows = groups[group_key]

        def unmet_need(split_name):
            return sum(targets[split_name][row["chapter"]]
                       - placed[split_name][row["chapter"]]
                       for row in group_rows)

        # ties break by SPLIT_NAMES order, so a tie always goes to train
        chosen = max(SPLIT_NAMES, key=unmet_need)
        assignment[group_key] = chosen
        for row in group_rows:
            placed[chosen][row["chapter"]] += 1

    # strongest rule goes first
    _pull_groups_to_cover(groups, assignment, placed, to_split="test",
                          chapter_floor=min_test_rows,
                          protected_floors={})
    _pull_groups_to_cover(groups, assignment, placed, to_split="val",
                          chapter_floor=MIN_REPRESENTATION_ROWS,
                          protected_floors={"test": min_test_rows})
    _pull_groups_to_cover(groups, assignment, placed, to_split="train",
                          chapter_floor=MIN_REPRESENTATION_ROWS,
                          protected_floors={"test": min_test_rows,
                                            "val": MIN_REPRESENTATION_ROWS})

    split_rows = {split_name: [] for split_name in SPLIT_NAMES}
    for row in rows:
        split_rows[assignment[leakage_group_key(row)]].append(row)
    return split_rows


def _pin_concentrated_blocks_to_test(groups, min_test_rows):
    # the case when the chapter is only in one source
    chapter_totals = collections.Counter()
    largest_pure_block = {}
    for group_key, group_rows in groups.items():
        chapters_present = {row["chapter"] for row in group_rows}
        for chapter in chapters_present:
            chapter_totals[chapter] += sum(row["chapter"] == chapter
                                           for row in group_rows)
        if len(chapters_present) == 1:
            chapter = next(iter(chapters_present))
            current = largest_pure_block.get(chapter)
            if current is None or len(group_rows) > len(groups[current]):
                largest_pure_block[chapter] = group_key

    pinned = {}
    need = 2 * MIN_REPRESENTATION_ROWS  # one row each for train and val
    for chapter, block_key in largest_pure_block.items():
        block_size = len(groups[block_key])
        scattered = chapter_totals[chapter] - block_size
        if block_size >= min_test_rows and scattered < min_test_rows + need:
            pinned[block_key] = "test"
    return pinned


def _pull_groups_to_cover(groups, assignment, placed, to_split,
                          chapter_floor, protected_floors):
    # move whole groups into to_split until every chapter hits its floor

    def may_leave(group_key, from_split):
        floor = protected_floors.get(from_split)
        if floor is None:
            return True
        taken = collections.Counter(row["chapter"]
                                    for row in groups[group_key])
        return all(placed[from_split][chapter] - count >= floor
                   for chapter, count in taken.items())

    for chapter in CHAPTER_SLUGS:
        while placed[to_split][chapter] < chapter_floor:
            shortfall = chapter_floor - placed[to_split][chapter]
            candidates = [
                group_key for group_key, split_name in assignment.items()
                if split_name != to_split
                and any(row["chapter"] == chapter for row in groups[group_key])
                and may_leave(group_key, split_name)
            ]
            if not candidates:
                break

            def wasted_rows(group_key):
                group_rows = groups[group_key]
                needed = sum(row["chapter"] == chapter for row in group_rows)
                useful = min(needed, shortfall)
                return (len(group_rows) - useful, -needed)

            moving = min(candidates, key=wasted_rows)
            old_split = assignment[moving]
            assignment[moving] = to_split
            for row in groups[moving]:
                placed[old_split][row["chapter"]] -= 1
                placed[to_split][row["chapter"]] += 1


def _require_known_chapters_and_unique_ids(rows):
    # ids must be unique before anything is grouped; a duplicate id silently
    # merges two questions into one
    id_counts = collections.Counter(row.get("id") for row in rows)
    bad_ids = sorted(str(row_id) for row_id, count in id_counts.items()
                     if count > 1 or row_id in (None, ""))
    if bad_ids:
        raise RowConservationError(
            f"input rows carry missing or duplicate ids: {bad_ids[:5]}")
    unknown = sorted({row["chapter"] for row in rows} - set(CHAPTER_SLUGS))
    if unknown:
        raise SplitIntegrityError(f"unknown chapter slugs in input: {unknown}")


# ---------------------------------------------------------------------------
# checks

def check_no_group_straddles_splits(split_rows):
    # Rule 1: no group ends up in two splits.
    first_seen_in = {}
    for split_name in SPLIT_NAMES:
        for row in split_rows[split_name]:
            group_key = leakage_group_key(row)
            earlier = first_seen_in.setdefault(group_key, split_name)
            if earlier != split_name:
                raise GroupLeakageError(
                    f"group {group_key} has rows in both '{earlier}' and "
                    f"'{split_name}'; questions from one paper must never "
                    f"face each other across the train/test fence")


def check_every_chapter_has_test_rows(split_rows,
                                      min_test_rows=DEFAULT_MIN_TEST_ROWS):
    # Rule 2: every chapter has enough test rows to be measured.
    test_counts = collections.Counter(row["chapter"]
                                      for row in split_rows["test"])
    shortfalls = {chapter: test_counts[chapter] for chapter in CHAPTER_SLUGS
                  if test_counts[chapter] < min_test_rows}
    if shortfalls:
        listing = ", ".join(f"'{chapter}' has {count}"
                            for chapter, count in shortfalls.items())
        raise ChapterTestShortfallError(
            f"every chapter needs at least {min_test_rows} test rows, but "
            f"{listing}; too few rows exist to evaluate these chapters")


def check_rows_conserved(input_rows, split_rows):
    # Rule 3: no row is lost, duplicated or invented.
    input_ids = collections.Counter(row["id"] for row in input_rows)
    output_ids = collections.Counter(row["id"] for split_name in SPLIT_NAMES
                                     for row in split_rows[split_name])
    lost = sorted(set(input_ids) - set(output_ids))
    duplicated = sorted(row_id for row_id, count in output_ids.items()
                        if count > input_ids[row_id])
    if lost or duplicated:
        raise RowConservationError(
            f"rows lost: {lost[:5]} ({len(lost)} total); "
            f"rows duplicated or invented: {duplicated[:5]} "
            f"({len(duplicated)} total)")


def check_split_is_reproducible(input_rows, split_rows, seed,
                                min_test_rows):
    # Rule 4: the same seed gives the same split every time.
    rerun = assign_groups_to_splits(input_rows, seed, min_test_rows)
    for split_name in SPLIT_NAMES:
        original_ids = [row["id"] for row in split_rows[split_name]]
        rerun_ids = [row["id"] for row in rerun[split_name]]
        if original_ids != rerun_ids:
            raise SplitNotReproducibleError(
                f"seed {seed} produced a different '{split_name}' split on a "
                f"second run; the assignment has a non-deterministic step")


# ---------------------------------------------------------------------------
# Summary

def summarise_assignment(split_rows, min_test_rows=DEFAULT_MIN_TEST_ROWS):
    # The summary exists to be read by eye before the numbers are trusted.
    total_rows = sum(len(assigned) for assigned in split_rows.values())
    counts_by_split = {
        split_name: collections.Counter(row["chapter"] for row in assigned)
        for split_name, assigned in split_rows.items()
    }

    lines = [f"{'split':<8}{'rows':>6}{'share':>9}{'target':>9}{'groups':>9}"]
    for split_name in SPLIT_NAMES:
        assigned = split_rows[split_name]
        group_count = len({leakage_group_key(row) for row in assigned})
        lines.append(f"{split_name:<8}{len(assigned):>6}"
                     f"{len(assigned) / total_rows:>9.1%}"
                     f"{SPLIT_FRACTIONS[split_name]:>9.0%}{group_count:>9}")

    ungroupable = sum(1 for assigned in split_rows.values() for row in assigned
                      if leakage_group_key(row)[0] == "row")
    lines.append(f"rows with neither provenance nor a source file, "
                 f"each standing as its own group: {ungroupable}")

    lines.append("")
    lines.append(f"{'chapter':<38}{'total':>7}{'train':>7}{'val':>6}"
                 f"{'test':>6}{'test share':>12}")
    for chapter in CHAPTER_SLUGS:
        chapter_total = sum(counts_by_split[split_name][chapter]
                            for split_name in SPLIT_NAMES)
        test_count = counts_by_split["test"][chapter]
        test_share = test_count / chapter_total if chapter_total else 0.0
        lines.append(f"{chapter:<38}{chapter_total:>7}"
                     f"{counts_by_split['train'][chapter]:>7}"
                     f"{counts_by_split['val'][chapter]:>6}"
                     f"{test_count:>6}{test_share:>12.1%}")

    thinnest = min(CHAPTER_SLUGS,
                   key=lambda chapter: counts_by_split["test"][chapter])
    lines.append("")
    lines.append(f"smallest test chapter: {thinnest} with "
                 f"{counts_by_split['test'][thinnest]} rows "
                 f"(floor is {min_test_rows})")

    # flag any chapter with zero rows in train or val
    for split_name in ("train", "val"):
        uncovered = [chapter for chapter in CHAPTER_SLUGS
                     if counts_by_split[split_name][chapter] == 0]
        if uncovered:
            lines.append(f"chapters with no {split_name} rows at all: "
                         f"{', '.join(uncovered)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Reading and writing
def read_rows(path):
    with Path(path).open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_split_files(input_path, seed=DEFAULT_SEED,
                      min_test_rows=DEFAULT_MIN_TEST_ROWS,
                      project_root=PROJECT_ROOT):
    # every check runs before a single file is written
    input_path = Path(input_path)
    output_dir = input_path.parent
    assert_inside_project(output_dir, project_root)

    rows = read_rows(input_path)
    split_rows = assign_groups_to_splits(rows, seed, min_test_rows)

    check_no_group_straddles_splits(split_rows)
    check_rows_conserved(rows, split_rows)
    check_every_chapter_has_test_rows(split_rows, min_test_rows)
    check_split_is_reproducible(rows, split_rows, seed, min_test_rows)

    for split_name in SPLIT_NAMES:
        output_path = output_dir / f"{split_name}.jsonl"
        with output_path.open("w", encoding="utf-8") as out:
            for row in split_rows[split_name]:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")

    header = (f"split of {input_path.name}: {len(rows)} rows, "
              f"seed {seed}, test floor {min_test_rows} rows per chapter")
    return header + "\n\n" + summarise_assignment(split_rows, min_test_rows)


def main():
    parser = argparse.ArgumentParser(
        description="Split the question dataset into train, validation and test.")
    parser.add_argument("input",
                        help=".jsonl of questions; writes train/val/test.jsonl beside it")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="seed for the split (default %(default)s)")
    parser.add_argument("--min-test-rows", type=int,
                        default=DEFAULT_MIN_TEST_ROWS,
                        help="minimum test rows needed per chapter (default %(default)s)")
    arguments = parser.parse_args()
    print(write_split_files(arguments.input, seed=arguments.seed,
                            min_test_rows=arguments.min_test_rows))


if __name__ == "__main__":
    main()
