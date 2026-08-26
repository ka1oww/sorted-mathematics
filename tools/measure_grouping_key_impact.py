"""Count what normalising the paper field changed about the grouping.

src/split.py used to build its group key from the raw `(school, year, paper)`
tuple. Schools print the same paper as `II` on one cover and `P2` on another,
so one paper handed back two group keys, and its questions could sit on both
sides of the train/test fence with all four integrity checks passing, because
each half really did sit in one split.

This measures the damage on the real corpus rather than arguing about it: how
many groups the two keys produce, how many questions moved, and - the number
that actually matters - how many of the frozen test questions came from a
paper whose other questions the model had already trained on.

It also prices the fix that was not made. The school field still goes into the
key raw, so a school recorded under more than one name across the years is
still more than one group. The alias fold is a measurement, not something
src/split.py uses.

Prints real school/year/paper triples, because that is what it is measuring.
Worth remembering before pasting its output anywhere public.

Run from the project root:  python3 tools/measure_grouping_key_impact.py
"""

import collections
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from merge import normalise_paper  # noqa: E402
from paths import PRIVATE_DATA  # noqa: E402

# The corpus records one school under three names: the short form, the same
# form with its JC section spelled out, and the name of a college absorbed into
# it years before any of these papers are dated. The names are real, so like
# the corpus they live in data/private rather than here:
#
#     data/private/school_aliases.json   {"XYZ(JC)": "XYZ", "XYZJC": "XYZ"}
#
# With the file absent the fold is a no-op and the residual reports zero.
# Deciding that two names are really one school is a judgement about the source
# papers rather than a house-style fold like the paper numbering, so
# src/split.py does not make it either way.
SCHOOL_ALIASES_FILE = PRIVATE_DATA / "school_aliases.json"


def read_school_aliases():
    if not SCHOOL_ALIASES_FILE.exists():
        return {}
    return json.loads(SCHOOL_ALIASES_FILE.read_text(encoding="utf-8"))


SCHOOL_ALIASES = read_school_aliases()


def read_rows(name):
    path = PRIVATE_DATA / f"{name}.jsonl"
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def straddling_groups(split_rows, **key_options):
    """Groups with rows on more than one side of the fence, and their test rows."""
    seen_in = collections.defaultdict(collections.Counter)
    for split_name, rows in split_rows.items():
        for row in rows:
            seen_in[group_key(row, **key_options)][split_name] += 1
    return {key: counts for key, counts in seen_in.items() if len(counts) > 1}


def group_key(row, normalise_the_paper=True, fold_school_aliases=False):
    """The splitter's key, with each of the two folds independently switchable."""
    school, year = row.get("school"), row.get("year")
    if fold_school_aliases and school:
        school = SCHOOL_ALIASES.get(school, school)
    if school is not None or year is not None:
        paper = row.get("paper")
        return ("paper", school, year,
                normalise_paper(paper) if normalise_the_paper else paper)

    source_file = row.get("source_file")
    if source_file:
        return ("file", source_file)
    row_id = row.get("id", "")
    if ":" in row_id:
        return ("file", row_id.rsplit(":", 1)[0])
    return ("row", row_id)


def count_groups(rows, **key_options):
    return collections.Counter(group_key(row, **key_options) for row in rows)


def report_group_counts(rows):
    """How many groups each key produces, and how many questions moved."""
    old = count_groups(rows, normalise_the_paper=False)
    new = count_groups(rows)
    old_papers = [key for key in old if key[0] == "paper"]
    new_papers = [key for key in new if key[0] == "paper"]

    print(f"{'=' * 72}")
    print(f"GROUPING KEY, {len(rows)} questions")
    print(f"{'=' * 72}")
    print(f"  raw paper field   {len(old):>4} groups, {len(old_papers):>4} of them papers")
    print(f"  normalised        {len(new):>4} groups, {len(new_papers):>4} of them papers")

    merged = {}
    for key in new_papers:
        halves = {group_key(row, normalise_the_paper=False) for row in rows
                  if group_key(row) == key}
        if len(halves) > 1:
            merged[key] = (halves, new[key])
    moved = sum(rows_in_group for _, rows_in_group in merged.values())
    print(f"\n  {len(merged)} papers were being counted as more than one group")
    print(f"  {moved} questions sit in them")
    for key, (halves, rows_in_group) in sorted(merged.items(),
                                               key=lambda item: -item[1][1]):
        spellings = sorted(str(half[3]) for half in halves)
        print(f"    {key[1]} {key[2]} {key[3]:<3} {rows_in_group:>3} questions, "
              f"filed under {', '.join(spellings)}")
    return merged


def report_straddling_in_frozen_split():
    """Price both keys against the frozen split: how much each one lets through.

    A group straddling the fence is the leak itself; the test rows inside one
    are the questions whose accuracy was not honestly earned, because their
    siblings from the same paper were in the training data.

    Both keys read zero on a split that was generated with the normalised key,
    which is what a fix looks like once it has been applied. The 25-of-241 the
    README records is a measurement of the split that existed before it, and it
    is a record rather than something this tool can re-run.
    """
    split_rows = {name: read_rows(name) for name in ("train", "val", "test")}
    print(f"\n{'-' * 72}\nWHAT EACH KEY LETS THROUGH, in the frozen split\n{'-' * 72}")
    for label, options in (("raw paper field", {"normalise_the_paper": False}),
                           ("normalised", {})):
        straddling = straddling_groups(split_rows, **options)
        contaminated = sum(counts["test"] for counts in straddling.values())
        for key, counts in sorted(straddling.items(),
                                  key=lambda item: -item[1]["test"]):
            spread = ", ".join(f"{split_name} {count}"
                               for split_name, count in sorted(counts.items()))
            print(f"    {key[1]} {key[2]} {key[3]:<3} {spread}")
        print(f"  {label:<18} {len(straddling)} papers on both sides, "
              f"{contaminated} of the {len(split_rows['test'])} test questions "
              f"in one of them")


def report_school_residual(rows):
    """Size the leak the school field still leaves open, since it is not folded.

    Reported rather than fixed. Folding two school names into one welds their
    papers together for good, and if the names really are two schools that
    costs stratification on real data to buy nothing.

    Two numbers, because they answer different questions. The corpus-level count
    is how many groups the fold would merge; the split-level count is how many
    test questions the unfolded key actually exposes, which is the one the
    README quotes.
    """
    print(f"\n{'-' * 72}\nWHAT THE SCHOOL FIELD STILL LEAVES OPEN\n{'-' * 72}")
    if not SCHOOL_ALIASES:
        print(f"  no alias map at {SCHOOL_ALIASES_FILE.relative_to(PROJECT_ROOT)}, "
              f"so there is nothing to fold and nothing to report")
        return

    without = count_groups(rows)
    with_aliases = count_groups(rows, fold_school_aliases=True)
    for key in with_aliases:
        if key[0] != "paper":
            continue
        halves = {group_key(row) for row in rows
                  if group_key(row, fold_school_aliases=True) == key}
        if len(halves) > 1:
            names = sorted(str(half[1]) for half in halves)
            print(f"  {key[1]} {key[2]} {key[3]:<3} {with_aliases[key]:>3} questions "
                  f"filed under {', '.join(names)}")
    print(f"\n  folding the aliases would merge "
          f"{len(without) - len(with_aliases)} further groups")

    split_rows = {name: read_rows(name) for name in ("train", "val", "test")}
    straddling = straddling_groups(split_rows, fold_school_aliases=True)
    exposed = sum(counts["test"] for counts in straddling.values())
    print("\n  in the frozen split, once the aliases are folded:")
    for key, counts in sorted(straddling.items(), key=lambda item: -item[1]["test"]):
        spread = ", ".join(f"{split_name} {count}"
                           for split_name, count in sorted(counts.items()))
        print(f"    {key[1]} {key[2]} {key[3]:<3} {spread}")
    print(f"  {len(straddling)} papers straddle the fence, exposing {exposed} of "
          f"the {len(split_rows['test'])} test questions")


def main():
    rows = read_rows("questions")
    report_group_counts(rows)
    report_straddling_in_frozen_split()
    report_school_residual(rows)


if __name__ == "__main__":
    main()
