"""Take a fraction of the training data, whole papers at a time.

The question this exists for is whether more data is the binding constraint.
Answering it means training each approach on a quarter, a half, three quarters
and all of the training pool, and watching what the test accuracy does. If it is
still climbing at the last point, more questions would buy accuracy; if it has
flattened, they would not, and the ceiling is somewhere else.

Two rules keep that measurement honest, and both are the same rules src/split.py
already enforces on the split itself:

* The test split never moves. Every point on the curve is scored on the same
  sealed 246 questions, so the points are comparable to each other and to the
  numbers in RESULTS.md. Only the training pool shrinks.
* A subsample takes whole groups, never single questions. A group is one paper,
  or one tutorial for questions with no provenance (`split.leakage_group_key`).
  Dropping half the questions of a paper and keeping the rest would leave the
  model a paper's house style and context at a quarter of the row count, which
  is not what a quarter of the data looks like.

Train and validation are subsampled independently at the same fraction, so the
procedure at every point is the one both approaches are measured under: tune on
validation, refit on training plus validation, score the sealed test rows. The
ratio between the two halves stays where the split put it.
"""

import random

from split import SPLIT_NAMES, leakage_group_key

FULL = 1.0


class SubsampleError(Exception): pass


def _groups_in_order(rows, seed, split_name):
    """The rows' groups, shuffled reproducibly.

    Sorted before the shuffle because the caller's row order decides dict
    insertion order, and a subsample that depends on which order the file
    happened to be read in is not reproducible from its seed.
    """
    groups = {}
    for row in rows:
        groups.setdefault(leakage_group_key(row), []).append(row)
    ordering = sorted(groups, key=repr)
    random.Random(f"{seed}:{split_name}").shuffle(ordering)
    return groups, ordering


def subsample_rows(rows, fraction, seed, split_name="train"):
    """A fraction of these rows, taking whole groups until the target is met.

    The last group accepted may carry the count past the target rather than
    stopping short of it, because a group is indivisible: the alternative is to
    split a paper, which is the one thing this must not do.
    """
    if not 0 < fraction <= FULL:
        raise SubsampleError(f"fraction must be in (0, 1], got {fraction}")
    if fraction == FULL:
        return list(rows)
    if not rows:
        return []

    target = max(1, round(fraction * len(rows)))
    groups, ordering = _groups_in_order(rows, seed, split_name)
    kept_keys, kept = [], 0
    for group_key in ordering:
        if kept >= target:
            break
        kept_keys.append(group_key)
        kept += len(groups[group_key])

    keep = set(kept_keys)
    return [row for row in rows if leakage_group_key(row) in keep]


def subsample_training_pool(split_rows, fraction, seed):
    """Both halves of the training pool at one fraction, test left alone.

    Returns a dict of the same shape as src/split.py's, so the callers below it
    cannot tell a subsampled pool from a whole one.
    """
    reduced = {"test": list(split_rows["test"])}
    for split_name in ("train", "val"):
        reduced[split_name] = subsample_rows(
            split_rows[split_name], fraction, seed, split_name)
    check_subsample(split_rows, reduced, fraction)
    return reduced


# ---------------------------------------------------------------------------
# checks, run before anything is trained on the result

def check_subsample(split_rows, reduced, fraction):
    """Four things that must hold of every point on the curve."""
    # Rule 1: the test split is untouched, or the points are not comparable.
    if [row["id"] for row in reduced["test"]] != [row["id"] for row in split_rows["test"]]:
        raise SubsampleError("the test split moved; every point on the curve "
                             "must be scored on the same sealed questions")

    # Rule 2: no group is half kept. Whole papers in or out, never split.
    for split_name in ("train", "val"):
        kept_ids = {row["id"] for row in reduced[split_name]}
        by_group = {}
        for row in split_rows[split_name]:
            by_group.setdefault(leakage_group_key(row), []).append(row)
        for group_key, group_rows in by_group.items():
            present = sum(row["id"] in kept_ids for row in group_rows)
            if present not in (0, len(group_rows)):
                raise SubsampleError(
                    f"group {group_key} is {present} of {len(group_rows)} rows "
                    f"in the '{split_name}' subsample; a paper is kept whole or "
                    f"dropped whole")

    # Rule 3: every kept row was in the split it is now in, and only once.
    for split_name in SPLIT_NAMES:
        original = [row["id"] for row in split_rows[split_name]]
        kept = [row["id"] for row in reduced[split_name]]
        if len(set(kept)) != len(kept) or not set(kept) <= set(original):
            raise SubsampleError(
                f"the '{split_name}' subsample invented or duplicated rows")

    # Rule 4: something is left to train on and something to tune on. A
    # validation split emptied by the subsample would silently turn the epoch
    # choice into a coin toss rather than raising.
    for split_name in ("train", "val"):
        if not reduced[split_name]:
            raise SubsampleError(
                f"fraction {fraction} left the '{split_name}' split empty; "
                f"the curve cannot have a point here")


def pool_rows(reduced):
    """Rows the model is finally refit on: training plus validation."""
    return len(reduced["train"]) + len(reduced["val"])
