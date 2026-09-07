"""Generate LEARNING-CURVE.md and its plot from the run ledger.

The question the curve answers is whether more data is the binding constraint.
Every number here is read from the ledger in data/private/runs/ (see
src/runs.py), where each point is a from-scratch run under the same procedure
as the headline comparison, scored on the same sealed test split, differing
only in how much of the training pool it saw. A record made against another
split is ignored by the same hash guard that protects RESULTS.md, so a point
cannot be earned on rows the model trained on.

The shape verdict is chosen by a rule fixed in this file, before the runs were
made: a curve is still climbing at the last point only when the paired test
between the previous point and the last is below p < 0.05 for every seed with
the same side ahead each time. That is the rule RESULTS.md already uses to
decide whether two sets of predictions differ, applied to two points on one
curve rather than to two approaches.

Run from the project root:  python3 tools/report_learning_curve.py
The ledger is filled by:     python3 tools/learning_curve.py --record
"""

import argparse
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from features import read_split  # noqa: E402
from intervals import mcnemar_exact_p, paired_difference_interval, paired_outcomes  # noqa: E402
from paths import assert_inside_project  # noqa: E402
from report_results import (SIGNIFICANCE, dataset_fingerprint, duration,  # noqa: E402
                            interval_text, pct, points, seed_list,
                            summarise_runs)
from runs import (APPROACH_NAMES, APPROACH_ONE, APPROACH_TWO, METHOD,  # noqa: E402
                  load_runs)

from learning_curve import FULL  # noqa: E402  (src/, via the insert above)

CURVE_PATH = PROJECT_ROOT / "LEARNING-CURVE.md"
DOCS_DIR = PROJECT_ROOT / "docs"
PLOT_PATH = DOCS_DIR / "learning-curve.png"

FRACTIONS = (0.25, 0.50, 0.75, FULL)

# A gain smaller than this between the first and last point is called flat
# rather than growth, whichever way the paired test at the top falls. Half a
# point on 246 test questions is one question either way.
NEGLIGIBLE_GAIN = 0.005

# The points were trained hours apart on a laptop that was in use between them,
# some of them niced down to keep it responsive. Contention moves wall-clock by
# more than the data size does, so the train-time column below is a record of
# what each run cost on the day and not a measurement of how training scales.
# It changes no accuracy: nice level and thread count move speed, not weights.
TIMING_CAVEAT = ("Train time is wall-clock on a laptop that was in use between "
                 "runs, so it reflects contention as much as data size and is "
                 "not a measurement of how training scales.")


# ---------------------------------------------------------------------------
# Reading the ledger

def curve_points(approach, y_true, fractions=FRACTIONS):
    """One summary per fraction the ledger holds runs for, in order."""
    measured = []
    for fraction in fractions:
        runs = load_runs(approach, fraction=fraction)
        if not runs:
            continue
        summary = summarise_runs(y_true, runs)
        summary["fraction"] = fraction
        summary["train_rows"] = statistics.fmean(run.rows_trained_on()
                                                 for run in runs)
        summary["runs"] = {run.seed: run for run in runs}
        measured.append(summary)
    return measured


def missing_fractions(approach, fractions=FRACTIONS):
    return [fraction for fraction in fractions
            if not load_runs(approach, fraction=fraction)]


# ---------------------------------------------------------------------------
# The shape of a curve

def increments(measured):
    """The gain in accuracy from each point to the next."""
    return [(earlier["fraction"], later["fraction"],
             later["mean"] - earlier["mean"])
            for earlier, later in zip(measured, measured[1:])]


def paired_last_step(y_true, measured):
    """The last step of the curve, tested seed by seed on the same questions.

    Both sides are the same approach at the same seed, so the questions and the
    procedure are held fixed and the only difference is how much data the run
    was given. Seeds the ledger holds on one side only are skipped rather than
    paired against a different run.
    """
    if len(measured) < 2:
        return []
    earlier, later = measured[-2], measured[-1]
    rows = []
    for seed in sorted(set(earlier["runs"]) & set(later["runs"])):
        before = earlier["runs"][seed].predicted
        after = later["runs"][seed].predicted
        both, only_before, only_after, neither = paired_outcomes(
            y_true, before, after)
        rows.append({
            "seed": seed, "both": both, "only_before": only_before,
            "only_after": only_after, "neither": neither,
            "difference": (only_after - only_before) / len(y_true),
            "interval": paired_difference_interval(only_after, only_before,
                                                   len(y_true)),
            "p": mcnemar_exact_p(only_after, only_before),
        })
    return rows


def shape(measured, pairs):
    """Climbing, flat or mixed at the last point, by the rule fixed above.

    Returns (verdict, sentence). The verdict is decided by the paired tests,
    not by whether the mean happened to tick up: on 246 questions a point of
    movement is two or three questions, and this report exists to say which
    kind of movement it was.
    """
    if len(measured) < 2:
        return "incomplete", (
            "**The shape of this curve is not measurable yet.** The ledger "
            "holds one point against the current test split, and a shape needs "
            "at least two.")
    first, last = measured[0], measured[-1]
    total_gain = last["mean"] - first["mean"]
    step = last["mean"] - measured[-2]["mean"]
    span = f"{first['fraction']:.0%} to {last['fraction']:.0%}"
    last_step = f"{measured[-2]['fraction']:.0%} to {last['fraction']:.0%}"
    p_values = [row["p"] for row in pairs]
    p_range = (f"p = {min(p_values):.2f}" if len(pairs) == 1
               else f"p between {min(p_values):.2f} and {max(p_values):.2f}")
    significant = [row for row in pairs if row["p"] < SIGNIFICANCE]
    climbing = {row["difference"] > 0 for row in significant}
    growth = (f"Accuracy rises {points(total_gain, signed=True)} across {span}, "
              f"and the last step, {last_step}, is "
              f"{points(step, signed=True)}.")

    if abs(total_gain) < NEGLIGIBLE_GAIN:
        return "flat", (
            f"**The curve is flat: more data is not the binding constraint "
            f"here.** {growth} Nothing in the range measured moves the accuracy.")
    if not significant:
        return "flattened", (
            f"**The curve has flattened: more data is not the binding "
            f"constraint.** {growth} The paired test on that last step does not "
            f"reach p < {SIGNIFICANCE} for any of the {len(pairs)} seeds "
            f"({p_range}), so the last quarter of the training pool bought "
            f"nothing measurable, and a further quarter would be expected to "
            f"buy no more.")
    if len(significant) == len(pairs) and len(climbing) == 1:
        if climbing.pop():
            return "climbing", (
                f"**The curve is still climbing at {last['fraction']:.0%}: more "
                f"data would help.** {growth} The paired test on that last step "
                f"is below p < {SIGNIFICANCE} for every one of the {len(pairs)} "
                f"seeds ({p_range}), so the last quarter of the training pool "
                f"was still buying accuracy when the corpus ran out.")
        return "falling", (
            f"**The curve turns down at the last point.** {growth} The paired "
            f"test is below p < {SIGNIFICANCE} for every one of the "
            f"{len(pairs)} seeds ({p_range}) with the smaller training pool "
            f"ahead, which is a result about this corpus rather than about how "
            f"much data the approach needs.")
    return "mixed", (
        f"**The evidence at the last point is mixed.** {growth} The paired test "
        f"is below p < {SIGNIFICANCE} for {len(significant)} of the "
        f"{len(pairs)} seeds ({p_range}), which is not enough to say the curve "
        f"is still climbing or that it has flattened.")


def curves_differ(one, two):
    """Whether the two approaches' curves have the same shape, and how they differ."""
    one_gain = one[-1]["mean"] - one[0]["mean"]
    two_gain = two[-1]["mean"] - two[0]["mean"]
    difference = two_gain - one_gain
    steeper = (APPROACH_NAMES[APPROACH_TWO] if difference > 0
               else APPROACH_NAMES[APPROACH_ONE])
    if abs(difference) < NEGLIGIBLE_GAIN:
        return (f"The two curves climb by the same amount over the range "
                f"measured: {points(one_gain, signed=True)} for Approach 1 "
                f"against {points(two_gain, signed=True)} for Approach 2. "
                f"Neither approach is the one waiting on more data.")
    return (f"The two curves are not the same shape. Over the range measured "
            f"Approach 1 gains {points(one_gain, signed=True)} and Approach 2 "
            f"{points(two_gain, signed=True)}, so {steeper} is the one that "
            f"benefits more from data, by {points(abs(difference))}. That "
            f"comparison is between two means of three seeds each on 246 test "
            f"questions, so it is a difference in slope worth naming rather "
            f"than a precise measurement of one.")


# ---------------------------------------------------------------------------
# Output

def curve_table(measured):
    lines = ["| Training pool | Rows trained on | Mean test accuracy | "
             "Lowest to highest | 95% interval | Macro-F1 | Train time |",
             "|---|---|---|---|---|---|---|"]
    for point in measured:
        lines.append(
            f"| {point['fraction']:.0%} | {point['train_rows']:,.0f} | "
            f"**{pct(point['mean'])}** | {pct(point['low'])} to "
            f"{pct(point['high'])} | {interval_text(point['interval'])} | "
            f"{point['macro_f1']:.2f} | {duration(point['train_seconds'])} |")
    return "\n".join(lines)


def increment_table(measured):
    lines = ["| Step | Rows added | Accuracy gained |", "|---|---|---|"]
    for earlier, later in zip(measured, measured[1:]):
        lines.append(
            f"| {earlier['fraction']:.0%} to {later['fraction']:.0%} | "
            f"{later['train_rows'] - earlier['train_rows']:,.0f} | "
            f"{points(later['mean'] - earlier['mean'], signed=True)} |")
    return "\n".join(lines)


def paired_table(pairs):
    lines = ["| Seed | Both right | Only the smaller pool right | "
             "Only the full pool right | Both wrong | Difference | "
             "95% interval on the difference | Exact p |",
             "|---|---|---|---|---|---|---|---|"]
    for row in pairs:
        lines.append(
            f"| {row['seed']} | {row['both']} | {row['only_before']} | "
            f"{row['only_after']} | {row['neither']} | "
            f"{points(row['difference'], signed=True)} | "
            f"{points(row['interval'][0], signed=True)} to "
            f"{points(row['interval'][1], signed=True)} | {row['p']:.2f} |")
    return "\n".join(lines)


def draw_curve(measured_by_approach, destination):
    """Mean accuracy against rows trained on, one line per approach."""
    figure, axes = plt.subplots(figsize=(7.5, 4.5))
    for approach, measured in measured_by_approach.items():
        x = [point["train_rows"] for point in measured]
        y = [point["mean"] * 100 for point in measured]
        # clamped, because a point whose seeds all agree can land a hair
        # below zero in floating point and matplotlib refuses a negative bar
        low = [max(0.0, (point["mean"] - point["low"]) * 100) for point in measured]
        high = [max(0.0, (point["high"] - point["mean"]) * 100) for point in measured]
        axes.errorbar(x, y, yerr=[low, high], marker="o", capsize=4,
                      label=APPROACH_NAMES[approach])
    axes.set_xlabel("questions trained on (training plus validation)")
    axes.set_ylabel("test accuracy, %")
    axes.set_title("Learning curve on the sealed test split")
    axes.grid(alpha=0.3)
    axes.legend()
    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    plt.close(figure)


def approach_section(approach, measured, missing, pairs):
    name = APPROACH_NAMES[approach]
    lines = [f"## {name}", ""]
    if not measured:
        return lines + [
            f"The ledger holds no {name} runs against the current test split, "
            f"so this curve is not measurable. "
            f"`python3 tools/learning_curve.py --approach {approach} --record` "
            f"puts it back."]
    lines += [
        f"Runs at seeds {seed_list(measured[0]['seeds'])}, each point a "
        f"from-scratch run on that fraction of the training pool.",
        "",
        curve_table(measured),
        "",
        "Timed on: " + "; ".join(sorted(
            {hardware for point in measured for hardware in point["hardware"]}))
        + ". " + TIMING_CAVEAT,
        "",
    ]
    if len(measured) > 1:
        lines += [increment_table(measured), ""]
    if missing:
        lines += [
            "**This curve is incomplete.** The ledger holds no runs at "
            + ", ".join(f"{fraction:.0%}" for fraction in missing)
            + ", so those points are absent rather than estimated.",
            "",
        ]
    if pairs:
        lines += [
            f"The last step of the curve, tested question by question. Each row "
            f"pairs one seed's {measured[-2]['fraction']:.0%} run against the "
            f"same seed's {measured[-1]['fraction']:.0%} run on the same test "
            f"questions, so nothing differs between the two but the amount of "
            f"training data.",
            "",
            paired_table(pairs),
            "",
        ]
    lines.append(shape(measured, pairs)[1])
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dated", default="",
                        help="date to stamp on the report, e.g. 2026-09-07")
    arguments = parser.parse_args()

    assert_inside_project(CURVE_PATH)
    assert_inside_project(DOCS_DIR)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    y_true = read_split("test")[1]
    fingerprint = dataset_fingerprint()

    measured = {approach: curve_points(approach, y_true)
                for approach in (APPROACH_ONE, APPROACH_TWO)}
    pairs = {approach: paired_last_step(y_true, points_)
             for approach, points_ in measured.items()}
    drawable = {approach: points_ for approach, points_ in measured.items()
                if len(points_) > 1}
    if drawable:
        draw_curve(drawable, PLOT_PATH)

    stamp = f" Generated {arguments.dated}." if arguments.dated else ""
    sections = [
        "# Learning curve",
        "",
        f"Generated by `python3 tools/report_learning_curve.py` from the run "
        f"ledger.{stamp} Nothing here is typed by hand.",
        "",
        "## What was varied, and what was not",
        "",
        f"Each point trains one approach on a fraction of the training pool and "
        f"scores it on the sealed test split. The procedure is the one "
        f"RESULTS.md measures both approaches under: {METHOD}. Only the amount "
        f"of training data changes.",
        "",
        f"A subsample takes whole papers, never single questions, by the same "
        f"grouping `src/split.py` uses: dropping half of a paper's questions "
        f"and keeping the rest would leave the model that paper's house style "
        f"and context at a quarter of the row count, which is not what a "
        f"quarter of the data looks like. Training and validation are "
        f"subsampled independently at the same fraction, so the split's own "
        f"proportions hold at every point.",
        "",
        f"The test split never moves. It is `test.jsonl`, "
        f"{fingerprint['counts']['test']} questions, sha256 "
        f"`{fingerprint['test_hash']}`, out of the frozen "
        f"`data/private/questions.jsonl`, {fingerprint['rows']} labelled rows, "
        f"sha256 `{fingerprint['hash']}`. Every run in the ledger carries that "
        f"test hash and the fraction it was trained on; a run against any other "
        f"split is ignored, and a fractional run cannot reach RESULTS.md.",
        "",
    ]
    if drawable:
        sections += ["![Learning curve](docs/learning-curve.png)", ""]

    for approach in (APPROACH_ONE, APPROACH_TWO):
        sections += approach_section(approach, measured[approach],
                                     missing_fractions(approach),
                                     pairs[approach]) + [""]

    sections += ["## Do the two curves have the same shape?", ""]
    if all(len(points_) > 1 for points_ in measured.values()):
        sections.append(curves_differ(measured[APPROACH_ONE],
                                      measured[APPROACH_TWO]))
    else:
        sections.append(
            "Only one approach has a curve against the current test split, so "
            "there is nothing to compare its shape to.")

    CURVE_PATH.write_text("\n".join(sections) + "\n", encoding="utf-8")
    print(f"wrote {CURVE_PATH.relative_to(PROJECT_ROOT)}")
    if drawable:
        print(f"wrote {PLOT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
