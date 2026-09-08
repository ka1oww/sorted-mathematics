"""Generate RESULTS.md, the confusion-matrix images and the abstention figure.

Every number in RESULTS.md comes from this script reading the frozen dataset,
so nothing in it is typed by hand. DATA-PLAN.md section 7 asks for exactly
this: a row count, per-chapter counts and a content hash, with every accuracy
quoted against a named frozen dataset.

The two approaches are compared symmetrically. Both are read from the run
ledger in data/private/runs/ (see src/runs.py), where each is the same three
from-scratch seed runs under the same procedure on the same frozen split, so
neither side is a single draw. Every accuracy carries a 95% interval, both
models' predictions on the same 246 questions are put through an exact paired
test, and the verdict paragraph is chosen by a rule fixed in this file, not
typed after the numbers were seen. Both sides of that paired test are ledger
runs, so both were scored against the split the hash names; the committed
models/chapter_classifier.joblib supplies only Approach 1's pictures and tables,
and the report says whether it still reproduces the ledger runs. Approach 2 is
included only if the ledger holds transformer runs against the current test
split; without them the report covers Approach 1 and says so.

Run from the project root:  python3 tools/report_results.py
The ledger is filled by:     python3 tools/compare_runs.py seeds --record
"""

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

import joblib
import matplotlib
from sklearn.metrics import classification_report, f1_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from report_abstention import (  # noqa: E402
    ABSTENTION_FIGURE,
    RefitPredictionMismatch,
    abstention_section,
    build_abstention,
    draw_abstention_curve,
)

from abstention import abstention_not_measurable_section  # noqa: E402
from chapters import CHAPTERS, CHAPTER_SLUGS  # noqa: E402
from features import read_split, to_features  # noqa: E402
from intervals import (intervals_overlap, mcnemar_exact_p,  # noqa: E402
                       paired_difference_interval, paired_outcomes, spread,
                       wilson_interval)
from paths import MODELS_DIR, PRIVATE_DATA, assert_inside_project  # noqa: E402
from predict import classify  # noqa: E402
from runs import (APPROACH_ONE, APPROACH_TWO, DEFAULT_SEEDS, METHOD,  # noqa: E402
                  file_hash, load_runs, resplit_tfidf, split_hash)
from split import read_rows  # noqa: E402

DOCS_DIR = PROJECT_ROOT / "docs"
RESULTS_PATH = PROJECT_ROOT / "RESULTS.md"
CLASSIFIER_PATH = MODELS_DIR / "chapter_classifier.joblib"

SPLIT_NAMES = ("train", "val", "test")
TOP_CONFUSIONS = 8

# The two approaches are called distinguishable only when every transformer
# seed's paired test clears this, with the same model ahead each time.
SIGNIFICANCE = 0.05

# Approach 1 is also refit on this many fresh group-aware splits, to measure
# how much the accuracy moves with which papers land in the test set. Ten is
# enough to see the size of that movement and costs about twenty seconds.
RESPLIT_SEEDS = tuple(range(1, 11))

# Questions written for this repository rather than taken from any paper, so
# the examples can be published. One per chapter family, and the last two are
# deliberately on a boundary: a curve-sketching question that is solved by
# differentiating, and a series question that could be filed either way.
EXAMPLE_QUESTIONS = [
    ("The time a student takes to finish a practice paper is normally "
     "distributed with mean 95 minutes and standard deviation 12 minutes. Two "
     "students are chosen at random. Find the probability that exactly one of "
     "them finishes in under 90 minutes."),
    ("The complex number w satisfies |w - 3i| = |w + 4|. Sketch the locus of w "
     "on an Argand diagram, and find the least possible value of |w|."),
    ("The plane p has equation r . (2i - j + 2k) = 6. The line l passes through "
     "the point with position vector i + 3j and is parallel to i + k. Find the "
     "acute angle between l and p."),
    ("A machine is set to fill packets with 500 g of flour. A random sample of "
     "40 packets has mean mass 497.2 g and standard deviation 6.1 g. Test, at "
     "the 5% significance level, whether the machine is underfilling."),
    ("Using the substitution u = 1 + tan x, find the exact value of the "
     "integral of (sec^2 x) / (1 + tan x)^3 with respect to x, from x = 0 to "
     "x = pi/4."),
    ("Water leaks from a tank so that the depth h cm satisfies dh/dt = -k root h, "
     "where k is a positive constant. Given that the depth falls from 64 cm to "
     "36 cm in 10 minutes, find the time taken for the tank to empty."),
    ("A committee of 5 is chosen from 6 men and 7 women. Find the number of "
     "ways the committee can be formed if it must contain at least 2 men, and "
     "the oldest woman refuses to serve alongside the youngest man."),
    ("The curve C has equation y = (x^2 - 4) / (x - 1). Find the coordinates of "
     "the stationary points of C, and hence sketch C, stating the equations of "
     "its asymptotes."),
    ("The first three terms of a geometric progression are x, x + 6 and 4x - 6. "
     "Find the possible values of x, and the sum to infinity of the progression "
     "that converges."),
]


def dataset_fingerprint():
    """Row counts and a content hash, so a number can name the data it came from."""
    corpus = PRIVATE_DATA / "questions.jsonl"
    counts = {name: sum(1 for line in (PRIVATE_DATA / f"{name}.jsonl")
                        .open(encoding="utf-8") if line.strip())
              for name in SPLIT_NAMES}
    per_chapter = Counter()
    with corpus.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                per_chapter[json.loads(line)["chapter"]] += 1
    return {"counts": counts, "per_chapter": per_chapter,
            "rows": sum(counts.values()), "hash": file_hash(corpus),
            "test_hash": split_hash("test")}


def approach_one_predictions():
    """The committed model's test predictions, for the matrices and tables only.

    The headline, the paired test and the verdict all come from the ledger
    instead; see main().
    """
    question_vectoriser, chapter_classifier = joblib.load(CLASSIFIER_PATH)
    test_questions, y_test = read_split("test")
    predicted = chapter_classifier.predict(
        to_features(question_vectoriser, test_questions))
    return y_test, list(predicted)


def confusion_counts(y_true, y_predicted):
    index = {slug: position for position, slug in enumerate(CHAPTER_SLUGS)}
    matrix = [[0] * len(CHAPTER_SLUGS) for _ in CHAPTER_SLUGS]
    for true, predicted in zip(y_true, y_predicted):
        matrix[index[true]][index[predicted]] += 1
    return matrix


def draw_confusion_matrix(matrix, title, destination):
    """A 21x21 heatmap. Unreadable in a terminal, which is why it is a file."""
    labels = [CHAPTERS[slug] for slug in CHAPTER_SLUGS]
    figure, axes = plt.subplots(figsize=(11, 9.5))
    image = axes.imshow(matrix, cmap="Blues", vmin=0)

    axes.set_xticks(range(len(labels)), labels, rotation=60, ha="right", fontsize=8)
    axes.set_yticks(range(len(labels)), labels, fontsize=8)
    axes.set_xlabel("predicted chapter")
    axes.set_ylabel("true chapter")
    axes.set_title(title, pad=14)

    ceiling = max(max(row) for row in matrix) or 1
    for row_index, row in enumerate(matrix):
        for column_index, count in enumerate(row):
            if count:
                axes.text(column_index, row_index, str(count),
                          ha="center", va="center", fontsize=7,
                          color="white" if count > ceiling * 0.55 else "#22303c")
    figure.colorbar(image, ax=axes, shrink=0.75, label="questions")
    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    plt.close(figure)


def per_chapter_table(y_true, y_predicted):
    report = classification_report(y_true, y_predicted, zero_division=0,
                                   output_dict=True, labels=list(CHAPTER_SLUGS))
    lines = ["| Chapter | Precision | Recall | F1 | Test questions |",
             "|---|---|---|---|---|"]
    for slug in CHAPTER_SLUGS:
        scores = report[slug]
        lines.append(f"| {CHAPTERS[slug]} | {scores['precision']:.2f} | "
                     f"{scores['recall']:.2f} | {scores['f1-score']:.2f} | "
                     f"{int(scores['support'])} |")
    return "\n".join(lines)


def confusion_table(y_true, y_predicted):
    confusions = Counter((true, predicted)
                         for true, predicted in zip(y_true, y_predicted)
                         if true != predicted)
    lines = ["| True chapter | Called it | Times |", "|---|---|---|"]
    for (true, predicted), count in confusions.most_common(TOP_CONFUSIONS):
        lines.append(f"| {CHAPTERS[true]} | {CHAPTERS[predicted]} "
                     f"| {count} |")
    singles = sum(1 for count in confusions.values() if count == 1)
    return "\n".join(lines), singles, len(confusions)


def example_table():
    """Real predict.py output on the questions above, top three each."""
    lines = []
    for question in EXAMPLE_QUESTIONS:
        ranked = classify(question)
        lines.append(f"> {question}\n")
        for chapter, confidence in ranked:
            lines.append(f"    {confidence:6.1%}  {CHAPTERS[chapter]}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Formatting

def pct(value):
    return f"{value:.1%}"


def interval_text(interval):
    return f"{pct(interval[0])} to {pct(interval[1])}"


def points(value, signed=False):
    return f"{value * 100:+.1f} points" if signed else f"{value * 100:.1f} points"


def duration(seconds):
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, rest = divmod(round(seconds), 60)
    return f"{minutes} min {rest} s"


def macro_f1(y_true, predicted):
    return f1_score(y_true, predicted, average="macro", zero_division=0)


# ---------------------------------------------------------------------------
# The symmetric comparison

def default_seed_run(runs):
    """The one run the tables and the pairing are drawn from, not a mean."""
    return next((run for run in runs if run.seed == DEFAULT_SEEDS[0]), runs[0])


def summarise_runs(y_true, runs):
    """Mean, spread and interval for one approach's recorded seed runs.

    The interval is a Wilson score interval on the mean number of correct
    answers over the test rows. It measures the noise from which 246 questions
    were drawn, which is a different noise from the seed spread and is reported
    beside it rather than folded into it.
    """
    accuracies = [run.accuracy(y_true) for run in runs]
    summary = spread(accuracies)
    mean_correct = statistics.fmean(run.correct(y_true) for run in runs)
    summary["interval"] = wilson_interval(mean_correct, len(y_true))
    summary["macro_f1"] = statistics.fmean(macro_f1(y_true, run.predicted)
                                           for run in runs)
    summary["train_seconds"] = statistics.fmean(run.train_seconds for run in runs)
    summary["hardware"] = sorted({run.hardware for run in runs})
    summary["seeds"] = [run.seed for run in runs]
    summary["parameters"] = runs[0].parameters
    summary["features"] = runs[0].features
    return summary


def paired_comparisons(y_true, baseline_predicted, transformer_runs):
    """Approach 1 against each transformer seed, on the same questions.

    baseline_predicted is a ledger run's predictions, so both sides of every
    pair were scored against the split named by the same hash.
    """
    rows = []
    for run in transformer_runs:
        both, only_one, only_two, neither = paired_outcomes(
            y_true, baseline_predicted, run.predicted)
        rows.append({
            "seed": run.seed, "both": both, "only_one": only_one,
            "only_two": only_two, "neither": neither,
            "difference": (only_one - only_two) / len(y_true),
            "interval": paired_difference_interval(only_one, only_two, len(y_true)),
            "p": mcnemar_exact_p(only_one, only_two),
        })
    return rows


def cheaper_first(one, two):
    """The two approaches ordered by training time, with their names.

    Which of them is the cheap one is a fact about the measurement, not
    something to assume: the sentences below read off this rather than naming
    Approach 1 and hoping.
    """
    if one["train_seconds"] <= two["train_seconds"]:
        return ("Approach 1", one), ("Approach 2", two)
    return ("Approach 2", two), ("Approach 1", one)


def cost_sentence(one, two):
    (cheap_name, cheap), (dear_name, dear) = cheaper_first(one, two)
    return (f"{cheap_name} trains in {duration(cheap['train_seconds'])} "
            f"against {duration(dear['train_seconds'])} for {dear_name}")


def cost_and_size_sentence(one, two):
    """The cost sentence, plus the parameter ratio attributed to the right side."""
    (cheap_name, cheap), (_, dear) = cheaper_first(one, two)
    factor = dear["train_seconds"] / cheap["train_seconds"]
    if one["parameters"] <= two["parameters"]:
        small_name, small, big = "Approach 1", one, two
    else:
        small_name, small, big = "Approach 2", two, one
    ratio = big["parameters"] / small["parameters"]
    size = (f", with {ratio:,.0f} times fewer parameters."
            if small is cheap else
            f". {small_name} carries {ratio:,.0f} times fewer parameters.")
    return f"{cost_sentence(one, two)}, a factor of {factor:,.0f}{size}"


def verdict(one, two, pairs):
    """Decide what the numbers say, by a rule fixed before they were seen.

    Distinguishable: every seed's paired test is below SIGNIFICANCE and the
    same approach is ahead each time. Indistinguishable: no seed's is. Anything
    in between is reported as mixed, with the counts, rather than rounded to
    whichever story is tidier.
    """
    overlap = intervals_overlap(one["interval"], two["interval"])
    significant = [row for row in pairs if row["p"] < SIGNIFICANCE]
    leaders = {row["difference"] > 0 for row in significant}
    p_values = [row["p"] for row in pairs]
    p_range = (f"p = {min(p_values):.2f}" if len(pairs) == 1
               else f"p between {min(p_values):.2f} and {max(p_values):.2f}")
    overlap_text = ("the 95% intervals on the two accuracies overlap"
                    if overlap else
                    "the 95% intervals on the two accuracies do not overlap")
    gap = one["mean"] - two["mean"]
    ahead = "Approach 1" if gap >= 0 else "Approach 2"

    if not significant:
        return "indistinguishable", (
            f"**On accuracy, the two approaches are statistically "
            f"indistinguishable on this test set.** {overlap_text.capitalize()}, "
            f"and the paired test does not reach p < {SIGNIFICANCE} for any of "
            f"the {len(pairs)} transformer seeds ({p_range}). The mean gap of "
            f"{points(abs(gap), signed=True)} in {ahead}'s "
            f"favour is inside the noise of a test set this size. "
            f"What separates the two is cost, not correctness: "
            f"{cost_and_size_sentence(one, two)}"
        )
    if len(significant) == len(pairs) and len(leaders) == 1:
        winner = "Approach 1" if leaders.pop() else "Approach 2"
        return "distinguishable", (
            f"**On accuracy, {winner} is ahead, and the evidence holds up.** "
            f"The paired test is below p < {SIGNIFICANCE} for every one of the "
            f"{len(pairs)} transformer seeds ({p_range}), and {overlap_text}. "
            f"The mean gap is {points(abs(gap))}. "
            f"Even so, {cost_sentence(one, two)}."
        )
    return "mixed", (
        f"**The evidence on accuracy is mixed.** The paired test is below "
        f"p < {SIGNIFICANCE} for {len(significant)} of the {len(pairs)} "
        f"transformer seeds ({p_range}), and {overlap_text}. That is not "
        f"enough to call either approach more accurate. What separates them "
        f"is cost: {cost_sentence(one, two)}."
    )


def resplit_summary():
    """Approach 1 on fresh group-aware splits, to size the test-set noise."""
    rows = read_rows(PRIVATE_DATA / "questions.jsonl")
    results = [resplit_tfidf(rows, seed) for seed in RESPLIT_SEEDS]
    summary = spread([correct / total for correct, total in results])
    summary["test_rows"] = sorted({total for _, total in results})
    return summary


# ---------------------------------------------------------------------------
# Sections

def seed_list(seeds):
    seeds = [str(seed) for seed in seeds]
    if len(seeds) == 1:
        return seeds[0]
    return f"{', '.join(seeds[:-1])} and {seeds[-1]}"


def seeds_sentence(one, two):
    """Name the seeds the ledger actually holds, not the ones it was asked for.

    An interrupted recording run leaves fewer, and a paragraph claiming three
    above a table headed "mean of 2 seeds" is the kind of small untruth this
    report exists to avoid.
    """
    if two is None or one["seeds"] == two["seeds"]:
        return (f"Each is run from scratch at seeds {seed_list(one['seeds'])} on "
                f"the frozen split, and every cell below is the mean over those "
                f"runs.")
    return (f"Approach 1 is run from scratch at seeds {seed_list(one['seeds'])} "
            f"and Approach 2 at seeds {seed_list(two['seeds'])} on the frozen "
            f"split, and every cell below is the mean over those runs.")


def headline_section(y_true, one, two):
    lines = [
        "## Headline",
        "",
        f"Both approaches are measured the same way: {METHOD}. "
        f"{seeds_sentence(one, two)} The interval is a 95% Wilson score interval "
        f"on the mean accuracy over {len(y_true)} test questions: it says how far "
        f"the number could move with a different draw of test questions, which "
        f"is a different question from the seed spread beneath it.",
        "",
        "| | Approach 1 | Approach 2 |",
        "|---|---|---|",
    ]

    def parameter_text(summary):
        if summary["features"]:
            return (f"{summary['features']:,} features × {len(CHAPTER_SLUGS)} "
                    f"chapters, {summary['parameters']:,} weights")
        return f"{summary['parameters']:,}"

    def row(label, one_text, two_text):
        return f"| {label} | {one_text} | {two_text if two else 'not measurable'} |"

    seeds = (f"mean of {one['n']} seeds"
             if two is None or one["n"] == two["n"] else "mean across seeds")
    lines += [
        row(f"Test accuracy, {seeds}", f"**{pct(one['mean'])}**",
            two and pct(two["mean"])),
        row("95% interval on that accuracy", interval_text(one["interval"]),
            two and interval_text(two["interval"])),
        row("Across seeds, lowest to highest",
            f"{pct(one['low'])} to {pct(one['high'])}",
            two and f"{pct(two['low'])} to {pct(two['high'])}"),
        row("Standard deviation across seeds", points(one["stdev"]),
            two and points(two["stdev"])),
        row(f"Macro-F1, {seeds}", f"**{one['macro_f1']:.2f}**",
            two and f"{two['macro_f1']:.2f}"),
        row("Training time, mean per run", duration(one["train_seconds"]),
            two and duration(two["train_seconds"])),
        row("Parameters", parameter_text(one), two and parameter_text(two)),
        f"| Test questions | {len(y_true)} | {len(y_true)} |",
        "",
    ]
    hardware = [f"Approach 1 on {', '.join(one['hardware'])}"]
    if two:
        hardware.append(f"Approach 2 on {', '.join(two['hardware'])}")
    lines.append(f"Timed on: {'; '.join(hardware)}.")
    if two and {h.rsplit(', ', 1)[0] for h in one['hardware']} != \
            {h.rsplit(', ', 1)[0] for h in two['hardware']}:
        lines.append("")
        lines.append("**The two timings were taken on different machines and "
                     "are not comparable.**")
    lines += [
        "",
        "Approach 1's spread across seeds is exactly zero because nothing in it "
        "is random: the TF-IDF counts are fixed by the corpus and the solver "
        "converges to the same weights every time. The three runs are recorded "
        "rather than asserted, so that both columns are the same kind of number.",
    ]
    return lines


def comparison_section(y_true, one, two, pairs, one_seeds_agree=True):
    if not two:
        return [
            "## Are they distinguishable on accuracy?",
            "",
            "Not measurable: the ledger holds no transformer runs against the "
            "current test split, so there is nothing to pair Approach 1 with. "
            "`python3 tools/compare_runs.py seeds --approach distilbert --record` "
            "puts them back.",
        ]
    label, paragraph = verdict(one, two, pairs)
    pairing = ("Approach 1 has one set of predictions (its seeds are "
               "identical), so it is paired against each transformer seed in "
               "turn." if one_seeds_agree else
               f"Approach 1's seed runs do not agree, so its seed "
               f"{one['seeds'][0]} run is the one paired against each "
               f"transformer seed in turn.")
    lines = [
        "## Are they distinguishable on accuracy?",
        "",
        f"Both models answer the same {len(y_true)} questions, so the paired "
        f"comparison is the one that counts: the questions both get right or "
        f"both get wrong say nothing about the difference between them, and the "
        f"evidence is entirely in the questions exactly one of them gets right. "
        f"{pairing} The test is McNemar's, "
        f"exact rather than approximated because the counts are small.",
        "",
        "| Approach 2 seed | Both right | Only Approach 1 right | Only Approach 2 right | "
        "Both wrong | Approach 1 minus Approach 2 | 95% interval on the difference | Exact p |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in pairs:
        low, high = row["interval"]
        lines.append(
            f"| {row['seed']} | {row['both']} | {row['only_one']} | {row['only_two']} | "
            f"{row['neither']} | {points(row['difference'], signed=True)} | "
            f"{points(low, signed=True)} to {points(high, signed=True)} | "
            f"{row['p']:.2f} |")
    lines += [
        "",
        f"The rule was fixed before the numbers were seen: the two approaches are "
        f"called distinguishable only when every seed's paired test is below "
        f"p < {SIGNIFICANCE} with the same approach ahead each time, "
        f"indistinguishable when none is, and mixed otherwise.",
        "",
        paragraph,
    ]
    return lines


def resplit_section(one, resplits):
    half_width = (one["interval"][1] - one["interval"][0]) / 2
    test_rows = " or ".join(str(count) for count in resplits["test_rows"])
    return [
        "## How much the test set itself moves the number",
        "",
        f"The interval above is a formula. This is the same noise measured: "
        f"Approach 1 refit from scratch on {resplits['n']} fresh group-aware "
        f"splits of the corpus at seeds {RESPLIT_SEEDS[0]} to {RESPLIT_SEEDS[-1]}, "
        f"papers kept whole and the same four checks run as `src/split.py` runs "
        f"before it writes. The frozen split is untouched; these splits exist "
        f"only in memory. Each has {test_rows} test questions.",
        "",
        "| | Approach 1 across resplits |",
        "|---|---|",
        f"| Mean accuracy | {pct(resplits['mean'])} |",
        f"| Lowest to highest | {pct(resplits['low'])} to {pct(resplits['high'])} |",
        f"| Standard deviation | {points(resplits['stdev'])} |",
        f"| Half-width of the 95% interval above, for comparison | {points(half_width)} |",
        "",
        f"A different draw of test papers moves the accuracy by "
        f"{points(resplits['stdev'])} on its own, before any model changes. "
        f"Approach 2 is not re-run here: at {duration(one['train_seconds'])} a "
        f"run Approach 1 can afford {resplits['n']} refits, and this section "
        f"measures the test set rather than the models.",
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dated", default="",
                        help="date to stamp on the report, e.g. 2026-08-26")
    arguments = parser.parse_args()

    assert_inside_project(DOCS_DIR)
    assert_inside_project(RESULTS_PATH)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    fingerprint = dataset_fingerprint()

    y_true, approach_one = approach_one_predictions()
    draw_confusion_matrix(confusion_counts(y_true, approach_one),
                          "Approach 1 - TF-IDF and logistic regression",
                          DOCS_DIR / "confusion-approach-1.png")

    one_runs = load_runs(APPROACH_ONE)
    two_runs = load_runs(APPROACH_TWO)
    if not one_runs:
        raise SystemExit("the ledger holds no Approach 1 runs against the current "
                         "test split; run python3 tools/compare_runs.py seeds "
                         "--approach tfidf --record first")
    one = summarise_runs(y_true, one_runs)
    two = summarise_runs(y_true, two_runs) if two_runs else None

    # Both halves of the paired comparison come from the ledger, whose records
    # are hash-bound to this test split. The committed joblib is not: it can
    # outlive a resplit, and pairing against it would let a model that had
    # memorised today's test rows produce the verdict, which is exactly the leak
    # the hash guard exists to stop.
    one_default = default_seed_run(one_runs)
    one_predicted = one_default.predicted
    one_seeds_agree = all(run.predicted == one_predicted for run in one_runs)
    pairs = paired_comparisons(y_true, one_predicted, two_runs)
    resplits = resplit_summary()

    committed_matches_ledger = all(run.predicted == list(approach_one)
                                   for run in one_runs)

    # The transformer's confusion matrix and tables come from its default-seed
    # run, so they are one model's mistakes, like Approach 1's.
    approach_two = None
    if two_runs:
        default_run = default_seed_run(two_runs)
        approach_two = default_run.predicted
        two_seed = default_run.seed
        draw_confusion_matrix(confusion_counts(y_true, approach_two),
                              f"Approach 2 - fine-tuned DistilBERT, seed {two_seed}",
                              DOCS_DIR / "confusion-approach-2.png")

    one_confusions, one_singles, one_pairs = confusion_table(y_true, approach_one)
    stamp = f" Generated {arguments.dated}." if arguments.dated else ""

    sections = [
        "# Results",
        "",
        f"Generated by `python3 tools/report_results.py` against the frozen "
        f"dataset below.{stamp} Nothing here is typed by hand.",
        "",
        "## The frozen dataset",
        "",
        f"`data/private/questions.jsonl`, {fingerprint['rows']} labelled rows, "
        f"sha256 `{fingerprint['hash']}`. The test split it was scored on is "
        f"`test.jsonl`, sha256 `{fingerprint['test_hash']}`; every run in the "
        f"ledger carries that hash, and a run against any other split is ignored.",
        "",
        "| Split | Rows |",
        "|---|---|",
    ]
    for name in SPLIT_NAMES:
        sections.append(f"| {name} | {fingerprint['counts'][name]} |")
    sections += [
        "",
        "Per-chapter row counts across the whole corpus:",
        "",
        "| Chapter | Rows |",
        "|---|---|",
    ]
    for slug in CHAPTER_SLUGS:
        sections.append(f"| {CHAPTERS[slug]} | {fingerprint['per_chapter'][slug]} |")

    sections += [""] + headline_section(y_true, one, two)
    sections += [""] + comparison_section(y_true, one, two, pairs,
                                          one_seeds_agree)
    sections += [""] + resplit_section(one, resplits)

    abstained = None
    try:
        abstained = build_abstention()
    except RefitPredictionMismatch as error:
        sections += [""] + abstention_not_measurable_section(error)
    else:
        draw_abstention_curve(abstained["val_points"], abstained["cutoff"],
                              abstained["test_selective"], ABSTENTION_FIGURE)
        sections += [""] + abstention_section(abstained)

    sections += ["", "## Confusion matrices", ""]
    if committed_matches_ledger:
        sections.append(
            "Approach 1's matrix is drawn from the committed "
            "`models/chapter_classifier.joblib`, whose predictions match the "
            "three ledger runs exactly.")
    else:
        sections.append(
            "**The committed `models/chapter_classifier.joblib` does not "
            "reproduce the ledger runs' predictions.** It is stale against the "
            "current split; `python3 src/train.py` refits it.")
    sections.append("")
    if approach_two:
        sections.append(
            f"Approach 2's is drawn from its seed {two_seed} run, so that both "
            f"matrices show one model's mistakes.")
    else:
        sections.append(
            "The ledger holds no transformer runs against the current test "
            "split, so Approach 2's matrix is withheld rather than drawn from a "
            "model that outlived its split. `python3 tools/compare_runs.py seeds "
            "--approach distilbert --record` puts it back.")
    sections += ["", "![Approach 1 confusion matrix](docs/confusion-approach-1.png)"]
    if approach_two:
        sections.append("")
        sections.append("![Approach 2 confusion matrix](docs/confusion-approach-2.png)")
    sections += [
        "",
        "## What Approach 1 confuses",
        "",
        one_confusions,
        "",
        f"{one_pairs} distinct pairs in all, {one_singles} of them confused exactly once.",
    ]
    if approach_two:
        two_confusions, two_singles, two_pairs = confusion_table(y_true, approach_two)
        sections += [
            "",
            f"## What Approach 2 confuses, seed {two_seed}",
            "",
            two_confusions,
            "",
            f"{two_pairs} distinct pairs in all, {two_singles} of them confused exactly once.",
        ]
    sections += [
        "",
        "## Approach 1 per chapter",
        "",
        per_chapter_table(y_true, approach_one),
    ]
    if approach_two:
        sections += ["", f"## Approach 2 per chapter, seed {two_seed}", "",
                     per_chapter_table(y_true, approach_two)]
    sections += [
        "",
        "## Example predictions",
        "",
        "Questions written for this repository, so nothing here quotes a real "
        "paper. Output is `src/predict.py`'s, top three chapters each.",
        "",
        example_table(),
    ]

    RESULTS_PATH.write_text("\n".join(sections) + "\n", encoding="utf-8")
    print(f"wrote {RESULTS_PATH.relative_to(PROJECT_ROOT)}")
    if abstained is not None:
        print(f"wrote {ABSTENTION_FIGURE.relative_to(PROJECT_ROOT)}")
    print(f"wrote {(DOCS_DIR / 'confusion-approach-1.png').relative_to(PROJECT_ROOT)}")
    if approach_two:
        print(f"wrote {(DOCS_DIR / 'confusion-approach-2.png').relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
