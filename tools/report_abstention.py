"""Sweep an abstention threshold for Approach 1 and score the chosen cutoff once.

Approach 1 always answers, even when its confidence is spread thin across
several chapters. This tool measures what a refusal option would cost and buy.
The confidence is the maximum of the predict_proba the pipeline already
produces, so there is no new model, no new library and no second scoring pass.

The protocol runs in order, so the sealed test split is scored exactly once,
at one cutoff chosen without looking at it:

1. Fit Approach 1 on the training rows alone and read validation confidences
   from that model.
2. Sweep the cutoff across its full range on validation and choose one by the
   rule fixed in src/abstention.py.
3. Refit on training plus validation, which is the model the headline numbers
   measure, and check the refit reproduces the ledger runs exactly.
4. Score the sealed test split once at the chosen cutoff.

Run from the project root:  python3 tools/report_abstention.py
The figure it draws is picked up by tools/report_results.py, which prints the
section below into RESULTS.md, so nothing there is typed by hand.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from abstention import (  # noqa: E402
    SWEEP_THRESHOLDS,
    TARGET_SELECTIVE_ACCURACY,
    calibration_sentence,
    reliability_bins,
    reliability_table,
    score_at_cutoff,
    select_cutoff,
    selective_interval,
    sweep_points,
    sweep_table,
    trade_sentence,
)
from features import read_split, to_features  # noqa: E402
from paths import assert_inside_project  # noqa: E402
from runs import (  # noqa: E402
    APPROACH_ONE,
    fit_tfidf,
    fit_train_only_tfidf,
    load_runs,
    questions_and_chapters,
    training_pool,
)

DOCS_DIR = PROJECT_ROOT / "docs"
ABSTENTION_FIGURE = DOCS_DIR / "abstention-curve.png"


class RefitPredictionMismatch(SystemExit):
    pass


def validation_confidences():
    """Approach 1 fitted on training rows alone, read back on validation.

    The full procedure refits on training plus validation before anything is
    scored, which leaves no held out row to choose a cutoff from. These
    confidences come from the same fit stopped halfway, so the cutoff choice
    never looks at the test rows. Nothing here resamples anything: the frozen
    split files are read as they stand, whole papers where the split put them.
    """
    pool = training_pool()
    train_questions, y_train = questions_and_chapters(pool["train"])
    val_questions, y_val = questions_and_chapters(pool["val"])
    vectoriser, classifier, regularisation = fit_train_only_tfidf(
        train_questions, y_train, val_questions, y_val, quiet=True
    )
    features = to_features(vectoriser, val_questions)
    probabilities = classifier.predict_proba(features)
    predicted = list(classifier.classes_[probabilities.argmax(axis=1)])
    confidences = [float(value) for value in probabilities.max(axis=1)]
    return {
        "y_val": y_val,
        "predicted": predicted,
        "confidences": confidences,
        "regularisation": regularisation,
        "train_rows": len(train_questions),
        "val_rows": len(val_questions),
    }


def refit_and_score_once(cutoff):
    """The headline model, refit and scored at the chosen cutoff, once.

    The refit is the procedure RESULTS.md measures Approach 1 under: tune on
    validation, refit from scratch on training plus validation. The guard below
    refuses to continue unless that refit reproduces the ledger runs exactly,
    so the abstention numbers describe the headline model with a refusal
    option rather than another model's predictions. The test rows pass through
    the model a single time, and the single row returned is the only test
    scoring this tool performs.
    """
    pool = training_pool()
    train_questions, y_train = questions_and_chapters(pool["train"])
    val_questions, y_val = questions_and_chapters(pool["val"])
    test_questions, y_test = read_split("test")
    vectoriser, classifier, _ = fit_tfidf(
        train_questions, y_train, val_questions, y_val, quiet=True
    )

    one_runs = load_runs(APPROACH_ONE)
    if not one_runs:
        raise SystemExit(
            "the ledger holds no Approach 1 runs against the current test "
            "split. Run python3 tools/compare_runs.py seeds --approach tfidf "
            "--record first"
        )
    probabilities = classifier.predict_proba(to_features(vectoriser, test_questions))
    predicted = list(classifier.classes_[probabilities.argmax(axis=1)])
    if not all(run.predicted == predicted for run in one_runs):
        raise RefitPredictionMismatch(
            "the refit model does not reproduce the ledger runs' predictions. "
            "Refusing to report abstention numbers for another model's "
            "predictions"
        )
    confidences = [float(value) for value in probabilities.max(axis=1)]
    row = score_at_cutoff(y_test, predicted, confidences, cutoff)
    base = sum(true == guess for true, guess in zip(y_test, predicted)) / len(y_test)
    return {
        "y_test": y_test,
        "predicted": predicted,
        "confidences": confidences,
        "row": row,
        "base_accuracy": base,
    }


def build_abstention():
    """The whole measurement: sweep on validation, one scoring on test."""
    validation = validation_confidences()
    val_points = sweep_points(
        validation["y_val"], validation["predicted"], validation["confidences"]
    )
    cutoff = select_cutoff(val_points)
    test = refit_and_score_once(cutoff)
    test_row = test["row"]
    answered = test_row["answered"]
    correct = round(
        (test_row["selective_accuracy"] or 0.0) * answered
    )
    bins = reliability_bins(
        validation["y_val"], validation["predicted"], validation["confidences"]
    )
    gaps = [
        abs(entry["mean_confidence"] - entry["accuracy"])
        for entry in bins
        if entry["questions"]
    ]
    signed = [
        entry["mean_confidence"] - entry["accuracy"]
        for entry in bins
        if entry["questions"]
    ]
    counts = [entry["questions"] for entry in bins if entry["questions"]]
    mean_gap = sum(gap * count for gap, count in zip(gaps, counts)) / sum(counts)
    signed_gap = sum(gap * count for gap, count in zip(signed, counts)) / sum(counts)
    return {
        "regularisation": validation["regularisation"],
        "train_rows": validation["train_rows"],
        "val_rows": validation["val_rows"],
        "val_points": val_points,
        "bins": bins,
        "mean_gap": mean_gap,
        "signed_gap": signed_gap,
        "cutoff": cutoff,
        "test_total": test_row["total"],
        "test_answered": answered,
        "test_coverage": test_row["coverage"],
        "test_selective": test_row["selective_accuracy"],
        "test_full": test_row["full_accuracy"],
        "test_correct": correct,
        "test_interval": (
            selective_interval(answered, correct) if answered else None
        ),
        "base_accuracy": test["base_accuracy"],
    }


def draw_abstention_curve(val_points, cutoff, test_selective, destination):
    """Coverage and both accuracies against the cutoff, validation in lines.

    The sealed test split appears exactly once: a single marker at the chosen
    cutoff, showing the accuracy on the questions answered there. Everything
    else on the plot is validation, which is the curve the cutoff was chosen
    from.
    """
    thresholds = [point["threshold"] for point in val_points]
    coverage = [point["coverage"] * 100 for point in val_points]
    selective = [
        (point["selective_accuracy"] * 100 if point["selective_accuracy"] is not None
         else float("nan"))
        for point in val_points
    ]
    full = [point["full_accuracy"] * 100 for point in val_points]

    figure, axes = plt.subplots(figsize=(7.5, 4.5))
    axes.plot(thresholds, selective, marker="o", markersize=3,
              label="accuracy on answered, validation")
    axes.plot(thresholds, full, marker="o", markersize=3,
              label="accuracy on full set, validation")
    axes.plot(thresholds, coverage, marker="o", markersize=3,
              label="coverage, validation")
    axes.axvline(cutoff, color="grey", linestyle="--",
                 label=f"chosen cutoff {cutoff:.2f}")
    if test_selective is not None:
        axes.scatter([cutoff], [test_selective * 100],
                     s=70, marker="D", color="black", zorder=5,
                     label="sealed test, scored once")
    axes.set_xlabel("minimum confidence to answer")
    axes.set_ylabel("%")
    axes.set_title("Abstention trade-off for Approach 1")
    axes.set_xlim(min(SWEEP_THRESHOLDS), max(SWEEP_THRESHOLDS))
    axes.set_ylim(0, 100)
    axes.grid(alpha=0.3)
    axes.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    plt.close(figure)


def pct(value):
    return f"{value:.1%}"


def interval_text(interval):
    return f"{pct(interval[0])} to {pct(interval[1])}"


def abstention_section(built):
    """The RESULTS.md section, in the voice of the surrounding document."""
    val_point = next(
        point for point in built["val_points"] if point["threshold"] == built["cutoff"]
    )
    val_selective = (
        pct(val_point["selective_accuracy"])
        if val_point["selective_accuracy"] is not None
        else "no questions answered"
    )
    test_selective = (
        pct(built["test_selective"])
        if built["test_selective"] is not None
        else "no questions answered"
    )
    lines = [
        "## When the classifier declines to answer",
        "",
        "Approach 1 always answers, even when its confidence is spread thin "
        "across several chapters. An abstention cutoff lets it decline "
        "instead: a question is answered only when the winning chapter's share "
        "of the model's confidence reaches the cutoff. The confidence is the "
        "maximum of the predict_proba the pipeline already produces, so this "
        "adds no model, no library and no second scoring pass.",
        "",
        f"The rule was fixed before the test split was scored: sweep the "
        f"cutoff from {min(SWEEP_THRESHOLDS):.2f} to "
        f"{max(SWEEP_THRESHOLDS):.2f} in steps of 0.05 on the "
        f"{built['val_rows']} validation questions and take the lowest cutoff "
        f"whose accuracy on the answered validation questions reaches "
        f"{TARGET_SELECTIVE_ACCURACY:.0%}. Where no cutoff reaches it, the "
        f"cutoff is 0.00 and the classifier answers everything. The model is "
        f"then refit on the {built['train_rows']} training questions plus "
        f"validation and scored at that cutoff once on the sealed test split. "
        f"The refit reproduces the ledger runs exactly, so these are the "
        f"headline model's numbers with a refusal option, not another model's.",
        "",
        "![Abstention curve](docs/abstention-curve.png)",
        "",
        "The sweep, on the validation split. Coverage is the share of "
        "questions still answered. Accuracy on answered is over those alone. "
        "Accuracy on the full set counts every declined question wrong. The "
        "diamond on the plot is the single test scoring at the chosen cutoff.",
        "",
        sweep_table(built["val_points"]),
        "",
        calibration_sentence(built["mean_gap"], built["signed_gap"]),
        "",
        reliability_table(built["bins"]),
        "",
        f"The cutoff the validation split chose is {built['cutoff']:.2f}. "
        f"There it answers {val_point['answered']} of {val_point['total']} "
        f"validation questions ({pct(val_point['coverage'])}) at "
        f"{val_selective} on the answered ones. Scored once on the "
        f"{built['test_total']} sealed test questions, it answers "
        f"{built['test_answered']} ({pct(built['test_coverage'])}) at "
        f"{test_selective} on the answered ones"
        + (
            f", 95% interval {interval_text(built['test_interval'])}"
            if built["test_interval"] is not None
            else ""
        )
        + f", with {pct(built['test_full'])} on the full set counting "
        f"declined questions wrong, against {pct(built['base_accuracy'])} "
        f"without the cutoff.",
        "",
        trade_sentence(
            built["cutoff"],
            built["test_total"] - built["test_answered"],
            built["test_total"],
            built["base_accuracy"],
            built["test_selective"] or 0.0,
            built["test_full"],
        ),
    ]
    return lines


def abstention_not_measurable_section(reason):
    return [
        "## When the classifier declines to answer",
        "",
        "**Abstention is not measurable.**",
        "",
        str(reason),
    ]


def main():
    assert_inside_project(DOCS_DIR)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    built = build_abstention()
    draw_abstention_curve(
        built["val_points"], built["cutoff"], built["test_selective"],
        ABSTENTION_FIGURE,
    )
    for line in abstention_section(built):
        print(line)
    print(f"wrote {ABSTENTION_FIGURE.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
