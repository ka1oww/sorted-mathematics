"""Let the classifier decline to answer, and measure what that costs and buys.

Approach 1 ends in a logistic regression, whose predict_proba is the pipeline's
own confidence: the share of the weight behind the winning chapter. This module
turns that number into a policy, with pure functions and no model of its own. A
question is answered only when the winning chapter's confidence reaches a
cutoff, and declined otherwise.

Three numbers describe any cutoff. Coverage is the share of questions still
answered. Selective accuracy is the accuracy on those answered questions. Full
accuracy is the accuracy on the whole set with every declined question counted
wrong, which puts the price of declining beside the gain.

Pure Python on purpose. The tests run on a machine with nothing but pytest
installed, and there is nothing here that needs more than arithmetic.
"""

from intervals import wilson_interval

# The thresholds the sweep walks, from answering everything to answering
# almost nothing. A confidence is a probability, so 0.00 answers every
# question and 1.00 answers only the rare question the model is certain of.
SWEEP_THRESHOLDS = tuple(round(index * 0.05, 2) for index in range(21))

# The bar an answered subset must clear on validation before the classifier is
# allowed to decline anything at all. Fixed here, before any test number was
# seen, so the cutoff choice cannot be tuned on the test split. The report
# quotes this rule word for word beside the numbers it chose.
TARGET_SELECTIVE_ACCURACY = 0.95

# Bins for the reliability check that the confidence means what it says.
RELIABILITY_BINS = 10

# The mean gap between stated confidence and accuracy, in points, below which
# the pipeline's probabilities are described as meaning what they say. Fixed
# here so the report cannot grade its own calibration on a curve.
CALIBRATION_TOLERANCE = 0.05


def abstention_not_measurable_section(reason):
    return [
        "## When the classifier declines to answer",
        "",
        "**Abstention is not measurable.**",
        "",
        str(reason),
    ]


def sweep_points(y_true, predicted, confidences, thresholds=SWEEP_THRESHOLDS):
    """One row per threshold: what is answered, and what it costs and buys.

    A question is answered when its confidence reaches the threshold. Rows come
    back in the order the thresholds were given. Selective accuracy is None
    where a threshold answers nothing, because an accuracy over no questions
    is not zero, it is undefined.
    """
    points = []
    for threshold in thresholds:
        answered = [confidence >= threshold for confidence in confidences]
        count = sum(answered)
        correct = sum(
            1
            for true, guess, keep in zip(y_true, predicted, answered)
            if keep and true == guess
        )
        total = len(y_true)
        points.append(
            {
                "threshold": threshold,
                "answered": count,
                "total": total,
                "coverage": count / total if total else 0.0,
                "selective_accuracy": correct / count if count else None,
                "full_accuracy": correct / total if total else 0.0,
            }
        )
    return points


def score_at_cutoff(y_true, predicted, confidences, cutoff):
    """The single row a sealed split is ever scored at: one cutoff, once."""
    return sweep_points(y_true, predicted, confidences, [cutoff])[0]


def select_cutoff(validation_points, target=TARGET_SELECTIVE_ACCURACY):
    """The cutoff the validation split earns, by the rule fixed above.

    Takes the lowest threshold whose accuracy on the answered validation
    questions reaches the target. Thresholds must arrive in ascending order,
    which is the order the sweep walks them in. Where no threshold reaches the
    target the answer is 0.00: the confidence never earns the right to decline,
    so the classifier answers everything. The test split takes no part in
    this, which is the whole value of the exercise.
    """
    for point in validation_points:
        selective = point["selective_accuracy"]
        if selective is not None and selective >= target:
            return point["threshold"]
    return 0.0


def reliability_bins(y_true, predicted, confidences, bin_count=RELIABILITY_BINS):
    """Validation questions grouped by confidence, least sure first.

    Each bin reports its mean confidence beside its accuracy, so a reader can
    see whether the model's confidence means what it says. Accuracy is None
    for a bin no question falls into. This is evidence about the pipeline's
    own probabilities, not a second calibration pass: nothing here refits or
    rescales anything.
    """
    bins = []
    for index in range(bin_count):
        lower = index / bin_count
        upper = (index + 1) / bin_count
        if index == bin_count - 1:
            members = [
                (true, guess, confidence)
                for true, guess, confidence in zip(y_true, predicted, confidences)
                if lower <= confidence <= upper
            ]
        else:
            members = [
                (true, guess, confidence)
                for true, guess, confidence in zip(y_true, predicted, confidences)
                if lower <= confidence < upper
            ]
        correct = sum(1 for true, guess, _ in members if true == guess)
        bins.append(
            {
                "lower": lower,
                "upper": upper,
                "questions": len(members),
                "mean_confidence": (
                    sum(confidence for _, _, confidence in members) / len(members)
                    if members
                    else None
                ),
                "accuracy": correct / len(members) if members else None,
            }
        )
    return bins


def selective_interval(answered, correct):
    """A 95% interval on an accuracy over the answered questions only.

    The same Wilson interval RESULTS.md puts on every other accuracy, so the
    answered subset is not quoted with less honesty than the whole set.
    """
    return wilson_interval(correct, answered)


def calibration_sentence(mean_gap, signed_gap,
                         tolerance=CALIBRATION_TOLERANCE):
    """What the reliability check found, in one sentence, by a fixed rule.

    Both gaps are weighted means over the non-empty bins, in fractions. Where
    the mean gap is within tolerance the confidence is described as meaning
    what it says. Otherwise the sentence says which way it leans. The report
    quotes this rather than grading its own calibration by eye.
    """
    if mean_gap < tolerance:
        return (
            f"The confidence means what it says: grouped by confidence, the "
            f"mean gap between stated confidence and accuracy across the "
            f"validation bins is {mean_gap * 100:.1f} points."
        )
    direction = "understating" if signed_gap < 0 else "overstating"
    return (
        f"The confidence does not track accuracy closely: grouped by "
        f"confidence, the mean gap between stated confidence and accuracy "
        f"across the validation bins is {mean_gap * 100:.1f} points, with "
        f"the model mostly {direction} how often it is right."
    )


def split_options(argv):
    """Read the abstention flag out of a predict.py command line.

    Returns the minimum confidence and the remaining arguments, which are the
    question. The default of 0.00 answers everything, so existing behaviour is
    unchanged unless the flag is passed. Anything above 1.00 declines every
    question, which is the degenerate case the tests pin down rather than a
    useful setting.
    """
    args = list(argv)
    min_confidence = 0.0
    if "--min-confidence" in args:
        position = args.index("--min-confidence")
        try:
            min_confidence = float(args[position + 1])
        except (IndexError, ValueError):
            raise ValueError(
                "--min-confidence needs a number, for example "
                "--min-confidence 0.5"
            )
        del args[position : position + 2]
    if not 0.0 <= min_confidence:
        raise ValueError(
            f"--min-confidence must be 0 or more, got {min_confidence}"
        )
    return min_confidence, args


def format_decline(ranked, cutoff):
    """The line predict.py prints when it declines, before the candidates.

    A bare refusal would hide where the model stood. This says plainly that it
    is not answering, names the best guess and how far short of the cutoff it
    fell, and hands over to the ranked candidates with their confidences,
    which the caller prints next.
    """
    best_name, best_confidence = ranked[0]
    return (
        f"not sure enough to answer this one (best guess {best_name} "
        f"at {best_confidence:.1%}, below the {cutoff:.0%} cutoff). "
        f"Leading candidates:"
    )


def sweep_table(points):
    """The sweep as a markdown table: coverage and both accuracies per cutoff."""
    lines = [
        "| Cutoff | Answered | Coverage | Accuracy on answered | "
        "Accuracy on full set |",
        "|---|---|---|---|---|",
    ]
    for point in points:
        selective = (
            f"{point['selective_accuracy']:.1%}"
            if point["selective_accuracy"] is not None
            else "no questions answered"
        )
        lines.append(
            f"| {point['threshold']:.2f} | {point['answered']} of {point['total']} | "
            f"{point['coverage']:.1%} | {selective} | "
            f"{point['full_accuracy']:.1%} |"
        )
    return "\n".join(lines)


def reliability_table(bins):
    """The reliability check as a markdown table."""
    lines = [
        "| Confidence | Questions | Mean confidence | Accuracy |",
        "|---|---|---|---|",
    ]
    for entry in bins:
        if entry["questions"]:
            lines.append(
                f"| {entry['lower']:.1f} to {entry['upper']:.1f} | "
                f"{entry['questions']} | {entry['mean_confidence']:.1%} | "
                f"{entry['accuracy']:.1%} |"
            )
        else:
            lines.append(
                f"| {entry['lower']:.1f} to {entry['upper']:.1f} | 0 | "
                f"no questions | no questions |"
            )
    return "\n".join(lines)


def trade_sentence(
    cutoff, declined, total, base_accuracy, selective_accuracy, full_accuracy
):
    """The trade in one plain sentence, for the close of the RESULTS.md section.

    Where the cutoff is 0.00 the classifier declines nothing and the sentence
    says so, rather than dressing an unchanged accuracy up as a gain.
    """
    if not declined:
        return (
            f"At a cutoff of {cutoff:.2f} the classifier declines nothing and "
            f"answers everything at the headline {base_accuracy:.1%}."
        )
    return (
        f"At a cutoff of {cutoff:.2f} the classifier declines {declined} of "
        f"{total} questions to lift accuracy on the questions it answers from "
        f"{base_accuracy:.1%} to {selective_accuracy:.1%}, while accuracy on "
        f"the full set with declined questions counted wrong is "
        f"{full_accuracy:.1%}."
    )
