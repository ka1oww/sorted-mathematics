"""The arithmetic behind "are these two accuracies actually different".

A test set of 246 questions puts roughly three percentage points of noise on
any accuracy measured against it, so two models a point or two apart cannot be
told apart by comparing the two numbers. Three things are needed instead, and
they are all here:

* a confidence interval on each accuracy, so the reader sees the noise;
* the spread across repeated runs, which is a different noise (where the
  weights started, not which questions were drawn) and is reported separately;
* a paired test. Both models answer the same 246 questions, so the question
  that matters is how often exactly one of them is right, and which one. That
  is McNemar's test, done exactly rather than with a chi-square approximation
  because the counts involved are small.

Pure Python on purpose. The tests run on a machine with nothing but pytest
installed, and there is nothing here that needs more than math.
"""

import math
import statistics

Z_95 = 1.959964  # two-sided 95% point of the standard normal


def wilson_interval(correct, total, z=Z_95):
    """95% Wilson score interval for a proportion of correct answers.

    Chosen over the textbook Wald interval because Wald misbehaves exactly
    where classifiers live, near 100%: it can run past 1 and it collapses to
    zero width when every answer is right. Wilson does neither.
    """
    if total <= 0:
        raise ValueError("an interval needs at least one trial")
    if not 0 <= correct <= total:
        raise ValueError("correct must lie between 0 and total")
    proportion = correct / total
    z_squared = z * z
    denominator = 1 + z_squared / total
    centre = (proportion + z_squared / (2 * total)) / denominator
    half_width = (z * math.sqrt(proportion * (1 - proportion) / total
                                + z_squared / (4 * total * total))
                  / denominator)
    return max(0.0, centre - half_width), min(1.0, centre + half_width)


def spread(values):
    """Mean, sample standard deviation and range across repeated runs.

    One run has no spread, and says so with a standard deviation of zero
    rather than an error, because a single run is a legitimate thing to have
    recorded on the way to three.
    """
    values = list(values)
    if not values:
        raise ValueError("spread needs at least one value")
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "low": min(values),
        "high": max(values),
    }


def paired_outcomes(y_true, predicted_a, predicted_b):
    """Count the four ways two models can agree or disagree on one question.

    Returns (both right, only A right, only B right, both wrong). The middle
    two are the discordant pairs, and they are the whole of the evidence about
    which model is more accurate: a question both get right, or both get wrong,
    says nothing about the difference between them.
    """
    if not len(y_true) == len(predicted_a) == len(predicted_b):
        raise ValueError("the two prediction lists must cover the same rows")
    both_right = only_a = only_b = both_wrong = 0
    for truth, a, b in zip(y_true, predicted_a, predicted_b):
        a_right, b_right = a == truth, b == truth
        if a_right and b_right:
            both_right += 1
        elif a_right:
            only_a += 1
        elif b_right:
            only_b += 1
        else:
            both_wrong += 1
    return both_right, only_a, only_b, both_wrong


def mcnemar_exact_p(only_a, only_b):
    """Two-sided exact McNemar p-value from the discordant counts.

    Under the hypothesis that the two models are equally accurate, each
    discordant question is a fair coin: as likely to be the one A got right as
    the one B got right. The p-value is the chance of a split at least this
    lopsided from that many fair coins. No discordant questions at all means
    the models agreed on everything, and the evidence for a difference is nil.
    """
    discordant = only_a + only_b
    if discordant == 0:
        return 1.0
    smaller = min(only_a, only_b)
    one_tail = sum(math.comb(discordant, k) for k in range(smaller + 1)) / 2 ** discordant
    return min(1.0, 2 * one_tail)


def paired_difference_interval(only_a, only_b, total, z=Z_95):
    """95% interval on (accuracy of A minus accuracy of B), in proportion units.

    Paired, so the questions both models get right or wrong cancel and only the
    discordant counts carry uncertainty. The standard Wald form for a paired
    difference of proportions; it is fine here because the total is in the
    hundreds and the difference is nowhere near the boundary.
    """
    if total <= 0:
        raise ValueError("an interval needs at least one trial")
    difference = (only_a - only_b) / total
    variance = ((only_a + only_b) - (only_a - only_b) ** 2 / total) / total ** 2
    half_width = z * math.sqrt(max(variance, 0.0))
    return difference - half_width, difference + half_width


def intervals_overlap(first, second):
    return first[0] <= second[1] and second[0] <= first[1]
