"""Re-run the two approaches under changed conditions and print the spread.

Three questions this answers, none of which a single run can:

* Does the gap between the two approaches survive a change of seed? A few
  points on a test set of this size is a handful of questions, and one run of
  each cannot tell a real difference from where the weights happened to start.
* What do the balanced class weights in the transformer's loss actually buy?
* What did truncating at 256 tokens cost, now that the limit is 512?

Every run here trains from scratch and scores the sealed test split, so this
is measurement and not tuning: nothing it prints is allowed to choose a
setting. Where it changed one, the reasoning is in the comment beside that
setting in src/train_transformer.py.

The seeds experiment is the one RESULTS.md is built from. With --record, each
of its runs is written to the ledger in data/private/runs/ (see src/runs.py),
and tools/report_results.py reads the two approaches from there so that both
sides of the comparison are the mean of the same seeds under the same
procedure, timed on the same machine.

Run from the project root, and expect the transformer runs to take minutes
each:

    python3 tools/compare_runs.py seeds --record
    python3 tools/compare_runs.py seeds --approach tfidf --record
    python3 tools/compare_runs.py class-weights
    python3 tools/compare_runs.py truncation
"""

import argparse
import statistics
import sys
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from features import read_split  # noqa: E402
from runs import (APPROACH_NAMES, DEFAULT_SEEDS, record_run, run_tfidf,  # noqa: E402
                  run_transformer)


def score(run):
    """Accuracy and macro-F1, the pair the evaluators already report."""
    _, y_test = read_split("test")
    return (accuracy_score(y_test, run.predicted),
            f1_score(y_test, run.predicted, average="macro", zero_division=0))


def report(heading, results):
    """One block per experiment: every run, then the spread across them."""
    print(f"\n{'=' * 72}\n{heading}\n{'=' * 72}")
    print(f"  {'run':<28}{'accuracy':>11}{'macro-F1':>11}{'train time':>13}")
    for label, run in results:
        accuracy, macro_f1 = score(run)
        print(f"  {label:<28}{accuracy:>11.1%}{macro_f1:>11.3f}"
              f"{run.train_seconds:>12.1f}s")
    if len(results) > 1:
        scores = [score(run) for _, run in results]
        accuracies = [accuracy for accuracy, _ in scores]
        macro_f1s = [macro_f1 for _, macro_f1 in scores]
        print(f"  {'mean':<28}{statistics.fmean(accuracies):>11.1%}"
              f"{statistics.fmean(macro_f1s):>11.3f}")
        print(f"  {'spread, worst to best':<28}"
              f"{max(accuracies) - min(accuracies):>10.1%} "
              f"{max(macro_f1s) - min(macro_f1s):>10.3f}")
    hardware = {run.hardware for _, run in results}
    print(f"  timed on: {', '.join(sorted(hardware))}")


def compare_seeds(approaches, record):
    """The gap between the approaches, across the same three seeds each."""
    runners = {"distilbert": ("DistilBERT", run_transformer),
               "tfidf": ("Logistic regression", run_tfidf)}
    for approach in approaches:
        label, runner = runners[approach]
        results = []
        for seed in DEFAULT_SEEDS:
            run = runner(seed=seed)
            if record:
                path = record_run(run)
                print(f"recorded {APPROACH_NAMES[approach]} seed {seed} -> "
                      f"{path.relative_to(PROJECT_ROOT)}")
            results.append((f"seed {seed}", run))
        report(f"{label}, {len(DEFAULT_SEEDS)} seeds", results)


def compare_class_weights():
    """What the balanced class weights in the transformer's loss buy."""
    report("DistilBERT, balanced class weights", [
        ("weighted loss", run_transformer(use_class_weights=True)),
        ("unweighted loss", run_transformer(use_class_weights=False)),
    ])


def compare_truncation():
    """What truncating questions at 256 tokens was costing."""
    report("DistilBERT, truncation limit", [
        ("512 tokens", run_transformer(max_tokens=512)),
        ("256 tokens", run_transformer(max_tokens=256)),
    ])


def main():
    parser = argparse.ArgumentParser(
        description="Re-run the two approaches under changed conditions.")
    parser.add_argument("experiment", choices=("seeds", "class-weights", "truncation"),
                        help="which comparison to run")
    parser.add_argument("--approach", choices=("both", "tfidf", "distilbert"),
                        default="both",
                        help="seeds only: which approach to re-run (default both)")
    parser.add_argument("--record", action="store_true",
                        help="seeds only: write each run to data/private/runs/ "
                             "for tools/report_results.py")
    arguments = parser.parse_args()
    if arguments.experiment == "seeds":
        approaches = (("distilbert", "tfidf") if arguments.approach == "both"
                      else (arguments.approach,))
        compare_seeds(approaches, arguments.record)
    elif arguments.experiment == "class-weights":
        compare_class_weights()
    else:
        compare_truncation()


if __name__ == "__main__":
    main()
