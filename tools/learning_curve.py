"""Train each approach on a quarter, a half, three quarters and all of the data.

The question is whether more data is the binding constraint. One accuracy
cannot answer it; the shape of the curve can. Every point here is the same
procedure the headline comparison uses (tune on validation, refit from scratch
on training plus validation, score the sealed test split) with one thing
changed: how much of the training pool the run is given. Whole papers are
dropped, never single questions, and the test split never moves, so the points
are comparable to each other and to RESULTS.md.

Runs are written to the same ledger as the headline runs (data/private/runs/,
see src/runs.py), each stamped with its fraction and the hash of the test split
it was scored against. The fraction is a filter as strict as the hash: a
quarter-data run cannot reach RESULTS.md, and the full-data runs already in the
ledger are the curve's 100% points rather than being trained again.

A transformer point costs a quarter of an hour, so this resumes: a run already
in the ledger is reused unless --redo says otherwise.

Run from the project root:

    python3 tools/learning_curve.py --approach tfidf --record
    python3 tools/learning_curve.py --approach distilbert --record
"""

import argparse
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from features import read_split
from runs import (
    APPROACH_NAMES,
    APPROACH_ONE,
    APPROACH_TWO,
    DEFAULT_SEEDS,
    load_runs,
    record_run,
    run_tfidf,
    run_transformer,
    same_fraction,
)

FRACTIONS = (0.25, 0.50, 0.75, 1.00)

RUNNERS = {APPROACH_ONE: run_tfidf, APPROACH_TWO: run_transformer}

# Approach 1 prints its regularisation search; at twelve runs that buries the
# curve, and the run itself is unchanged by silencing it.
EXTRA_ARGUMENTS = {APPROACH_ONE: {"quiet": True}, APPROACH_TWO: {}}


def recorded_run(approach, fraction, seed):
    """The ledger's run at this fraction and seed, if it holds one."""
    for run in load_runs(approach, fraction=fraction):
        if run.seed == seed:
            return run
    return None


def curve(approach, fractions, seeds, record, redo):
    """One approach at every fraction, reusing what the ledger already has."""
    _, y_test = read_split("test")
    print(f"\n{'=' * 72}\n{APPROACH_NAMES[approach]}, learning curve\n{'=' * 72}")
    print(f"  {'point':<22}{'train rows':>12}{'accuracy':>11}{'train time':>13}")
    for fraction in fractions:
        accuracies = []
        for seed in seeds:
            run = None if redo else recorded_run(approach, fraction, seed)
            reused = run is not None
            if run is None:
                run = RUNNERS[approach](
                    seed=seed, fraction=fraction, **EXTRA_ARGUMENTS[approach]
                )
                if record:
                    record_run(run)
            accuracies.append(run.accuracy(y_test))
            label = f"{fraction:.0%} seed {seed}" + (" (ledger)" if reused else "")
            print(
                f"  {label:<22}{run.rows_trained_on():>12}{accuracies[-1]:>11.1%}"
                f"{run.train_seconds:>12.1f}s",
                flush=True,
            )
        if len(accuracies) > 1:
            print(
                f"  {'mean of ' + str(len(accuracies)) + ' seeds':<22}"
                f"{'':>12}{statistics.fmean(accuracies):>11.1%}"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--approach",
        choices=("both", APPROACH_ONE, APPROACH_TWO),
        default="both",
        help="which approach to run (default both)",
    )
    parser.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        default=list(FRACTIONS),
        help="fractions of the training pool (default %(default)s)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="seeds to repeat each point over (default %(default)s)",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="write each new run to data/private/runs/ for "
        "tools/report_learning_curve.py",
    )
    parser.add_argument(
        "--redo", action="store_true", help="re-run points the ledger already holds"
    )
    arguments = parser.parse_args()

    approaches = (
        (APPROACH_ONE, APPROACH_TWO)
        if arguments.approach == "both"
        else (arguments.approach,)
    )
    fractions = [1.0 if same_fraction(f, 1.0) else f for f in arguments.fractions]
    for approach in approaches:
        curve(approach, fractions, arguments.seeds, arguments.record, arguments.redo)


if __name__ == "__main__":
    main()
