"""One from-scratch run of either approach, and the ledger that remembers it.

The comparison between the two approaches has to be symmetric to mean anything:
the same frozen split, the same procedure (tune on validation, refit on
training plus validation, score the sealed test rows), repeated over the same
seeds, timed on the same machine. This module is the one definition of that
run, so tools/compare_runs.py and tools/report_results.py cannot drift apart.

A transformer run costs about fifteen minutes, so runs are written to a ledger
under data/private/runs/ (gitignored, like everything else in data/private).
Each record carries the sha256 of the test split it was scored against, and
the reader ignores any record whose hash is not the current split's. That is
the leak guard: a model that outlives its split scores far too well, because
rows that were its training data have become test data, and the hash catches
that where a timestamp only guesses at it.

The predictions are stored positionally against test.jsonl, so a record holds
chapter slugs and numbers and nothing else: no question text, no weights.
"""

import contextlib
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from paths import PRIVATE_DATA, assert_inside_project

# features, split and train pull in scikit-learn, and train_transformer pulls in
# torch. Neither is imported here: the ledger below is plain JSON handling, and
# the tests that cover its leak guard run on a machine with only pytest
# installed. Each runner imports what it needs when it is called.

RUNS_DIR = PRIVATE_DATA / "runs"

APPROACH_ONE = "tfidf"
APPROACH_TWO = "distilbert"
APPROACH_NAMES = {APPROACH_ONE: "Approach 1", APPROACH_TWO: "Approach 2"}

# The seeds tools/compare_runs.py has always used. The transformer's published
# mean was over these three, so the baseline is measured over the same three.
DEFAULT_SEEDS = (42, 43, 44)

METHOD = ("train on the training split, tune on validation, refit from scratch "
          "on training plus validation, score the sealed test split")

# The conditions a run must have been made under to count towards the headline.
# tools/compare_runs.py also runs the transformer unweighted and truncated; those
# are experiments about the transformer, not measurements of the comparison.
DEFAULT_CLASS_WEIGHTS = True

# Only reached on a machine without torch, where no transformer run can be made
# and so nothing can disagree with it. Everywhere else the limit is read from
# train_transformer, which owns it.
FALLBACK_MAX_TOKENS = 512

# load_runs' sentinel for "the conditions below"; None means filter on none.
_DEFAULT = object()


def default_max_tokens():
    """train_transformer's truncation limit, read rather than copied.

    Importing train_transformer pulls in torch, which neither the ledger nor
    Approach 1 needs, so it is imported here rather than at the top. Reading the
    value instead of repeating it means that changing the limit there does not
    quietly filter every newly recorded run out of the headline.
    """
    try:
        from train_transformer import MAX_TOKENS
    except ImportError:
        return FALLBACK_MAX_TOKENS
    return MAX_TOKENS


def default_conditions():
    return {"class_weights": DEFAULT_CLASS_WEIGHTS,
            "max_tokens": default_max_tokens()}


@dataclass
class Run:
    """Everything the report needs from one run, and nothing private."""
    approach: str
    seed: int
    predicted: list
    test_hash: str
    test_rows: int
    train_seconds: float
    parameters: int
    hardware: str
    recorded: str
    method: str = METHOD
    features: int = None
    conditions: dict = field(default_factory=default_conditions)

    def correct(self, y_true):
        return sum(t == p for t, p in zip(y_true, self.predicted))

    def accuracy(self, y_true):
        return self.correct(y_true) / len(y_true)


# ---------------------------------------------------------------------------
# What the run is stamped with

def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def split_hash(split_name="test"):
    return file_hash(PRIVATE_DATA / f"{split_name}.jsonl")


def hardware_description(compute):
    """The machine a timing was taken on, so two timings can be told apart.

    A training time is only comparable to another taken on the same hardware,
    and the report prints both strings side by side so a reader can check.
    """
    chip = platform.processor() or platform.machine()
    if platform.system() == "Darwin":
        try:
            chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True,
                                  check=True).stdout.strip() or chip
        except (OSError, subprocess.CalledProcessError):
            pass
    try:
        memory_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30
        memory = f"{memory_gb:.0f} GB"
    except (ValueError, OSError, AttributeError):
        memory = "memory unknown"
    return f"{chip}, {memory}, {compute}"


def _now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# ---------------------------------------------------------------------------
# Approach 1

def fit_tfidf(train_questions, y_train, val_questions, y_val, quiet=False):
    """Approach 1's whole training procedure, timed as one piece.

    Mirrors src/train.py exactly: vocabulary and regularisation are chosen from
    the training and validation rows, then the model is refit on both. The
    clock covers all of it, which is the same span the transformer's clock
    covers in run_transformer, so the two times are the same kind of number.
    """
    from features import fit_question_vectoriser, to_features
    from train import choose_regularisation, train_chapter_classifier

    started = time.perf_counter()
    silence = (contextlib.redirect_stdout(io.StringIO()) if quiet
               else contextlib.nullcontext())
    with silence:
        question_vectoriser = fit_question_vectoriser(train_questions)
        regularisation, _ = choose_regularisation(
            to_features(question_vectoriser, train_questions), y_train,
            to_features(question_vectoriser, val_questions), y_val)
        final_questions = train_questions + val_questions
        question_vectoriser = fit_question_vectoriser(final_questions)
        chapter_classifier = train_chapter_classifier(
            to_features(question_vectoriser, final_questions), y_train + y_val,
            regularisation)
    return question_vectoriser, chapter_classifier, time.perf_counter() - started


def run_tfidf(seed=42, quiet=False):
    """Refit Approach 1 from scratch on the frozen split and score the test rows.

    The seed is accepted, recorded, and not used, because there is nothing here
    for it to move: the TF-IDF counts are fixed by the corpus and lbfgs converges
    to the same weights every time. That is the honest answer to "how much of
    the gap is run-to-run noise" for this half of the comparison, and it is
    worth recording three identical runs to show it rather than asserting it.
    """
    from features import read_split, to_features

    train_questions, y_train = read_split("train")
    val_questions, y_val = read_split("val")
    test_questions, y_test = read_split("test")
    question_vectoriser, chapter_classifier, seconds = fit_tfidf(
        train_questions, y_train, val_questions, y_val, quiet=quiet)
    predicted = chapter_classifier.predict(
        to_features(question_vectoriser, test_questions))
    return Run(
        approach=APPROACH_ONE, seed=seed, predicted=[str(p) for p in predicted],
        test_hash=split_hash(), test_rows=len(y_test), train_seconds=seconds,
        parameters=int(chapter_classifier.coef_.size
                       + chapter_classifier.intercept_.size),
        features=len(question_vectoriser.vocabulary_),
        hardware=hardware_description("CPU"), recorded=_now(),
    )


def resplit_tfidf(rows, seed):
    """Approach 1 on a fresh group-aware split at another seed.

    The frozen split stays frozen; this assigns the same whole papers to new
    sides of the fence in memory, runs the same four-check discipline
    src/split.py runs before it writes, and reports how many test rows the
    refit model gets right. Across seeds that is a direct measurement of how
    much the accuracy moves with which papers happen to land in the test set,
    which is the noise a confidence interval is only estimating.
    """
    from features import normalise_question, to_features
    from split import (DEFAULT_MIN_TEST_ROWS, assign_groups_to_splits,
                       check_every_chapter_has_test_rows,
                       check_no_group_straddles_splits, check_rows_conserved,
                       check_split_is_reproducible)

    split_rows = assign_groups_to_splits(rows, seed, DEFAULT_MIN_TEST_ROWS)
    check_no_group_straddles_splits(split_rows)
    check_rows_conserved(rows, split_rows)
    check_every_chapter_has_test_rows(split_rows, DEFAULT_MIN_TEST_ROWS)
    check_split_is_reproducible(rows, split_rows, seed, DEFAULT_MIN_TEST_ROWS)

    def questions(split_name):
        return [normalise_question(row["text"]) for row in split_rows[split_name]]

    def chapters(split_name):
        return [row["chapter"] for row in split_rows[split_name]]

    question_vectoriser, chapter_classifier, _ = fit_tfidf(
        questions("train"), chapters("train"),
        questions("val"), chapters("val"), quiet=True)
    predicted = chapter_classifier.predict(
        to_features(question_vectoriser, questions("test")))
    y_test = chapters("test")
    return sum(t == p for t, p in zip(y_test, predicted)), len(y_test)


# ---------------------------------------------------------------------------
# Approach 2

def run_transformer(seed=42, use_class_weights=True, max_tokens=None):
    """Fine-tune DistilBERT from scratch on the frozen split and score the test rows.

    torch is imported here rather than at the top so that the ledger and the
    baseline can be used on a machine without it.
    """
    import torch
    from evaluate_transformer import predict_chapters
    from features import read_split
    from train_transformer import (MAX_TOKENS, encode, fit_on_the_full_dataset,
                                   pick_device)

    if max_tokens is None:
        max_tokens = MAX_TOKENS
    device = pick_device()
    started = time.perf_counter()
    question_tokeniser, chapter_transformer = fit_on_the_full_dataset(
        device, use_class_weights=use_class_weights, max_tokens=max_tokens,
        seed=seed)
    seconds = time.perf_counter() - started

    test_questions, y_test = read_split("test")
    test_dataset = encode(question_tokeniser, test_questions, y_test, max_tokens)
    predicted = predict_chapters(chapter_transformer, test_dataset, device)
    parameters = sum(p.numel() for p in chapter_transformer.parameters())

    # Each run holds a fresh 66-million-parameter model, and three of them at
    # once is more than the laptop's shared memory wants to carry.
    del chapter_transformer
    if device.type == "mps":
        torch.mps.empty_cache()

    return Run(
        approach=APPROACH_TWO, seed=seed, predicted=list(predicted),
        test_hash=split_hash(), test_rows=len(y_test), train_seconds=seconds,
        parameters=int(parameters), features=None,
        hardware=hardware_description(device.type.upper()), recorded=_now(),
        conditions={"class_weights": bool(use_class_weights),
                    "max_tokens": int(max_tokens)},
    )


# ---------------------------------------------------------------------------
# The ledger

def record_path(run, runs_dir=None):
    return ((runs_dir or RUNS_DIR)
            / f"{run.approach}-seed{run.seed}-{run.test_hash}.json")


def record_run(run, runs_dir=None):
    """Write one run to the ledger, replacing an earlier run of the same seed."""
    path = assert_inside_project(record_path(run, runs_dir))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(run), indent=1), encoding="utf-8")
    return path


def read_record(path):
    """One ledger file as a Run, or None with a note on stderr saying why not.

    The ledger is gitignored, long-lived and expensive to rebuild, so it will
    outlive a change to the Run schema. A record written under an older one is
    skipped by name rather than aborting the whole report with a traceback that
    does not say which file was at fault.
    """
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as problem:
        print(f"ignoring ledger record {path}: unreadable ({problem})",
              file=sys.stderr)
        return None
    if not isinstance(record, dict):
        print(f"ignoring ledger record {path}: not a run record", file=sys.stderr)
        return None
    try:
        return Run(**record)
    except TypeError as problem:
        print(f"ignoring ledger record {path}: written under another schema "
              f"({problem})", file=sys.stderr)
        return None


def load_runs(approach, test_hash=None, conditions=_DEFAULT, runs_dir=None):
    """Every recorded run of one approach against the current test split.

    Records made against another split, or under other conditions, are left out
    without comment; the caller sees only runs that can be compared. Pass
    conditions=None to filter on the split alone.
    """
    if test_hash is None:
        test_hash = split_hash()
    if conditions is _DEFAULT:
        conditions = default_conditions()
    runs = []
    for path in sorted((runs_dir or RUNS_DIR).glob(f"{approach}-seed*-*.json")):
        run = read_record(path)
        if run is None:
            continue
        if run.test_hash != test_hash:
            continue
        if conditions is not None and run.conditions != conditions:
            continue
        runs.append(run)
    runs.sort(key=lambda run: run.seed)
    return runs
