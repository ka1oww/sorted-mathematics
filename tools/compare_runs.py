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

Run from the project root, and expect the transformer runs to take minutes
each:

    python3 tools/compare_runs.py seeds
    python3 tools/compare_runs.py class-weights
    python3 tools/compare_runs.py truncation
"""

import argparse
import statistics
import sys
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluate_transformer import predict_chapters  # noqa: E402
from features import fit_question_vectoriser, read_split, to_features  # noqa: E402
from train import choose_regularisation, train_chapter_classifier  # noqa: E402
from train_transformer import (MAX_TOKENS, encode, fit_on_the_full_dataset,  # noqa: E402
                               pick_device)

SEEDS = (42, 43, 44)


def score(true_chapters, predicted_chapters):
    """Accuracy and macro-F1, the pair the evaluators already report."""
    return (accuracy_score(true_chapters, predicted_chapters),
            f1_score(true_chapters, predicted_chapters, average="macro",
                     zero_division=0))


def run_transformer(seed=42, use_class_weights=True, max_tokens=MAX_TOKENS):
    """Fine-tune under one set of conditions and score the sealed test split."""
    device = pick_device()
    question_tokeniser, chapter_transformer = fit_on_the_full_dataset(
        device, use_class_weights=use_class_weights, max_tokens=max_tokens,
        seed=seed)
    test_questions, y_test = read_split("test")
    test_dataset = encode(question_tokeniser, test_questions, y_test, max_tokens)
    predicted = predict_chapters(chapter_transformer, test_dataset, device)

    # Each run holds a fresh 66-million-parameter model, and three of them at
    # once is more than the laptop's shared memory wants to carry.
    del chapter_transformer
    if device.type == "mps":
        torch.mps.empty_cache()
    return score(y_test, predicted)


def run_logistic_regression(seed=42):
    """Refit Approach 1 end to end, including its choice of regularisation.

    The seed is accepted and then not used, because there is nothing here for
    it to move: the TF-IDF counts are fixed by the corpus and lbfgs converges
    to the same weights every time. That is the honest answer to "how much of
    the gap is run-to-run noise" for this half of the comparison, and it is
    worth printing three identical rows to show it rather than asserting it.
    """
    train_questions, y_train = read_split("train")
    val_questions, y_val = read_split("val")
    question_vectoriser = fit_question_vectoriser(train_questions)
    regularisation, _ = choose_regularisation(
        to_features(question_vectoriser, train_questions), y_train,
        to_features(question_vectoriser, val_questions), y_val)

    final_questions = train_questions + val_questions
    question_vectoriser = fit_question_vectoriser(final_questions)
    chapter_classifier = train_chapter_classifier(
        to_features(question_vectoriser, final_questions), y_train + y_val,
        regularisation)

    test_questions, y_test = read_split("test")
    predicted = chapter_classifier.predict(
        to_features(question_vectoriser, test_questions))
    return score(y_test, predicted)


def report(heading, results):
    """One block per experiment: every run, then the spread across them."""
    print(f"\n{'=' * 72}\n{heading}\n{'=' * 72}")
    print(f"  {'run':<28}{'accuracy':>11}{'macro-F1':>11}")
    for label, (accuracy, macro_f1) in results:
        print(f"  {label:<28}{accuracy:>11.1%}{macro_f1:>11.3f}")
    if len(results) > 1:
        accuracies = [accuracy for _, (accuracy, _) in results]
        macro_f1s = [macro_f1 for _, (_, macro_f1) in results]
        print(f"  {'mean':<28}{statistics.fmean(accuracies):>11.1%}"
              f"{statistics.fmean(macro_f1s):>11.3f}")
        print(f"  {'spread, worst to best':<28}"
              f"{max(accuracies) - min(accuracies):>10.1%} "
              f"{max(macro_f1s) - min(macro_f1s):>10.3f}")


def compare_seeds():
    """The gap between the approaches, across three seeds each."""
    report("DistilBERT, three seeds",
           [(f"seed {seed}", run_transformer(seed=seed)) for seed in SEEDS])
    report("Logistic regression, three seeds",
           [(f"seed {seed}", run_logistic_regression(seed)) for seed in SEEDS])


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
    experiments = {
        "seeds": compare_seeds,
        "class-weights": compare_class_weights,
        "truncation": compare_truncation,
    }
    parser = argparse.ArgumentParser(
        description="Re-run the two approaches under changed conditions.")
    parser.add_argument("experiment", choices=sorted(experiments),
                        help="which comparison to run")
    arguments = parser.parse_args()
    experiments[arguments.experiment]()


if __name__ == "__main__":
    main()
