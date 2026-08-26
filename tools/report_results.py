"""Generate RESULTS.md and the confusion-matrix images from the frozen split.

Every number in RESULTS.md comes from this script reading the frozen dataset,
so nothing in it is typed by hand. DATA-PLAN.md section 7 asks for exactly
this: a row count, per-chapter counts and a content hash, with every accuracy
quoted against a named frozen dataset.

Approach 2 is included only if models/chapter_transformer exists; without it
the report covers Approach 1 and says so.

Run from the project root:  python3 tools/report_results.py
"""

import argparse
import hashlib
import json
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

from chapters import CHAPTERS, CHAPTER_SLUGS  # noqa: E402
from features import read_split, to_features  # noqa: E402
from paths import MODELS_DIR, PRIVATE_DATA, assert_inside_project  # noqa: E402
from predict import classify  # noqa: E402

DOCS_DIR = PROJECT_ROOT / "docs"
RESULTS_PATH = PROJECT_ROOT / "RESULTS.md"
CLASSIFIER_PATH = MODELS_DIR / "chapter_classifier.joblib"
TRANSFORMER_DIR = MODELS_DIR / "chapter_transformer"

SPLIT_NAMES = ("train", "val", "test")
TOP_CONFUSIONS = 8

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


def content_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


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
            "rows": sum(counts.values()), "hash": content_hash(corpus)}


def approach_one_predictions():
    question_vectoriser, chapter_classifier = joblib.load(CLASSIFIER_PATH)
    test_questions, y_test = read_split("test")
    predicted = chapter_classifier.predict(
        to_features(question_vectoriser, test_questions))
    return y_test, list(predicted)


def model_predates_the_split(model_file):
    """True when a saved model is older than the split it would be scored on.

    A model that outlives its split scores far too well, because rows that were
    its training data have since become test data. There is no way to see that
    in the accuracy itself, so it is caught here on the timestamps instead and
    the number is withheld rather than printed with a caveat nobody reads.
    """
    if not model_file.exists():
        return True
    split_file = PRIVATE_DATA / "test.jsonl"
    return model_file.stat().st_mtime < split_file.stat().st_mtime


def approach_two_predictions():
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from train_transformer import BATCH_SIZE, encode, id2label, pick_device

    device = pick_device()
    tokeniser = AutoTokenizer.from_pretrained(TRANSFORMER_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(
        TRANSFORMER_DIR).to(device)
    test_questions, y_test = read_split("test")
    dataset = encode(tokeniser, test_questions, y_test)

    model.eval()
    predicted = []
    with torch.no_grad():
        for input_ids, attention_mask, _ in DataLoader(dataset, batch_size=BATCH_SIZE):
            outputs = model(input_ids=input_ids.to(device),
                            attention_mask=attention_mask.to(device))
            predicted.extend(outputs.logits.argmax(dim=-1).cpu().tolist())
    return y_test, [id2label[index] for index in predicted]


def accuracy(y_true, y_predicted):
    return sum(t == p for t, p in zip(y_true, y_predicted)) / len(y_true)


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

    approach_two = None
    transformer_is_stale = model_predates_the_split(
        TRANSFORMER_DIR / "model.safetensors")
    if TRANSFORMER_DIR.is_dir() and not transformer_is_stale:
        _, approach_two = approach_two_predictions()
        draw_confusion_matrix(confusion_counts(y_true, approach_two),
                              "Approach 2 - fine-tuned DistilBERT",
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
        f"sha256 `{fingerprint['hash']}`.",
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

    sections += [
        "",
        "## Headline",
        "",
        "| | Approach 1 | Approach 2 |",
        "|---|---|---|",
    ]
    two_accuracy = (f"{accuracy(y_true, approach_two):.1%}" if approach_two
                    else "not measurable")
    two_f1 = (f"{f1_score(y_true, approach_two, average='macro', zero_division=0):.2f}"
              if approach_two else "not measurable")
    sections += [
        f"| Test accuracy | **{accuracy(y_true, approach_one):.1%}** | {two_accuracy} |",
        f"| Macro-F1 | **{f1_score(y_true, approach_one, average='macro', zero_division=0):.2f}** "
        f"| {two_f1} |",
        f"| Test questions | {len(y_true)} | {len(y_true)} |",
        "",
        "Single runs at the default seed.",
        "",
        "## Confusion matrices",
        "",
    ]
    if not approach_two:
        sections += [
            "The saved transformer is older than the split it would be scored "
            "against, so its accuracy would be a training-set score wearing a "
            "test set's name, and this report withholds it rather than print it "
            "under a caveat nobody reads. Retraining it puts the number back. "
            "The README's Approach 2 figure is a three-seed mean recorded when "
            "the model and the split still matched.",
            "",
        ]
    sections.append("![Approach 1 confusion matrix](docs/confusion-approach-1.png)")
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
        "",
        "## Approach 1 per chapter",
        "",
        per_chapter_table(y_true, approach_one),
    ]
    if approach_two:
        sections += ["", "## Approach 2 per chapter", "",
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
    print(f"wrote {(DOCS_DIR / 'confusion-approach-1.png').relative_to(PROJECT_ROOT)}")
    if approach_two:
        print(f"wrote {(DOCS_DIR / 'confusion-approach-2.png').relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
