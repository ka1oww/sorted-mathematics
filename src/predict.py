"""Classify a question typed, pasted or piped in at the command line, or a
whole exam paper handed over as a PDF.

    python3 src/predict.py "Find the exact value of the volume generated when R
                            is rotated about the y-axis."

    cat question.txt | python3 src/predict.py

    python3 src/predict.py --pdf paper.pdf
    python3 src/predict.py --pdf paper.pdf --pages 1-5

Three chapters are returned rather than one, because a question that genuinely
straddles two chapters should say so instead of committing.

With --min-confidence, the classifier may decline instead: a question is
answered only when the winning chapter's confidence reaches the cutoff.
Declining prints a plain refusal line first, then the top candidates with
their confidences, so the model's standing is never hidden. The default of
0.00 answers everything, which leaves existing behaviour unchanged.

The `--pdf` form is the whole product in one line: `reader.read_paper` cuts the
paper into questions, each one comes back here, and a PDF goes in and
chapter-labelled questions come out. The reader refuses to emit a paper its
confidence signals do not trust, so a mis-cut paper stops here rather than
arriving labelled and wrong.
"""

import sys

import joblib

from abstention import format_decline, split_options
from chapters import CHAPTERS
from features import normalise_question
from paths import MODELS_DIR

MODEL_PATH = MODELS_DIR / "chapter_classifier.joblib"

DEFAULT_HOW_MANY = 3


def classify(question_text, how_many=DEFAULT_HOW_MANY):
    """Return the most likely chapters and how confident the model is in each."""
    if not MODEL_PATH.exists():
        print("no model at models/chapter_classifier.joblib. Train one with "
              "python3 src/train.py, which needs the private corpus.")
        raise SystemExit(1)
    question_vectoriser, chapter_classifier = joblib.load(MODEL_PATH)

    # same normalisation the training rows went through
    features = question_vectoriser.transform([normalise_question(question_text)])

    confidences = chapter_classifier.predict_proba(features)[0]
    ranked = confidences.argsort()[::-1][:how_many]
    return [(chapter_classifier.classes_[index], confidences[index]) for index in ranked]


def classify_with_abstention(question_text, how_many=DEFAULT_HOW_MANY,
                             min_confidence=0.0):
    """Classify, declining when the best confidence falls short of the cutoff.

    Returns whether the question is answered and the ranked candidates. A
    cutoff of 0.00 answers everything, because confidences are probabilities.
    Anything above 1.00 declines everything. A negative cutoff is refused.
    """
    if not 0.0 <= min_confidence:
        raise ValueError(
            f"min_confidence must be 0 or more, got {min_confidence}"
        )
    ranked = classify(question_text, how_many=how_many)
    answered = bool(ranked) and ranked[0][1] >= min_confidence
    return answered, ranked


def classify_paper(path, pages=None, how_many=2):
    """Read one exam paper and classify every question in it.

    Yields (question, ranked chapters). pymupdf is imported through `reader`
    only on this path, so classifying a pasted question needs nothing but the
    model.
    """
    from reader import read_paper

    for question in read_paper(path, pages):
        yield question, classify(question.text, how_many)


def parse_pages(text):
    """"1-5" or "1,3,5" or both, zero-based, as PyMuPDF counts pages."""
    if not text:
        return None
    pages = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            first, last = part.split("-", 1)
            pages.extend(range(int(first), int(last) + 1))
        else:
            pages.append(int(part))
    return pages


def report_paper(path, pages):
    for question, ranked in classify_paper(path, pages):
        opening = " ".join(question.text.split())[:58]
        guesses = "   ".join(f"{CHAPTERS[chapter]} {confidence:.0%}"
                             for chapter, confidence in ranked)
        print(f"Q{question.number:>2}  pages={str(question.pages):<10} "
              f"{opening:<60}  ->  {guesses}")


def main():
    try:
        min_confidence, arguments = split_options(sys.argv[1:])
    except ValueError as problem:
        print(problem)
        raise SystemExit(2)
    if arguments and arguments[0] == "--pdf":
        if len(arguments) < 2:
            print("--pdf needs a path")
            raise SystemExit(1)
        pages = None
        if "--pages" in arguments:
            pages_index = arguments.index("--pages")
            if pages_index + 1 == len(arguments):
                print("--pages needs a page range")
                raise SystemExit(1)
            try:
                pages = parse_pages(arguments[pages_index + 1])
            except ValueError:
                print("--pages needs page numbers such as 1-5")
                raise SystemExit(1)
        report_paper(arguments[1], pages)
        return

    question_text = " ".join(arguments) or sys.stdin.read()
    if not question_text.strip():
        print("give me a question as an argument or on stdin, or a paper with "
              "--pdf")
        raise SystemExit(1)
    try:
        answered, ranked = classify_with_abstention(
            question_text, min_confidence=min_confidence)
    except ValueError as problem:
        print(problem)
        raise SystemExit(2)
    if not answered:
        print(format_decline(
            [(CHAPTERS[chapter], confidence) for chapter, confidence in ranked],
            min_confidence))
    for chapter, confidence in ranked:
        bar = "#" * round(confidence * 30)
        print(f"  {confidence:6.1%}  {CHAPTERS[chapter]:<38} {bar}")


if __name__ == "__main__":
    main()
