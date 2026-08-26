"""Classify a question typed, pasted or piped in at the command line.

    python3 src/predict.py "Find the exact value of the volume generated when R
                            is rotated about the y-axis."

    cat question.txt | python3 src/predict.py

Three chapters are returned rather than one, because a question that genuinely
straddles two chapters should say so instead of committing.
"""

import sys

import joblib

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


def main():
    question_text = " ".join(sys.argv[1:]) or sys.stdin.read()
    if not question_text.strip():
        print("give me a question, as an argument or on stdin")
        raise SystemExit(1)
    for chapter, confidence in classify(question_text):
        bar = "#" * round(confidence * 30)
        print(f"  {confidence:6.1%}  {CHAPTERS[chapter]:<38} {bar}")


if __name__ == "__main__":
    main()
