"""Measure the chapter classifier on the held-back test questions.

Overall accuracy is one number and it hides everything worth knowing. Four
things are printed here, and the last two are the ones to actually read.

Run: python3 src/evaluate.py
"""

from collections import Counter

import joblib
from sklearn.metrics import classification_report

from chapters import CHAPTERS
from features import read_split, to_features
from paths import MODELS_DIR

MODEL_PATH = MODELS_DIR / "chapter_classifier.joblib"

MIN_CONFUSIONS_TO_REPORT = 2
WORDS_PER_CHAPTER = 12


def report_confusions(y_test, predicted_chapters):
    """List the chapter pairs the model actually mixes up, worst first.

    A confusion matrix over 21 chapters is 441 cells and unreadable in a
    terminal. What anyone wants from it is the short list of pairs that collide
    often, so that is what this prints. It answers "what did it say instead",
    which is the question that leads somewhere.
    """
    confusions = Counter(
        (true, predicted)
        for true, predicted in zip(y_test, predicted_chapters)
        if true != predicted
    )
    print(f"\n{'-' * 72}\nWHAT IT CONFUSES, worst first\n{'-' * 72}")
    for (true, predicted), count in confusions.most_common():
        if count >= MIN_CONFUSIONS_TO_REPORT:
            print(f"  {count:>3}x   {CHAPTERS[true]:<38} called it  {CHAPTERS[predicted]}")
    singles = sum(1 for count in confusions.values() if count < MIN_CONFUSIONS_TO_REPORT)
    print(f"  ...plus {singles} pairs confused only once")


def report_strongest_words(question_vectoriser, chapter_classifier):
    """Print the words arguing hardest for each chapter.

    Two jobs. It reads like a summary of the subject, which is a good sign in
    itself. More usefully it is a leak detector: if a chapter's strongest
    feature turns out to be a school name, a year, or one of the mangled glyphs
    the topical bank leaves behind, then the model learned to recognise where a
    question came from rather than what it is about. That scores beautifully
    and is worthless, and this printout is how it gets caught.
    """
    vocabulary = question_vectoriser.get_feature_names_out()
    print(f"\n{'-' * 72}\nWHAT IT LEARNED, strongest words per chapter\n{'-' * 72}")
    for chapter, weights in zip(chapter_classifier.classes_, chapter_classifier.coef_):
        strongest = weights.argsort()[-WORDS_PER_CHAPTER:][::-1]
        words = ", ".join(vocabulary[index] for index in strongest)
        print(f"\n  {CHAPTERS[chapter]}\n    {words}")


def main():
    question_vectoriser, chapter_classifier = joblib.load(MODEL_PATH)
    test_questions, y_test = read_split("test")
    X_test = to_features(question_vectoriser, test_questions)
    predicted_chapters = chapter_classifier.predict(X_test)

    correct = sum(t == p for t, p in zip(y_test, predicted_chapters))
    print(f"{'=' * 72}")
    print(f"TEST ACCURACY  {correct / len(y_test):.1%}   ({correct} of {len(y_test)} questions)")
    print(f"{'=' * 72}")

    print(f"\n{'-' * 72}\nPER CHAPTER\n{'-' * 72}")
    print(classification_report(y_test, predicted_chapters, zero_division=0))

    report_confusions(y_test, predicted_chapters)
    report_strongest_words(question_vectoriser, chapter_classifier)


if __name__ == "__main__":
    main()
