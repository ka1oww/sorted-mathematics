"""TF-IDF features, fitted on the training rows and nothing else.

Everything a model knows about a question arrives through this file. One rule
governs it: the vocabulary and the rarity weights come from the training split
only. Fitting on the whole corpus lets the test questions' wording reach the
model before it is tested on them, which raises the score and destroys its
meaning. Exactly one function here calls fit, and its name says so.
"""

import json
import re

from sklearn.feature_extraction.text import TfidfVectorizer

from paths import PRIVATE_DATA

DEFAULT_MAX_FEATURES = 20000


def normalise_question(question_text):
    """Put text into the one shape the model is ever shown.

    Deliberately light, because the extractors already did the real cleaning.
    Its job is to guarantee that a question pasted in at prediction time is
    processed identically to a question that went through training. A model fed
    raw text when it learned on collapsed text is being asked a question in a
    language it was never taught, and the failure is silent.
    """
    return re.sub(r"\s+", " ", (question_text or "")).strip()


def read_split(split_name):
    """Read one split file and return its questions and their chapters."""
    path = PRIVATE_DATA / f"{split_name}.jsonl"
    questions, chapters = [], []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            questions.append(normalise_question(row["text"]))
            chapters.append(row["chapter"])
    return questions, chapters


def fit_question_vectoriser(training_questions, max_features=DEFAULT_MAX_FEATURES):
    """Learn the vocabulary and rarity weights from the TRAINING questions only.

    This is the only function in the project permitted to call fit on text.
    Every other use transforms against the vocabulary decided here.
    """
    question_vectoriser = TfidfVectorizer(
        # Single words and pairs. "random variable" carries far more signal as
        # one feature than "random" and "variable" do separately.
        ngram_range=(1, 2),
        # A word appearing in exactly one question out of a thousand cannot
        # generalise to a new question. It can only help the model memorise.
        min_df=2,
        # A word used ten times is not ten times as meaningful as one used once.
        sublinear_tf=True,
        # A ceiling, so the feature count stays near twenty times the training
        # row count rather than drifting into the hundreds of thousands.
        max_features=max_features,
        strip_accents="unicode",
    )
    question_vectoriser.fit(training_questions)
    return question_vectoriser


def to_features(question_vectoriser, questions):
    """Turn questions into numbers using a vocabulary that is already fixed."""
    return question_vectoriser.transform(questions)
