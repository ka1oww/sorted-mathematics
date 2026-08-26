"""Approach 1: tune the regularisation on validation, refit, save the model.

Run: python3 src/train.py
"""

import joblib
from sklearn.linear_model import LogisticRegression

from features import fit_question_vectoriser, read_split, to_features
from paths import MODELS_DIR, assert_inside_project

MODEL_PATH = MODELS_DIR / "chapter_classifier.joblib"

# Regularisation strengths to try. Low values force the weights to stay small
# and spread out, so the model must rely on broad patterns; high values let
# individual weights grow large, which is how a model memorises its training set.
REGULARISATION_CANDIDATES = (0.1, 0.5, 1.0, 3.0, 10.0, 30.0)


def train_chapter_classifier(X_train, y_train, regularisation):
    """Learn one weight per word per chapter.

    class_weight balanced is not a nicety here. Complex Numbers has 152
    questions and Correlation and Regression has 32, so a model could raise its
    overall score by quietly never predicting the small chapters while becoming
    useless for exactly the questions a teacher would struggle to place. This
    prices that shortcut out by making an error on a rare chapter cost more.
    """
    chapter_classifier = LogisticRegression(
        C=regularisation,
        class_weight="balanced",
        max_iter=3000,
    )
    chapter_classifier.fit(X_train, y_train)
    return chapter_classifier


def choose_regularisation(X_train, y_train, X_val, y_val):
    """Pick the regularisation the VALIDATION questions prefer.

    The test questions take no part in this. Every choice made by looking at a
    score is a way of fitting to whatever produced that score, so the set used
    for tuning can never be the set used for reporting.
    """
    best_score, best_regularisation = -1.0, None
    for regularisation in REGULARISATION_CANDIDATES:
        candidate = train_chapter_classifier(X_train, y_train, regularisation)
        score = candidate.score(X_val, y_val)
        marker = ""
        if score > best_score:
            best_score, best_regularisation, marker = score, regularisation, "  <- best so far"
        print(f"   C={regularisation:>5}   validation accuracy {score:.3f}{marker}")
    return best_regularisation, best_score


def main():
    train_questions, y_train = read_split("train")
    val_questions, y_val = read_split("val")
    print(f"training on {len(train_questions)} questions, tuning on {len(val_questions)}")

    question_vectoriser = fit_question_vectoriser(train_questions)
    X_train = to_features(question_vectoriser, train_questions)
    X_val = to_features(question_vectoriser, val_questions)
    print(f"vocabulary learned from the training questions: {X_train.shape[1]} features\n")

    regularisation, val_score = choose_regularisation(X_train, y_train, X_val, y_val)
    print(f"\nchosen C={regularisation} at {val_score:.3f} validation accuracy")

    # Refit on training plus validation now that tuning is finished. The
    # validation rows have done their job, and holding them back would waste
    # every one of them. The test rows stay sealed.
    final_questions = train_questions + val_questions
    final_chapters = y_train + y_val
    question_vectoriser = fit_question_vectoriser(final_questions)
    chapter_classifier = train_chapter_classifier(
        to_features(question_vectoriser, final_questions), final_chapters, regularisation
    )

    # The vectoriser and the classifier are saved as one object. A model whose
    # feature 4,182 means "argand" is meaningless beside a vectoriser that
    # thinks 4,182 means "asymptote"; they are only correct together.
    assert_inside_project(MODEL_PATH)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump((question_vectoriser, chapter_classifier), MODEL_PATH)
    print(f"saved to models/{MODEL_PATH.name}, refit on {len(final_questions)} questions")


if __name__ == "__main__":
    main()
