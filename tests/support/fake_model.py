"""Create a non-corpus model artifact for MCP tests."""

from pathlib import Path

import joblib
import numpy
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from chapters import CHAPTER_SLUGS


def write_fake_model(destination: Path) -> Path:
    """Write a deterministic 21-class model using synthetic training rows."""
    vectoriser = TfidfVectorizer().fit(["synthetic routing fixture"])
    classifier = DummyClassifier(strategy="prior")
    classifier.fit(
        numpy.arange(len(CHAPTER_SLUGS)).reshape(-1, 1), list(CHAPTER_SLUGS)
    )
    joblib.dump((vectoriser, classifier), destination)
    return destination
