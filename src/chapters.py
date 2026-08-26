"""The 21 H2 Mathematics chapters this classifier predicts.

The label set mirrors the Ember teaching chapters exactly, so a prediction can
be read directly as "which Ember package does this question belong to".

CHAPTERS maps a stable slug to the display name. The slug is what gets written
into data files and model checkpoints; the display name is what a human sees.
They are separate on purpose: renaming a chapter for teaching should never
silently invalidate every label already stored on disk.

Chapter order here is the canonical label order. Position in CHAPTER_SLUGS is
the integer class id the model actually predicts, so this order must not be
rearranged once a model has been trained against it.
"""

CHAPTERS = {
    # Pure Mathematics (13)
    "apgp": "APGP",
    "sequences-and-series": "SEQUENCES AND SERIES",
    "equations-and-inequalities": "EQUATIONS AND INEQUALITIES",
    "functions": "FUNCTIONS",
    "curve-sketching": "CURVE SKETCHING",
    "vectors-1": "VECTORS 1",
    "vectors-2": "VECTORS 2",
    "complex-numbers": "COMPLEX NUMBERS",
    "differentiation-and-its-applications": "DIFFERENTIATION AND ITS APPLICATIONS",
    "maclaurin-series": "MACLAURIN SERIES",
    "integration-techniques": "INTEGRATION TECHNIQUES",
    "applications-of-integration": "APPLICATIONS OF INTEGRATION",
    "differential-equations": "DIFFERENTIAL EQUATIONS",
    # Statistics (8)
    "permutations-and-combinations": "PERMUTATIONS AND COMBINATIONS",
    "probability": "PROBABILITY",
    "discrete-random-variables": "DISCRETE RANDOM VARIABLES",
    "binomial-distribution": "BINOMIAL DISTRIBUTION",
    "normal-distribution": "NORMAL DISTRIBUTION",
    "sampling": "SAMPLING",
    "hypothesis-testing": "HYPOTHESIS TESTING",
    "correlation-and-regression": "CORRELATION AND REGRESSION",
}

CHAPTER_SLUGS = tuple(CHAPTERS)

# The same chapter is spelled several ways across the Ember source folders.
# Every alias here is lowercased before lookup, so only distinct spellings and
# abbreviations need an entry. The "differenciation" spellings are sic: the
# folders really are misspelled, and a loader has to match what is on disk.
FOLDER_ALIASES = {
    "sns": "sequences-and-series",
    "sequence and series": "sequences-and-series",
    "equation and inequalities": "equations-and-inequalities",
    "equations-inequalities": "equations-and-inequalities",
    "graph and transformation": "curve-sketching",
    "curve sketching + transformations": "curve-sketching",
    "transformations": "curve-sketching",
    "differentiation": "differentiation-and-its-applications",
    "differenciation": "differentiation-and-its-applications",
    "differentiation-applications": "differentiation-and-its-applications",
    "differenciation and its application": "differentiation-and-its-applications",
    "maclaurin": "maclaurin-series",
    "integration": "integration-techniques",
    "integration application": "applications-of-integration",
    "pnc": "permutations-and-combinations",
    "p and c": "permutations-and-combinations",
    "drv": "discrete-random-variables",
    "correlation-regression": "correlation-and-regression",
    "c and r": "correlation-and-regression",
    "de": "differential-equations",
}


def resolve(name):
    """Return the chapter slug for a folder name, or None if it is not a chapter.

    Accepts a slug, a display name, or any known folder spelling. Returning None
    rather than raising lets a loader skip directories such as `_snapshots` and
    `misc` without treating them as an error.
    """
    key = name.strip().lower()
    if key in CHAPTERS:
        return key
    if key in FOLDER_ALIASES:
        return FOLDER_ALIASES[key]
    normalised = key.replace(" ", "-").replace("_", "-")
    if normalised in CHAPTERS:
        return normalised
    for slug, display in CHAPTERS.items():
        if key == display.lower():
            return slug
    return None
