# Data plan — sorted(mathematics)

Frozen 2026-08-14. This document is the contract for how the training data is
built. Change it deliberately, not in passing.

## Rule zero: the sources are read-only

No script in this project may create, modify, move or delete anything under a
source location. Sources are opened for reading only; every output is written
inside this project folder.

Every writer calls `assert_inside_project` in `src/paths.py` before it opens a
file, so a path bug fails loudly instead of writing somewhere it should not.
That is one function, not one per loader. No LaTeX is ever compiled, because
compiling drops `.aux` and `.log` files beside the sources.

The source roots live in `src/paths.py` and are overridable, so nothing in the
pipeline assumes my directory layout:

| Variable | Default | Used by |
|---|---|---|
| `EMBER_SOURCE_ROOT` | `~/Downloads/TUITION/H2 MATH EMBER` | `extract_latex.py`, `extract_pdf.py` |
| `TRIPLEMATH_PDF` | `~/Documents/GitHub/Untitled/triple-math-exercises.pdf` | `extract_triplemath.py` |

## The 21 chapters

Locked. Defined in `src/chapters.py`, mirroring the Ember teaching chapters so a
prediction reads directly as "which Ember package does this belong to".

Position in `CHAPTER_SLUGS` is the model's integer class id, so that order must
not be rearranged once a model has been trained.

## Sources in scope

All five are labelled at source, which is the reason they are cheap. Paths
below are relative to `EMBER_SOURCE_ROOT`, except source 5.

| # | Source | Questions | Path | Label from |
|---|--------|-----------|------|------------|
| 1 | Revision packages | 239 | `revision-packages/<CHAPTER>/parts/*_q.tex` | folder, plus `\qend{}` |
| 2 | Tutorials (LaTeX) | 196 | `tutorials/<chapter>/*.tex` | folder |
| 3 | Tutorials (PDF) | 87 | `tutorials/<chapter>/*.pdf` | folder |
| 4 | JC MATH TOPICAL | 648 | `misc/JC MATH TOPICAL.pdf` | topic page ranges |
| 5 | TripleMath exercises | ~602 | `TRIPLEMATH_PDF` | bookmark page ranges |
| | **Total** | **~1,772** | | |

Source 1 was originally recorded as 241, which counts the raw `*_q.tex` file
total. Two of those sit in my own `dropped/` and `dupes/`
subfolders and are correctly excluded: one was cut from its package, and one
duplicates a question already kept, which would have planted a train/test leak.
Usable total is 239.

Source 5 is a published third-party book, *Triple Math — Exercises*. Private
training data only, exactly like the school material: never published, no
examples in the repository. Roughly half the book is 9649 Further Mathematics
and 9820 H3 and is out of scope; only the sections mapping onto the 21 chapters
are used.
Its whole-past-paper section is also excluded, because a paper spans many
chapters so its heading is not a valid label.

Counts 1 and 2 are exact (one file per question, and `\Q` markers). Count 3 was
measured with a question counter validated against three tutorials whose LaTeX
counts were already known: 12/12, 17/17, 13/13. Count 4 counts `Q<n>.` markers.

### What is held out, and how

**The holdout is by whole paper, not by year.** An earlier version of this plan
said the 2025 prelims were never training input. They partly are. The revision
packages are organised by chapter rather than by year, so 2025 questions came
in with the rest of their chapter, carrying their provenance tags with them:
`data/private/questions.jsonl` holds 31 questions tagged 2025, from 13 schools
across 12 chapters, and the split deals them out like any other rows, 22 to
train, 6 to validation, 3 to test.

That is not a leak, because the year was never what protected the test set.
`src/split.py` assigns whole papers to one side of the fence, so a 2025
question sits in the test set only when every other question from its paper
does too. Holding a year back would have been a second, weaker rule on top of
that one, and the packages made it untrue before it could be useful.

A whole unseen prelim paper is still the production use case and still the
right demo. It has to be a paper that is not in the corpus at all, though,
rather than any paper that happens to be dated 2025.

### Deliberately excluded

| Source | Why |
|--------|-----|
| One school's reference folder (75 PDFs) | Inconsistent layouts; parsing cost exceeds the yield |
| One school's topical PDFs (~70 files) | Heterogeneous, 20 have no text layer at all |
| Topical `.docx` (13 files) | Pasted screenshots, zero extractable text, OCR only |
| 2023 / 2024 prelims | Unlabelled; would need ~800 hand labels for training data I do not lack |

Revisit only if measured accuracy says more data is the binding constraint.

## Row schema

One row per question.

| Field | Notes |
|-------|-------|
| `id` | Stable, derived from source and position |
| `chapter` | One of the 21 slugs |
| `text` | Normalised plain text: what the model sees |
| `raw_text` | Exactly as found, LaTeX or extracted PDF text |
| `source` | Which of the five sources |
| `year`, `school`, `level`, `exam`, `paper`, `question_no` | Parsed from provenance where present |
| `marks` | From `\mk{}` where present |

Keeping both `text` and `raw_text` costs nothing at parse time and turns
"LaTeX or plain text" into a two-second experiment rather than a re-parse.

## Preparation steps

### 1. Extract, read-only

Per source, one row per question.

- Source 1: read each `*_q.tex`. Ignore every `*_s.tex`. Those are solutions.
- Source 2: read each tutorial `.tex` whose name does not contain `Solutions`,
  split on the `\Q` macro.
- Source 3: extract the text layer, split on lines matching `^\s*\d+[.)]`.
- Source 4: walk the 22 topic page ranges, split on `Q<n>.`, map each topic onto
  a chapter slug.

### 2. Exclude solutions

A solution states its own method constantly, so a model trained on solutions
scores well and is worthless. Two places they hide: the `*_s.tex` files, and
solution pages interleaved through the topical bank.

### 3. Normalise to plain text

The decision, and the reasons in order of weight:

1. **It matches production.** A teacher pastes text, never LaTeX. Training on
   LaTeX means training on a distribution that never occurs at inference.
2. **The pre-trained encoder has never seen LaTeX.** `\int` tokenises to
   `\` + `in` + `t` and reassembles into nothing, while `integral` carries a
   rich pre-trained representation.
3. **Two surface forms would leak.** Source 4 is PDF text and sources 1–2 are
   LaTeX, and they cover different chapter mixes, so a model can learn
   "has backslashes" as a proxy for chapter.

What normalisation does: map LaTeX commands to words (`\int` to `integral`,
`\sum` to `sum`), strip formatting macros, drop PDF page-number artefacts in
symbol fonts, collapse whitespace.

Known limitation, accepted: the topical bank loses arithmetic operators in
extraction: `6i + 4j - 3k` arrives as `6 4 3 i j k`. The prose survives intact,
and the chapter signal lives in the prose nouns, so this is tolerable. It is
recorded here so it is never mistaken for a parsing bug later.

### 4. Deduplicate

Sources 1 and 4 both draw on prelim papers, so the same question can appear
twice. Duplicates split across train and test would inflate accuracy invisibly.

- First on the provenance tag, which 807 questions carry and which is exact.
- Then a near-duplicate text pass for the remainder.

### 5. Split by group, never at random

Group by `(school, year, paper)` and assign whole groups to train, validation
and test. Never split a group. Two questions from one paper are not independent
samples.

### 6. Audit before freezing

- Per-chapter counts, and flag any chapter under 20 rows.
- Text length distribution; inspect the shortest and longest rows by hand.
- Empty, near-empty, and non-question rows.
- Confirm no solution text survived.
- Confirm the train/test provenance sets are disjoint.

### 7. Freeze

Record the row count, per-chapter counts and a content hash. Every later
accuracy number is quoted against a named frozen dataset.

`tools/report_results.py` does this and writes `RESULTS.md`, so the numbers in
the README have a generated source rather than a typed one.

## Expected class balance

From source 4, which is the largest and best measured. Range 14 to 58 per
topic before merging into the 21 chapters: healthy, with no class so rare it
cannot be learned.

## What is not decided yet

- Whether Vectors 1 and Vectors 2 stay separate. Kept separate for now, matching
  the Ember packages. Expect this to be the largest single source of confusion.
- Train/validation/test proportions.
- Whether the near-duplicate pass uses exact text hashing or a similarity score.
