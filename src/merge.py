"""Combines the three extractor outputs into one corpus.

Drops the out-of-syllabus topics, collapses duplicates on provenance first and
text second, and tags every row with the group the split will hold together.

Run: python3 src/merge.py
"""

import json
import re

from chapters import CHAPTER_SLUGS
from paths import PRIVATE_DATA, assert_inside_project

EXTRACTOR_OUTPUTS = ("ember_latex.jsonl", "topical_pdf.jsonl", "triplemath.jsonl")

MERGED_OUTPUT = "questions.jsonl"
# Unlabelled rows are kept rather than dropped. Relabelling them later is a
# mapping change, not a re-extraction.
UNLABELLED_OUTPUT = "questions_unlabelled.jsonl"

# Topics the bank teaches that no chapter of the current syllabus covers.
# "Thoerem" is sic, the bank's own spelling
TOPICS_OUTSIDE_SYLLABUS = {"Mathematical Induction", "Binomial Thoerem"}
POISSON_TOPIC = "Binomial & Poisson Distributions"

# these two rows share a provenance tag that is mistyped onto one of them, so
# collapsing on it would throw away a question that is not a duplicate. Keyed by
# row id, because the tag itself names a real paper.
ROWS_WITH_MISTYPED_PROVENANCE = {"topical-t10-q029", "topical-t13-q011"}


MIN_CHARS_FOR_TEXT_MATCH = 40


def normalise_paper(paper):
    # One school's cover prints II where another prints P2. Folding them stops a
    # duplicate surviving on house style alone.
    if not paper:
        return None
    compact = str(paper).strip().upper().replace(" ", "")
    return {"I": "P1", "II": "P2", "III": "P3", "1": "P1", "2": "P2"}.get(compact, compact)


def normalise_question_number(question_no):
    # A tag reading 9(a) or 9 (modified) names the same question as 9.
    if not question_no:
        return None
    match = re.match(r"\s*q?\s*(\d+)\s*\(?\s*([a-z])?", str(question_no).lower())
    if match:
        return match.group(1) + (match.group(2) or "")
    return str(question_no).strip().lower()


def provenance_identity(row):
    # All four fields, because a partial tag names the paper and not the question
    # inside it.
    school = str(row.get("school")).strip().upper().replace(" ", "") if row.get("school") else None
    year, paper = row.get("year"), normalise_paper(row.get("paper"))
    question_no = normalise_question_number(row.get("question_no"))
    if school and year and paper and question_no:
        return (school, str(year), paper, question_no)
    return None


def comparable_text(row):
    # strips down to just words for comparison
    stripped = re.sub(r"[^a-z0-9]+", " ", (row.get("text") or "").lower())
    return re.sub(r"\s+", " ", stripped).strip()


def grouping_hint(row):
    source, row_id = row.get("source"), row["id"]
    if source == "tutorial_latex":
        return ":".join(row_id.split(":")[:2])
    if source == "tutorial_pdf":
        return row_id.rsplit("-", 1)[0]
    return row_id


def is_poisson_question(row):
    return "poisson" in (row.get("text") or "").lower()


def read_extractions():
    rows = []
    for filename in EXTRACTOR_OUTPUTS:
        path = PRIVATE_DATA / filename
        if not path.exists():
            raise FileNotFoundError(f"extractor output missing: {path}")
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def drop_rows_outside_syllabus(rows):
    # Poisson and the two retired topics have no chapter to go to
    kept, dropped_by_reason = [], {}
    for row in rows:
        topic = row.get("topic")
        if topic in TOPICS_OUTSIDE_SYLLABUS:
            reason = topic
        elif topic == POISSON_TOPIC and is_poisson_question(row):
            reason = "Poisson"
        else:
            kept.append(row)
            continue
        dropped_by_reason[reason] = dropped_by_reason.get(reason, 0) + 1
    return kept, dropped_by_reason


def collapse_duplicates(rows):
    kept, seen_tags = [], set()
    provenance_collapsed = 0
    for row in rows:
        tag = provenance_identity(row)
        if tag and row.get("id") not in ROWS_WITH_MISTYPED_PROVENANCE:
            if tag in seen_tags:
                provenance_collapsed += 1
                continue
            seen_tags.add(tag)
        kept.append(row)

    deduplicated, seen_text = [], set()
    text_collapsed = 0
    for row in kept:
        text = comparable_text(row)
        if len(text) >= MIN_CHARS_FOR_TEXT_MATCH:
            if text in seen_text:
                text_collapsed += 1
                continue
            seen_text.add(text)
        deduplicated.append(row)

    return deduplicated, provenance_collapsed, text_collapsed


def write_corpus(rows, destination):
    resolved = assert_inside_project(destination)
    with resolved.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    rows = read_extractions()
    print(f"read {len(rows)} rows from {len(EXTRACTOR_OUTPUTS)} extractions")

    in_syllabus, dropped = drop_rows_outside_syllabus(rows)
    for reason, count in sorted(dropped.items()):
        print(f"  dropped {count:>3}  {reason} (no chapter teaches it)")

    corpus, by_provenance, by_text = collapse_duplicates(in_syllabus)
    print(f"  collapsed {by_provenance:>3}  duplicates matched on provenance")
    print(f"  collapsed {by_text:>3}  duplicates matched on identical text")

    for row in corpus:
        row["source_file"] = grouping_hint(row)

    seen_ids = set()
    for row in corpus:
        if row["id"] in seen_ids:
            raise ValueError(f"duplicate id in merged corpus: {row['id']}")
        seen_ids.add(row["id"])

    labelled = [row for row in corpus if row.get("chapter")]
    unlabelled = [row for row in corpus if not row.get("chapter")]
    write_corpus(labelled, PRIVATE_DATA / MERGED_OUTPUT)
    write_corpus(unlabelled, PRIVATE_DATA / UNLABELLED_OUTPUT)

    groups = len({row["source_file"] for row in labelled})
    print(f"\nwrote {len(labelled)} trainable rows to data/private/{MERGED_OUTPUT}")
    print(f"      {len(unlabelled)} rows awaiting a chapter to {UNLABELLED_OUTPUT}")
    print(f"  {groups} groups for the split")

    per_chapter = {slug: 0 for slug in CHAPTER_SLUGS}
    for row in labelled:
        per_chapter[row["chapter"]] += 1
    thinnest = min(per_chapter.items(), key=lambda item: item[1])
    print(f"  thinnest chapter: {thinnest[0]} at {thinnest[1]} rows")


if __name__ == "__main__":
    main()
