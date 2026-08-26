"""Extract labelled questions from the Triple Math exercises book (source 5).

A 1,003-page PDF covering three A-Level syllabuses. Only the 9758 H2 material
maps onto the 21 chapters, so roughly half the book is out of scope: every
Further Maths / H3 section is skipped, and so is the past-paper section at the
back, because a whole paper spans many chapters and its heading is not a label.

Its bookmarks give exact page ranges, and inside a section the layout
alternates strictly:

    Problem 1. <question text>
    Solution.
    <worked answer>
    *****
    Problem 2. ...

A question runs from its "Problem n." marker to the next "Solution." marker and
no further. Cutting at "Solution." is the one step that cannot be compromised
on: a model trained on worked answers learns the answers' phrasing of the
method, scores well, and is useless on real questions.

Two label decisions were settled by reading the sections rather than trusting
their titles:

  * A3/A4 (Sequences and Series I/II) both map to sequences-and-series, not
    apgp. A3 interleaves AP/GP questions with recurrence and sequence-behaviour
    questions rather than keeping them in a block, so there is no clean cut
    that would justify splitting the two sections between the two chapters.
  * A15B (Special Continuous Random Variables) serves both syllabuses at once:
    11 of its 21 problems are normal-distribution questions, the other 10 are
    uniform/exponential/general-pdf questions that belong to Further Maths.
    Only the normal-distribution problems are extracted, listed explicitly in
    A15B_NORMAL below.

raw_text is the problem's text exactly as extracted, including any running page
headers that fall inside it when a problem spans a page break. text is the
cleaned version: header lines cut, artefacts dropped, whitespace collapsed.

Output: data/private/triplemath.jsonl, one JSON object per question.

Run: python3 src/extract_triplemath.py
"""

import json
import re

import fitz  # PyMuPDF

from chapters import resolve
from paths import PRIVATE_DATA, assert_inside_project, triplemath_pdf

SOURCE_PDF = triplemath_pdf()
OUTPUT_PATH = PRIVATE_DATA / "triplemath.jsonl"

# In-scope sections, keyed by their exact level-2 bookmark title. Everything
# not listed here is Further Maths or H3 material and is skipped. The values
# are chapter slugs; resolve() re-checks each one so a typo here cannot put
# an unknown label into the data.
SECTIONS = {
    "A1 Equations and Inequalities": "equations-and-inequalities",
    # A3 and A4 both go to sequences-and-series; see the module docstring.
    "A3 Sequences and Series I": "sequences-and-series",
    "A4 Sequences and Series II": "sequences-and-series",
    "A7 Vectors I - Basic Properties and Vector Algebra": "vectors-1",
    "A8 Vectors II - Lines": "vectors-2",
    "A9 Vectors III - Planes": "vectors-2",
    "A10.1 Complex Numbers - Complex Numbers in Cartesian Form": "complex-numbers",
    "A10.2 Complex Numbers - Complex Numbers in Polar Form": "complex-numbers",
    "A10.3 Complex Numbers - Geometrical Effects and De Moivre's Theorem": "complex-numbers",
    "A10.4 Complex Numbers - Loci in Argand Diagram": "complex-numbers",
    "A11 Permutations and Combinations": "permutations-and-combinations",
    "A12 Probability": "probability",
    "A14A Discrete Random Variables": "discrete-random-variables",
    "A14B Special Discrete Random Variables": "binomial-distribution",
    "A15B Special Continuous Random Variables": "normal-distribution",
    "A16 Sampling": "sampling",
    "A18A Hypothesis Testing I - Single Value": "hypothesis-testing",
    "B1 Graphs and Transformations I": "curve-sketching",
    "B2 Graphs and Transformations II": "curve-sketching",
    "B3 Functions": "functions",
    "B4 Differentiation": "differentiation-and-its-applications",
    "B5 Applications of Differentiation": "differentiation-and-its-applications",
    "B6 Maclaurin Series": "maclaurin-series",
    "B7 Integration Techniques": "integration-techniques",
    "B8 Applications of Integration I - Area and Volume": "applications-of-integration",
    "B10 Applications of Integration III - Trapezium and Simpson's Rule": "applications-of-integration",
    "B12 Separable Differential Equations": "differential-equations",
}

# A15B problems that are genuinely about the normal distribution, decided by
# reading all 21 problems. The excluded ones (Tutorial 1, 2, 5-10 and
# Assignment 1, 2) are uniform, exponential or general-pdf questions - 9649
# Further Maths material with no place among the 21 chapters.
A15B_NORMAL = {
    "Tutorial A15B": {3, 4, 11, 12, 13, 14, 15, 16, 17},
    "Assignment A15B": {3, 4},
}

# Problems dropped by hand. Tutorial A9 problem 1 is a True/False table with
# the answers printed inside the question itself, so there is no way to keep
# the question without keeping its answers.
DROPPED_IDS = {"triplemath-tutorial-a9-p01"}

# The past-paper section that is skipped but whose size is worth knowing:
# it holds real questions that could be hand-labelled later.
PAST_PAPER_SECTION = "9758 H2 Mathematics"

PROBLEM_RE = re.compile(r"^Problem (\d+)\.", re.M)
SOLUTION_RE = re.compile(r"^Solution\.", re.M)
# Five asterisk-operator glyphs separate consecutive problems. In most
# subsections a worked solution sits between the question and the separator,
# but the Self-Practice subsections of seven chapters print bare questions
# with no solutions, so the separator is the only terminator there.
SEPARATOR = "∗∗∗∗∗"

# Symbol-font artefacts that carry no text: Unicode private-use glyphs
# (pieces of large parentheses) and stray control characters that stand in
# for stretched delimiters.
ARTEFACT_RE = re.compile(r"[\ue000-\uf8ff\x00-\x09\x0b-\x1f]")


def section_ranges(toc):
    """Pair each in-scope section with its page range and subsection titles.

    A section runs from its own start page to the start of the next level-1
    or level-2 bookmark. Subsections are its level-3 bookmarks, kept in
    order; problem numbering restarts at 1 in each, which is how problems
    are later matched back to their subsection.
    """
    sections = []
    for i, (level, title, page) in enumerate(toc):
        if level != 2 or title not in SECTIONS:
            continue
        end = None
        subsections = []
        for lvl, sub, _ in toc[i + 1 :]:
            if lvl <= 2:
                break
            subsections.append(sub)
        for lvl, _, start in toc[i + 1 :]:
            if lvl <= 2:
                end = start
                break
        sections.append((title, page, end, subsections))
    return sections


def load_pages(doc, first, last):
    """Return the joined text of pages [first, last) with a page lookup.

    offsets[i] is where page (first + i) begins in the joined string, so a
    character position can be mapped back to the PDF page it was printed
    on - every output row records that page for hand-checking.
    """
    chunks, offsets, pos = [], [], 0
    for pno in range(first, last):
        text = doc[pno - 1].get_text()
        chunks.append(text)
        offsets.append(pos)
        pos += len(text)
    return "".join(chunks), offsets


def page_of(offsets, first_page, pos):
    page = first_page
    for i, start in enumerate(offsets):
        if start > pos:
            break
        page = first_page + i
    return page


def normalise(raw, section_title, subsection_titles):
    """Clean one problem's text without touching its wording.

    Running page headers land inside a problem whenever it crosses a page
    break: a bare page number plus the chapter or subsection title. Those
    lines belong to the page, not the question, so they are cut here (raw_text
    keeps them). Everything else is light: drop symbol-font artefacts and
    collapse whitespace - the PDF has real minus signs and intact symbols,
    so unlike the topical bank there is nothing to reconstruct.
    """
    headers = {section_title} | set(subsection_titles)
    lines = []
    for line in raw.split("\n"):
        stripped = line.strip()
        if stripped.isdigit() or stripped in headers:
            continue
        lines.append(line)
    text = ARTEFACT_RE.sub(" ", "\n".join(lines))
    return " ".join(text.split())


def subsection_kind(subsection_title):
    """'Tutorial B1A' -> 'Tutorial', 'Self-Practice A7' -> 'Self-Practice'."""
    kind = subsection_title.split()[0]
    if kind not in {"Tutorial", "Self-Practice", "Assignment"}:
        raise ValueError(f"unrecognised subsection: {subsection_title}")
    return kind


def problem_id(subsection_title, number):
    slug = re.sub(r"[^a-z0-9]+", "-", subsection_title.lower()).strip("-")
    return f"triplemath-{slug}-p{number:02d}"


def extract_section(doc, title, first, last, subsections):
    """Yield one row per problem in a section.

    Problems are found by their "Problem n." markers over the whole section
    at once, because a subsection can start mid-page and pure page ranges
    would then attach boundary problems to the wrong subsection. The number
    restarting at 1 marks each subsection boundary instead, and the count of
    restarts is checked against the bookmarks so a numbering surprise fails
    loudly rather than mislabelling silently.
    """
    text, offsets = load_pages(doc, first, last)
    markers = list(PROBLEM_RE.finditer(text))
    # Numbering restarts at 1 in every subsection, so starting prev_number
    # high makes the very first problem open the first subsection.
    sub_index, prev_number = -1, float("inf")
    keep_normal = A15B_NORMAL if title.startswith("A15B") else None

    for i, marker in enumerate(markers):
        number = int(marker.group(1))
        if number <= prev_number:
            sub_index += 1
        prev_number = number
        subsection = subsections[sub_index]

        span_end = markers[i + 1].start() if i + 1 < len(markers) else len(text)
        span = text[marker.end() : span_end]
        # The question ends at whichever comes first: its solution, or the
        # separator when the subsection prints no solutions at all. A span
        # with neither is the last problem of such a subsection and runs to
        # the next problem marker.
        cut = len(span)
        solution = SOLUTION_RE.search(span)
        if solution is not None:
            cut = solution.start()
        separator = span.find(SEPARATOR)
        if 0 <= separator < cut:
            cut = separator
        raw = span[:cut].strip()

        if keep_normal is not None and number not in keep_normal[subsection]:
            continue
        row_id = problem_id(subsection, number)
        if row_id in DROPPED_IDS:
            continue

        yield {
            "id": row_id,
            "chapter": SECTIONS[title],
            "section": title,
            "subsection": subsection_kind(subsection),
            "text": normalise(raw, title, subsections),
            "raw_text": raw,
            "source": "triplemath",
            "page": page_of(offsets, first, marker.start()),
            "year": None,
            "school": None,
            "level": None,
            "exam": None,
            "paper": None,
            "question_no": None,
        }

    if sub_index + 1 != len(subsections):
        raise ValueError(
            f"{title}: numbering restarts gave {sub_index + 1} subsections, "
            f"bookmarks say {len(subsections)}"
        )


def count_past_paper_problems(doc, toc):
    """Count the questions left behind in the skipped past-paper section."""
    for i, (level, title, page) in enumerate(toc):
        if level == 2 and title == PAST_PAPER_SECTION:
            end = next(start for lvl, _, start in toc[i + 1 :] if lvl <= 2)
            text, _ = load_pages(doc, page, end)
            return len(PROBLEM_RE.findall(text))
    return 0


def main():
    assert_inside_project(OUTPUT_PATH)

    for title, slug in SECTIONS.items():
        assert resolve(slug) == slug, f"{title!r} maps to unknown chapter {slug!r}"

    doc = fitz.open(SOURCE_PDF)
    toc = doc.get_toc()

    rows = []
    for title, first, last, subsections in section_ranges(toc):
        rows.extend(extract_section(doc, title, first, last, subsections))

    ids = [row["id"] for row in rows]
    assert len(ids) == len(set(ids)), "duplicate ids"
    assert not any("Solution." in row["text"] for row in rows), "solution text leaked"

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    per_section = {}
    for row in rows:
        per_section[row["section"]] = per_section.get(row["section"], 0) + 1
    print(f"{len(rows)} questions -> {OUTPUT_PATH}")
    for section, count in per_section.items():
        print(f"  {count:4d}  {section}")
    skipped = count_past_paper_problems(doc, toc)
    print(f"skipped past-paper section '{PAST_PAPER_SECTION}': {skipped} questions")


if __name__ == "__main__":
    main()
