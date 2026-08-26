"""Extract labelled questions from the two PDF sources (3 and 4 in DATA-PLAN.md).

Source 3 is seven statistics tutorials, one question PDF each; files whose
names contain "Sol" are solutions and are never opened. Source 4 is a
971-page topical compilation laid out as question pages, then an answer key,
then worked solutions repeating every question. Only the question pages are
wanted: a model trained on solution text learns the solutions' phrasing of
the method, scores well, and is useless on real questions.

Known extraction loss, accepted and recorded in DATA-PLAN.md: the topical
bank renders arithmetic operators in symbol fonts with no text mapping, so
"6i + 4j - 3k" arrives as "6 4 3 i j k". No attempt is made to reconstruct
the operators; guessing them would be inventing content. The prose, which
carries the chapter signal, survives intact.

raw_text is the text of the pages a question spans exactly as extracted,
except that each page's printed page-number line, and the run of duplicated
glyphs that follows it in the topical bank, are cut at page level; they
belong to the page, not to any question. text is the cleaned version.

Output: data/private/topical_pdf.jsonl, one JSON object per question.

Run: python3 src/extract_pdf.py
"""

import json
import re

import fitz  # PyMuPDF

from chapters import resolve
from paths import PRIVATE_DATA, PROJECT_ROOT, assert_inside_project, ember_source_root

SOURCE_ROOT = ember_source_root()
OUTPUT_PATH = PRIVATE_DATA / "topical_pdf.jsonl"

# The seven tutorial chapters that exist only as PDF. Every other tutorial
# folder has a .tex source and is handled by extract_latex.py instead.
TUTORIAL_FOLDERS = [
    "binomial-distribution",
    "discrete-random-variables",
    "hypothesis-testing",
    "normal-distribution",
    "PNC",
    "probability",
    "sampling",
]

# The 22 topics of the topical bank: 0-indexed start page (verified by hand
# against the PDF; a topic runs to the start of the next) and the topic name
# exactly as the running header prints it, typos included.
#
# The topics do not map one-to-one onto the 21 teaching chapters, so each
# entry carries its chapter slug explicitly, with the reasoning:
#
#   - Names resolve() already knows map through it (marked "via resolve").
#   - "Arithmetic & Geometric Progressions" is the APGP chapter under its
#     full name; "Graphing Techniques" is the curve-sketching chapter;
#     "Sampling Methods" / "Correlation and Linear Regression" are spelling
#     variants of their chapters.
#   - "Sigma Notation" sits inside the SEQUENCES AND SERIES chapter (sigma
#     notation and method of differences are that chapter's core), so it is
#     merged there.
#   - "Inequalities" and "Systems of Linear Equations" are both halves of
#     the EQUATIONS AND INEQUALITIES chapter.
#   - Five topics get chapter None because the call is not this script's to
#     make. "Vectors" spans both VECTORS 1 and VECTORS 2. "Integration & its
#     Applications" spans INTEGRATION TECHNIQUES and APPLICATIONS OF
#     INTEGRATION. "Binomial & Poisson Distributions" mixes BINOMIAL
#     DISTRIBUTION questions with Poisson ones that no current chapter
#     teaches. "Binomial Thoerem" (sic) and "Mathematical Induction" cover
#     material with no chapter of its own in the current syllabus. Their
#     rows keep chapter null and carry the topic name, so they can be
#     labelled later without re-extracting anything.
TOPICS = [
    (1, "Arithmetic & Geometric Progressions", "apgp"),
    (41, "Binomial Thoerem", None),
    (58, "Sigma Notation", "sequences-and-series"),
    (94, "Mathematical Induction", None),
    (123, "Graphing Techniques", "curve-sketching"),
    (156, "Functions", resolve("Functions")),  # via resolve
    (201, "Transformations", resolve("Transformations")),  # via resolve
    (222, "Inequalities", "equations-and-inequalities"),
    (236, "Systems of Linear Equations", "equations-and-inequalities"),
    (251, "Vectors", None),
    (322, "Differentiation & its Applications", "differentiation-and-its-applications"),
    (400, "Maclaurin’s Series", "maclaurin-series"),
    (446, "Integration & its Applications", None),
    (536, "Differential Equations", resolve("Differential Equations")),  # via resolve
    (621, "Complex Numbers", resolve("Complex Numbers")),  # via resolve
    (716, "Permutations and Combinations", resolve("Permutations and Combinations")),  # via resolve
    (737, "Probability", resolve("Probability")),  # via resolve
    (774, "Binomial & Poisson Distributions", None),
    (826, "Normal Distribution", resolve("Normal Distribution")),  # via resolve
    (861, "Sampling Methods", "sampling"),
    (883, "Hypothesis Testing", resolve("Hypothesis Testing")),  # via resolve
    (928, "Correlation and Linear Regression", "correlation-and-regression"),
]

# A question marker: "Q7." at the start of a line.
QUESTION_MARK = re.compile(r"^\s*Q(\d+)\.(.*)$", re.MULTILINE)

# A provenance tag on its own line, e.g. [2013/Prelim/ABCJC/II/5]. The shape
# is kept strict (must contain a year) so bracketed instructions inside
# questions, like "[Give all answers correct to the nearest dollar.]", are
# not mistaken for one.
# "Pelim" appears once in the source ([ABCJC/Pelim2013/P2/Q2(part)]); the
# typo is the source's and is accepted rather than corrected.
PROVENANCE_LINE = re.compile(
    r"^\s*\[\s*((?:[A-Z]{1,6}(?:\(JC\))?/)?(?:(?:Prelim|Pelim|Promo)\s*)?(?:19|20)\d{2}[ /][^\[\]]{0,55})\]\s*$",
    re.MULTILINE,
)

# Characters the topical bank's symbol fonts leave behind. The paired
# counts in the corpus give them away as glyph fragments, not content:
# large-bracket and fraction pieces arrive as runs like "§ · ¨ ¸ © ¹".
BANK_ARTEFACTS = re.compile(r"[§¨©ª«¬®¯±³´¶·¸¹º»¼½¾¿¦]")

# Control bytes (C0 and C1 ranges) are unmapped glyphs; the U+0D00 block
# and the private-use area are two more fonts whose page numbers and math
# symbols land on arbitrary codepoints.
JUNK_CHARS = re.compile(r"[\x00-\x09\x0b-\x1f\x7f-\x9f\xad\u0D00-\u0D7F\uE000-\uF8FF]")

# Page numbers in the bank's margins are set in a Greek-letter font
# (tau=0, upsilon=1, ... according to position), so they extract as short
# lines like "υψχ". Only whole lines of these letters are dropped; real
# Greek in the questions (pi, theta, lambda) uses other codepoints.
GREEK_DIGIT_LINE = re.compile(r"^[τ-ύ\s]{1,6}$")

LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl"}


def read_page_without_tail(page, page_index):
    """Return a page's text with its printed page number and what follows cut.

    Both sources print the 1-indexed page number as its own line near the
    bottom. In the topical bank a second, garbled copy of part of the page
    (an artefact of a duplicated text layer) follows that line, so cutting
    from the number line onward removes the number and the garble in one go.
    """
    text = page.get_text()
    printed_number = re.compile(rf"^\s*{page_index + 1}\s*$", re.MULTILINE)
    matches = list(printed_number.finditer(text))
    if matches:
        return text[: matches[-1].start()]
    return text


def is_solution_page(text):
    """Worked-solution pages carry headers and watermarks in a font whose
    space character extracts as \\x03. Question pages measure at most 2
    stray occurrences; solution pages dozens. The threshold of 3 sits in
    that gap.
    """
    return text.count("\x03") > 3


def is_answer_key_page(text):
    """The condensed answer key between questions and worked solutions:
    difficulty headings with bare-numbered answers, but no question
    markers and no provenance tags.
    """
    if QUESTION_MARK.search(text) or PROVENANCE_LINE.search(text):
        return False
    return bool(re.search(r"^\s*(Level\s*[123]|Answers?)\s*$", text, re.MULTILINE))


def parse_provenance(tag):
    """Split a source reference into its parts.

    Formats observed in the corpus, by frequency:

      2013/Prelim/ABCJC/II/5       year / exam / school / paper / question
      2013/DEFJC/I/Q6              year / school / paper / question
      GHIJC/Prelim2013/P2/Q3       school / exam+year / paper / question
      JKLJC/2013/Prelim/P1/Q2      school / year / exam / paper / question
      XYZMI/2013/Prelim/PU3/P2/Q1  school / year / exam / level / paper / question
      ABCJC/2013/Promo/Q4          school / year / exam / question
      Prelim 2015/MNOJC/P2/11      exam year / school / paper / question
      2015 PQRJC Prelim P2/10      year school exam paper / question

    School codes here are placeholders; the real ones stay with the corpus.

    Sub-part suffixes such as "3(a)" or "9(modified)" stay attached to the
    question number rather than being second-guessed. Anything that fits no
    known shape is left unparsed with every field null.
    """
    fields = {"year": None, "school": None, "level": None,
              "exam": None, "paper": None, "question_no": None}
    if not tag:
        return fields

    def is_year(s):
        return bool(re.fullmatch(r"(?:19|20)\d{2}", s))

    def exam_and_year(s):
        # "Prelim 2015" or the glued "Prelim2013"
        m = re.fullmatch(r"(Prelim|Pelim|Promo)\s*((?:19|20)\d{2})", s)
        return (m.group(1), int(m.group(2))) if m else None

    def is_paper(s):
        return bool(re.fullmatch(r"P\d|I{1,3}", s))

    def question_no(s):
        return s[1:] if re.fullmatch(r"Q\d.*", s) else s

    parts = [p.strip() for p in tag.split("/")]

    if len(parts) == 2 and " " in parts[0]:
        # 2015 PQRJC Prelim P2/10: everything but the question number shares
        # one space-separated field.
        for token in parts[0].split():
            if is_year(token):
                fields["year"] = int(token)
            elif token.lower() in ("prelim", "promo", "mye", "eoy"):
                fields["exam"] = token
            elif is_paper(token):
                fields["paper"] = token
            else:
                fields["school"] = token
        fields["question_no"] = question_no(parts[1])
    elif exam_and_year(parts[0]) and len(parts) == 4:
        # Prelim 2015/MNOJC/P2/11
        exam, year = exam_and_year(parts[0])
        fields.update(exam=exam, year=year, school=parts[1],
                      paper=parts[2], question_no=question_no(parts[3]))
    elif is_year(parts[0]) and len(parts) == 5:
        fields.update(year=int(parts[0]), exam=parts[1], school=parts[2],
                      paper=parts[3], question_no=question_no(parts[4]))
    elif is_year(parts[0]) and len(parts) == 4:
        fields.update(year=int(parts[0]), school=parts[1],
                      paper=parts[2], question_no=question_no(parts[3]))
    elif len(parts) == 6 and is_year(parts[1]):
        fields.update(school=parts[0], year=int(parts[1]), exam=parts[2],
                      level=parts[3], paper=parts[4],
                      question_no=question_no(parts[5]))
    elif len(parts) == 5 and is_year(parts[1]):
        fields.update(school=parts[0], year=int(parts[1]), exam=parts[2],
                      paper=parts[3], question_no=question_no(parts[4]))
    elif len(parts) == 4 and exam_and_year(parts[1]):
        exam, year = exam_and_year(parts[1])
        fields.update(school=parts[0], exam=exam, year=year,
                      paper=parts[2], question_no=question_no(parts[3]))
    elif len(parts) == 4 and is_year(parts[1]):
        # School/2013/Promo/Q4 has an exam in third place; a paper such as
        # P1 could sit there instead, so check which it is.
        fields.update(school=parts[0], year=int(parts[1]),
                      question_no=question_no(parts[3]))
        if is_paper(parts[2]):
            fields["paper"] = parts[2]
        else:
            fields["exam"] = parts[2]
    return fields


def clean_text(lines, drop_line_patterns, strip_bank_artefacts):
    """Normalise a question's lines into the plain prose a teacher would
    paste: drop running headers and page-number lines, strip glyph junk,
    and collapse the whitespace extraction scatters everywhere.
    """
    kept = []
    for line in lines:
        stripped = line.strip()
        if any(p.fullmatch(stripped) for p in drop_line_patterns):
            continue
        kept.append(line)
    text = "\n".join(kept)
    text = JUNK_CHARS.sub(" ", text)
    if strip_bank_artefacts:
        text = BANK_ARTEFACTS.sub(" ", text)
    for ligature, ascii_form in LIGATURES.items():
        text = text.replace(ligature, ascii_form)
    return " ".join(text.split())


def bank_drop_patterns(topic_name):
    # The provenance tag is metadata, recorded in its own fields; a teacher
    # pasting a question would not include it, so it stays out of text.
    return [
        re.compile(r"Topic\s+\d+"),
        re.compile(re.escape(topic_name)),
        re.compile(r"Level\s*[123]"),
        GREEK_DIGIT_LINE,
        PROVENANCE_LINE,
    ]


TUTORIAL_DROP_PATTERNS = [
    re.compile(r"Tutorial\s*[—|].*"),
    re.compile(r"H2 MATHEMATICS"),
    re.compile(r"·"),
    re.compile(r"9758"),
    re.compile(r"♦.*"),
]


def first_provenance(lines):
    for line in lines:
        m = PROVENANCE_LINE.match(line)
        if m:
            return m.group(1).strip()
    return None


def extract_tutorials():
    """Source 3: one PDF per chapter, questions numbered 1., 2., ... on
    their own lines. The numbering ascends through the whole file, so a
    marker is only accepted when it continues the sequence; that rejects
    look-alikes such as a bare "25." from a question's own data."""
    rows = []
    for folder in TUTORIAL_FOLDERS:
        chapter = resolve(folder)
        pdfs = [p for p in sorted((SOURCE_ROOT / "tutorials" / folder).glob("*.pdf"))
                if "Sol" not in p.name]
        if len(pdfs) != 1:
            raise SystemExit(f"expected exactly one question PDF in {folder}, found {len(pdfs)}")
        doc = fitz.open(pdfs[0])

        lines = []  # (page_index, line)
        for i, page in enumerate(doc):
            for line in read_page_without_tail(page, i).splitlines():
                lines.append((i, line))

        marker = re.compile(r"^\s*(\d{1,2})\.\s*$")
        segments = []  # (question_number, start_page, lines)
        expected = 1
        for page_index, line in lines:
            m = marker.match(line)
            if m and int(m.group(1)) == expected:
                segments.append((expected, page_index, []))
                expected += 1
            elif segments:
                segments[-1][2].append(line)

        for number, start_page, body in segments:
            raw = "\n".join(body)
            provenance = first_provenance(body)
            fields = parse_provenance(provenance)
            rows.append({
                "id": f"tutorial-{folder.lower()}-q{number:02d}",
                "chapter": chapter,
                "topic": None,
                "text": clean_text(body, TUTORIAL_DROP_PATTERNS, strip_bank_artefacts=False),
                "raw_text": raw,
                "source": "tutorial_pdf",
                **fields,
                "page": start_page,
            })
        doc.close()
    return rows


def extract_topical_bank():
    """Source 4: for each topic, keep the run of question pages at the top
    and stop at the first answer-key or worked-solution page; the rest of
    the topic is answers. Questions are split on Q-markers, and also on a
    provenance tag when the running question already has one: a few
    questions (the bank's Level 3 sections) are numbered in a margin
    column that extraction garbles, so their tag is the only reliable
    start-of-question signal left."""
    doc = fitz.open(SOURCE_ROOT / "misc" / "JC MATH TOPICAL.pdf")
    rows = []
    pages_excluded = {"solutions": 0, "answer_key": 0}

    boundaries = [start for start, _, _ in TOPICS] + [len(doc)]
    for topic_index, (start, name, chapter) in enumerate(TOPICS):
        stop = boundaries[topic_index + 1]
        header = doc[start].get_text()
        if f"Topic {topic_index + 1}" not in header:
            raise SystemExit(f"page {start} does not open topic {topic_index + 1}")

        kept = []  # (page_index, text)
        cut = stop
        for i in range(start, stop):
            text = doc[i].get_text()
            if is_solution_page(text):
                cut = i
                break
            if is_answer_key_page(text):
                cut = i
                break
            kept.append((i, read_page_without_tail(doc[i], i)))

        # Everything from the cut onward must really be answers. A question
        # page misread as an answer key would silently drop the rest of the
        # topic, so fail loudly if any dropped non-solution page still looks
        # like it holds questions.
        for i in range(cut, stop):
            text = doc[i].get_text()
            if is_solution_page(text):
                pages_excluded["solutions"] += 1
            elif QUESTION_MARK.search(text) or PROVENANCE_LINE.search(text):
                raise SystemExit(f"page {i} looks like questions after the cut in topic {topic_index + 1}")
            else:
                pages_excluded["answer_key"] += 1

        lines = []  # (page_index, line)
        for page_index, text in kept:
            for line in text.splitlines():
                lines.append((page_index, line))

        segments = []  # (start_page, lines)
        segment_has_tag = False
        for page_index, line in lines:
            q = QUESTION_MARK.match(line)
            tag_line = PROVENANCE_LINE.match(line)
            if q:
                segments.append((page_index, [q.group(2)] if q.group(2).strip() else []))
                segment_has_tag = False
                continue
            if tag_line and segments and segment_has_tag:
                # A second tag while a question is already underway means a
                # new question whose margin number did not survive extraction.
                segments.append((page_index, [line]))
                segment_has_tag = True
                continue
            if segments:
                segments[-1][1].append(line)
                segment_has_tag = segment_has_tag or bool(tag_line)

        for sequence, (start_page, body) in enumerate(segments, start=1):
            provenance = first_provenance(body)
            fields = parse_provenance(provenance)
            rows.append({
                "id": f"topical-t{topic_index + 1:02d}-q{sequence:03d}",
                "chapter": chapter,
                "topic": name,
                "text": clean_text(body, bank_drop_patterns(name), strip_bank_artefacts=True),
                "raw_text": "\n".join(body),
                "source": "topical_bank",
                **fields,
                "page": start_page,
            })
    doc.close()
    return rows, pages_excluded


def write_rows(rows):
    resolved = assert_inside_project(OUTPUT_PATH)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with open(resolved, "w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    tutorial_rows = extract_tutorials()
    bank_rows, pages_excluded = extract_topical_bank()
    rows = tutorial_rows + bank_rows

    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate row ids in the extracted rows; two questions "
                         "would silently merge downstream")

    write_rows(rows)

    print(f"wrote {len(rows)} rows to {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"  tutorial_pdf: {len(tutorial_rows)}")
    print(f"  topical_bank: {len(bank_rows)}")
    print(f"topical bank pages excluded: {pages_excluded['solutions']} worked-solution, "
          f"{pages_excluded['answer_key']} answer-key")

    by_chapter = {}
    for row in tutorial_rows:
        by_chapter[row["chapter"]] = by_chapter.get(row["chapter"], 0) + 1
    print("tutorial questions per chapter:")
    for chapter, count in sorted(by_chapter.items()):
        print(f"  {chapter}: {count}")

    print("topical bank questions per topic:")
    for _, name, chapter in TOPICS:
        topic_rows = [r for r in bank_rows if r["topic"] == name]
        tagged = sum(1 for r in topic_rows if r["question_no"] is not None)
        print(f"  {name}: {len(topic_rows)} ({tagged} with provenance) -> {chapter}")

    unmapped = sum(1 for r in rows if r["chapter"] is None)
    tagged = sum(1 for r in rows if r["question_no"] is not None)
    print(f"rows with provenance: {tagged}")
    print(f"rows with chapter null (mapping left open): {unmapped}")

    shortest = min(rows, key=lambda r: len(r["text"]))
    longest = max(rows, key=lambda r: len(r["text"]))
    print(f"shortest text: {shortest['id']} ({len(shortest['text'])} chars): {shortest['text'][:120]!r}")
    print(f"longest text: {longest['id']} ({len(longest['text'])} chars)")


if __name__ == "__main__":
    main()
