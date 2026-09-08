"""The one rule that cuts a paper into questions: a question opens with a
number sitting in the question column, and the numbers ascend.

This is the `stops` variant of the scout baseline, which scored 43/43 questions
with 43/43 exact page sets and no invented boundaries over four papers from two
examination boards. Both rungs run this code; only the source of the lines
differs, and the one divergence is the column fit (see `skewed_column`).

Two guards are what make a deliberately permissive marker pattern safe:

  the column   a marker line must begin at the question column, a left edge
               learnt from the document itself rather than assumed. "12 cm"
               inside a question is indented and dies here.
  the sequence the number must be the next one expected. A stray column number
               that breaks the run dies here.

A rule that instead insists on punctuation - the `\\d+\\.` everybody writes
first - finds nothing at all on a real exam paper: SEAB separates the number
from the text with a tab and Cambridge with a single space.
"""

import re
from collections import Counter

DOTTED = re.compile(r"\.{15,}")

# A number, then any separator or nothing at all. Permissive on purpose; the
# column and sequence guards above are what keep it honest.
MARKER = re.compile(r"^(\d{1,2})(?:[.)]\s*|\s+|$)(.*)$", re.DOTALL)

# Lines that close the running question without opening a new one: the end
# matter every board prints after the last question, and the section headers an
# H2 paper 2 uses to split pure from statistics. Treating these as furniture
# between questions rather than as the tail of the previous one is what moved
# the baseline from 40/43 to 43/43 exact page sets.
STOP_LINE = re.compile(
    r"^(Section\s+[A-Z]\b"
    r"|BLANK PAGE"
    r"|Additional page"
    r"|Permission to reproduce items)",
    re.IGNORECASE,
)

# The footer band as a fraction of page height. "[Turn over" sits a little above
# the printed footer and is page furniture just the same.
FOOTER_FRACTION = 0.92

# Glyphs smaller than this are barcodes and scan marks, never body text.
MINIMUM_BODY_SIZE = 6.5

# A marker's own row is not a single line: a stacked fraction puts its numerator
# a few points higher than the line it sits in, and a tall display integral
# belonging to the question a marker opens can start several points above the
# marker's own baseline. Cut on a band around the marker rather than on its
# baseline.
CUT_BAND = 6.0

# When a marker is the first thing on its page, widen that band to the whole
# page top: nothing above it can belong to a question that ended on the previous
# page. This one line took the held-out paper from 9/11 to 11/11 exact.
TOP_OF_PAGE = 0.12

# How far from the learnt column a line may start and still count as a marker.
# Points on a typeset page; pixels on a raster, where OCR box jitter is larger.
TEXT_LAYER_TOLERANCE = 2.0
OCR_TOLERANCE = 8.0


def repeated_texts(rows, page_count, threshold=0.5):
    """Text appearing on at least half the pages belongs to the page, not to a
    question. Short strings are exempt: a bare "1" is a question marker on one
    page and a fraction numerator on another."""
    pages_with = {}
    for row in rows:
        pages_with.setdefault(row["text"].strip(), set()).add(row["page"])
    floor = max(2, threshold * page_count)
    return {
        text
        for text, pages in pages_with.items()
        if len(text) >= 5 and len(pages) >= floor
    }


def is_furniture(row, repeated, top_margin):
    """True for anything printed by the page rather than by a question."""
    text = row["text"].strip()
    if not text:
        return True
    size = row.get("size")
    if size is not None and size < MINIMUM_BODY_SIZE:  # barcodes, scan marks
        return True
    if DOTTED.search(text):  # write-on answer rules
        return True
    if row["y0"] < top_margin:  # running header band
        return True
    if row["y0"] > row["page_height"] * FOOTER_FRACTION:  # footer band
        return True
    return text in repeated  # anything else repeated


def question_column(rows):
    """The left margin of the question column: the smallest x0 that at least
    three body lines share. Learnt per document, so a school with its own
    margins is handled without configuration."""
    counts = Counter(row["x0"] for row in rows)
    shared = sorted(x0 for x0, count in counts.items() if count >= 3)
    return shared[0] if shared else None


def skewed_column(rows, tolerance):
    """The column of a scanned page is not a vertical line but a diagonal.

    A page fed through a scanner a fraction of a degree off square puts the
    column tens of pixels further right at the foot than at the head: at 0.7
    degrees over a 2339 px page the left edge drifts about 28 px, far outside
    any sane fixed tolerance. Fitting the column as a least-squares line in
    (y, x) recovers it, and costs nothing on a straight page, where the fitted
    slope comes back at about -0.0002.

    Returns (a predicate on a row, the fitted slope).
    """
    left = min(row["x0"] for row in rows)
    band = [row for row in rows if row["x0"] <= left + 4 * tolerance]
    if len(band) < 3:
        column = question_column(rows)
        if column is None:
            return (lambda row: False), 0.0
        return (lambda row: abs(row["x0"] - column) <= tolerance), 0.0
    count = len(band)
    mean_y = sum(row["y0"] for row in band) / count
    mean_x = sum(row["x0"] for row in band) / count
    variance = sum((row["y0"] - mean_y) ** 2 for row in band)
    covariance = sum((row["y0"] - mean_y) * (row["x0"] - mean_x) for row in band)
    slope = covariance / variance if variance else 0.0
    intercept = mean_x - slope * mean_y
    return (
        lambda row: abs(row["x0"] - (slope * row["y0"] + intercept)) <= tolerance,
        slope,
    )


def column_test(rows, tolerance, skew_aware):
    """Pick the column fit. Returns (predicate, description)."""
    if skew_aware:
        in_column, slope = skewed_column(rows, tolerance)
        return in_column, f"skew-aware fit, slope {slope:+.4f} per unit"
    column = question_column(rows)
    if column is None:
        return (lambda row: False), None
    return (lambda row: abs(row["x0"] - column) <= tolerance), column


def find_markers(body, in_column):
    """Every line that opens a question, as (page, y0, number, rest, identity).

    `identity` is the line's index in `body`, so the marker line can be told
    apart from an identical-looking line elsewhere on the page.
    """
    markers, expected = [], 1
    for index, row in enumerate(body):
        if not in_column(row):
            continue
        text = row["text"].strip()
        if STOP_LINE.match(text):
            continue
        match = MARKER.match(text)
        if match and int(match.group(1)) == expected:
            markers.append(
                (row["page"], row["y0"], expected, match.group(2).strip(), index)
            )
            expected += 1
    return markers


def _page_height(body, page):
    for row in body:
        if row["page"] == page:
            return row["page_height"]
    return 842.0


def _cut_points(markers, body):
    """Where each question starts, as a (page, y) below which its lines lie."""
    cuts, pages_seen = [], set()
    for page, y0, _number, _rest, _index in markers:
        if page not in pages_seen and y0 < _page_height(body, page) * TOP_OF_PAGE:
            cuts.append((page, 0.0))
        else:
            cuts.append((page, y0 - CUT_BAND))
        pages_seen.add(page)
    return cuts


def segment_lines(rows, page_count, top_margin, tolerance, skew_aware=False):
    """Cut a paper's lines into questions.

    `rows` is every line of the paper in one coordinate system, in reading
    order. Returns (questions, column description), where each question carries
    its number, the page it opens on, the exact set of pages it spans, and its
    text.
    """
    repeated = repeated_texts(rows, page_count)
    body = [row for row in rows if not is_furniture(row, repeated, top_margin)]
    if not body:
        return [], None

    in_column, column = column_test(body, tolerance, skew_aware)
    markers = find_markers(body, in_column)
    if not markers:
        return [], column

    stops = [
        (row["page"], row["y0"]) for row in body if STOP_LINE.match(row["text"].strip())
    ]
    cuts = _cut_points(markers, body)

    questions = [
        {
            "number": number,
            "start_page": page,
            "pages": {page},
            "lines": [rest] if rest else [],
        }
        for page, _y0, number, rest, _index in markers
    ]

    for index, row in enumerate(body):
        here = (row["page"], row["y0"])
        owner = None
        for position, cut in enumerate(cuts):
            if here >= cut:
                owner = position
        if owner is None:
            continue  # above the first marker: front matter
        if index == markers[owner][4]:
            continue  # the marker line itself, already added
        if STOP_LINE.match(row["text"].strip()):
            continue  # a stop line belongs to no question
        marker_at = (markers[owner][0], markers[owner][1])
        if any(here >= stop > marker_at for stop in stops):
            continue  # past a stop line, so past the question
        questions[owner]["lines"].append(row["text"].strip())
        questions[owner]["pages"].add(row["page"])

    for question in questions:
        question["pages"] = sorted(question["pages"])
        question["text"] = " ".join(line for line in question["lines"] if line)
        del question["lines"]
    return questions, column
