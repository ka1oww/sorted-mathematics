"""The one line representation every rung reads, and where a line comes from.

A page arrives as a list of dicts, each a horizontal run of text with its
bounding box:

    {"x0", "y0", "x1", "y1", "size", "text", "page", "page_height"}

Both rungs consume exactly this shape, which is what lets `question_rule` be a
single rule rather than one rule per source. `size` is None when the source
cannot report it (OCR knows pixels, not points), and the rule treats a missing
size as "not a barcode" rather than guessing.

Coordinates are whatever unit the source works in - PostScript points for the
text layer, pixels for OCR - so a document must never mix the two. That is why
the rung is chosen per paper in `reader`, not per page.
"""

import pymupdf

# A page whose text layer yields fewer than this many characters has no usable
# text layer. Measured on the three public papers: the emptiest real page (a
# copyright-and-blank end page) still carries 344 characters, and a rasterised
# page carries 0, so anything in this range is unambiguous.
SCAN_CHARACTER_FLOOR = 40

# The running-header band of a typeset exam page, in points. Measured constant
# from the scout baseline rather than a fraction of the page, because that is
# the form it was scored in.
TEXT_LAYER_TOP_MARGIN = 50.0


def page_lines(page):
    """Every text line on one PyMuPDF page, sorted the way a reader sees them."""
    lines = []
    for block in page.get_text("dict")["blocks"]:
        if block["type"] != 0:  # 0 is text; 1 is an image
            continue
        for line in block["lines"]:
            text = "".join(span["text"] for span in line["spans"])
            if not text.strip():
                continue
            x0, y0, x1, y1 = line["bbox"]
            lines.append(
                {
                    "x0": round(x0, 1),
                    "y0": round(y0, 1),
                    "x1": round(x1, 1),
                    "y1": round(y1, 1),
                    "size": round(line["spans"][0]["size"], 1),
                    "text": text,
                }
            )
    # y first, then x: a marker and the text beside it must arrive in that order
    lines.sort(key=lambda line: (round(line["y0"], 0), line["x0"]))
    return lines


def text_layer_lines(document, page_indices):
    """Lines from the PDF text layer, in points. Empty on an image-only PDF."""
    rows = []
    for index in page_indices:
        page = document[index]
        height = page.rect.height
        for line in page_lines(page):
            rows.append({**line, "page": index, "page_height": height})
    return rows


def page_character_counts(document, page_indices):
    """How much text each page carries. The scan test reads this and nothing else."""
    return {index: len(document[index].get_text()) for index in page_indices}


def scanned_pages(character_counts):
    """The pages with no usable text layer."""
    return sorted(
        index
        for index, count in character_counts.items()
        if count < SCAN_CHARACTER_FLOOR
    )


def open_document(path):
    return pymupdf.open(path)


def all_pages(document):
    return list(range(document.page_count))
