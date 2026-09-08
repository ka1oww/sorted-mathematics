"""Rung 2: lines from pixels, for papers mostly without text layers.

Rung 1 is provably inert on an image-only PDF - it extracts 0 characters and
finds 0 questions - and that is the entire reason this module exists. What it
produces is the same line shape `page_lines` produces, so `question_rule` runs
over it unchanged.

Every import here is deferred into the functions. The product's own dependency
is pymupdf and nothing else; docTR, torch, torchvision and opencv are only
needed when a scanned paper actually turns up, and a text-layer-only install
must keep working without them.

What OCR costs is character accuracy, not boundaries: a degraded scan drops the
ball from `160 cm` to `1600 cm`. Segmentation needs only the leading integer
and the left edge, both of which survive; a solution string does not.
"""

# Rendering resolution for the scan path. 200 dpi is what the scan proxies were
# measured at and is the usual floor for reliable OCR of body text.
RENDER_DPI = 200

_PREDICTOR = None

MISSING_STACK = (
    "the scan path needs python-doctr==1.1.0, torch==2.14.0, "
    "torchvision==0.29.0 and opencv-python==5.0.0.93, none of which the "
    "text-layer path requires. See the OCR path notes in CLAUDE.md."
)


class OcrUnavailable(RuntimeError):
    """docTR and its stack are not installed in this interpreter."""


def _import_doctr():
    """The one place the optional stack is imported, so a missing install
    reports itself rather than surfacing as a bare ModuleNotFoundError from
    somewhere inside the reader."""
    try:
        import torch
        from doctr.io import DocumentFile
        from doctr.models import ocr_predictor
    except ImportError as error:
        raise OcrUnavailable(MISSING_STACK) from error
    return torch, DocumentFile, ocr_predictor


def _load_predictor():
    """Load docTR once per process; the model costs seconds to build."""
    global _PREDICTOR
    if _PREDICTOR is not None:
        return _PREDICTOR
    torch, _document_file, ocr_predictor = _import_doctr()
    predictor = ocr_predictor(pretrained=True)
    if torch.backends.mps.is_available():
        predictor = predictor.to("mps")
    _PREDICTOR = predictor
    return predictor


def render_pages(document, page_indices, dpi=RENDER_DPI):
    """Rasterise pages to PNG bytes, and report the pixel height of each."""
    images, heights = [], {}
    for index in page_indices:
        pixmap = document[index].get_pixmap(dpi=dpi)
        images.append(pixmap.tobytes("png"))
        heights[index] = float(pixmap.height)
    return images, heights


def ocr_lines(document, page_indices, dpi=RENDER_DPI):
    """Lines read off the pixels, in the shared shape, measured in pixels.

    `size` is None throughout: OCR reports no font size, and the rule treats a
    missing size as "not a barcode" rather than guessing one.
    """
    _torch, DocumentFile, _ocr_predictor = _import_doctr()

    images, heights = render_pages(document, page_indices, dpi)
    predictor = _load_predictor()
    result = predictor(DocumentFile.from_images(images))

    rows = []
    for index, page in zip(page_indices, result.pages):
        height, width = page.dimensions
        page_rows = []
        for block in page.blocks:
            for line in block.lines:
                (x0, y0), (x1, y1) = line.geometry
                text = " ".join(word.value for word in line.words)
                if not text.strip():
                    continue
                page_rows.append(
                    {
                        "x0": round(x0 * width, 1),
                        "y0": round(y0 * height, 1),
                        "x1": round(x1 * width, 1),
                        "y1": round(y1 * height, 1),
                        "size": None,
                        "text": text,
                        "page": index,
                        "page_height": heights.get(index, float(height)),
                    }
                )
        page_rows.sort(key=lambda row: (round(row["y0"], 0), row["x0"]))
        rows.extend(page_rows)
    return rows


def ocr_top_margin(page_height):
    """The header band of a raster page, as the fraction the scan path was
    measured with. The text layer uses an absolute 50 pt instead, because that
    is the form the baseline was scored in."""
    return page_height * 0.06
