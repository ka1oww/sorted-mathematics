"""Manufacture a scanned paper from a digital one, so the OCR path can be
measured without a real scan.

No public exam paper is a scan, so the rung-2 numbers were measured against a
proxy: pages rasterised and re-wrapped as an image-only PDF, optionally rotated,
speckled and re-encoded to imitate a scanner. It is a kind stand-in - it has no
camera blur, no bleed-through from the reverse, no staple shadow - but the
skewed variant is the harshest thing a clean source can be made to produce, and
it is what `skewed_column` was fitted against.

    python3 tools/make_scan_proxy.py in.pdf out.pdf --pages 1,2,3
    python3 tools/make_scan_proxy.py in.pdf out.pdf --skew 0.7 --noise 6 --quality 60
"""

import argparse
import io
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import pymupdf                                                  # noqa: E402

from paths import assert_inside_project                         # noqa: E402


def degrade(png_bytes, skew, noise, quality):
    """Rotate, speckle and re-encode one page, the way a scanner would."""
    import numpy
    from PIL import Image

    image = Image.open(io.BytesIO(png_bytes)).convert("L")
    if skew:
        image = image.rotate(skew, resample=Image.BICUBIC, fillcolor=255,
                             expand=False)
    if noise:
        array = numpy.asarray(image).astype(numpy.float32)
        array += numpy.random.default_rng(0).normal(0, noise, array.shape)
        image = Image.fromarray(numpy.clip(array, 0, 255).astype(numpy.uint8))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def make_proxy(source, destination, pages=None, dpi=200,
               skew=0.0, noise=0.0, quality=95):
    document = pymupdf.open(source)
    indices = pages if pages is not None else range(document.page_count)
    scanned = pymupdf.open()
    for index in indices:
        source_page = document[index]
        pixmap = source_page.get_pixmap(dpi=dpi)
        image = pixmap.tobytes("png")
        if skew or noise or quality < 95:
            image = degrade(image, skew, noise, quality)
        page = scanned.new_page(width=source_page.rect.width,
                                height=source_page.rect.height)
        page.insert_image(page.rect, stream=image)
    scanned.save(str(destination))
    scanned.close()
    document.close()
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("destination")
    parser.add_argument("--pages", help="comma-separated zero-based page indices")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--skew", type=float, default=0.0, help="degrees")
    parser.add_argument("--noise", type=float, default=0.0, help="gaussian sigma")
    parser.add_argument("--quality", type=int, default=95, help="jpeg quality")
    arguments = parser.parse_args()

    pages = ([int(p) for p in arguments.pages.split(",")]
             if arguments.pages else None)
    destination = assert_inside_project(arguments.destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    make_proxy(arguments.source, destination, pages, arguments.dpi,
               arguments.skew, arguments.noise, arguments.quality)
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
