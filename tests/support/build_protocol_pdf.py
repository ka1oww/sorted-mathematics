"""Build a tiny text-layer PDF for MCP protocol tests."""

from pathlib import Path

import pymupdf


def write_protocol_pdf(destination: Path) -> Path:
    document = pymupdf.open()
    for number in (1, 2, 3):
        page = document.new_page()
        page.insert_text((50.0, 120.0), str(number), fontsize=11)
        page.insert_text(
            (90.0, 120.0), f"Synthetic protocol fixture {number}.", fontsize=11
        )
        page.insert_text(
            (90.0, 140.0), f"Neutral placeholder words {number}.", fontsize=11
        )
    document.save(destination)
    document.close()
    return destination


def write_untrusted_pdf(destination: Path) -> Path:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((90.0, 120.0), "Synthetic prose without question markers.", fontsize=11)
    document.save(destination)
    document.close()
    return destination


def write_image_only_pdf(destination: Path) -> Path:
    source = destination.with_name("source.pdf")
    write_protocol_pdf(source)
    original = pymupdf.open(source)
    scanned = pymupdf.open()
    for original_page in original:
        pixmap = original_page.get_pixmap(dpi=120)
        page = scanned.new_page(
            width=original_page.rect.width, height=original_page.rect.height
        )
        page.insert_image(page.rect, pixmap=pixmap)
    scanned.save(destination)
    scanned.close()
    original.close()
    source.unlink()
    return destination
