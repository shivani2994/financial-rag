"""
Loads raw text out of a PDF with PyMuPDF.

The 10-K/10-Q PDFs in this corpus were saved via "print to PDF" from a
browser, so PyMuPDF's extraction includes one repeating block of browser
furniture per page: a timestamp, the filing's URL slug, the page URL, and a
"page/total" counter. That block gets inserted mid-paragraph wherever a page
break happens to fall, which would otherwise break chunk coherence. It's
stripped here, once, at load time, so every downstream step works with clean
prose.
"""

import re
from pathlib import Path

import fitz

# One line each: timestamp, filename slug, source URL, "N/total" page counter.
# Observed identically once per page across every 10-K/10-Q PDF in the corpus.
_PAGE_FURNITURE_RE = re.compile(
    r"\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}\s*[AP]M\n"
    r".+\n"
    r"https?://\S+\n"
    r"\d+/\d+\n?"
)


def load_pdf_text(pdf_path: Path) -> str:
    """Return the full cleaned text of a PDF, pages concatenated in order."""
    doc = fitz.open(pdf_path)
    try:
        pages = [page.get_text() for page in doc]
    finally:
        doc.close()

    cleaned_pages = [_PAGE_FURNITURE_RE.sub("", page) for page in pages]
    return "".join(cleaned_pages)
