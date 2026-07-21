"""
Section-aware chunking.

Splits a document's full text into sections first (10-K/10-Q filing items,
transcript speaker turns), then packs each section into chunks bounded by
`max_chars` without ever cutting mid-sentence. Filing items and speaker
turns become the chunk's `section_or_speaker` metadata.

If a document doesn't match either structured pattern (e.g. a memo, or a
transcript in some other format), it falls back to one unlabeled section
covering the whole document, so ingestion always produces something usable
instead of raising on an unexpected layout.
"""

import re

from src.common.metadata import DocType

# A genuine filing item header (e.g. "ITEM 7.  MANAGEMENT'S DISCUSSION...")
# starts at a line boundary and is followed on the same line by its title.
# Inline references to items within a sentence (e.g. "...as discussed in
# Item 8. Financial Statements...") are always preceded by other prose, not
# a newline, so anchoring on '^' (line start) filters those out.
_ITEM_HEADER_RE = re.compile(
    r"^item\s+(?P<num>\d+)(?P<letter>[a-z]?)\.[ \t\xa0]+\S",
    re.MULTILINE | re.IGNORECASE,
)

# A transcript speaker turn: "Speaker Name: " at the start of a line, the
# standard convention for earnings-call transcripts (e.g. "Operator: ...",
# "Henrique Braun: ..."). A handful of non-dialogue lines that happen to
# share this "Capitalized Words:" shape -- glossary entries, ad copy on a
# scraped page -- can false-positive match and get labeled as a "speaker";
# this is a known limitation of a page-layout-agnostic heuristic, not a
# crash risk, and doesn't affect 10-K/10-Q chunking at all.
_SPEAKER_TURN_RE = re.compile(
    r"^(?P<name>[A-Z][A-Za-z.'-]+(?: [A-Z][A-Za-z.'-]+){0,3}):\s",
    re.MULTILINE,
)


def _normalize_whitespace(text: str) -> str:
    """Collapse the PDF's hard line-wraps back into flowing prose."""
    return re.sub(r"\s+", " ", text).strip()


def pack_into_chunks(text: str, max_chars: int) -> list[str]:
    """Greedily pack sentences into chunks no larger than `max_chars`.

    Never splits inside a sentence -- if a single sentence alone exceeds
    max_chars, it's kept whole as its own (oversized) chunk rather than cut.
    """
    normalized = _normalize_whitespace(text)
    if not normalized:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _split_filing_sections(text: str) -> list[tuple[str | None, str]]:
    matches = list(_ITEM_HEADER_RE.finditer(text))
    if not matches:
        return []

    sections: list[tuple[str | None, str]] = []
    if matches[0].start() > 0:
        sections.append(("Front Matter", text[: matches[0].start()]))

    for i, match in enumerate(matches):
        label = f"Item {match.group('num')}{match.group('letter').upper()}"
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((label, text[start:end]))
    return sections


def _split_transcript_sections(text: str) -> list[tuple[str | None, str]]:
    matches = list(_SPEAKER_TURN_RE.finditer(text))
    if not matches:
        return []

    sections: list[tuple[str | None, str]] = []
    if matches[0].start() > 0:
        sections.append((None, text[: matches[0].start()]))

    for i, match in enumerate(matches):
        label = match.group("name").strip()
        start = match.end()  # exclude the "Name\nROLE\n#N\n" label itself
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((label, text[start:end]))
    return sections


def chunk_document(
    text: str, doc_type: DocType, max_chars: int
) -> list[tuple[str, str | None]]:
    """Split a document's full text into (chunk_text, section_or_speaker) pairs."""
    if doc_type in (DocType.TEN_K, DocType.TEN_Q):
        sections = _split_filing_sections(text)
    elif doc_type == DocType.TRANSCRIPT:
        sections = _split_transcript_sections(text)
    else:
        sections = []

    if not sections:
        sections = [(None, text)]

    chunks: list[tuple[str, str | None]] = []
    for label, section_text in sections:
        for piece in pack_into_chunks(section_text, max_chars):
            chunks.append((piece, label))
    return chunks
