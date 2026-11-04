
import re


def chunk_text(
    text,
    page_number,
    chunk_size=700,
    overlap=120
):
    """
    Split page text into overlapping chunks.

    Prefers paragraph, sentence and word boundaries.
    Every chunk retains its original PDF page number.
    """

    if not isinstance(text, str):
        return []

    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size."
        )

    chunks = []

    start = 0
    chunk_id = 1

    while start < len(text):

        end = min(
            start + chunk_size,
            len(text)
        )

        if end < len(text):

            # Prefer a paragraph boundary.
            boundary = text.rfind(". ", start, end)

            # If no suitable sentence boundary exists,
            # prefer a word boundary.
            if boundary <= start:
                boundary = text.rfind(" ", start, end)

            if boundary > start:
                end = boundary + 1

        chunk = text[start:end].strip()

        if chunk:

            chunks.append({
                "chunk_id": chunk_id,
                "page_number": page_number,
                "text": chunk
            })

            chunk_id += 1

        if end >= len(text):
            break

        next_start = max(
            start + 1,
            end - overlap
        )

        # Avoid starting in the middle of a word.
        next_space = text.find(" ", next_start)

        if next_space != -1 and next_space < end:
            next_start = next_space + 1

        start = next_start

    return chunks
