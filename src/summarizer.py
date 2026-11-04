import re
import pymupdf

from src.llm import ask_ollama, ask_gemini

def mark_ocr_uncertainty(text):
    """Flag known OCR-corrupted symbols without guessing their meaning."""

    suspicious_patterns = [
        r"\bJor\b",
        r"\bJ[;¢]\b",
        r"\bJ[0-9]\b",
    ]

    for pattern in suspicious_patterns:
        text = re.sub(
            pattern,
            lambda match: f"[OCR UNCERTAIN: {match.group(0)}]",
            text,
            flags=re.IGNORECASE
        )

    return text



def run_ai(prompt, mode):
    if mode == "Offline":
        return ask_ollama(prompt), "Ollama (Offline)"

    if mode == "Online":
        return ask_gemini(prompt), "Gemini (Online)"

    try:
        return ask_gemini(prompt), "Gemini (Online)"
    except Exception:
        return ask_ollama(prompt), "Ollama (Offline fallback)"


def get_chapters(pdf_bytes, chunks=None):
    import re

    if not chunks:
        raise ValueError("PDF chunks are not available.")

    # Combine OCR chunks belonging to the same physical PDF page.
    pages = {}

    for chunk in chunks:
        page_no = chunk.get("page_number")
        if page_no is not None:
            pages.setdefault(page_no, []).append(
                chunk.get("text", "")
            )

    page_text = {
        page_no: " ".join(texts)
        for page_no, texts in pages.items()
    }

    # Recognize both numeric and word-based chapter numbering.
    numbers = {
        "1": 1, "one": 1,
        "2": 2, "two": 2,
        "3": 3, "three": 3,
        "4": 4, "four": 4,
        "5": 5, "five": 5,
        "6": 6, "six": 6,
        "7": 7, "seven": 7,
        "8": 8, "eight": 8,
        "9": 9, "nine": 9,
        "10": 10, "ten": 10
    }

    detected = {}

    # Ignore the table of contents and locate actual chapter headings.
    for page_no in sorted(page_text):
        if page_no <= 8:
            continue

        text = page_text[page_no]

        pattern = re.compile(
            r"\bchapter\s+(1|one|2|two|3|three|4|four|"
            r"5|five|6|six|7|seven|8|eight|9|nine|10|ten)\b",
            re.IGNORECASE
        )

        for match in pattern.finditer(text):
            number = numbers[match.group(1).lower()]

            # Keep the first occurrence after the contents pages.
            if number not in detected:
                detected[number] = page_no

    if not detected:
        raise ValueError(
            "No chapter headings found after the contents pages."
        )

    # Use the actual heading pages to calculate chapter boundaries.
    chapter_numbers = sorted(detected)

    chapters = []

    for i, number in enumerate(chapter_numbers):
        start = detected[number]

        if i + 1 < len(chapter_numbers):
            end = detected[chapter_numbers[i + 1]] - 1
        else:
            end = max(page_text)

        # Use the heading text as the chapter title.
        text = page_text[start]
        heading_match = re.search(
            rf"\bchapter\s+(?:{number}|{list(numbers.keys())[list(numbers.values()).index(number)]})\b"
            rf"(.{{0,180}})",
            text,
            re.IGNORECASE
        )

        heading = (
            heading_match.group(0).strip()
            if heading_match
            else f"Chapter {number}"
        )

        chapters.append({
            "title": f"Chapter {number}",
            "start": start,
            "end": end
        })

    return chapters

def summarize_text(text, mode):
    prompt = f"""
You are StudyMate AI, a PDF study tutor.

Summarize ONLY the supplied PDF content.

Rules:
1. Do not invent information.
2. Preserve technical terminology.
3. Include important definitions and concepts.
4. Use simple student-friendly language.
5. Avoid repetition.
6. Do not invent formulas.

PDF CONTENT:
{text}

Create a clear, structured summary.
"""
    return run_ai(prompt, mode)


def summarize_chunks(chunks, mode="Offline", batch_size=8):
    if not chunks:
        raise ValueError("No PDF content is available.")

    summaries = []
    engine = ""

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]

        text = "\n\n".join(
            f"[PDF Page {c['page_number']}]\n{c['text']}"
            for c in batch
        )

        summary, engine = summarize_text(text, mode)
        summaries.append(summary)

    final_prompt = f"""
Combine these section summaries into one complete,
student-friendly summary of the PDF.

Use only the supplied summaries.
Remove repeated points.
Preserve important technical concepts.
Do not invent facts or formulas.

SECTION SUMMARIES:
{chr(10).join(summaries)}
"""

    final_summary, engine = run_ai(final_prompt, mode)
    return final_summary, engine



def summarize_chapter(chunks, chapter, mode="Offline"):
    selected = [
        c for c in chunks
        if chapter["start"] <= c["page_number"] <= chapter["end"]
    ]

    if not selected:
        return "No readable content found in this chapter.", "None"

    selected.sort(key=lambda c: (c["page_number"], c["chunk_id"]))

    chapter_title = chapter["title"]
    page_range = f"{chapter['start']}–{chapter['end']}"

    summaries = []
    engine = ""

    # Summarize the chapter in batches.
    for i in range(0, len(selected), 30):
        batch = selected[i:i + 30]

        text = "\n\n".join(
    f"[PDF Page {c['page_number']}]\n"
    f"{mark_ocr_uncertainty(c['text'])}"
    for c in batch
)

        prompt = f"""
You are StudyMate AI, summarizing a section of a student's PDF.

SOURCE RESTRICTIONS:
- Use only information explicitly present in PDF CONTENT.
- Do not introduce information from other subjects or chapters.
- Do not use general knowledge to complete missing explanations.
- Do not invent definitions, examples, technical terms or abbreviations.
- Preserve technical terms exactly as extracted.
- Do not infer the meaning of corrupted OCR text.
- Do not generate, reproduce, derive or reconstruct equations.
- Do not include mathematical expressions as definitions.
- If information is unclear, write:
  "Unclear in extracted PDF text."
- Avoid duplicate explanations.
- Include the physical PDF page number for each important concept.
- Treat [OCR UNCERTAIN: ...] as unreadable text.
- Never define, expand, or interpret an OCR UNCERTAIN item.
- Do not use uncertain symbols in technical definitions.

IMPORTANT:
If a topic is not clearly supported by PDF CONTENT, leave it out.

Write concise, accurate study notes.

PDF CONTENT:
{text}
"""

        summary, engine = run_ai(prompt, mode)
        summaries.append(summary)

    # Combine all section summaries.
    final_prompt = f"""
You are StudyMate AI, creating exam-friendly notes.

Create one complete summary for:
{chapter_title}

PDF page range: {page_range}

Use ONLY the section summaries below.

ORGANIZE THE ANSWER INTO:

1. Chapter Overview
2. Important Concepts and Explanations
3. Important Definitions
4. Key Technical Points
5. Equation Verification
6. Conclusion

STRICT RULES:
- Preserve the technical meaning from the source.
- Do not add outside knowledge.
- Do not invent or expand abbreviations.
- Do not introduce topics from other chapters.
- Remove duplicate explanations.
- Keep important details; do not oversimplify.
- Preserve page references supplied in the section summaries.
- Do not reproduce equations from section summaries.
- Do not generate equations using your own knowledge.
- Write: "Equations require verification from the original PDF."
EQUATION SAFETY:
- Do not generate equations from memory or general knowledge.
- Include an equation only if it appears clearly and completely
  in the supplied PDF text.
- Never derive, repair, or reconstruct an equation.
- If an equation cannot be verified, do not display it.
- Write: "Equations require verification from the original PDF."

SECTION SUMMARIES:
{chr(10).join(summaries)}
"""

    final_summary, final_engine = run_ai(final_prompt, mode)

    return final_summary, final_engine
