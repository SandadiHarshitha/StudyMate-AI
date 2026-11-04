
import re


# ==========================================
# CLEAN ANSWER STRUCTURE
# ==========================================

def clean_empty_sections(answer):

    if not answer:
        return ""

    lines = answer.splitlines()
    cleaned = []
    skip_section = False

    for line in lines:

        heading = re.match(
            r"^\s*#{1,3}\s+(.+?)\s*$",
            line
        )

        if heading:
            skip_section = False

            title = heading.group(1).strip().lower()

            if title in {
                "information not found",
                "information not found (if applicable)"
            }:
                skip_section = True
                continue

            cleaned.append(line)
            continue

        if skip_section:
            continue

        if line.strip().lower() in {
            "none",
            "not applicable",
            "n/a"
        }:
            continue

        cleaned.append(line)

    return "\n".join(cleaned).strip()


# ==========================================
# BUILD EVIDENCE VERIFICATION PROMPT
# ==========================================

def build_validation_prompt(question, answer, context):

    return f"""
You are an evidence verification assistant for StudyMate AI.

Your task is to verify and improve an answer using ONLY
the supplied PDF passages.

STUDENT QUESTION:
{question}

ORIGINAL ANSWER:
{answer}

PDF EVIDENCE:
{context}

STRICT VERIFICATION RULES:

1. Check every factual claim against the PDF evidence.
2. Remove claims that are not supported by the passages.
3. Correct claims that contradict the passages.
4. Do not introduce facts from general knowledge.
5. Do not invent formulas or mathematical relationships.
6. Do not confuse energy with power.
7. Preserve correct technical terminology.
8. Keep the original page citations only when supported.
9. Never invent page numbers.
10. If evidence is insufficient, state that clearly.
11. Do not claim the entire PDF lacks information.
12. Do not repeat the same explanation.
13. Do not add a conclusion or bibliography.

OUTPUT FORMAT:

## Explanation
Explain the answer in simple language with page citations.

## Important Points
List the important supported facts without repetition.

## Formula
Write exactly:
Formula could not be reliably extracted from the retrieved PDF passages. Mathematical expressions require verification against the original PDF page.

Do not generate any mathematical formula.

## Information Not Found
Include this section only if an important part of the question
is not supported by the passages. Otherwise omit it.

Return ONLY the revised answer.
"""


# ==========================================
# ANSWER VALIDATION
# ==========================================

def validate_answer(
    question,
    answer,
    context,
    verification_function
):

    if not answer or not answer.strip():
        return "No answer was generated."

    prompt = build_validation_prompt(
        question,
        answer,
        context
    )

    try:

        verified_answer = verification_function(prompt)

        if verified_answer and verified_answer.strip():
            answer = verified_answer.strip()

    except Exception:
        # Keep the original answer if verification fails.
        pass

    return clean_empty_sections(answer)
