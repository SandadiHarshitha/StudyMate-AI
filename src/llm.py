
import os
import re
import requests
from dotenv import load_dotenv
from google import genai

from src.answer_validator import validate_answer

load_dotenv()

# ==========================================
# OLLAMA CONFIGURATION
# ==========================================

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:3b"
OLLAMA_TIMEOUT = 180

# ==========================================
# GEMINI CONFIGURATION
# ==========================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.8-flash"
).strip()

client = (
    genai.Client(api_key=GEMINI_API_KEY)
    if GEMINI_API_KEY
    else None
)


# ==========================================
# PREPARE PDF CONTEXT
# ==========================================

def prepare_context(context_chunks):

    passages = []

    for chunk in context_chunks:

        page = chunk.get("page_number", "Unknown")
        text = chunk.get("text", "")

        if isinstance(text, str) and text.strip():

            passages.append(
                f"[PDF Page {page}]\n{text.strip()}"
            )

    return "\n\n".join(passages)



# ==========================================
# BUILD ANSWER PROMPT
# ==========================================

def build_prompt(question, context):

    return f"""
You are StudyMate AI, a reliable PDF-based study tutor.

Answer the student's question using ONLY the supplied
PDF passages.

SOURCE RULES:

1. Do not invent facts, definitions, examples or formulas.
2. Do not use outside knowledge.
3. Preserve the meaning of the PDF.
4. Treat unclear OCR text as uncertain.
5. Do not silently repair OCR errors.
6. Cite important claims using supplied page numbers.
7. Never invent page numbers.
8. Avoid repeating information.
9. Do not confuse energy and power.
10. Do not derive formulas yourself.

OUTPUT FORMAT:

## Explanation
Explain the topic simply with page citations.

## Important Points
List 3 to 6 useful points without repetition.

## Formula
Write exactly:
Formula could not be reliably extracted from the retrieved PDF passages. Mathematical expressions require verification against the original PDF page.

Do not generate a formula.

## Information Not Found
Include this only if an important part of the question
is not sufficiently covered by the passages.

Do not add a conclusion or bibliography.

PDF PASSAGES:
{context}

STUDENT QUESTION:
{question}

Now prepare the answer.
"""


# ==========================================
# CLEAN DUPLICATE HEADINGS
# ==========================================

def clean_answer(answer):

    if not answer:
        return "No answer was generated."

    seen = set()
    lines = []

    heading_pattern = re.compile(
        r"^\s*#{0,3}\s*"
        r"(Explanation|Important Points|Formula"
        r"|Information Not Found)\s*:?\s*$",
        re.IGNORECASE
    )

    for line in answer.splitlines():

        match = heading_pattern.match(line)

        if match:

            heading = match.group(1).lower()

            if heading in seen:
                continue

            seen.add(heading)

            lines.append(
                f"## {match.group(1)}"
            )

        else:
            lines.append(line)

    return "\n".join(lines).strip()


# ==========================================
# FORMULA POLICY
# ==========================================

def apply_formula_policy(answer):

    message = (
        "Formula could not be reliably extracted from the "
        "retrieved PDF passages. Mathematical expressions "
        "require verification against the original PDF page."
    )

    pattern = re.compile(
        r"(?im)^#{0,3}\s*Formula(?:\s*\([^\\n]*\))?\s*$"
    )

    match = pattern.search(answer)

    if not match:

        return answer + "\n\n## Formula\n" + message

    next_section = re.search(
        r"(?m)^#{1,3}\s+",
        answer[match.end():]
    )

    replacement = "## Formula\n" + message

    if next_section:

        end = match.end() + next_section.start()

        answer = (
            answer[:match.start()]
            + replacement
            + "\n\n"
            + answer[end:]
        )

    else:

        answer = answer[:match.start()] + replacement

    return answer.strip()


# ==========================================
# OLLAMA AI ENGINE
# ==========================================

def ask_ollama(prompt):

    try:

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0,
                    "num_ctx": 8192,
                    "num_predict": 1200,
                    "repeat_penalty": 1.1
                }
            },
            timeout=OLLAMA_TIMEOUT
        )

        response.raise_for_status()

        answer = response.json().get(
            "response", ""
        ).strip()

        if not answer:
            raise ValueError(
                "Ollama returned an empty answer."
            )

        return answer

    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Cannot connect to Ollama. "
            "Make sure Ollama is running."
        )

    except requests.exceptions.Timeout:
        raise RuntimeError(
            "Ollama took too long to respond."
        )


# ==========================================
# GEMINI AI ENGINE
# ==========================================

def ask_gemini(prompt):

    if client is None:
        raise RuntimeError(
            "Gemini API key is missing from .env."
        )

    interaction = client.interactions.create(
        model=GEMINI_MODEL,
        input=prompt,
        generation_config={
            "max_output_tokens": 2048,
            "thinking_level": "low"
        }
    )

    answer = (
        interaction.output_text or ""
    ).strip()

    if not answer:
        raise RuntimeError(
            "Gemini returned an empty answer."
        )

    return answer


# ==========================================
# FINAL ANSWER PROCESSING
# ==========================================

def finalize_answer(
    question,
    answer,
    context,
    verification_function
):

    answer = clean_answer(answer)
    answer = apply_formula_policy(answer)

    answer = validate_answer(
        question=question,
        answer=answer,
        context=context,
        verification_function=verification_function
    )

    answer = clean_answer(answer)
    answer = apply_formula_policy(answer)

    return answer


# ==========================================
# MAIN ANSWER GENERATION FUNCTION
# ==========================================



def generate_answer(question, context_chunks, mode="Offline"):

    if not context_chunks:
        return (
            "This information was not found in the retrieved PDF content.",
            "None"
        )

    context = prepare_context(context_chunks)
    prompt = build_prompt(question, context)

    if mode == "Offline":
        ai_function = ask_ollama
        engine = "Ollama (Offline)"

    elif mode == "Online":
        ai_function = ask_gemini
        engine = "Gemini (Online)"

    elif mode == "Auto":
        try:
            answer = ask_gemini(prompt)
            verification_function = ask_gemini
            engine = "Gemini (Online)"
        except Exception:
            answer = ask_ollama(prompt)
            verification_function = ask_ollama
            engine = "Ollama (Offline fallback)"

        answer = finalize_answer(
            question,
            answer,
            context,
            verification_function
        )

        return answer, engine

    else:
        return "Please select a valid AI mode.", "None"

    answer = ai_function(prompt)

    answer = finalize_answer(
        question,
        answer,
        context,
        ai_function
    )

    return answer, engine
