
import hashlib
import json
from io import BytesIO
from pathlib import Path

import pymupdf
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from src.chunker import chunk_text


# ==========================================
# TESSERACT CONFIGURATION
# ==========================================

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

OCR_LANGUAGE = "eng"
OCR_DPI = 180
OCR_CONFIG = "--oem 3 --psm 6"

# Change this whenever the PDF processing logic changes.
CACHE_VERSION = "v3"


# ==========================================
# IMAGE PREPROCESSING
# ==========================================

def preprocess_image(image):

    # Convert to grayscale
    image = ImageOps.grayscale(image)

    # Improve contrast
    image = ImageOps.autocontrast(image)

    # Slightly sharpen text
    image = image.filter(ImageFilter.SHARPEN)

    # Improve contrast further
    image = ImageEnhance.Contrast(image).enhance(1.5)

    return image


# ==========================================
# OCR EXTRACTION
# ==========================================

def extract_text_with_ocr(page):

    pix = page.get_pixmap(
        dpi=OCR_DPI,
        colorspace=pymupdf.csGRAY,
        alpha=False
    )

    image = Image.open(
        BytesIO(pix.tobytes("png"))
    )

    image = preprocess_image(image)

    text = pytesseract.image_to_string(
        image,
        lang=OCR_LANGUAGE,
        config=OCR_CONFIG
    )

    return text.strip()


# ==========================================
# PROCESS PDF
# ==========================================

def process_pdf(pdf_bytes, progress_callback=None):

    chunks = []

    pdf = pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    total_pages = len(pdf)

    try:

        for page_index, page in enumerate(pdf):

            page_number = page_index + 1

            # First try normal digital text extraction
            text = page.get_text("text").strip()

            # Use OCR when the page has no digital text
            if not text:

                text = extract_text_with_ocr(page)

            # Create chunks for non-empty pages
            if text:

                page_chunks = chunk_text(
                    text,
                    page_number
                )

                for chunk in page_chunks:

                    chunk["chunk_id"] = len(chunks) + 1

                    chunks.append(chunk)

            # Update progress
            if progress_callback:

                progress_callback(
                    page_number / total_pages,
                    page_number,
                    total_pages
                )

    finally:

        pdf.close()

    return chunks


# ==========================================
# PROCESS PDF WITH CACHE
# ==========================================

def process_pdf_with_cache(
    pdf_bytes,
    progress_callback=None
):

    # Generate unique PDF hash
    pdf_hash = hashlib.sha256(
        pdf_bytes
    ).hexdigest()

    # Find project folder
    project_folder = Path(__file__).resolve().parent.parent

    # Create cache folder
    cache_folder = (
        project_folder / "data" / "processed"
    )

    cache_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    # Include cache version to avoid loading old OCR results
    cache_file = cache_folder / (
        f"{pdf_hash}_{CACHE_VERSION}.json"
    )

    print("Cache file location:", cache_file.resolve())

    # Load cached chunks if available
    if cache_file.exists():

        print("Loading previously processed PDF...")

        with open(
            cache_file,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    # Process PDF
    print("Processing PDF with improved OCR...")

    chunks = process_pdf(
        pdf_bytes,
        progress_callback=progress_callback
    )

    # Save extracted chunks
    with open(
        cache_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=4
        )

    print("PDF processing completed and saved!")

    return chunks
