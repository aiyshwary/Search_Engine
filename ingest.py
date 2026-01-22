import os
import json
from pathlib import Path
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from docx import Document
from lingua import Language, LanguageDetectorBuilder
import fasttext

# ------------------------------------------------------------
# ISO 639-1 → ISO 639-3 NORMALIZATION (CRITICAL FIX)
# ------------------------------------------------------------
LANG_1_TO_3 = {
    "en": "eng",
    "ta": "tam",
    "kn": "kan",
    "hi": "hin",
    "te": "tel",
    "ml": "mal",
    "bn": "ben",
    "mr": "mar",
    "gu": "guj",
    "pa": "pan",
    "or": "ori",
    "ur": "urd"
}

# ------------------------------------------------------------
# Load fastText LID176
# ------------------------------------------------------------
FASTTEXT_MODEL_PATH = "lid.176.ftz"

if not os.path.exists(FASTTEXT_MODEL_PATH):
    import urllib.request
    urllib.request.urlretrieve(
        "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz",
        FASTTEXT_MODEL_PATH,
    )

fasttext_model = fasttext.load_model(FASTTEXT_MODEL_PATH)

def detect_lang_fasttext(text):
    cleaned_text = text.replace("\n", " ").strip()
    if not cleaned_text:
        return "und", 0.0
    label, prob = fasttext_model.predict(cleaned_text, k=1)
    return label[0].replace("__label__", ""), prob[0]

# ------------------------------------------------------------
# Lingua fallback
# ------------------------------------------------------------
LINGUA_LANGS = [
    Language.ENGLISH,
    Language.HINDI,
    Language.TAMIL,
    Language.BENGALI,
    Language.TELUGU,
    Language.MARATHI,
    Language.GUJARATI,
]

lingua_detector = LanguageDetectorBuilder.from_languages(*LINGUA_LANGS).build()

def detect_lang_lingua(text):
    lang = lingua_detector.detect_language_of(text)
    return lang.iso_code_639_1.name.lower() if lang else None

# ------------------------------------------------------------
# Hybrid detector
# ------------------------------------------------------------
def detect_language(text):
    ft_lang, ft_conf = detect_lang_fasttext(text)
    if ft_conf > 0.60:
        return ft_lang
    if len(text.strip()) > 50:
        lg = detect_lang_lingua(text)
        if lg:
            return lg
    return ft_lang

# ------------------------------------------------------------
# PDF extractor with OCR confidence
# ------------------------------------------------------------
def extract_pdf(path):
    pages = []
    doc = fitz.open(path)

    OCR_LANG_CODES = "eng+hin+tam+ben+tel+mar+guj+mal+pan+ori+urd+kan"

    for page in doc:
        text = page.get_text("text")
        if text.strip():
            pages.append({"text": text, "ocr_confidence": 1.0})
        else:
            pix = page.get_pixmap(dpi=300)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text = pytesseract.image_to_string(img, lang=OCR_LANG_CODES)
            pages.append({"text": text, "ocr_confidence": 0.6})
    return pages

# ------------------------------------------------------------
# DOCX extractor
# ------------------------------------------------------------
def extract_docx(path):
    doc = Document(path)
    text = "\n".join(p.text for p in doc.paragraphs)
    return [{"text": text, "ocr_confidence": 1.0}]

# ------------------------------------------------------------
# Dispatcher
# ------------------------------------------------------------
def extract_text(path):
    ext = path.suffix.lower()
    if ext == ".pdf":
        return extract_pdf(str(path))
    if ext == ".docx":
        return extract_docx(str(path))
    return []

# ------------------------------------------------------------
# Main ingestion
# ------------------------------------------------------------
def ingest_folder(folder_path, output_file_path):
    doc_id = 1
    Path(output_file_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file_path, "w", encoding="utf-8") as out:
        for full_path in Path(folder_path).rglob("*"):
            if not full_path.is_file():
                continue
            if full_path.suffix.lower() not in [".pdf", ".docx"]:
                continue

            pages = extract_text(full_path)
            if not pages:
                continue

            combined = "\n".join(p["text"] for p in pages)
            snippet = combined[:4096]

            lang_6391 = detect_language(snippet)
            lang = LANG_1_TO_3.get(lang_6391, lang_6391)

            record = {
                "doc_id": doc_id,
                "filename": full_path.name,
                "path": str(full_path),
                "language": lang,
                "page_count": len(pages),
                "pages": [
                    {"page": i + 1, **p} for i, p in enumerate(pages)
                ],
            }

            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            doc_id += 1

# ------------------------------------------------------------
# Run
# ------------------------------------------------------------
if __name__ == "__main__":
    ingest_folder("dataset", "output/documents.jsonl")
