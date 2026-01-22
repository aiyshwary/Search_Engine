import os
import json
import warnings
from pathlib import Path
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from docx import Document
from lingua import Language, LanguageDetectorBuilder
import fasttext
import concurrent.futures

# ------------------------------------------------------------
# SILENCE WARNINGS
# ------------------------------------------------------------
# Silence the repetitive FastText model load warning that occurs during multiprocessing on Windows
warnings.filterwarnings("ignore", message="`load_model` does not return WordVectorModel")

# ------------------------------------------------------------
# GLOBAL CONFIG
# ------------------------------------------------------------
# NOTE FOR WINDOWS USERS: If Tesseract is not in your PATH, uncomment and adjust the line below:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

OCR_LANG_CODES = "eng+hin+tam+ben+tel+mar+guj+mal+pan+ori+urd+kan+san"
OCR_DPI = 300
FASTTEXT_CONF_THRESHOLD = 0.60
SNIPPET_LENGTH = 4096

# ISO 639-1 → ISO 639-3 (PIPELINE STANDARD)
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
    "ur": "urd",
    "sa": "san"
}

# ------------------------------------------------------------
# FASTTEXT LOAD
# ------------------------------------------------------------
FASTTEXT_MODEL_PATH = "lid.176.ftz"

# Wrap model loading in a try/except block for robustness during multiprocessing
try:
    if not os.path.exists(FASTTEXT_MODEL_PATH):
        # Only print download message in the main process (rough check)
        if __name__ == "__main__":
             print("Downloading FastText model...")
        import urllib.request
        urllib.request.urlretrieve(
            "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz",
            FASTTEXT_MODEL_PATH,
        )
        if __name__ == "__main__":
             print("Download complete.")

    fasttext_model = fasttext.load_model(FASTTEXT_MODEL_PATH)
except Exception as e:
    # In workers, this might fail silently if file is locked, but usually okay on read
    pass

def detect_lang_fasttext(text):
    # Ensure model is loaded in worker process if needed
    global fasttext_model
    if 'fasttext_model' not in globals():
         fasttext_model = fasttext.load_model(FASTTEXT_MODEL_PATH)

    cleaned = text.replace("\n", " ").strip()
    if not cleaned:
        return "und", 0.0
    try:
        label, prob = fasttext_model.predict(cleaned, k=1)
        return label[0].replace("__label__", ""), prob[0]
    except Exception:
        return "unknown", 0.0

# ------------------------------------------------------------
# LINGUA FALLBACK
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

# Initialize Lingua globally so it's shared/copied to workers
lingua_detector = LanguageDetectorBuilder.from_languages(*LINGUA_LANGS).build()

def detect_lang_lingua(text):
    try:
        lang = lingua_detector.detect_language_of(text)
        return lang.iso_code_639_1.name.lower() if lang else None
    except Exception:
        return None

# ------------------------------------------------------------
# HYBRID DETECTOR
# ------------------------------------------------------------
def detect_language(text):
    ft_lang, ft_conf = detect_lang_fasttext(text)

    if ft_conf > FASTTEXT_CONF_THRESHOLD:
        return ft_lang

    if len(text.strip()) > 30:
        lg = detect_lang_lingua(text)
        if lg:
            return lg

    return ft_lang

# ------------------------------------------------------------
# PDF EXTRACTOR (WITH OCR CONFIDENCE) - CORRECTED
# ------------------------------------------------------------
def extract_pdf(path):
    pages = []
    doc = None
    try:
        doc = fitz.open(path)
        found_digital_text = False

        # Iterate through pages to extract text
        for i, page in enumerate(doc):
            # 1. Try digital extraction first (this is the correct way)
            text = page.get_text("text")

            if text.strip():
                # Found digital text on this page
                found_digital_text = True
                pages.append({
                    "text": text,
                    "ocr_confidence": 1.0
                })
            else:
                # 2. No digital text on page, try OCR (fallback)
                try:
                    pix = page.get_pixmap(dpi=OCR_DPI)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    text = pytesseract.image_to_string(img, lang=OCR_LANG_CODES)
                    # Even OCR might return empty string if page is blank
                    conf = 0.6 if text.strip() else 0.0
                    pages.append({
                        "text": text,
                        "ocr_confidence": conf
                    })
                except pytesseract.TesseractNotFoundError:
                     # print(f"WARNING: Tesseract not found. Skipping OCR for image page {i+1} in {path.name}")
                     pages.append({"text": "", "ocr_confidence": 0.0})
                except Exception as e:
                     # print(f"OCR Error on page {i+1} of {path.name}: {e}")
                     pages.append({"text": "", "ocr_confidence": 0.0})

        # Determine final status based on what we found during iteration
        status = "PDF_TEXT" if found_digital_text else "PDF_OCR"

        # Handle edge case where PDF opens but has absolutely no content (e.g. 0 pages)
        if not pages:
             status = "EMPTY_PDF"

        return pages, status

    except Exception as e:
        print(f"Error opening PDF {path.name}: {e}")
        return [], f"ERROR: {e}"
    finally:
        if doc:
            doc.close()

# ------------------------------------------------------------
# DOCX EXTRACTOR
# ------------------------------------------------------------
def extract_docx(path):
    try:
        doc = Document(path)
        text = "\n".join(p.text for p in doc.paragraphs)
        if not text.strip():
             return [], "EMPTY_DOCX"
        return [{
            "text": text,
            "ocr_confidence": 1.0
        }], "DOCX"
    except Exception as e:
        print(f"Error opening DOCX {path.name}: {e}")
        return [], f"ERROR: {e}"

# ------------------------------------------------------------
# DISPATCH
# ------------------------------------------------------------
def extract_text(path):
    ext = path.suffix.lower()
    if ext == ".pdf":
        return extract_pdf(path)
    if ext == ".docx":
        return extract_docx(path)
    return [], "SKIP"

# ------------------------------------------------------------
# WORKER
# ------------------------------------------------------------
def process_file(full_path):
    try:
        # Re-initialize FastText in worker if necessary (Windows specific safety)
        global fasttext_model
        if 'fasttext_model' not in globals() or fasttext_model is None:
             try:
                fasttext_model = fasttext.load_model(FASTTEXT_MODEL_PATH)
             except:
                 pass # Fallback will handle it or fail gracefully later

        pages, status = extract_text(full_path)
        if not pages:
            return None

        combined = "\n".join(p["text"] for p in pages)
        snippet = combined[:SNIPPET_LENGTH]

        if not snippet.strip():
             # Document might be OCR'd but result in empty text
             return None

        lang_6391 = detect_language(snippet)
        lang = LANG_1_TO_3.get(lang_6391, lang_6391)

        return {
            "filename": full_path.name,
            "path": str(full_path),
            "language": lang,
            "page_count": len(pages),
            "extraction_method": status,
            "pages": [
                {"page": i + 1, **p} for i, p in enumerate(pages)
            ],
        }
    except Exception as e:
        print(f"Worker process failed for {full_path.name}: {e}")
        return None

# ------------------------------------------------------------
# MAIN INGEST
# ------------------------------------------------------------
def ingest_folder(folder_path, output_file_path):
    input_dir = Path(folder_path)
    output_path = Path(output_file_path)

    if not input_dir.exists():
        print(f"ERROR: Input directory does not exist: {input_dir.absolute()}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Scanning for PDF and DOCX files in: {input_dir.absolute()} ...")
    files = [
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in [".pdf", ".docx"]
    ]

    if not files:
        print("❌ ERROR: No .pdf or .docx files found!")
        return

    print(f"Found {len(files)} files found. Starting processing...")

    # Reduce workers slightly to avoid system choking
    max_workers = max(1, (os.cpu_count() or 2) - 2)
    print(f"Starting pool with {max_workers} workers.")

    records = []

    try:
        # We use a list here to ensure all futures complete before moving on
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as exe:
            # map handles iteration and result retrieval cleanly
            results = exe.map(process_file, files)
            for rec in results:
                if rec:
                    records.append(rec)
                    # print(".", end="", flush=True) # Uncomment for progress dots

    except KeyboardInterrupt:
        print("\nStopping processing due to keyboard interrupt...")
    except Exception as e:
        print(f"\nAn error occurred during multiprocessing: {e}")

    print(f"\nWriting results to {output_path.absolute()}...")
    with open(output_path, "w", encoding="utf-8") as out:
        for i, rec in enumerate(records, 1):
            rec["doc_id"] = i
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"✅ Finished. Ingested {len(records)} documents successfully.")

# ------------------------------------------------------------
# RUN
# ------------------------------------------------------------
if __name__ == "__main__":
    # Ensure this matches the exact folder name next to the script
    DATASET_FOLDER = "dataset"
    OUTPUT_FILE = "output/documents.jsonl"

    print("Initializing...")
    # Ensure model is loaded in main process first
    if not os.path.exists(FASTTEXT_MODEL_PATH):
         print("Downloading FastText model...")
         import urllib.request
         urllib.request.urlretrieve(
            "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz",
            FASTTEXT_MODEL_PATH,
         )
         print("Download complete.")
    fasttext_model = fasttext.load_model(FASTTEXT_MODEL_PATH)

    ingest_folder(DATASET_FOLDER, OUTPUT_FILE)