import os
import math
import sqlite3
import datetime as dt
import json
import re
import threading
from difflib import SequenceMatcher
import smtplib
import ssl
import secrets
import hashlib
import io
import csv
import uuid
import logging
import tempfile
import time
import tracemalloc
import gc
import asyncio
import shutil
import subprocess

try:
    import pdfplumber
    HAVE_PDFPLUMBER = True
except Exception:
    pdfplumber = None
    HAVE_PDFPLUMBER = False

# ✅ Prefer PyMuPDF (fitz) when available: faster + usually lower RAM than pdfplumber
try:
    import fitz  # PyMuPDF
    HAVE_PYMUPDF = True
except Exception:
    fitz = None
    HAVE_PYMUPDF = False


from fastapi import UploadFile, File, Form
from email.message import EmailMessage
from pathlib import Path
from typing import List, Dict, Any, Optional
import html


try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    HAVE_REPORTLAB = True
except Exception:
    HAVE_REPORTLAB = False

try:
    from weasyprint import HTML as WEASY_HTML
    HAVE_WEASYPRINT = True
except Exception:
    HAVE_WEASYPRINT = False

from fastapi import FastAPI, HTTPException, Request, Form, Query, Body, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent   # this is your /backend folder

# Where the HTML files actually live, same as old app.py
FRONTEND_DIR = BASE_DIR.parent / "frontend"

SNF_DEFAULT_PIN = (os.getenv("SNF_DEFAULT_PIN") or "").strip()


def require_admin(request: Request):
    """Simple header-based admin guard via ADMIN_TOKEN env var."""
    token = os.getenv("ADMIN_TOKEN")
    if not token:
        return
    header = request.headers.get("x-admin-token") or ""
    if header.strip() != token.strip():
        raise HTTPException(status_code=403, detail="invalid admin token")


def require_pad_api_key(request: Request):
    """
    Simple header-based PAD guard via PAD_API_KEY env var.
    - If PAD_API_KEY is not set, PAD endpoints are left open (same pattern as require_admin).
    - PAD should send header: x-pad-api-key: <PAD_API_KEY>
    """
    token = os.getenv("PAD_API_KEY")
    if not token:
        return
    header = request.headers.get("x-pad-api-key") or ""
    if header.strip() != token.strip():
        raise HTTPException(status_code=403, detail="invalid PAD api key")


# ============================
# Email helper utilities
# ============================
def normalize_email_list(raw: str) -> str:
    """
    Accepts commas, semicolons, or newlines.
    Returns a clean, comma-separated email list.
    """
    parts = re.split(r"[,\n;]+", (raw or "").strip())
    parts = [p.strip() for p in parts if p.strip()]
    return ", ".join(parts)

# --- TIMEZONE DISPLAY HELPERS (UTC -> US/Eastern) ---
from zoneinfo import ZoneInfo

_EASTERN_TZ = ZoneInfo("America/New_York")
_UTC_TZ = dt.timezone.utc

def utc_text_to_eastern_display(value: Any) -> str:
    """
    Convert common timestamp strings (UTC) into US/Eastern for DISPLAY ONLY.
    - Accepts: "YYYY-MM-DD HH:MM:SS", "YYYY-MM-DDTHH:MM:SS", "...Z", etc.
    - If timezone info is missing, we assume the stored value is UTC.
    - Returns: "YYYY-MM-DD HH:MM:SS" (Eastern)
    """
    if value is None:
        return ""
    s = str(value).strip()
    if not s:
        return ""

    # Date-only values should remain as-is
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return s

    try:
        raw = s

        # Handle trailing Z
        if raw.endswith("Z"):
            raw = raw[:-1]

        # Handle SQLite "YYYY-MM-DD HH:MM:SS"
        if " " in raw and "T" not in raw and len(raw) >= 19:
            d = dt.datetime.strptime(raw[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=_UTC_TZ)
        else:
            # ISO-ish
            d = dt.datetime.fromisoformat(raw)
            if d.tzinfo is None:
                d = d.replace(tzinfo=_UTC_TZ)

        return d.astimezone(_EASTERN_TZ).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        # If parsing fails, return original string unchanged
        return s


def normalize_date_to_iso(value: Any) -> Optional[str]:
    """
    Normalize common date inputs into ISO YYYY-MM-DD so SQLite date() works.
    Accepts:
      - YYYY-MM-DD
      - MM/DD/YYYY or MM-DD-YYYY
      - MM/DD/YY or MM-DD-YY   (assumes 20YY for 00-69, else 19YY)
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None

    # Already ISO date
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return s

    cleaned = s.replace("-", "/")

    m = re.fullmatch(r"(\d{2})/(\d{2})/(\d{4})", cleaned)
    if m:
        mm, dd, yyyy = m.group(1), m.group(2), m.group(3)
        return f"{yyyy}-{mm}-{dd}"

    m = re.fullmatch(r"(\d{2})/(\d{2})/(\d{2})", cleaned)
    if m:
        mm, dd, yy = m.group(1), m.group(2), int(m.group(3))
        yyyy = 2000 + yy if yy <= 69 else 1900 + yy
        return f"{yyyy:04d}-{mm}-{dd}"

    # If it’s some other format we don’t recognize, don’t destroy it
    return s

def log_pad_request_debug(request: Request, raw_text: str, payload: dict):
    """
    Debug helper to see exactly what PAD sent.
    Logs are intentionally short and safe for Render.
    """
    headers = dict(request.headers)

    print("====== PAD DEBUG START ======")
    print("Path:", request.url.path)
    print("Content-Type:", headers.get("content-type"))
    print("Content-Length:", headers.get("content-length"))
    print("Raw body (first 500 chars):")
    print(raw_text[:500])
    print("Parsed payload:", payload)
    print("====== PAD DEBUG END ======")


# ---------------------------------------------------------------------------
# DB PATH – wired for Render, but still overrideable via DB_PATH
# ---------------------------------------------------------------------------
RENDER_DEFAULT_DB = "/opt/render/project/data/evolv.db"
_env_db = os.getenv("DB_PATH")
if _env_db:
    DB_PATH = _env_db
elif os.path.exists(RENDER_DEFAULT_DB):
    DB_PATH = RENDER_DEFAULT_DB
else:
    # Local/dev fallback if you're running off the uploaded DB
    DB_PATH = str(BASE_DIR / "evolv(3).db")

# HTML files (mirror of your old app.py wiring)
ASK_HTML = FRONTEND_DIR / "TEST Ask.html"
ADMIN_HTML = FRONTEND_DIR / "TEST Admin.html"
FACILITY_HTML = FRONTEND_DIR / "TEST  Facility_Details.html"  # note double space
SNF_HTML = FRONTEND_DIR / "TEST SNF_Admissions.html"
HOSPITAL_DISCHARGE_HTML = FRONTEND_DIR / "Hospital_Discharge.html"
CENSUS_HTML = FRONTEND_DIR / "Census.html"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("evolv")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("evolv")

PERF_LOG_ENABLED = (os.getenv("PERF_LOG", "0") == "1")

def _rss_mb() -> float | None:
    """
    Returns current process RSS (resident memory) in MB.
    Linux/Render-friendly. If unavailable, returns None.
    """
    try:
        with open("/proc/self/statm", "r") as f:
            parts = f.read().strip().split()
        # statm: size resident shared text lib data dt
        resident_pages = int(parts[1])
        page_size = os.sysconf("SC_PAGE_SIZE")  # bytes
        return (resident_pages * page_size) / (1024 * 1024)
    except Exception:
        return None

def _perf_log(job_id: str, filename: str, step: str, t0: float | None = None) -> float:
    """
    Logs timing + memory snapshots. Returns current perf_counter timestamp.
    Set env PERF_LOG=1 to enable.
    """
    now = time.perf_counter()
    if not PERF_LOG_ENABLED:
        return now

    dt_s = (now - t0) if t0 is not None else None
    rss = _rss_mb()

    if tracemalloc.is_tracing():
        cur_b, peak_b = tracemalloc.get_traced_memory()
        cur_mb = cur_b / (1024 * 1024)
        peak_mb = peak_b / (1024 * 1024)
    else:
        cur_mb = None
        peak_mb = None

    logger.info(
        "PERF job_id=%s file=%s step=%s dt=%.3fs rss=%sMB tracemalloc_cur=%sMB tracemalloc_peak=%sMB",
        job_id,
        filename,
        step,
        dt_s if dt_s is not None else -1.0,
        f"{rss:.1f}" if rss is not None else "na",
        f"{cur_mb:.1f}" if cur_mb is not None else "na",
        f"{peak_mb:.1f}" if peak_mb is not None else "na",
    )
    return now


CENSUS_UPLOAD_TMP = Path(
    os.getenv("CENSUS_UPLOAD_TMP", os.path.join(tempfile.gettempdir(), "evolv_census_uploads"))
)

CENSUS_UPLOAD_TMP.mkdir(parents=True, exist_ok=True)



OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("openai_api_key")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY env var is required")

client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------------------------------------------------------------------
# Email / SMTP config for Facility Details intake notifications
# ---------------------------------------------------------------------------
INTAKE_EMAIL_FROM = os.getenv("INTAKE_EMAIL_FROM", "")
INTAKE_EMAIL_TO   = os.getenv("INTAKE_EMAIL_TO", "")
SMTP_HOST         = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT         = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER         = os.getenv("SMTP_USER", INTAKE_EMAIL_FROM or "")
SMTP_PASSWORD     = os.getenv("SMTP_PASSWORD", "")

# Optional: direct URL back to the Facility Details page (?f=&t=)
FACILITY_FORM_BASE_URL = os.getenv("FACILITY_FORM_BASE_URL", "")


# ---------------------------------------------------------------------------
# Email / SMTP config for Facility Details intake notifications
# ---------------------------------------------------------------------------
INTAKE_EMAIL_FROM = os.getenv("INTAKE_EMAIL_FROM", "")
INTAKE_EMAIL_TO   = os.getenv("INTAKE_EMAIL_TO", "")
SMTP_HOST         = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT         = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER         = os.getenv("SMTP_USER", INTAKE_EMAIL_FROM or "")
SMTP_PASSWORD     = os.getenv("SMTP_PASSWORD", "")

# Optional: direct URL back to the Facility Details page (?f=&t=)
FACILITY_FORM_BASE_URL = os.getenv("FACILITY_FORM_BASE_URL", "")

# Public base URL for links in outbound emails (set in Render as PUBLIC_APP_BASE_URL)
PUBLIC_APP_BASE_URL = (os.getenv("PUBLIC_APP_BASE_URL") or "").strip().rstrip("/")

# How long (in hours) a secure SNF link should remain valid (set in Render as SNF_LINK_TTL_HOURS)
try:
    SNF_LINK_TTL_HOURS = int(os.getenv("SNF_LINK_TTL_HOURS", "48"))
except Exception:
    SNF_LINK_TTL_HOURS = 48



# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Evolv Copilot (Fresh Backend)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # internal app; okay to be open
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    ms = int((time.time() - start) * 1000)

    # ✅ reduce log spam: do not log the high-frequency polling endpoint
    if request.url.path == "/admin/census/upload-status":
        return response

    ct = response.headers.get("content-type", "")
    cl = response.headers.get("content-length", "")

    logger.info(
        "[APP] %s %s status=%s ms=%s ct=%s cl=%s",
        request.method,
        request.url.path,
        response.status_code,
        ms,
        ct,
        cl,
    )
    return response

from fastapi.responses import JSONResponse

# ✅ Force JSON for unexpected server errors (prevents frontend JSON.parse crashes)
@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error path=%s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"ok": False, "detail": "Internal Server Error"},
    )


def _safe(s: str) -> str:
    return (s or "").strip()

def _split_name(full: str) -> tuple[str, str]:
    full = _safe(full)
    if not full:
        return ("", "")

    # Drop common noise like preferred names in parentheses: "SMITH, JOHN (JACK)"
    full = re.sub(r"\([^)]*\)", " ", full)

    # ✅ Remove "Name Alert" noise that sometimes leaks into names
    full = re.sub(r"\bNAME\s+ALERT\b", " ", full, flags=re.IGNORECASE)

    full = re.sub(r"\s+", " ", full).strip()

    # Helper: keep unicode letters + space + apostrophe + hyphen
    def _clean_name(s: str) -> str:
        s = _safe(s)
        out = []
        for ch in s:
            if ch.isalpha() or ch in {" ", "-", "'"}:
                out.append(ch)
        s2 = "".join(out)
        s2 = re.sub(r"\s{2,}", " ", s2).strip(" ,")
        return s2

    def _is_middle_initial(tok: str) -> bool:
        tok = _safe(tok).strip()
        return bool(re.fullmatch(r"[A-Za-z]\.?", tok))

    suffixes = {"JR", "SR", "II", "III", "IV", "V"}

    # Case 1: "LAST NAME, FIRST M"  (also handles "LAST M, FIRST" where M should move to first name)
    if "," in full:
        last_raw, rest = [x.strip() for x in full.split(",", 1)]

        # Remove trailing suffix from the last name side if present
        last_tokens = [t for t in re.split(r"\s+", last_raw) if t]

        mi_from_last = ""
        # ✅ If last side ends with a middle initial, move it to the first name side
        if len(last_tokens) >= 2 and _is_middle_initial(last_tokens[-1]):
            mi_from_last = last_tokens[-1].strip(".")
            last_tokens = last_tokens[:-1]

        if last_tokens and last_tokens[-1].upper().strip(".") in suffixes:
            last_tokens = last_tokens[:-1]

        last = _clean_name(" ".join(last_tokens))

        rest_parts = [p for p in rest.split() if p]
        if not rest_parts:
            return ("", last)

        first_parts = [rest_parts[0]]

        # If rest already has middle initial (e.g. "JOHN C"), keep it
        if len(rest_parts) >= 2 and _is_middle_initial(rest_parts[1]):
            first_parts.append(rest_parts[1].strip("."))

        # If we extracted a middle initial from the last side, append it after first name
        if mi_from_last and (mi_from_last not in [p.strip(".") for p in first_parts]):
            first_parts.append(mi_from_last)

        first = _clean_name(" ".join(first_parts))
        return (first, last)

    # Case 2: no comma — assume "First ... Last", but preserve particles like "de la", "van", "st"
    parts = [p for p in full.split() if p]
    if len(parts) == 1:
        return ("", _clean_name(parts[0]))

    particles = {
        "DE", "DEL", "DELA", "DA", "DI", "DOS", "DAS",
        "VAN", "VON", "LA", "LE", "DU", "ST", "ST."
    }

    # ✅ If second token is a middle initial, keep it with the first name (e.g., "JOHN C SMITH")
    first_end = 1
    first_parts = [parts[0]]
    if len(parts) >= 3 and _is_middle_initial(parts[1]):
        first_parts.append(parts[1].strip("."))
        first_end = 2

    first = _clean_name(" ".join(first_parts))

    # Build last name from the end, pulling in preceding particle tokens
    last_parts = [parts[-1]]
    i = len(parts) - 2
    while i >= first_end and parts[i].upper() in particles:
        last_parts.insert(0, parts[i])
        i -= 1

    last = _clean_name(" ".join(last_parts))
    return (first, last)


class ScannedPdfError(Exception):
    """Raised when a PDF has no extractable text (likely scanned/image-only)."""
    pass

# ✅ NEW: parse directly from a PDF file path (avoids loading PDF bytes into RAM)
class PdfTooLargeError(Exception):
    """Raised when extracted PDF text is too large (prevents OOM)."""
    pass

def iter_pcc_admission_records_from_pdf_path(pdf_path: str, job_id: str | None = None, filename: str | None = None):
    """
    Streaming version of PCC parser:
      - extracts text page-by-page
      - buffers only enough text to complete an "ADMISSION RECORD" chunk
      - yields rows as soon as chunks are ready (no giant list)

    If job_id/filename are provided, we emit page-level PERF logs so you can see
    exactly which page causes CPU/RAM spikes.
    """
    MAX_TEXT_CHARS = 8_000_000
    DELIM = "ADMISSION RECORD"

    saw_any_text = False
    total_chars = 0

    carry = ""
    carry_parts: list[str] = []
    carry_parts_chars = 0
    saw_delim = False

    def _perf(step: str, t0: float | None = None):
        if PERF_LOG_ENABLED and job_id and filename:
            _perf_log(job_id, filename, step, t0)

    def _flush_parts_into_carry() -> None:
        nonlocal carry, carry_parts, carry_parts_chars
        if carry_parts:
            carry = carry + "".join(carry_parts)
            carry_parts = []
            carry_parts_chars = 0

    def _yield_completed_chunks_from_carry():
        """
        Yields parsed rows for completed DELIM-delimited chunks from `carry`,
        leaving only trailing partial chunk in `carry`.
        """
        nonlocal carry, saw_delim

        first = carry.find(DELIM)
        if first == -1:
            return

        saw_delim = True
        if first > 0:
            carry = carry[first:]

        while True:
            next_pos = carry.find(DELIM, len(DELIM))
            if next_pos == -1:
                break

            chunk_body = carry[len(DELIM):next_pos]
            if chunk_body.strip():
                for rec in parse_pcc_admission_records_from_pdf_text(f"{DELIM}\n{chunk_body}"):
                    yield rec

            carry = carry[next_pos:]

        # keep carry bounded to avoid pathological memory growth
        if len(carry) > 1_500_000:
            carry = carry[-1_500_000:]

    # -----------------------
    # 1) Prefer PyMuPDF (fitz)
    # -----------------------
    if HAVE_PYMUPDF:
        t0_open = time.perf_counter()
        doc = fitz.open(pdf_path)
        _perf("pymupdf_open", t0_open)

        try:
            for i in range(doc.page_count):
                t0_page = time.perf_counter()
                page = doc.load_page(i)
                page_text = page.get_text("text") or ""
                # explicitly drop page refs ASAP
                del page

                _perf(f"pymupdf_page_{i+1}_extract", t0_page)

                if page_text.strip():
                    saw_any_text = True

                total_chars += len(page_text) + 1
                if total_chars > MAX_TEXT_CHARS:
                    raise PdfTooLargeError(
                        f"PDF too large after text extraction (>{MAX_TEXT_CHARS:,} chars). "
                        "Re-export as text PDF or split into smaller files."
                    )

                carry_parts.append(page_text)
                carry_parts.append("\n")
                carry_parts_chars += len(page_text) + 1

                # flush when delimiter appears OR buffer grows
                if (DELIM in page_text) or (carry_parts_chars >= 512_000):
                    t0_flush = time.perf_counter()
                    _flush_parts_into_carry()
                    _perf("flush_to_carry", t0_flush)

                    t0_parse = time.perf_counter()
                    yield from _yield_completed_chunks_from_carry()
                    _perf("yield_completed_chunks", t0_parse)

        finally:
            try:
                doc.close()
            except Exception:
                pass

    # -----------------------
    # 2) Fallback: pdfplumber
    # -----------------------
    else:
        if not HAVE_PDFPLUMBER:
            raise HTTPException(
                status_code=500,
                detail="PDF parsing support is not installed (pdfplumber). Please add 'pdfplumber' to your environment."
            )

        with pdfplumber.open(pdf_path) as pdf:
            for idx, page in enumerate(pdf.pages, start=1):
                t0_page = time.perf_counter()
                page_text = (page.extract_text() or "")
                _perf(f"pdfplumber_page_{idx}_extract", t0_page)

                if page_text.strip():
                    saw_any_text = True

                total_chars += len(page_text) + 1
                if total_chars > MAX_TEXT_CHARS:
                    raise PdfTooLargeError(
                        f"PDF too large after text extraction (>{MAX_TEXT_CHARS:,} chars). "
                        "Re-export as text PDF or split into smaller files."
                    )

                carry_parts.append(page_text)
                carry_parts.append("\n")
                carry_parts_chars += len(page_text) + 1

                if (DELIM in page_text) or (carry_parts_chars >= 512_000):
                    t0_flush = time.perf_counter()
                    _flush_parts_into_carry()
                    _perf("flush_to_carry", t0_flush)

                    t0_parse = time.perf_counter()
                    yield from _yield_completed_chunks_from_carry()
                    _perf("yield_completed_chunks", t0_parse)

    _flush_parts_into_carry()

    if not saw_any_text:
        raise ScannedPdfError("Scanned/image PDF detected (no extractable text)")

    if not saw_delim:
        text = carry
        if not text.strip():
            raise ScannedPdfError("Scanned/image PDF detected (no extractable text)")
        for rec in parse_pcc_admission_records_from_pdf_text(text):
            yield rec
        return

    yield from _yield_completed_chunks_from_carry()

    if carry.strip():
        for rec in parse_pcc_admission_records_from_pdf_text(carry):
            yield rec



def parse_pcc_admission_records_from_pdf_path(pdf_path: str) -> list[dict]:
    """
    FAST PATH:
      - Prefer `pdftotext` (poppler) streamed extraction: much lower RAM and usually far less CPU than pdfplumber.
      - Fallback to existing pdfplumber logic if pdftotext is not available.
    """
    # Safety cap to prevent Render OOM (tune as needed)
    MAX_TEXT_CHARS = 8_000_000  # ~8MB of text
    DELIM = "ADMISSION RECORD"

    saw_any_text = False
    total_chars = 0
    out: list[dict] = []

    # We avoid repeated giant string reallocations.
    carry = ""
    carry_parts: list[str] = []
    carry_parts_chars = 0
    saw_delim = False

    def _flush_parts_into_carry() -> None:
        nonlocal carry, carry_parts, carry_parts_chars
        if carry_parts:
            carry = carry + "".join(carry_parts)
            carry_parts = []
            carry_parts_chars = 0

    def _parse_completed_chunks_from_carry() -> None:
        """
        Consumes complete DELIM-delimited chunks from `carry` into `out`,
        leaving only the trailing partial chunk in `carry`.
        """
        nonlocal carry, saw_delim, out

        first = carry.find(DELIM)
        if first == -1:
            return

        saw_delim = True
        if first > 0:
            carry = carry[first:]

        while True:
            next_pos = carry.find(DELIM, len(DELIM))
            if next_pos == -1:
                break

            chunk_body = carry[len(DELIM):next_pos]
            if chunk_body.strip():
                out.extend(parse_pcc_admission_records_from_pdf_text(f"{DELIM}\n{chunk_body}"))

            carry = carry[next_pos:]

        # keep carry bounded
        if len(carry) > 1_500_000:
            carry = carry[-1_500_000:]

    # -----------------------
    # 1) FAST PATH: pdftotext
    # -----------------------
    pdftotext_bin = shutil.which("pdftotext")
    if pdftotext_bin:
        # -layout preserves column alignment better for these PCC reports
        # "-" outputs text to stdout so we can stream it without storing a giant blob
        cmd = [pdftotext_bin, "-layout", pdf_path, "-"]

        try:
            with subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,   # we WILL read it
                text=True,
                errors="replace",
                bufsize=1,   # line-buffered stdout
            ) as proc:
                assert proc.stdout is not None
                assert proc.stderr is not None

                for line in proc.stdout:
                    if line.strip():
                        saw_any_text = True

                    total_chars += len(line)
                    if total_chars > MAX_TEXT_CHARS:
                        try:
                            proc.kill()
                        except Exception:
                            pass
                        raise PdfTooLargeError(
                            f"PDF too large after text extraction (>{MAX_TEXT_CHARS:,} chars). "
                            "Re-export as text PDF or split into smaller files."
                        )

                    carry_parts.append(line)
                    carry_parts_chars += len(line)

                    # flush when delimiter appears OR buffer grows
                    if (DELIM in line) or (carry_parts_chars >= 512_000):
                        _flush_parts_into_carry()
                        _parse_completed_chunks_from_carry()

                # Read stderr and verify the command succeeded (prevents silent failures / hangs)
                stderr_text = proc.stderr.read()
                rc = proc.wait(timeout=30)
                if rc != 0:
                    raise RuntimeError(f"pdftotext failed (rc={rc}): {stderr_text[-2000:]}")

            _flush_parts_into_carry()

            if not saw_any_text:
                raise ScannedPdfError("Scanned/image PDF detected (no extractable text)")

            if not saw_delim:
                text = carry
                if not text.strip():
                    raise ScannedPdfError("Scanned/image PDF detected (no extractable text)")
                return parse_pcc_admission_records_from_pdf_text(text)

            _parse_completed_chunks_from_carry()

            if carry.strip():
                out.extend(parse_pcc_admission_records_from_pdf_text(carry))

            return out

        except PdfTooLargeError:
            raise
        except ScannedPdfError:
            raise
        except Exception:
            # fall through to pdfplumber fallback if pdftotext errors for any reason
            pass

    # -----------------------
    # 2) FALLBACK: pdfplumber
    # -----------------------
    if not HAVE_PDFPLUMBER:
        raise HTTPException(
            status_code=500,
            detail="PDF parsing support is not installed (pdfplumber). Please add 'pdfplumber' to your environment."
        )

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = (page.extract_text() or "")
            if page_text.strip():
                saw_any_text = True

            total_chars += len(page_text) + 1
            if total_chars > MAX_TEXT_CHARS:
                raise PdfTooLargeError(
                    f"PDF too large after text extraction (>{MAX_TEXT_CHARS:,} chars). "
                    "Re-export as text PDF or split into smaller files."
                )

            carry_parts.append(page_text)
            carry_parts.append("\n")
            carry_parts_chars += len(page_text) + 1

            if (DELIM in page_text) or (carry_parts_chars >= 512_000):
                _flush_parts_into_carry()
                _parse_completed_chunks_from_carry()

    _flush_parts_into_carry()

    if not saw_any_text:
        raise ScannedPdfError("Scanned/image PDF detected (no extractable text)")

    if not saw_delim:
        text = carry
        if not text.strip():
            raise ScannedPdfError("Scanned/image PDF detected (no extractable text)")
        return parse_pcc_admission_records_from_pdf_text(text)

    _parse_completed_chunks_from_carry()

    if carry.strip():
        out.extend(parse_pcc_admission_records_from_pdf_text(carry))

    return out


# ✅ Shared parser core after text extraction (used by BOTH path + bytes)
def _extract_pcc_resident_info(chunk: str) -> tuple[str, str, str, str, list[str], float]:
    warnings: list[str] = []
    confidence = 0.0

    # Locate the RESIDENT/PATIENT INFORMATION header row (allow indentation + case)
    m_hdr = re.search(
        r"^\s*(RESIDENT|PATIENT) INFORMATION\s*$",
        chunk,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    if not m_hdr:
        warnings.append("missing_resident_information_header")
        return ("", "", "", "", warnings, confidence)

    after = chunk[m_hdr.end():]

    # Labels vary a LOT by facility:
    # - single line: "Resident Name ... Resident #"
    # - wrapped: "Resident Name ... Orig.Adm.Date" then next line "Resident #"
    # - sometimes "Patient" instead of "Resident"
    #
    # So we anchor on the FIRST occurrence of "Resident #" / "Patient #" anywhere after the header,
    # and start parsing values immediately after that marker.
    m_hash = re.search(r"(Resident|Patient)\s*#", after, flags=re.IGNORECASE)
    if not m_hash:
        warnings.append("missing_resident_information_labels")
        return ("", "", "", "", warnings, confidence)

    tail = after[m_hash.end():]

    cand_lines = [l.strip() for l in tail.splitlines() if l.strip()]

    # Real date pattern (MM/DD/YYYY) used throughout this function
    date_pat = r"\b(0[1-9]|1[0-2])/(0[1-9]|[12][0-9]|3[01])/(19|20)\d{2}\b"

    # ✅ NEW: Arcadia split-text fix:
    # If the table row is split across lines (e.g. "Cart 4" on its own line),
    # the resident name is consistently the non-empty line immediately ABOVE the first real date.
    resident_line = ""
    first_date_idx = -1
    for i, l in enumerate(cand_lines[:40]):
        if re.search(date_pat, l):
            first_date_idx = i
            break

    if first_date_idx > 0:
        name_line = cand_lines[first_date_idx - 1].strip()
        date_line = cand_lines[first_date_idx].strip()

        # Require a comma to confirm it's a "LAST, FIRST" name line
        if "," in name_line:
            # Build a synthetic resident_line so the rest of the parser can keep working
            resident_line = f"{name_line} {date_line}"
            warnings.append("resident_name_from_line_above_first_date")

    # Existing behavior (covers PDFs where the whole row stays on one line)
    if not resident_line:
        for l in cand_lines[:10]:
            if "," in l and re.search(date_pat, l):
                resident_line = l
                break

    # Keep your compacting fallback, but only if it looks like a real one-line record
    # (this is what was accidentally pulling in "Cart ..." into the last name sometimes)
    if not resident_line:
        compact = " ".join(cand_lines[:15])
        if "," in compact and re.search(date_pat, compact):
            resident_line = compact.strip()
            warnings.append("resident_line_compacted_from_multiline")

    # Fallback: any "Last, First" line after labels (but avoid matching City, ST like "Arcadia, CA")
    if not resident_line:
        m_name = re.search(r"([A-Za-z'\-\. ]+,\s*[A-Za-z'\-\. ]+)", tail, flags=re.MULTILINE)
        if m_name:
            candidate = m_name.group(1).strip()
            post = candidate.split(",", 1)[-1].strip().split()[0] if "," in candidate else ""
            if not re.fullmatch(r"[A-Z]{2}", post):
                resident_line = candidate
                warnings.append("resident_line_fallback_used")

    if not resident_line:
        warnings.append("resident_line_not_found")
        return ("", "", "", "", warnings, confidence)

    # Resident / Patient # (can be numeric like "9080" OR alphanumeric like "AS201641")
    resident_num = ""
    last_tok = (resident_line.split() or [""])[-1].strip()

    # Guard: avoid accidentally treating a date as the ID
    if last_tok and not re.fullmatch(r"\d{2}/\d{2}/\d{4}", last_tok):
        # Common PCC IDs are 4+ chars/digits
        if re.fullmatch(r"[A-Za-z0-9]{4,}", last_tok):
            resident_num = last_tok

    # ✅ NEW: Arcadia split-text fallback
    # When text extraction splits columns into separate lines, the resident # can appear later.
    if not resident_num and cand_lines:
        for l in cand_lines[:60]:
            # Skip obvious unit lines like "Cart 4"
            if re.fullmatch(r"Cart\s*\d+\b", l, flags=re.IGNORECASE):
                continue

            # Look for a standalone 4+ alnum token on a line
            toks = re.findall(r"\b[A-Za-z0-9]{4,}\b", l)
            for t in toks:
                # Avoid capturing pure dates like 12252025 in weird cases (rare, but safe)
                if re.fullmatch(r"\d{8}", t):
                    continue
                resident_num = t
                break

            if resident_num:
                warnings.append("resident_number_scanned_from_lines")
                break

    if resident_num:
        confidence += 0.25
    else:
        warnings.append("missing_resident_number")

    # Admission date (PAD-style):
    # Look in the Resident/Patient Info value block immediately after the labels line.
    # PAD does: split after "Resident #"/"Patient #", then takes FIRST MM/DD/YYYY found.
    admission_date = ""
    date_pat = r"\b(0[1-9]|1[0-2])/(0[1-9]|[12][0-9]|3[01])/(19|20)\d{2}\b"

    # Build a small "value window" from the lines right after the labels.
    # This catches PDFs where the name and dates are on separate lines (like ASPIRE).
    value_window = "\n".join(cand_lines[:30]) if cand_lines else ""

    m_adm = re.search(date_pat, value_window)
    if m_adm:
        admission_date = m_adm.group(0).strip()
    else:
        # Fallback to old behavior if needed (some PDFs keep everything on one line)
        m_adm2 = re.search(date_pat, resident_line)
        admission_date = m_adm2.group(0).strip() if m_adm2 else ""

    if admission_date:
        confidence += 0.25
    else:
        warnings.append("missing_admission_date")


    # Resident/Patient name (PAD-style)
    # IMPORTANT: Arcadia has multi-word last names like "DAMIAN VARGAS, NESTOR"
    # so we must NOT collapse last name to only the final token.
    resident_name = ""

    # Everything before comma = last name (possibly multi-word)
    m_last = re.search(r"^[^,]+", resident_line)
    last_raw = (m_last.group(0) if m_last else "").strip()

    # ✅ Strip leading location prefixes that sometimes leak into the "last name" area
    # Examples: "Unit 2 SMITH" -> "SMITH", "Wing 1 DAMIAN VARGAS" -> "DAMIAN VARGAS"
    last_raw = re.sub(r"^(?:Wing|Unit)\s*\d+\s+", "", last_raw, flags=re.IGNORECASE)

    # ✅ Also strip leading room tokens if they appear before the name
    # Example: "319-B SMITH" -> "SMITH"
    last_raw = re.sub(r"^\d{1,4}[A-Z]?(?:-[A-Za-z0-9]{1,6})\s+", "", last_raw, flags=re.IGNORECASE)

    # Remove trailing suffix token if present (keep the full last name otherwise)
    suffixes = {"Jr", "Sr", "II", "III"}
    last_tokens = [t for t in re.split(r"\s+", last_raw) if t]

    # Remove trailing suffix token if present (keep the full last name otherwise)
    suffixes = {"Jr", "Sr", "II", "III"}
    last_tokens = [t for t in re.split(r"\s+", last_raw) if t]
    if last_tokens and last_tokens[-1] in suffixes:
        last_tokens = last_tokens[:-1]
    last_keep = " ".join(last_tokens).strip()

    # Clean last name but KEEP unicode letters + spaces/apostrophes/hyphens
    def _clean_name(s: str) -> str:
        s = (s or "").strip()
        out = []
        for ch in s:
            if ch.isalpha() or ch in {" ", "-", "'"}:
                out.append(ch)
        s2 = "".join(out)
        s2 = re.sub(r"\s{2,}", " ", s2).strip(" ,")
        return s2

    last_clean = _clean_name(last_keep)

    # First name: FIRST token after comma (unicode-safe, stops at whitespace)
    m_first = re.search(r",\s*(?P<first>[^\W\d_][^\s,]*)", resident_line)
    first_clean = _clean_name(m_first.group("first")) if m_first else ""

    if last_clean and first_clean:
        resident_name = f"{last_clean}, {first_clean}"

    if resident_name:
        confidence += 0.25
    else:
        warnings.append("missing_resident_name")

    # Room/bed: Ventura sometimes extracts the room AFTER the admission date
    # (e.g. "... 12/02/2022119-A ..."), so we must not rely on "token before date".
    room_bed = ""

    # Strict PCC room patterns like "319-B", "202-A", "147-1"
    # NOTE: no \b boundaries so it still matches if glued to digits (e.g. "...2022119-A...")
    room_pat = re.compile(r"\d{1,4}[A-Z]?(?:-[A-Za-z0-9]{1,6})")

    room_scan = resident_line or value_window or ""

    # Normalize a common Ventura glue case: "Wing 1Corea" -> "Wing 1 Corea"
    room_scan = re.sub(
        r"((?:Wing|Unit)\s*\d)(?=[A-Za-z])",
        r"\1 ",
        room_scan,
        flags=re.IGNORECASE,
    )

    matches = list(room_pat.finditer(room_scan))
    if matches:
        # Prefer first room token AFTER "Wing/Unit <n>" if possible
        m_unit = re.search(r"(?:Wing|Unit)\s*\d", room_scan, flags=re.IGNORECASE)
        if m_unit:
            after_unit = [m for m in matches if m.start() > m_unit.end()]
            room_bed = (after_unit[0] if after_unit else matches[0]).group(0).strip()
        else:
            room_bed = matches[0].group(0).strip()

    # Fallback: keep old behavior if we still didn't find it
    if (not room_bed) and admission_date:
        room_source = resident_line if admission_date in resident_line else value_window
        pre = room_source.split(admission_date, 1)[0]

        matches = list(room_pat.finditer(pre))
        if matches:
            room_bed = matches[-1].group(0).strip()
        else:
            toks = pre.rstrip().split()
            if toks:
                room_bed = toks[-1].strip()
                if room_bed in {"-", "—"} and len(toks) >= 2:
                    room_bed = toks[-2].strip()

    if room_bed:
        confidence += 0.25
    else:
        warnings.append("missing_room_bed")

    return (resident_name, room_bed, admission_date, resident_num, warnings, min(confidence, 1.0))


def _extract_pcc_address_phone(chunk: str) -> tuple[str, str, str, str, str, list[str]]:
    """
    PCC PDFs: Address is most reliably found under the "Previous address" column
    on the row beneath:
        "Previous address  Previous Phone #  Legal Mailing address"

    We parse that row first (Previous Address + Phone),
    then fall back to the "Legal Mailing address ... Sex" window only if needed.

    Returns: (address, city, state, zip_code, home_phone, warnings)
    """
    warnings: list[str] = []
    address = city = state = zip_code = home_phone = ""

    def _fmt_phone(s: str) -> str:
        s = re.sub(r"\D+", "", (s or ""))
        if len(s) == 10:
            return f"({s[0:3]}) {s[3:6]}-{s[6:10]}"
        return (s or "").strip()

    def _first_phone(text: str) -> str:
        m = re.search(r"(\(?\d{3}\)?[ -]?\d{3}[ -]?\d{4}|\b\d{10}\b)", text or "")
        return _fmt_phone(m.group(1)) if m else ""

    def _collapse_ws(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "")).strip()

    def _strip_same_as(s: str) -> str:
        s = _collapse_ws(s)

        # Remove common PCC "marker" prefixes that sometimes get glued into the address
        # Examples: "# Same as 8069 KEY WEST LANE..." or "# United States ..."
        s = re.sub(r"#\s*United\s+States\b", " ", s, flags=re.IGNORECASE)
        s = re.sub(r"#\s*Same\s+as\b", " ", s, flags=re.IGNORECASE)

        # Existing Ventura/PCC marker cleanup
        s = re.sub(r"\bSame as Previous Address\b", " ", s, flags=re.IGNORECASE)
        s = re.sub(r"^\s*Same as\b", " ", s, flags=re.IGNORECASE)

        return _collapse_ws(s).strip(" ,")

    def _parse_addr_text(addr_text: str):
        nonlocal address, city, state, zip_code
        addr_text = _strip_same_as(addr_text)

        # If it was only "Same as ..." (no real address), treat as blank
        if not addr_text:
            warnings.append("address_marked_same_as_previous")
            return

        m_addr = re.search(
            r"^(?P<street>.+?),\s*(?P<city>[A-Za-z .'\-]+),\s*(?P<state>[A-Z]{2})\s*,?\s*(?P<zip>(?:\d{5}(?:-\d{4})?|\d{9}))\b",
            addr_text,
        )
        if m_addr:
            address = m_addr.group("street").strip()
            city = m_addr.group("city").strip()
            state = m_addr.group("state").strip()

            zip_raw = (m_addr.group("zip") or "").strip()
            if re.fullmatch(r"\d{9}", zip_raw):
                zip_code = f"{zip_raw[:5]}-{zip_raw[5:]}"
            else:
                zip_code = zip_raw
        else:
            warnings.append("address_city_state_zip_parse_failed")
            address = addr_text

    # ------------------------------------------------------------
    # 1) PRIMARY: row under "Previous address  Previous Phone #  Legal Mailing address"
    # ------------------------------------------------------------
    m_hdr = re.search(
        r"^Previous address\s+Previous Phone\s*(?:#|No\.?)?\s+Legal Mailing address\s*$",
        chunk,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    if m_hdr:
        tail = chunk[m_hdr.end():]
        lines = [l.strip() for l in tail.splitlines() if l.strip()]

        # Prefer a line with a phone (if present), otherwise accept an address-only line.
        line_with_phone = ""
        line_with_address_only = ""

        # A "real address" heuristic: has ", <City>, <ST> <ZIP>"
        addr_like = re.compile(r",\s*[A-Za-z .'\-]+,\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?\b")

        for l in lines[:10]:
            if re.search(r"(\(?\d{3}\)?[ -]?\d{3}[ -]?\d{4}|\b\d{10}\b)", l):
                line_with_phone = l
                break
            if (not line_with_address_only) and addr_like.search(l):
                line_with_address_only = l

        picked = line_with_phone or line_with_address_only

        if picked:
            # Strip emails (Sinai sometimes merges these into the row)
            picked = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", " ", picked)

            # Strip header labels that sometimes leak into the captured row
            picked = re.sub(
                r"\bPrevious\s+address\b"
                r"|\bPrevious\s+Phone\s*(?:#|No\.?)?\b"
                r"|\bLegal\s+Mailing\s+address\b",
                " ",
                picked,
                flags=re.IGNORECASE,
            )
            picked = _collapse_ws(picked)

            # Ventura PDFs sometimes tack "Same as Previous Address" onto the same line (or split across whitespace)
            # If it appears anywhere, drop it and anything after it.
            picked = re.split(r"\bSame\s+as\s+Previous\s+Address\b", picked, maxsplit=1, flags=re.IGNORECASE)[0].strip()

            # If we have a phone, split around it. Otherwise treat as address text.
            m_ph = re.search(r"(\(?\d{3}\)?[ -]?\d{3}[ -]?\d{4}|\b\d{10}\b)", picked)
            if m_ph:
                home_phone = _fmt_phone(m_ph.group(1))

                left_raw = picked[:m_ph.start()].strip(" ,")
                right_raw = picked[m_ph.end():].strip()

                left = _strip_same_as(left_raw)
                right = _strip_same_as(right_raw)

                # Choose the side that actually looks like an address (prevents "Same as ..." winning)
                chosen = ""
                if addr_like.search(left):
                    chosen = left
                elif addr_like.search(right):
                    chosen = right
                else:
                    chosen = left or right

                if chosen:
                    if (chosen == right) and (not left):
                        warnings.append("previous_address_blank_used_legal_mailing_fallback")
                    _parse_addr_text(chosen)

            else:
                # Address-only line (Ventura): parse the whole thing as address
                picked = _strip_same_as(picked)
                _parse_addr_text(picked)


    # ------------------------------------------------------------
    # 2) FALLBACK: your existing "Legal Mailing address ... Sex" window
    # ------------------------------------------------------------
    if not address:
        m_win = re.search(
            r"Legal Mailing address(?P<body>.*?)(?:\n\s*Sex\b|\s+Sex\b)",
            chunk,
            flags=re.IGNORECASE | re.DOTALL,
        )
        body = (m_win.group("body") if m_win else "").strip()
        if body:
            # Strip header labels that sometimes leak into this window
            body = re.sub(
                r"\bPrevious address\b|\bPrevious Phone\s*(?:#|No\.?)?\b|\bLegal Mailing address\b",
                " ",
                body,
                flags=re.IGNORECASE,
            )
            body = _collapse_ws(body)

            if not home_phone:
                home_phone = _first_phone(body)

            body_wo_phone = re.sub(r"(\(?\d{3}\)?[ -]?\d{3}[ -]?\d{4}|\b\d{10}\b)", " ", body)
            addr_text = _collapse_ws(body_wo_phone)
            addr_text = re.sub(r"\s*Same as Previous Address\s*$", "", addr_text, flags=re.IGNORECASE).strip()
            if addr_text:
                _parse_addr_text(addr_text)
        else:
            warnings.append("legal_mailing_window_not_found")

    # ------------------------------------------------------------
    # 3) Phone fallback: CONTACTS ... DIAGNOSIS
    # ------------------------------------------------------------
    if not home_phone:
        m_contacts = re.search(
            r"CONTACTS(?P<cbody>.*?)(?:\n\s*DIAGNOSIS\b|\s+DIAGNOSIS\b)",
            chunk,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if m_contacts:
            alt_phone = _first_phone(m_contacts.group("cbody"))
            if alt_phone:
                home_phone = alt_phone
                warnings.append("phone_fallback_from_contacts_used")

    if not home_phone:
        warnings.append("phone_parse_failed")

    return (address, city, state, zip_code, home_phone, warnings)


def _extract_pcc_birthdate(chunk: str) -> str:
    """
    PAD-style DOB extraction that works across PCC layouts.

    Reliable pattern in BOTH Arcadia + Seabranch:
      - Header line contains "Sex Birthdate"
      - Next line begins with M/F and then the DOB (MM/DD/YYYY)
        e.g. "M 03/12/1945 80 ..."
    """
    date_any = re.compile(r"\b(0?[1-9]|1[0-2])/(0?[1-9]|[12][0-9]|3[01])/(19|20)\d{2}\b")

    # 1) Best: find "Sex Birthdate" header, then grab DOB from the next M/F row
    m_hdr = re.search(r"^\s*Sex\s+Birthdate\b.*$", chunk, flags=re.IGNORECASE | re.MULTILINE)
    if m_hdr:
        after = chunk[m_hdr.end(): m_hdr.end() + 500]
        m_row = re.search(r"^[MF]\s+(?P<dob>\d{1,2}/\d{1,2}/\d{4})\b", after, flags=re.MULTILINE)
        if m_row:
            return (m_row.group("dob") or "").strip()

        # If the row is wrapped oddly, still allow first date after header
        m_any = date_any.search(after)
        if m_any:
            return m_any.group(0).strip()

    # 2) Next-best: a standalone "Birthdate" label (stacked columns)
    m_birth = re.search(r"^\s*Birthdate\s*$", chunk, flags=re.IGNORECASE | re.MULTILINE)
    if m_birth:
        after = chunk[m_birth.end(): m_birth.end() + 500]
        m = date_any.search(after)
        if m:
            return m.group(0).strip()

    # 3) Fallback anywhere: lines like "M 09/09/1951 74"
    m_line = re.search(r"^[MF]\s+(\d{1,2}/\d{1,2}/\d{4})\b", chunk, flags=re.MULTILINE)
    if m_line:
        return m_line.group(1).strip()

    return ""


def parse_pcc_admission_records_from_pdf_text(text: str) -> list[dict]:
    out: list[dict] = []

    if not (text or "").strip():
        raise ScannedPdfError("Scanned/image PDF detected (no extractable text)")

    chunks = [c.strip() for c in text.split("ADMISSION RECORD") if c.strip()]
    for chunk in chunks:
        lines = [l.strip() for l in chunk.splitlines() if l.strip()]

        facility_name = lines[0] if lines else ""

        report_dt = ""
        for l in lines[:15]:
            if (" ET" in f" {l} ") and re.search(r"\b\d{4}\b", l):
                report_dt = l
                break

        resident_name, room_number, admission_date, resident_num, res_warnings, res_conf = _extract_pcc_resident_info(chunk)

        # Extra fallback: any "Last, First" anywhere
        if not resident_name:
            m2 = re.search(
                r"\b([A-Za-z'\-\. ]+,\s*[A-Za-z'\-\. ]+)\b",
                chunk,
                flags=re.MULTILINE,
            )
            if m2:
                resident_name = m2.group(1).strip()
                res_warnings.append("resident_name_global_fallback_used")

        first_name, last_name = _split_name(resident_name)

        dob = _extract_pcc_birthdate(chunk)

        address, city, state, zip_code, home_phone, addr_warnings = _extract_pcc_address_phone(chunk)

        primary_ins = ""
        primary_number = ""

        # PAD-style: parse ONLY the "Primary Payer ..." line.
        # This works whether the PDF uses "Medicare #", "Policy #", "Group #", etc.
        # NOTE: some PDFs have no space (e.g. "Primary PayerPrivate Pay"), so allow \s*
        m_primary_line = re.search(
            r"^Primary Payer\s*(?P<rest>.+)$",
            chunk,
            flags=re.IGNORECASE | re.MULTILINE,
        )

        if m_primary_line:
            rest = re.sub(r"\s+", " ", (m_primary_line.group("rest") or "")).strip()

            # If "Insurance Name" is blank, the extracted rest is sometimes just the label "Insurance"
            if re.fullmatch(r"insurance(\s+name)?\:?", rest, flags=re.IGNORECASE):
                rest = ""

            # Private Pay normalization (PAD behavior)
            if re.search(r"\bprivate\s*pay\b", rest, flags=re.IGNORECASE):
                primary_ins = "Private Pay"
                primary_number = ""
            else:
                # Otherwise split on first "#": left side = payer name, right side begins with ID
                if "#" in rest:
                    left, right = rest.split("#", 1)
                    left = left.strip()
                    right = right.strip()

                    # Remove trailing token like "Medicare"/"Medicaid"/"Policy"/"Group" right before "#"
                    left_tokens = left.split()
                    if left_tokens and left_tokens[-1].lower() in {"medicare", "medicaid", "policy", "group"}:
                        left_tokens = left_tokens[:-1]
                    primary_ins = " ".join(left_tokens).strip() or left

                    # ID number: first token after "#"
                    primary_number = (right.split() or [""])[0].strip()
                else:
                    primary_ins = rest
                    primary_number = ""

        # Extra guardrails: never allow placeholder junk like "Insurance" or just digits
        if re.fullmatch(r"insurance(\s+name)?\:?", (primary_ins or "").strip(), flags=re.IGNORECASE):
            primary_ins = ""
        if re.fullmatch(r"\d+", re.sub(r"\D+", "", (primary_ins or "").strip())):
            primary_ins = ""

        # Fallback: separate "Policy #:" block (only for number, not payer name)
        if not primary_number:
            m_pol = re.search(r"^Policy\s*#:\s*\n?([A-Za-z0-9\-]+)\b", chunk, flags=re.MULTILINE)
            if m_pol:
                primary_number = m_pol.group(1).strip()

        # Guardrail: never allow nonsense insurance like "1"
        if primary_ins.strip() in {"1", "#1"}:
            primary_ins = ""

        attending = ""
        matt = re.search(r"^Attending Physician\s*\n([^\n]+)$", chunk, flags=re.MULTILINE)
        if matt:
            attending = matt.group(1).strip()

        primary_care = ""
        mpcp = re.search(r"^Primary Care Physician\s*\n([^\n]+)$", chunk, flags=re.MULTILINE)
        if mpcp:
            primary_care = mpcp.group(1).strip()

        reason = ""

        # PAD-style: DIAGNOSIS INFORMATION handling
        if re.search(r"DIAGNOSIS INFORMATION\s*\(No Data Found\)", chunk, flags=re.IGNORECASE):
            reason = "na"
        else:
            # Most reliable anchor in these PDFs (per your PAD flow)
            m_diag = re.search(
                r"Scale Rank Classification\s*\n(?P<line>.+)",
                chunk,
                flags=re.IGNORECASE,
            )
            if m_diag:
                line = (m_diag.group("line") or "").strip()
                # remove leading date if present (PAD removes MM/DD/YYYY)
                line = re.sub(r"^\d{2}/\d{2}/\d{4}\s*", "", line).strip()
                # if the heading leaks in, crop after it
                line = re.sub(r"^DIAGNOSIS INFORMATION\s*", "", line, flags=re.IGNORECASE).strip()
                reason = line

        # final fallback: old "Diagnosis" label
        if not reason:
            mdiag = re.search(r"^Diagnosis\s*\n([^\n]+)$", chunk, flags=re.MULTILINE)
            if mdiag:
                reason = mdiag.group(1).strip()

        facility_code = ""

        patient_key = "|".join([
            facility_name.upper(),
            (resident_num or "").upper(),
            last_name.upper(),
            first_name.upper(),
            dob
        ])

        parse_warnings = list(dict.fromkeys((res_warnings or []) + (addr_warnings or [])))

        out.append({
            "facility_name": facility_name,
            "facility_code": facility_code,
            "report_dt": report_dt,

            "First Name": first_name,
            "Last Name": last_name,
            "DOB": dob,
            "Home Phone": home_phone,
            "Address": address,
            "City": city,
            "State": state,
            "Zip": zip_code,
            "Primary Insurance": primary_ins,
            "Primary Number": primary_number,
            "Primary Care Physician": primary_care,
            "Tag": "",
            "Facility_Code": facility_code,
            "Admission Date": admission_date,
            "Discharge Date": "",
            "Room Number": room_number,
            "Reason for admission": reason,
            "Attending Physician": attending,

            "_patient_key": patient_key,
            "_parse_confidence": res_conf,
            "_parse_warnings": parse_warnings,
            "_raw": chunk[:8000],
        })

    return out



def parse_pcc_admission_records_from_pdf_bytes(pdf_bytes: bytes) -> list[dict]:
    """
    Parses PCC-style 'ADMISSION RECORD' PDFs like your samples.
    Returns rows already mapped to your Sensys CSV columns.
    """
    if not HAVE_PDFPLUMBER:
        raise HTTPException(
            status_code=500,
            detail="PDF parsing support is not installed (pdfplumber). Please add 'pdfplumber' to your environment."
        )

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        text = "\n".join((p.extract_text() or "") for p in pdf.pages)

    # If there's no selectable text, this is very likely a scanned/image PDF
    if not (text or "").strip():
        raise ScannedPdfError("Scanned/image PDF detected (no extractable text)")

    # Reuse shared core
    return parse_pcc_admission_records_from_pdf_text(text)


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def get_db() -> sqlite3.Connection:
    # timeout=30 makes sqlite wait (up to 30s) if the DB is temporarily locked
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row

    # Pragmas to reduce lock errors under bursty write traffic (PAD per-admission calls)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=10000;")  # 10s additional wait on locks
    except Exception:
        # Don't crash app if pragma fails for some reason
        pass

    return conn


def now_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _is_locked_error(e: Exception) -> bool:
    return isinstance(e, sqlite3.OperationalError) and "database is locked" in str(e).lower()


def execute_with_retry(
    cur: sqlite3.Cursor,
    sql: str,
    params: tuple = (),
    *,
    retries: int = 8,
    base_sleep_s: float = 0.08,
):
    """
    Retries on 'database is locked' with exponential backoff.
    Use for writes (INSERT/UPDATE/DELETE) and important reads in a write path.
    """
    for attempt in range(retries):
        try:
            return cur.execute(sql, params)
        except Exception as e:
            if _is_locked_error(e) and attempt < retries - 1:
                time.sleep(base_sleep_s * (2 ** attempt))
                continue
            raise


def commit_with_retry(conn: sqlite3.Connection, *, retries: int = 8, base_sleep_s: float = 0.08):
    for attempt in range(retries):
        try:
            conn.commit()
            return
        except Exception as e:
            if _is_locked_error(e) and attempt < retries - 1:
                time.sleep(base_sleep_s * (2 ** attempt))
                continue
            raise


def _normalize_note_text_for_hash(text: str) -> str:
    """
    Lightly normalize note text so small OCR whitespace differences
    don't create completely different hashes.
    """
    if not text:
        return ""
    t = text.lower()
    # collapse whitespace/newlines/tabs to single spaces
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def compute_note_hash(patient_mrn: str, note_datetime: str, note_text: str) -> str:
    """
    Compute a stable hash for a CM note using MRN, datetime, and normalized text.
    Used to dedupe repeated OCR of the same note.
    """
    base = f"{(patient_mrn or '').strip()}|{(note_datetime or '').strip()}|{_normalize_note_text_for_hash(note_text)}"
    return hashlib.md5(base.encode("utf-8")).hexdigest()

def _normalize_doc_text_for_hash(s: str) -> str:
    # collapse whitespace so trivial formatting changes don't create new hashes
    return " ".join((s or "").strip().split())

def compute_hospital_document_hash(
    hospital_name: str,
    document_type: str,
    visit_id: Optional[str],
    document_datetime: Optional[str],
    source_text: str,
) -> str:
    """
    Stable hash for hospital_documents to skip re-ingesting the exact same document.
    """
    base = (
        f"{(hospital_name or '').strip()}|"
        f"{(document_type or '').strip()}|"
        f"{(visit_id or '').strip()}|"
        f"{(document_datetime or '').strip()}|"
        f"{_normalize_doc_text_for_hash(source_text)}"
    )
    return hashlib.md5(base.encode('utf-8')).hexdigest()

def load_extraction_profile(conn: sqlite3.Connection, hospital_name: str, document_type: str) -> Optional[Dict[str, str]]:
    """
    Returns a dict like:
      { "hospital course": "hospital_course", "discharge medications": "discharge_meds", ... }

    Keys should be normalized (lowercase), because we'll match headings case-insensitively.
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT profile_json
        FROM hospital_extraction_profiles
        WHERE hospital_name = ?
          AND document_type = ?
          AND active = 1
        """,
        (hospital_name, document_type),
    )
    row = cur.fetchone()
    if not row:
        return None

    try:
        prof = json.loads(row["profile_json"] if isinstance(row, sqlite3.Row) else row[0])
    except Exception:
        return None

    # Normalize keys to lowercase for matching
    out: Dict[str, str] = {}
    if isinstance(prof, dict):
        for k, v in prof.items():
            if k and v:
                out[str(k).strip().lower()] = str(v).strip()
    return out or None

def split_document_into_sections(source_text: str, heading_map_override: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
    """
    Very simple, hospital-friendly section splitter.

    It looks for known headings and captures the text until the next heading.

    Returns:
      [
        { "section_key": "hospital_course", "section_title": "Hospital Course", "section_text": "..." , "order": 1 },
        ...
      ]
    """
    text = (source_text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.rstrip() for ln in text.split("\n")]

    # Headings we see in your JFK sample (you can add more later)
    # NOTE: Keep these as they appear in the text (case-insensitive match below)
    default_heading_map = {
        "hospital course": "hospital_course",
        "assessment and plan": "assessment_plan",
        "results": "results",
        "procedures": "procedures",
        "labs from last 24 hours": "labs",
        "objective": "objective",
        "discharge plan": "discharge_plan",
        "discharge medications": "discharge_meds",
        "follow up appointments": "follow_up",
    }

    # Use hospital profile if provided; otherwise fall back to default
    heading_map = heading_map_override or default_heading_map

    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip().lower())

    # Find heading line indexes
    heading_hits: List[Dict[str, Any]] = []
    for i, ln in enumerate(lines):
        key = norm(ln)
        if key in heading_map:
            heading_hits.append(
                {
                    "line_index": i,
                    "section_key": heading_map[key],
                    "section_title": ln.strip(),
                }
            )

    # If we found no headings, store entire thing as one section
    if not heading_hits:
        return [
            {
                "section_key": "full_text",
                "section_title": "Full Text",
                "section_text": (source_text or "").strip(),
                "order": 1,
            }
        ]

    # Always include a "header" section from top-of-note until first heading
    sections: List[Dict[str, Any]] = []
    first_heading_line = heading_hits[0]["line_index"]
    header_text = "\n".join(lines[:first_heading_line]).strip()
    if header_text:
        sections.append(
            {
                "section_key": "header",
                "section_title": "Header",
                "section_text": header_text,
                "order": 1,
            }
        )

    # Build each section from one heading to the next
    for idx, hit in enumerate(heading_hits):
        start = hit["line_index"] + 1
        end = heading_hits[idx + 1]["line_index"] if idx + 1 < len(heading_hits) else len(lines)
        body = "\n".join(lines[start:end]).strip()

        # skip empty sections
        if not body:
            continue

        sections.append(
            {
                "section_key": hit["section_key"],
                "section_title": hit["section_title"],
                "section_text": body,
                "order": len(sections) + 1,
            }
        )

    return sections



def extract_facility_code_from_filename(filename: str) -> str:
    """
    Extracts facility code from filename.
    Example: 'CLIFTON.Clifton Rehab.pcc.pdf' -> 'CLIFTON'
    """
    if not filename:
        return ""
    return filename.split(".", 1)[0].strip().upper()

def extract_facility_name_from_filename(filename: str) -> str:
    """
    Extracts facility name from filename.
    Example: 'FLHIEWCGHR.Westchester Gardens.pcc.pdf' -> 'Westchester Gardens'
    """
    if not filename:
        return ""
    parts = [p for p in filename.split(".") if p is not None]
    # expected: [CODE, FACILITY NAME, 'pcc', 'pdf'?]
    if len(parts) >= 2:
        return (parts[1] or "").strip()
    return ""

# ============================
# SNF Secure Link + PIN helpers
# ============================

def sha256_hex(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()

# ✅ NEW: hash a file without loading it all into memory
def sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def hash_pin(pin: str) -> str:
    """Hash a facility PIN for storage/comparison."""
    return sha256_hex((pin or "").strip())


def verify_pin(pin: str, expected_hash: str) -> bool:
    if not expected_hash:
        return False
    return hash_pin(pin) == expected_hash


def ensure_column(conn: sqlite3.Connection, table: str, col: str, col_sql: str) -> None:
    """Add a column to a SQLite table if it doesn't already exist."""
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    cols = {r[1] for r in cur.fetchall()}  # (cid, name, type, notnull, dflt_value, pk)
    if col not in cols:
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {col_sql}")
        conn.commit()


def get_public_base_url(request: Request) -> str:
    """Prefer PUBLIC_APP_BASE_URL, otherwise fall back to request.base_url."""
    if PUBLIC_APP_BASE_URL:
        return PUBLIC_APP_BASE_URL
    return str(request.base_url).rstrip("/")


def build_snf_secure_link_email_html(secure_url: str, ttl_hours: int) -> str:
    """HTML email body for the secure expiring link (matches snf_secure_link_email_preview.html)."""
    safe_url = html.escape(secure_url or "")
    ttl_hours = int(ttl_hours or 48)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>SNF Secure Link Email Preview</title>
  <style>
    /* Color swatches
       Primary/nav:        #0D3B66
       Links/highlights:   #4DA8DA
       Success/accents:    #A8E6CF
       Backgrounds/cards:  #F5F7FA
    */

    /* Email-safe CSS: keep it simple and inline-ish */
    body{{
      margin:0;
      padding:0;
      background:#F5F7FA; /* backgrounds/cards */
      font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;
      color:#111827;
    }}
    .wrap{{padding:28px 12px;}}
    .card{{
      max-width:560px;
      margin:0 auto;
      background:#ffffff;
      border-radius:14px;
      overflow:hidden;
      box-shadow:0 10px 28px rgba(0,0,0,.08);
      border:1px solid #e5e7eb;
    }}
    .topbar{{
      background:#0D3B66; /* primary/nav */
      padding:10px 0; /* thinner, empty bar */
    }}

    .content{{padding:26px 26px 18px 26px;}}
    h1{{
      margin:0 0 10px 0;
      font-size:22px;
      line-height:1.25;
      color:#0D3B66; /* primary/nav */
      letter-spacing:-0.01em;
    }}
    p{{margin:0 0 12px 0; font-size:14px; line-height:1.55; color:#374151;}}

    /* Optional inline link style if you add any clickable emails/urls later */
    a.inline-link{{
      color:#4DA8DA; /* links/highlights */
      text-decoration:underline;
    }}

    .callout{{
      background:#F5F7FA; /* backgrounds/cards */
      border:1px solid #A8E6CF; /* success/accents */
      padding:12px 14px;
      border-radius:12px;
      margin:14px 0 18px 0;
      font-size:13px;
      color:#1f2937;
    }}
    .pill{{
      display:inline-block;
      background:#A8E6CF; /* success/accents */
      color:#0D3B66;      /* primary/nav */
      padding:2px 8px;
      border-radius:999px;
      font-weight:700;
      font-size:12px;
    }}

    .btn-row{{padding:0 26px 24px 26px;}}
    .btn{{
      display:inline-block;
      background:#0D3B66; /* primary/nav */
      color:#ffffff !important;
      text-decoration:none;
      padding:12px 18px;
      border-radius:10px;
      font-weight:700;
      font-size:14px;
      box-shadow:0 8px 18px rgba(13,59,102,.18);
    }}
    .btn:hover{{background:#0b3357;}}

    .fine{{
      padding:0 26px 20px 26px;
      font-size:12px;
      color:#6b7280;
      line-height:1.5;
    }}
    .divider{{height:1px;background:#eef2f7;}}
    .footer{{
      padding:14px 26px 18px 26px;
      font-size:11px;
      color:#6b7280;
      line-height:1.4;
    }}
    .footer strong{{color:#111827}}
    .meta{{
      font-size:12px;
      color:#6b7280;
      margin-top:10px;
      line-height:1.45;
    }}
    code{{
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      font-size: 12px;
      background:#F5F7FA; /* backgrounds/cards */
      padding:2px 6px;
      border-radius:6px;
      border:1px solid #e5e7eb;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <div class="topbar" aria-hidden="true"></div>

      <div class="content">
        <h1>First Docs Notification</h1>
        <p>
          Our Hospitalists at HCA Florida JFK Hospital have identified upcoming patients expected to discharge to your facility.
          Please use the View List button below to download today's list (Facility PIN required), and assign the referral(s) to the correct First Docs provider.
        </p>

        <div class="callout">
          <div><strong>What you’ll need:</strong></div>
          <div style="margin-top:6px;">• Your facility PIN</div>
          <div style="margin-top:6px;">
            • Link expires in <span class="pill">{ttl_hours} hours</span>
          </div>
        </div>

        <p class="meta">
          Questions or need to add recipients to these notifications?
          Please contact Doug Neal (Doug.Neal@medrina.com) or Stephanie Sellers (ssellers@startevolv.com).
        </p>
      </div>

      <div class="btn-row">
        <a class="btn" href="{safe_url}" target="_blank" rel="noopener">
          View List
        </a>
      </div>

      <div class="fine">
      </div>

      <div class="divider"></div>

      <div class="footer">
        <div><strong>Powered by Evolv Health</strong></div>
        <div style="margin-top:6px;">
          Disclaimer: The information provided reflects anticipated or potential patient referrals and is shared for planning purposes only. Actual admissions are not guaranteed and may change at any time.
        </div>
      </div>
    </div>
  </div>
</body>
</html>"""




def next_qa_id(conn: sqlite3.Connection) -> str:
    """
    Generate a new QA id like Q0001, Q0002...
    Uses the existing qa.id values in SQLite.
    """
    cur = conn.cursor()
    cur.execute("SELECT id FROM qa")
    existing = {str(r["id"]) for r in cur.fetchall() if r["id"] is not None}
    i = 1
    while True:
        candidate = f"Q{i:04d}"
        if candidate not in existing:
            return candidate
        i += 1



def init_db():
    """Create core tables if missing (non-destructive to existing DB)."""
    conn = get_db()
    cur = conn.cursor()

    # Q&A
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS qa (
            id       TEXT PRIMARY KEY,
            section  TEXT,
            question TEXT NOT NULL,
            answer   TEXT NOT NULL,
            tags     TEXT,
            topics   TEXT
        )
        """
    )

    # Dictionary (key → canonical term, plus kind/notes/match_mode)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS dictionary (
            key        TEXT PRIMARY KEY,
            canonical  TEXT NOT NULL,
            kind       TEXT DEFAULT 'abbr',
            notes      TEXT DEFAULT '',
            match_mode TEXT DEFAULT 'exact'
        )
        """
    )


    # Store secure, expiring links for SNF PDFs (no PDF blob, just references)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS snf_pdf_links (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      token TEXT NOT NULL UNIQUE,
      snf_facility_id INTEGER NOT NULL,
      for_date TEXT NOT NULL,
      admission_ids_json TEXT NOT NULL,
      created_at TEXT NOT NULL DEFAULT (datetime('now')),
      expires_at TEXT NOT NULL
    )
    """)


    # Facility facts (link knowledge to specific facilities)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS fac_facts (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            facility_id TEXT NOT NULL,
            fact_text   TEXT NOT NULL,
            tags        TEXT DEFAULT '',
            created_at  TEXT
        )
        """
    )

    # User question log
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS user_qa_log (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            ts         TEXT,
            section    TEXT,
            q          TEXT,
            a          TEXT,
            promoted   INTEGER DEFAULT 0,
            a_quality  TEXT,
            debug_sql  TEXT

        )
        """
    )

    # Ensure debug_sql column exists on older DBs
    try:
        cur.execute("ALTER TABLE user_qa_log ADD COLUMN debug_sql TEXT")
    except sqlite3.Error:
        # Will fail with 'duplicate column name' once it already exists – that's fine.
        pass


    # AI settings (editable later via Admin)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS ai_settings (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )

    # PAD API run log (one row per API call, even if 0 inserts)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS pad_api_runs (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            endpoint      TEXT NOT NULL,
            received_at   TEXT NOT NULL,   -- UTC datetime string
            pad_run_id    TEXT,
            rows_received INTEGER DEFAULT 0,
            inserted      INTEGER DEFAULT 0,
            skipped       INTEGER DEFAULT 0,
            error_count   INTEGER DEFAULT 0,
            status        TEXT NOT NULL,   -- 'ok' | 'error'
            message       TEXT
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_pad_api_runs_received ON pad_api_runs (received_at DESC)")

    # PAD Flow run log (one row per PAD flow execution)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS pad_flow_runs (
            run_id            TEXT PRIMARY KEY,   -- UUID (links all events for this run)
            flow_name         TEXT NOT NULL,      -- friendly name e.g. "SNF OCR CM Notes"
            flow_key          TEXT,               -- stable key you choose e.g. "snf_cm_notes_ocr"
            flow_version      TEXT,               -- optional: version/build
            environment       TEXT,               -- optional: prod/dev, etc.

            machine_name      TEXT,               -- optional: PAD machine
            os_user           TEXT,               -- optional: windows user
            triggered_by      TEXT,               -- optional: scheduler/manual/etc.

            started_at        TEXT NOT NULL,      -- UTC datetime string
            ended_at          TEXT,               -- UTC datetime string
            status            TEXT NOT NULL DEFAULT 'running',  -- running | ok | error
            duration_ms       INTEGER,            -- computed on stop/error

            start_payload_json TEXT,              -- raw JSON string (optional)
            stop_payload_json  TEXT,              -- raw JSON string (optional)

            error_message     TEXT,
            error_details_json TEXT,

            created_at        TEXT DEFAULT (datetime('now')),
            updated_at        TEXT DEFAULT (datetime('now'))
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_pad_flow_runs_flow ON pad_flow_runs (flow_key, started_at DESC)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_pad_flow_runs_started ON pad_flow_runs (started_at DESC)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_pad_flow_runs_status ON pad_flow_runs (status)")

    # PAD Flow events (multiple rows per run_id: start/stop/error + optional steps)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS pad_flow_events (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id          TEXT NOT NULL,        -- FK to pad_flow_runs.run_id
            event_type      TEXT NOT NULL,        -- start | stop | error | info | step
            event_ts        TEXT NOT NULL,        -- UTC datetime string

            step_name       TEXT,                -- optional: "Extract PDF", "Upload", etc.
            message         TEXT,                -- short message
            details_json    TEXT,                -- JSON string (anything extra)

            created_at      TEXT DEFAULT (datetime('now')),
            FOREIGN KEY(run_id) REFERENCES pad_flow_runs(run_id)
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_pad_flow_events_run ON pad_flow_events (run_id, event_ts)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_pad_flow_events_type ON pad_flow_events (event_type, event_ts)")

    # Auto-update updated_at for older DBs (safe no-op if already exists)
    for stmt in [
        "ALTER TABLE pad_flow_runs ADD COLUMN flow_key TEXT",
        "ALTER TABLE pad_flow_runs ADD COLUMN flow_version TEXT",
        "ALTER TABLE pad_flow_runs ADD COLUMN environment TEXT",
        "ALTER TABLE pad_flow_runs ADD COLUMN machine_name TEXT",
        "ALTER TABLE pad_flow_runs ADD COLUMN os_user TEXT",
        "ALTER TABLE pad_flow_runs ADD COLUMN triggered_by TEXT",
        "ALTER TABLE pad_flow_runs ADD COLUMN ended_at TEXT",
        "ALTER TABLE pad_flow_runs ADD COLUMN duration_ms INTEGER",
        "ALTER TABLE pad_flow_runs ADD COLUMN start_payload_json TEXT",
        "ALTER TABLE pad_flow_runs ADD COLUMN stop_payload_json TEXT",
        "ALTER TABLE pad_flow_runs ADD COLUMN error_message TEXT",
        "ALTER TABLE pad_flow_runs ADD COLUMN error_details_json TEXT",
        "ALTER TABLE pad_flow_runs ADD COLUMN updated_at TEXT DEFAULT (datetime('now'))",
    ]:
        try:
            cur.execute(stmt)
        except sqlite3.Error:
            pass

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS census_runs (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            facility_name   TEXT NOT NULL,
            facility_code   TEXT,
            report_dt       TEXT,
            source_filename TEXT,
            source_sha256   TEXT,
            created_at      TEXT DEFAULT (datetime('now'))
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_census_runs_fac_dt ON census_runs (facility_name, created_at DESC)")

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS census_run_patients (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id             INTEGER NOT NULL,
            facility_name      TEXT NOT NULL,
            facility_code      TEXT,
            patient_key        TEXT NOT NULL,

            first_name         TEXT,
            last_name          TEXT,
            dob                TEXT,
            home_phone         TEXT,
            address            TEXT,
            city               TEXT,
            state              TEXT,
            zip                TEXT,
            primary_ins        TEXT,
            primary_number     TEXT,
            primary_care_phys  TEXT,
            attending_phys     TEXT,
            admission_date     TEXT,
            discharge_date     TEXT,
            room_number        TEXT,
            reason_admission   TEXT,
            tag                TEXT,

            raw_text           TEXT,
            created_at         TEXT DEFAULT (datetime('now')),
            FOREIGN KEY(run_id) REFERENCES census_runs(id)
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_census_patients_run ON census_run_patients (run_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_census_patients_key ON census_run_patients (facility_name, patient_key)")
    
    # ----------------------------
    # Sensys Census (CSV compare)
    # ----------------------------
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS sensys_census_jobs (
            job_id        TEXT PRIMARY KEY,
            facility_name TEXT NOT NULL,
            created_at    TEXT DEFAULT (datetime('now'))
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS sensys_census_job_rows (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id         TEXT NOT NULL,
            kind           TEXT NOT NULL, -- missing_admissions | missing_discharges | duplicates
            first_name     TEXT,
            last_name      TEXT,
            dob            TEXT,
            admission_date TEXT,
            discharge_date TEXT,
            room_number    TEXT,
            primary_phone  TEXT,
            note           TEXT,
            created_at     TEXT DEFAULT (datetime('now')),
            FOREIGN KEY(job_id) REFERENCES sensys_census_jobs(job_id)
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_sensys_rows_job ON sensys_census_job_rows(job_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_sensys_rows_kind ON sensys_census_job_rows(job_id, kind)")
    # ✅ Expand sensys_census_job_rows to support full Missing Admissions exports (safe no-op if already exists)
    for stmt in [
        "ALTER TABLE sensys_census_job_rows ADD COLUMN home_phone TEXT",
        "ALTER TABLE sensys_census_job_rows ADD COLUMN address TEXT",
        "ALTER TABLE sensys_census_job_rows ADD COLUMN city TEXT",
        "ALTER TABLE sensys_census_job_rows ADD COLUMN state TEXT",
        "ALTER TABLE sensys_census_job_rows ADD COLUMN zip TEXT",
        "ALTER TABLE sensys_census_job_rows ADD COLUMN primary_ins TEXT",
        "ALTER TABLE sensys_census_job_rows ADD COLUMN primary_number TEXT",
        "ALTER TABLE sensys_census_job_rows ADD COLUMN primary_care_phys TEXT",
        "ALTER TABLE sensys_census_job_rows ADD COLUMN tag TEXT",
        "ALTER TABLE sensys_census_job_rows ADD COLUMN facility_code TEXT",
        "ALTER TABLE sensys_census_job_rows ADD COLUMN reason_admission TEXT",
        "ALTER TABLE sensys_census_job_rows ADD COLUMN attending_phys TEXT",
    ]:
        try:
            cur.execute(stmt)
        except sqlite3.Error:
            pass

    # ✅ Ensure new columns exist on older census_run_patients tables (safe no-op if already exists)
    for stmt in [
        "ALTER TABLE census_run_patients ADD COLUMN first_name TEXT",
        "ALTER TABLE census_run_patients ADD COLUMN last_name TEXT",
        "ALTER TABLE census_run_patients ADD COLUMN dob TEXT",
        "ALTER TABLE census_run_patients ADD COLUMN home_phone TEXT",
        "ALTER TABLE census_run_patients ADD COLUMN address TEXT",
        "ALTER TABLE census_run_patients ADD COLUMN city TEXT",
        "ALTER TABLE census_run_patients ADD COLUMN state TEXT",
        "ALTER TABLE census_run_patients ADD COLUMN zip TEXT",
        "ALTER TABLE census_run_patients ADD COLUMN primary_ins TEXT",
        "ALTER TABLE census_run_patients ADD COLUMN primary_number TEXT",
        "ALTER TABLE census_run_patients ADD COLUMN primary_care_phys TEXT",
        "ALTER TABLE census_run_patients ADD COLUMN attending_phys TEXT",
        "ALTER TABLE census_run_patients ADD COLUMN admission_date TEXT",
        "ALTER TABLE census_run_patients ADD COLUMN discharge_date TEXT",
        "ALTER TABLE census_run_patients ADD COLUMN room_number TEXT",
        "ALTER TABLE census_run_patients ADD COLUMN reason_admission TEXT",
        "ALTER TABLE census_run_patients ADD COLUMN tag TEXT",
        "ALTER TABLE census_run_patients ADD COLUMN raw_text TEXT",
        "ALTER TABLE census_run_patients ADD COLUMN created_at TEXT",
    ]:
        try:
            cur.execute(stmt)
        except sqlite3.Error:
            pass

        # ✅ Also ensure newer columns exist on older census_runs tables (optional but recommended)
        for stmt in [
            "ALTER TABLE census_runs ADD COLUMN facility_code TEXT",
            "ALTER TABLE census_runs ADD COLUMN report_dt TEXT",
            "ALTER TABLE census_runs ADD COLUMN source_filename TEXT",
            "ALTER TABLE census_runs ADD COLUMN source_sha256 TEXT",
        ]:
            try:
                cur.execute(stmt)
            except sqlite3.Error:
                pass

    # ----------------------------
    # Census upload jobs (Option A)
    # ----------------------------
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS census_upload_jobs (
            job_id            TEXT PRIMARY KEY,
            status            TEXT NOT NULL, -- queued | running | done | error
            total_files       INTEGER DEFAULT 0,
            processed_files   INTEGER DEFAULT 0,
            message           TEXT,
            created_at        TEXT DEFAULT (datetime('now')),
            updated_at        TEXT DEFAULT (datetime('now'))
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS census_upload_job_files (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id           TEXT NOT NULL,
            filename         TEXT,
            status           TEXT NOT NULL, -- ok | skipped | error
            detail           TEXT,
            run_id           INTEGER,
            rows_inserted    INTEGER DEFAULT 0,
            created_at       TEXT DEFAULT (datetime('now')),
            FOREIGN KEY(job_id) REFERENCES census_upload_jobs(job_id)
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_census_upload_job_files_job ON census_upload_job_files (job_id)")


    # Seed default system prompt
    cur.execute("SELECT value FROM ai_settings WHERE key='system_prompt'")
    row = cur.fetchone()
    if not row:
        cur.execute(
            """
            INSERT INTO ai_settings (key, value)
            VALUES ('system_prompt', ?)
            """,
            (
                "You are Evolv's internal copilot. Use Evolv's database facts "
                "about facilities, processes, and tools FIRST. You may also "
                "answer general healthcare, discharge, and workflow questions, "
                "but never invent facility-specific facts. If you aren't sure, "
                "say what you do and don't know and suggest next steps.",
            ),
        )


    # -------------------------------------------------------------------
    # SNF pipeline tables
    # -------------------------------------------------------------------

    # Raw case management notes from PAD OCR
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS cm_notes_raw (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,

            -- Patient identity
            patient_mrn     TEXT NOT NULL,
            patient_name    TEXT,
            dob             TEXT,

            -- Encounter / context
            visit_id        TEXT,
            encounter_id    TEXT,
            hospital_name   TEXT,
            unit_name       TEXT,
            admission_date  TEXT,
            admit_date      TEXT,
            attending       TEXT,

            -- Note metadata
            note_datetime   TEXT NOT NULL,
            note_author     TEXT,
            note_type       TEXT,
            note_text       TEXT NOT NULL,

            -- PAD / OCR metadata
            source_system   TEXT,
            pad_run_id      TEXT,
            ocr_confidence  REAL,

            -- Dedup helper
            note_hash       TEXT,

            created_at      TEXT DEFAULT (datetime('now'))
        )
        """
    )

    # Ensure visit_id exists on older cm_notes_raw tables
    try:
        cur.execute("ALTER TABLE cm_notes_raw ADD COLUMN visit_id TEXT")
    except sqlite3.Error:
        pass

    try:
        cur.execute("ALTER TABLE cm_notes_raw ADD COLUMN admit_date TEXT")
    except sqlite3.Error:
        pass

    try:
        cur.execute("ALTER TABLE cm_notes_raw ADD COLUMN attending TEXT")
    except sqlite3.Error:
        pass

    try:
        cur.execute("ALTER TABLE cm_notes_raw ADD COLUMN dob TEXT")
    except sqlite3.Error:
        pass

    # NEW: allow soft-deleting/ignoring specific notes (UI cleanup + AI ignore)
    try:
        cur.execute("ALTER TABLE cm_notes_raw ADD COLUMN ignored INTEGER DEFAULT 0")
    except sqlite3.Error:
        pass

    try:
        cur.execute("ALTER TABLE cm_notes_raw ADD COLUMN ignored_at TEXT")
    except sqlite3.Error:
        pass

    # Indexes to speed up typical queries on cm_notes_raw
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_cm_notes_raw_mrn ON cm_notes_raw (patient_mrn)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_cm_notes_raw_datetime ON cm_notes_raw (note_datetime DESC)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_cm_notes_raw_hash ON cm_notes_raw (note_hash)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_cm_notes_raw_ignored ON cm_notes_raw (ignored)"
    )

    # -------------------------------------------------------------------
    # Hospital extraction profiles (per hospital + document type) ✅ NEW
    # -------------------------------------------------------------------
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS hospital_extraction_profiles (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            hospital_name  TEXT NOT NULL,
            document_type  TEXT NOT NULL,
            profile_json   TEXT NOT NULL,   -- JSON mapping of heading -> section_key
            active         INTEGER DEFAULT 1,
            updated_at     TEXT DEFAULT (datetime('now'))
        )
        """
    )
    cur.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_hexprof_unique ON hospital_extraction_profiles (hospital_name, document_type)"
    )



    # -------------------------------------------------------------------
    # Hospital Documents (Raw + Sectioned)  ✅ NEW
    # -------------------------------------------------------------------
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS hospital_documents (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,

            hospital_name      TEXT NOT NULL,
            document_type      TEXT NOT NULL,
            document_datetime  TEXT,
            patient_mrn        TEXT,
            patient_name       TEXT,
            dob                TEXT,
            visit_id           TEXT,   --

            admit_date         TEXT,
            dc_date            TEXT,

            attending          TEXT,
            pcp                TEXT,
            insurance          TEXT,

            source_text        TEXT NOT NULL,
            source_system      TEXT DEFAULT 'EMR',
            
            document_hash      TEXT,

            created_at         TEXT DEFAULT (datetime('now'))
        )
        """
    )

    # Ensure new columns exist on older hospital_documents tables (safe no-op if already exists)
    try:
        cur.execute("ALTER TABLE hospital_documents ADD COLUMN admit_date TEXT")
    except sqlite3.Error:
        pass

    try:
        cur.execute("ALTER TABLE hospital_documents ADD COLUMN dc_date TEXT")
    except sqlite3.Error:
        pass

    try:
        cur.execute("ALTER TABLE hospital_documents ADD COLUMN attending TEXT")
    except sqlite3.Error:
        pass

    try:
        cur.execute("ALTER TABLE hospital_documents ADD COLUMN pcp TEXT")
    except sqlite3.Error:
        pass

    try:
        cur.execute("ALTER TABLE hospital_documents ADD COLUMN insurance TEXT")
    except sqlite3.Error:
        pass

    try:
        cur.execute("ALTER TABLE hospital_documents ADD COLUMN document_hash TEXT")
    except sqlite3.Error:
        pass

    # -------------------------------------------------------------------
    # Legacy date normalization (hospital_documents): MM/DD/YY -> YYYY-MM-DD
    # Ensures SQLite date() filters work in reports.
    # -------------------------------------------------------------------
    try:
        rows = cur.execute(
            """
            SELECT id, admit_date, dc_date
            FROM hospital_documents
            WHERE admit_date IS NOT NULL OR dc_date IS NOT NULL
            """
        ).fetchall()

        for doc_id, admit_date_raw, dc_date_raw in rows:
            admit_date_norm = normalize_date_to_iso(admit_date_raw)
            dc_date_norm = normalize_date_to_iso(dc_date_raw)

            # Only write if something actually changes
            if admit_date_norm != admit_date_raw or dc_date_norm != dc_date_raw:
                cur.execute(
                    """
                    UPDATE hospital_documents
                    SET admit_date = ?, dc_date = ?
                    WHERE id = ?
                    """,
                    (admit_date_norm, dc_date_norm, doc_id),
                )
    except sqlite3.Error:
        # Never block app startup on a migration cleanup
        pass

    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_hosp_docs_hosp_type_dt ON hospital_documents (hospital_name, document_type, document_datetime DESC)"
    )
    
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_hosp_docs_mrn ON hospital_documents (patient_mrn)"
    )
    
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_hosp_docs_hash ON hospital_documents (document_hash)"
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS hospital_document_sections (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id    INTEGER NOT NULL,

            section_key    TEXT NOT NULL,     -- your normalized key like 'hospital_course'
            section_title  TEXT,              -- the raw heading like 'Hospital Course'
            section_order  INTEGER NOT NULL,  -- 1,2,3...
            section_text   TEXT NOT NULL,

            created_at     TEXT DEFAULT (datetime('now')),

            FOREIGN KEY(document_id) REFERENCES hospital_documents(id)
        )
        """
    )

    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_doc_sections_doc_order ON hospital_document_sections (document_id, section_order)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_doc_sections_key ON hospital_document_sections (section_key)"
    )

    # -------------------------------------------------------------------
    # Hospital Discharges hub table (one row per visit_id) ✅ NEW
    # -------------------------------------------------------------------
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS hospital_discharges (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,

            visit_id       TEXT NOT NULL,   -- group key (one "encounter / discharge packet")
            patient_mrn    TEXT,
            patient_name   TEXT,
            hospital_name  TEXT,

            admit_date     TEXT,
            dc_date        TEXT,

            attending      TEXT,
            pcp            TEXT,
            insurance      TEXT,

            disposition    TEXT,
            dc_agency      TEXT,
            dispo_source_dt     TEXT,
            dispo_source_doc_id INTEGER,
            updated_at     TEXT DEFAULT (datetime('now')),
            created_at     TEXT DEFAULT (datetime('now'))
        )
        """
    )

    cur.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_hosp_discharge_visit ON hospital_discharges (visit_id)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_hosp_discharge_patient ON hospital_discharges (patient_mrn, patient_name)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_hosp_discharge_hosp_dates ON hospital_discharges (hospital_name, admit_date, dc_date)"
    )
    
    # Ensure new columns exist on older hospital_discharges tables (safe no-op if already exists)
    try:
        cur.execute("ALTER TABLE hospital_discharges ADD COLUMN dc_agency TEXT")
    except sqlite3.Error:
        pass

    # Ensure dispo_source_dt / dispo_source_doc_id exist on older DBs
    try:
        cur.execute("ALTER TABLE hospital_discharges ADD COLUMN dispo_source_dt TEXT")
    except sqlite3.Error:
        pass

    try:
        cur.execute("ALTER TABLE hospital_discharges ADD COLUMN dispo_source_doc_id INTEGER")
    except sqlite3.Error:
        pass
 
    try:
        cur.execute("ALTER TABLE hospital_discharges ADD COLUMN attending TEXT")
    except sqlite3.Error:
        pass

    try:
        cur.execute("ALTER TABLE hospital_discharges ADD COLUMN pcp TEXT")
    except sqlite3.Error:
        pass

    try:
        cur.execute("ALTER TABLE hospital_discharges ADD COLUMN insurance TEXT")
    except sqlite3.Error:
        pass

    # -------------------------------------------------------------------
    # Legacy date normalization (hospital_discharges): MM/DD/YY -> YYYY-MM-DD
    # Ensures SQLite date() filters work in reports.
    # -------------------------------------------------------------------
    try:
        rows = cur.execute(
            """
            SELECT id, admit_date, dc_date
            FROM hospital_discharges
            WHERE admit_date IS NOT NULL OR dc_date IS NOT NULL
            """
        ).fetchall()

        for discharge_id, admit_date_raw, dc_date_raw in rows:
            admit_date_norm = normalize_date_to_iso(admit_date_raw)
            dc_date_norm = normalize_date_to_iso(dc_date_raw)

            # Only write if something actually changes
            if admit_date_norm != admit_date_raw or dc_date_norm != dc_date_raw:
                cur.execute(
                    """
                    UPDATE hospital_discharges
                    SET admit_date = ?, dc_date = ?
                    WHERE id = ?
                    """,
                    (admit_date_norm, dc_date_norm, discharge_id),
                )
    except sqlite3.Error:
        # Never block app startup on a migration cleanup
        pass

   
    # SNF admissions derived from CM notes
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS snf_admissions (
            id                          INTEGER PRIMARY KEY AUTOINCREMENT,

            -- Link back to source note
            raw_note_id                 INTEGER NOT NULL,

            -- Patient + context
            visit_id                    TEXT,
            patient_mrn                 TEXT NOT NULL,
            patient_name                TEXT,
            dob                         TEXT,
            attending                   TEXT,
            admit_date                  TEXT,
            hospital_name               TEXT,
            note_datetime               TEXT,

            -- AI interpretation
            ai_is_snf_candidate         INTEGER,     -- 0/1
            ai_snf_name_raw             TEXT,
            ai_snf_facility_id          TEXT,
            ai_expected_transfer_date   TEXT,
            ai_confidence               REAL,

            -- Human review / final decision
            status                      TEXT DEFAULT 'pending',  -- pending/confirmed/corrected/rejected/removed
            disposition                 TEXT,
            facility_free_text          TEXT,
            final_snf_facility_id       TEXT,
            final_snf_name_display      TEXT,
            final_expected_transfer_date TEXT,
            reviewed_by                 TEXT,
            reviewed_at                 TEXT,
            review_comments             TEXT,

            -- Activity tracking (which admissions are still active in PAD feeds)
            last_seen_active_date       TEXT,

            -- Email tracking
            emailed_at                  TEXT,
            email_run_id                TEXT,

            -- SNF Physician Assignment (NEW)
            assignment_confirmation     TEXT DEFAULT 'Unknown',  -- Unknown / Assigned / Assigned Out
            billing_confirmed           INTEGER DEFAULT 0,       -- 0/1
            confirmation_call_dt        TEXT,                    -- datetime-local string
            snf_staff_name              TEXT,
            physician_assigned          TEXT,
            assignment_notes            TEXT,

            created_at                  TEXT DEFAULT (datetime('now')),
            updated_at                  TEXT DEFAULT (datetime('now'))
        )
        """
    )


    # Ensure visit_id exists on older snf_admissions tables
    try:
        cur.execute("ALTER TABLE snf_admissions ADD COLUMN visit_id TEXT")
    except sqlite3.Error:
        pass

    # Ensure new columns exist on older snf_admissions tables
    try:
        cur.execute("ALTER TABLE snf_admissions ADD COLUMN disposition TEXT")
    except sqlite3.Error:
        pass

    # Ensure updated_at exists on older snf_admissions tables
    try:
        cur.execute("PRAGMA table_info(snf_admissions)")
        cols = [r[1] for r in cur.fetchall()]
        if "updated_at" not in cols:
            cur.execute("ALTER TABLE snf_admissions ADD COLUMN updated_at TEXT")
            # Backfill so existing rows don't have NULL
            cur.execute(
                "UPDATE snf_admissions SET updated_at = COALESCE(updated_at, created_at) WHERE updated_at IS NULL"
            )
    except sqlite3.Error:
        pass


    try:
        cur.execute("ALTER TABLE snf_admissions ADD COLUMN facility_free_text TEXT")
    except sqlite3.Error:
        pass

    try:
        cur.execute("ALTER TABLE snf_admissions ADD COLUMN last_seen_active_date TEXT")
    except sqlite3.Error:
        pass

    try:
        cur.execute("ALTER TABLE snf_admissions ADD COLUMN dob TEXT")
    except sqlite3.Error:
        pass

    try:
        cur.execute("ALTER TABLE snf_admissions ADD COLUMN attending TEXT")
    except sqlite3.Error:
        pass

    try:
        cur.execute("ALTER TABLE snf_admissions ADD COLUMN admit_date TEXT")
    except sqlite3.Error:
        pass

    try:
        cur.execute("ALTER TABLE snf_admissions ADD COLUMN notified_by_hospital INTEGER DEFAULT 0")
    except sqlite3.Error:
        pass

    try:
        cur.execute("ALTER TABLE snf_admissions ADD COLUMN notified_by TEXT")
    except sqlite3.Error:
        pass

    try:
        cur.execute("ALTER TABLE snf_admissions ADD COLUMN notification_dt TEXT")
    except sqlite3.Error:
        pass

    try:
        cur.execute("ALTER TABLE snf_admissions ADD COLUMN hospital_reported_facility TEXT")
    except sqlite3.Error:
        pass

    try:
        cur.execute("ALTER TABLE snf_admissions ADD COLUMN notification_details TEXT")
    except sqlite3.Error:
        pass

    # SNF Physician Assignment (NEW)
    try:
        cur.execute("ALTER TABLE snf_admissions ADD COLUMN assignment_confirmation TEXT DEFAULT 'Unknown'")
    except sqlite3.Error:
        pass

    try:
        cur.execute("ALTER TABLE snf_admissions ADD COLUMN billing_confirmed INTEGER DEFAULT 0")
    except sqlite3.Error:
        pass

    try:
        cur.execute("ALTER TABLE snf_admissions ADD COLUMN confirmation_call_dt TEXT")
    except sqlite3.Error:
        pass

    try:
        cur.execute("ALTER TABLE snf_admissions ADD COLUMN snf_staff_name TEXT")
    except sqlite3.Error:
        pass

    try:
        cur.execute("ALTER TABLE snf_admissions ADD COLUMN physician_assigned TEXT")
    except sqlite3.Error:
        pass

    try:
        cur.execute("ALTER TABLE snf_admissions ADD COLUMN assignment_notes TEXT")
    except sqlite3.Error:
        pass


    # Helpful indexes for SNF admissions queries
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_snf_adm_mrn ON snf_admissions (patient_mrn)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_snf_adm_status_date ON snf_admissions (status, final_expected_transfer_date)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_snf_adm_facility ON snf_admissions (final_snf_facility_id)"
    )

    # Refresh indexes for SNF admissions uniqueness & lookup:
    # - raw_note_id is now just a lookup index
    # - visit_id enforces "one row per admission"
    cur.execute("DROP INDEX IF EXISTS idx_snf_adm_raw_note")
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_snf_adm_raw_note ON snf_admissions (raw_note_id)"
    )
    cur.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_snf_adm_visit ON snf_admissions (visit_id)"
    )

    # Notification targets for SNF emails
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS snf_notification_targets (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            facility_id TEXT NOT NULL,
            email_to    TEXT NOT NULL,
            email_cc    TEXT,
            active      INTEGER DEFAULT 1,
            created_at  TEXT DEFAULT (datetime('now')),
            updated_at  TEXT DEFAULT (datetime('now'))
        )
        """
    )

    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_snf_notif_facility ON snf_notification_targets (facility_id)"
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS snf_admission_facilities (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            facility_name  TEXT NOT NULL,
            facility_phone TEXT,
            attending      TEXT,
            notes          TEXT,
            notes2         TEXT,
            aliases        TEXT,
            facility_emails TEXT
        )
        """
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_snf_adm_fac_name ON snf_admission_facilities(facility_name)"
    )


    # --- Secure link + facility PIN support ---
    # Store a per-facility PIN hash (optional). If blank, we fall back to SNF_DEFAULT_PIN env var.
    ensure_column(conn, "snf_admission_facilities", "pin_hash", "pin_hash TEXT")

    # NEW: facility phone
    ensure_column(conn, "snf_admission_facilities", "facility_phone", "facility_phone TEXT")


    # Secure expiring links table (store only token HASH, never the raw token)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS snf_secure_links (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            token_hash      TEXT NOT NULL,
            snf_facility_id INTEGER NOT NULL,
            admission_ids   TEXT NOT NULL,   -- JSON list of admission ids
            for_date        TEXT,
            expires_at      TEXT NOT NULL,   -- ISO string UTC
            created_at      TEXT DEFAULT (datetime('now')),
            used_at         TEXT
        )
        """
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_snf_secure_links_hash ON snf_secure_links(token_hash)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_snf_secure_links_fac ON snf_secure_links(snf_facility_id)"
    )

    # ---------------------------------------------------------
    # NEW: Add a detailed access log (each time someone enters a PIN)
    # ---------------------------------------------------------
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS snf_secure_link_access_log (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            secure_link_id  INTEGER NOT NULL,
            snf_facility_id INTEGER NOT NULL,
            pin_type        TEXT NOT NULL,   -- facility | universal | default
            accessed_at     TEXT DEFAULT (datetime('now')),
            ip              TEXT,
            user_agent      TEXT
        )
        """
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_snf_access_link ON snf_secure_link_access_log(secure_link_id)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_snf_access_fac ON snf_secure_link_access_log(snf_facility_id)"
    )

    # ---------------------------------------------------------
    # OPTIONAL (but recommended): tie the secure link back to the email send
    # (safe: this is just metadata, no PHI)
    # ---------------------------------------------------------
    ensure_column(conn, "snf_secure_links", "email_run_id", "email_run_id TEXT")
    ensure_column(conn, "snf_secure_links", "sent_to",     "sent_to TEXT")
    ensure_column(conn, "snf_secure_links", "sent_at",     "sent_at TEXT")

    
# -------------------------------------------------------------------
    # v_facility_listables view: single place for list + aggregate queries
    # NOTE: When you add new columns you want to filter on, edit this SQL.
    # -------------------------------------------------------------------
    try:
        cur.executescript(
            '''
            DROP VIEW IF EXISTS v_facility_listables;

            CREATE VIEW v_facility_listables AS

            -- 1) One row per facility with summary metrics
            SELECT
                f.facility_id,
                f.facility_name,
                f.city,
                f.state,
                f.zip,
                f.county,
                f.corporate_group,
                f.emr,
                f.emr_other,
                f.pt_emr,
                vs.orders,
                vs.orders_other,
                vs.outpatient_pt,
                vs.short_beds,
                vs.ltc_beds,
                vs.avg_dcs,
                'facility' AS item_type,
                ''         AS item_subtype,
                f.facility_name AS display_name,
                TRIM(
                    COALESCE(f.city, '') ||
                    CASE
                        WHEN f.city IS NOT NULL AND f.city <> '' AND f.state IS NOT NULL AND f.state <> ''
                            THEN ', '
                        ELSE ''
                    END ||
                    COALESCE(f.state, '') ||
                    CASE
                        WHEN f.corporate_group IS NOT NULL AND f.corporate_group <> ''
                            THEN ' – ' || f.corporate_group
                        ELSE ''
                    END
                ) AS details
            FROM facilities f
            LEFT JOIN v_facility_summary vs
                ON vs.facility_id = f.facility_id

            UNION ALL

            -- 2) Additional services
            SELECT
                fas.facility_id,
                f.facility_name,
                f.city,
                f.state,
                f.zip,
                f.county,
                f.corporate_group,
                NULL AS emr,
                NULL AS emr_other,
                NULL AS pt_emr,
                NULL AS orders,
                NULL AS orders_other,
                NULL AS outpatient_pt,
                NULL AS short_beds,
                NULL AS ltc_beds,
                NULL AS avg_dcs,
                'additional_service' AS item_type,
                ''                   AS item_subtype,
                fas.service          AS display_name,
                ''                   AS details
            FROM facility_additional_services fas
            JOIN facilities f
                ON f.facility_id = fas.facility_id

            UNION ALL

            -- 3) Insurance plans
            SELECT
                fip.facility_id,
                f.facility_name,
                f.city,
                f.state,
                f.zip,
                f.county,
                f.corporate_group,
                NULL AS emr,
                NULL AS emr_other,
                NULL AS pt_emr,
                NULL AS orders,
                NULL AS orders_other,
                NULL AS outpatient_pt,
                NULL AS short_beds,
                NULL AS ltc_beds,
                NULL AS avg_dcs,
                'insurance_plan' AS item_type,
                ''               AS item_subtype,
                fip.plan         AS display_name,
                ''               AS details
            FROM facility_insurance_plans fip
            JOIN facilities f
                ON f.facility_id = fip.facility_id

            UNION ALL

            -- 4) Community partners (home health, hospice, DME, etc.)
            SELECT
                fp.facility_id,
                f.facility_name,
                f.city,
                f.state,
                f.zip,
                f.county,
                f.corporate_group,
                NULL AS emr,
                NULL AS emr_other,
                NULL AS pt_emr,
                NULL AS orders,
                NULL AS orders_other,
                NULL AS outpatient_pt,
                NULL AS short_beds,
                NULL AS ltc_beds,
                NULL AS avg_dcs,
                'community_partner' AS item_type,
                fp.type             AS item_subtype,
                fp.name             AS display_name,
                TRIM(
                    CASE
                        WHEN fp.ins_only IS NOT NULL AND fp.ins_only <> ''
                            THEN 'Insurance only: ' || fp.ins_only
                        ELSE ''
                    END ||
                    CASE
                        WHEN (fp.ins_only IS NOT NULL AND fp.ins_only <> '')
                             AND (fp.insurance IS NOT NULL AND fp.insurance <> '')
                            THEN ' – '
                        ELSE ''
                    END ||
                    COALESCE(fp.insurance, '')
                ) AS details
            FROM facility_partners fp
            JOIN facilities f
                ON f.facility_id = fp.facility_id

            UNION ALL

            -- 5) Contacts (Administrator, DON, UR, etc.)
            SELECT
                fc.facility_id,
                f.facility_name,
                f.city,
                f.state,
                f.zip,
                f.county,
                f.corporate_group,
                NULL AS emr,
                NULL AS emr_other,
                NULL AS pt_emr,
                NULL AS orders,
                NULL AS orders_other,
                NULL AS outpatient_pt,
                NULL AS short_beds,
                NULL AS ltc_beds,
                NULL AS avg_dcs,
                'contact'   AS item_type,
                fc.type     AS item_subtype,
                fc.name     AS display_name,
                TRIM(
                    COALESCE(fc.phone, '') ||
                    CASE
                        WHEN fc.phone IS NOT NULL AND fc.phone <> '' AND fc.email IS NOT NULL AND fc.email <> ''
                            THEN ' | '
                        ELSE ''
                    END ||
                    COALESCE(fc.email, '')
                ) AS details
            FROM facility_contacts fc
            JOIN facilities f
                ON f.facility_id = fc.facility_id;
            '''
        )
    except sqlite3.Error as e:
        # Don't break app startup if the view can't be created (e.g. missing tables in a fresh DB)
        print("[init_db] Warning: could not create v_facility_listables view:", e)

    conn.commit()
    conn.close()



init_db()


class ManualCmNoteIn(BaseModel):
    patient_mrn: str
    patient_name: Optional[str] = None
    dob: Optional[str] = None

    visit_id: str
    encounter_id: Optional[str] = None
    hospital_name: Optional[str] = None
    unit_name: Optional[str] = None
    admission_date: Optional[str] = None
    admit_date: Optional[str] = None
    attending: Optional[str] = None

    note_datetime: str
    note_author: Optional[str] = None
    note_type: Optional[str] = "Case Management"
    note_text: str

    source_system: Optional[str] = "EMR"
    pad_run_id: Optional[str] = None
    ocr_confidence: Optional[float] = None

class ManualHospitalDocumentIn(BaseModel):
    # must match /api/hospital-documents/ingest payload keys
    hospital_name: str
    document_type: str
    source_text: str

    # recommended for Hospital Discharge UI
    visit_id: Optional[str] = None

    # optional metadata (same as ingest supports)
    document_datetime: Optional[str] = None
    patient_mrn: Optional[str] = None
    patient_name: Optional[str] = None
    dob: Optional[str] = None
    admit_date: Optional[str] = None
    dc_date: Optional[str] = None
    
    attending: Optional[str] = None
    pcp: Optional[str] = None
    insurance: Optional[str] = None
    
    source_system: Optional[str] = "EMR"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str
    top_k: int = 4
    section_hint: Optional[str] = None


class AskResponse(BaseModel):
    answer: str
    supporting_points: List[str]
    sources: List[str]
    section_used: str
    log_id: Optional[int]


# ---------------------------------------------------------------------------
# PAD → CM notes ingest endpoint
# ---------------------------------------------------------------------------

@app.post("/api/pad/cm-notes/bulk")
async def pad_cm_notes_bulk(
    request: Request,
):
    """
    Bulk ingest of OCR'd Case Management notes from PAD.

    Accepts a few JSON shapes to be robust to PAD quirks:

      1) Raw array:
         [ { patient_mrn, note_datetime, note_text, ... }, ... ]

      2) Wrapped from PAD DataTable:
         { "DataTable": [ { patient_mrn, note_datetime, note_text, ... }, ... ] }

    We manually read and parse the request body instead of letting FastAPI
    auto-parse JSON, so that even slightly odd wire formats from PAD
    can still be handled.
    """
    require_pad_api_key(request)

    # Read raw body bytes from the request
    raw_bytes = await request.body()
    raw_text = raw_bytes.decode("utf-8", errors="ignore").strip()

    if not raw_text:
        raise HTTPException(status_code=400, detail="Empty request body")

    # TEMP: uncomment this to see exactly what PAD is sending in your Render logs
    print("[pad_cm_notes_bulk] raw body:", raw_text[:500])

    # First try: parse the whole body as JSON
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        # Fallback: try to extract the first {...} block and parse that
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise HTTPException(
                status_code=400,
                detail="Could not parse request body as JSON (no JSON object found)",
            )
        try:
            payload = json.loads(raw_text[start : end + 1])
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Could not parse request body as JSON: {e.msg}",
            )

    # Allow both raw list and {"DataTable": [...]} wrapper from PAD
    if isinstance(payload, dict) and "DataTable" in payload:
        notes = payload.get("DataTable") or []
    else:
        notes = payload

    if not isinstance(notes, list) or not notes:
        raise HTTPException(
            status_code=400,
            detail="Request body must be a non-empty JSON array or an object with a non-empty 'DataTable' array",
        )

    conn = get_db()
    cur = conn.cursor()

    received_at = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    # Best-effort: take a pad_run_id from the first row (if present)
    pad_run_id = None
    try:
        if isinstance(notes, list) and notes and isinstance(notes[0], dict):
            pad_run_id = notes[0].get("pad_run_id")
    except Exception:
        pad_run_id = None

    cur.execute(
        """
        INSERT INTO pad_api_runs (endpoint, received_at, pad_run_id, rows_received, status, message)
        VALUES (?, ?, ?, ?, 'ok', 'received')
        """,
        (request.url.path, received_at, pad_run_id, len(notes)),
    )
    run_log_id = cur.lastrowid
    conn.commit()

    inserted = 0
    skipped = 0
    errors: List[Dict[str, Any]] = []
    touched_visit_ids: set[str] = set()

    try:
        for idx, row in enumerate(notes):
            # Defensive: ensure row is a dict
            if not isinstance(row, dict):
                skipped += 1
                errors.append({"index": idx, "error": "row is not an object"})
                continue

            # Normalize incoming fields from PAD – they might be numbers, etc.
            raw_patient_mrn = row.get("patient_mrn")
            raw_note_datetime = row.get("note_datetime")
            raw_visit_id = row.get("visit_id")

            # Convert MRN + datetime to strings safely
            patient_mrn = str(raw_patient_mrn or "").strip()
            note_datetime = str(raw_note_datetime or "").strip()
            note_text = row.get("note_text") or ""

            # visit_id might come in as a float like 1234564.0 from PAD DataTable
            if isinstance(raw_visit_id, (int, float)):
                # If it's 1234564.0, store "1234564"
                if isinstance(raw_visit_id, float) and raw_visit_id.is_integer():
                    visit_id = str(int(raw_visit_id))
                else:
                    visit_id = str(raw_visit_id)
            else:
                visit_id = str(raw_visit_id or "").strip()

            # Every time PAD sends this visit_id, treat it as "still active" today
            if visit_id:
                cur.execute(
                    "UPDATE snf_admissions SET last_seen_active_date = date('now') WHERE visit_id = ?",
                    (visit_id,),
                )

            if not patient_mrn or not note_datetime or not note_text:
                skipped += 1
                errors.append(
                    {
                        "index": idx,
                        "error": "patient_mrn, note_datetime, and note_text are required",
                    }
                )
                continue

            note_hash = compute_note_hash(patient_mrn, note_datetime, note_text)

            # Skip if we've already seen this hash
            cur.execute(
                "SELECT 1 FROM cm_notes_raw WHERE note_hash = ?",
                (note_hash,),
            )
            if cur.fetchone():
                skipped += 1
                continue

            cur.execute(
                """
                INSERT INTO cm_notes_raw (
                    patient_mrn,
                    patient_name,
                    dob,
                    visit_id,
                    encounter_id,
                    hospital_name,
                    unit_name,
                    admission_date,
                    admit_date,
                    attending,
                    note_datetime,
                    note_author,
                    note_type,
                    note_text,
                    source_system,
                    pad_run_id,
                    ocr_confidence,
                    note_hash,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                """,
                (
                    patient_mrn,
                    row.get("patient_name"),
                    row.get("dob"),
                    visit_id or None,
                    row.get("encounter_id"),
                    row.get("hospital_name"),
                    row.get("unit_name"),
                    row.get("admission_date"),
                    row.get("admit_date") or row.get("admission_date"),
                    row.get("attending") or row.get("hospitalist") or row.get("note_author"),
                    note_datetime,
                    row.get("note_author"),
                    row.get("note_type") or "Case Management",
                    note_text,
                    row.get("source_system") or "EMR",
                    row.get("pad_run_id"),
                    row.get("ocr_confidence"),
                    note_hash,
                ),
            )

            inserted += 1

            conn.commit()

            if visit_id:
                touched_visit_ids.add(str(visit_id))


        # Finalize run log USING THE SAME CONNECTION (avoids extra lock contention)
        try:
            cur.execute(
                """
                UPDATE pad_api_runs
                SET inserted = ?, skipped = ?, error_count = ?, status = ?, message = ?
                WHERE id = ?
                """,
                (inserted, skipped, len(errors), "ok", "completed", run_log_id),
            )
            conn.commit()
        except Exception as e:
            print("[pad_cm_notes_bulk] failed to update pad_api_runs:", e)

    except Exception as e:
        # Mark run as error (best-effort)
        try:
            cur.execute(
                """
                UPDATE pad_api_runs
                SET inserted = ?, skipped = ?, error_count = ?, status = ?, message = ?
                WHERE id = ?
                """,
                (inserted, skipped, len(errors) + 1, "error", f"failed: {type(e).__name__}", run_log_id),
            )
            conn.commit()
        except Exception as e2:
            print("[pad_cm_notes_bulk] failed to mark run as error:", e2)
        raise

    finally:
        conn.close()

    # Kick off SNF recompute ONLY for the visit_ids we just received.
    # This still uses ALL historical notes for that visit_id (snf_recompute_for_admission queries cm_notes_raw by visit_id).
    try:
        visit_ids_to_process = sorted(touched_visit_ids)

        def _process_visits(v_ids: list[str]):
            for vid in v_ids:
                try:
                    snf_recompute_for_admission(visit_id=vid)
                except Exception as e:
                    print(f"[pad_cm_notes_bulk] snf_recompute_for_admission failed visit_id={vid}:", e)

        if visit_ids_to_process:
            threading.Thread(
                target=_process_visits,
                args=(visit_ids_to_process,),
                daemon=True,
            ).start()

    except Exception as e:
        print("[pad_cm_notes_bulk] failed to trigger SNF recompute:", e)

    return {
        "ok": True,
        "inserted": inserted,
        "skipped": skipped,
        "errors": errors,
        "run_log_id": run_log_id,
        "received_at": received_at,
    }


# ---------------------------------------------------------------------------
# Hospital Document ingest (raw text -> document + sections) ✅ NEW
# ---------------------------------------------------------------------------


# PAD → Hospital documents bulk ingest endpoint  ✅ NEW
@app.post("/api/pad/hospital-documents/bulk")
async def pad_hospital_documents_bulk(request: Request):
    """
    Bulk ingest of Hospital Documents from PAD.

    Accepts:
      1) Raw array: [ { hospital_name, document_type, source_text, ... }, ... ]
      2) PAD DataTable2 wrapper: { "DataTable2": [ {...}, ... ] }
    """
    require_pad_api_key(request)

    raw_bytes = await request.body()
    raw_text = raw_bytes.decode("utf-8", errors="ignore").strip()
    if not raw_text:
        raise HTTPException(status_code=400, detail="Empty request body")

    # TEMP debug
    print("[pad_hospital_documents_bulk] raw body:", raw_text[:500])

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise HTTPException(status_code=400, detail="Could not parse request body as JSON")
        payload = json.loads(raw_text[start : end + 1])

    def _to_clean_id(val) -> Optional[str]:
        # Handles PAD floats like 12345678.0 -> "12345678"
        if val is None:
            return None
        if isinstance(val, (int, float)):
            if isinstance(val, float) and val.is_integer():
                return str(int(val))
            return str(val)
        s = str(val).strip()
        return s or None

    # Accept multiple wrapper keys from PAD
    if isinstance(payload, dict):
        if "DataTable2" in payload:
            docs = payload.get("DataTable2") or []
        elif "discharge_visits" in payload:
            docs = payload.get("discharge_visits") or []
        else:
            docs = payload
    else:
        docs = payload

    if not isinstance(docs, list) or not docs:
        raise HTTPException(
            status_code=400,
            detail="Request body must be a non-empty JSON array or an object with a non-empty 'DataTable2' array",
        )

    conn = get_db()
    inserted = 0
    skipped = 0
    errors: List[Dict[str, Any]] = []

    try:
        for idx, row in enumerate(docs):
            if not isinstance(row, dict):
                skipped += 1
                errors.append({"index": idx, "error": "row is not an object"})
                continue

            # Required
            hospital_name = str(row.get("hospital_name") or "").strip()
            document_type = str(row.get("document_type") or "").strip()
            source_text = str(row.get("source_text") or "").strip()

            if not hospital_name or not document_type or not source_text:
                skipped += 1
                errors.append(
                    {"index": idx, "error": "hospital_name, document_type, source_text are required"}
                )
                continue

            # Optional fields supported by your existing ingest endpoint
            doc_payload = {
                "hospital_name": hospital_name,
                "document_type": document_type,
                "source_text": source_text,
                "document_datetime": (str(row.get("document_datetime") or "").strip() or None),
                "patient_mrn": _to_clean_id(row.get("patient_mrn")),
                "patient_name": (str(row.get("patient_name") or "").strip() or None),
                "dob": (str(row.get("dob") or "").strip() or None),
                "visit_id": _to_clean_id(row.get("visit_id")),
                "admit_date": (str(row.get("admit_date") or "").strip() or None),
                "dc_date": (str(row.get("dc_date") or "").strip() or None),
                "attending": (str(row.get("attending") or "").strip() or None),
                "pcp": (str(row.get("pcp") or "").strip() or None),
                "insurance": (str(row.get("insurance") or "").strip() or None),
                "source_system": (str(row.get("source_system") or "EMR").strip() or "EMR"),
            }

            # Reuse the same logic as /api/hospital-documents/ingest (inline)
            profile_map = load_extraction_profile(conn, hospital_name, document_type)
            sections = split_document_into_sections(source_text, heading_map_override=profile_map)

            cur = conn.cursor()

            # Dedup (hash + skip)
            doc_hash = compute_hospital_document_hash(
                hospital_name=doc_payload["hospital_name"],
                document_type=doc_payload["document_type"],
                visit_id=doc_payload["visit_id"],
                document_datetime=doc_payload["document_datetime"],
                source_text=doc_payload["source_text"],
            )

            execute_with_retry(cur, "SELECT id FROM hospital_documents WHERE document_hash = ? LIMIT 1", (doc_hash,))
            existing = cur.fetchone()
            if existing:
                # Skip this row; continue to next
                continue

            execute_with_retry(
                cur,
                """
                INSERT INTO hospital_documents (
                    hospital_name, document_type, document_datetime,
                    patient_mrn, patient_name, dob, visit_id,
                    admit_date, dc_date,
                    attending, pcp, insurance,
                    source_text, source_system,
                    document_hash
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc_payload["hospital_name"],
                    doc_payload["document_type"],
                    doc_payload["document_datetime"],
                    doc_payload["patient_mrn"],
                    doc_payload["patient_name"],
                    doc_payload["dob"],
                    doc_payload["visit_id"],
                    doc_payload["admit_date"],
                    doc_payload["dc_date"],
                    doc_payload["attending"],
                    doc_payload["pcp"],
                    doc_payload["insurance"],
                    doc_payload["source_text"],
                    doc_payload["source_system"],
                    doc_hash,
                ),
            )
            document_id = cur.lastrowid

            # Upsert hub row (same behavior as your existing ingest endpoint)
            if doc_payload["visit_id"]:
                cur.execute(
                    """
                    INSERT INTO hospital_discharges (
                        visit_id, patient_mrn, patient_name, hospital_name,
                        admit_date, dc_date,
                        attending, pcp, insurance,
                        updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                    ON CONFLICT(visit_id) DO UPDATE SET
                        patient_mrn   = COALESCE(excluded.patient_mrn, hospital_discharges.patient_mrn),
                        patient_name  = COALESCE(excluded.patient_name, hospital_discharges.patient_name),
                        hospital_name = COALESCE(excluded.hospital_name, hospital_discharges.hospital_name),
                        admit_date    = COALESCE(excluded.admit_date, hospital_discharges.admit_date),
                        dc_date       = COALESCE(excluded.dc_date, hospital_discharges.dc_date),
                        attending     = COALESCE(excluded.attending, hospital_discharges.attending),
                        pcp           = COALESCE(excluded.pcp, hospital_discharges.pcp),
                        insurance     = COALESCE(excluded.insurance, hospital_discharges.insurance),
                        updated_at    = datetime('now')
                    """,
                    (
                        doc_payload["visit_id"],
                        doc_payload["patient_mrn"],
                        doc_payload["patient_name"],
                        doc_payload["hospital_name"],
                        doc_payload["admit_date"],
                        doc_payload["dc_date"],
                        doc_payload["attending"],
                        doc_payload["pcp"],
                        doc_payload["insurance"],
                    ),
                )

                # ✅ UPDATED: derive disposition + dc_agency, but only let the "latest doc" win
                dispo = analyze_discharge_disposition_with_llm(doc_payload["source_text"])
                new_dispo = (dispo.get("disposition") if dispo else None)
                new_agency = (dispo.get("dc_agency") if dispo else None)

                # Determine this document's effective timestamp:
                # 1) document_datetime (from PAD) if present
                # 2) else fallback to hospital_documents.created_at
                cur.execute(
                    "SELECT document_datetime, created_at FROM hospital_documents WHERE id = ?",
                    (document_id,),
                )
                _doc_row = cur.fetchone() or {}
                eff_dt = normalize_sortable_dt((_doc_row.get("document_datetime") if isinstance(_doc_row, dict) else _doc_row[0]) or None)
                if not eff_dt:
                    # fallback to created_at
                    created_at_val = (_doc_row.get("created_at") if isinstance(_doc_row, dict) else _doc_row[1]) if _doc_row else None
                    eff_dt = normalize_sortable_dt(created_at_val)

                # Fetch current "winning" disposition source
                cur.execute(
                    "SELECT disposition, dispo_source_dt, dispo_source_doc_id FROM hospital_discharges WHERE visit_id = ?",
                    (doc_payload["visit_id"],),
                )
                curr = cur.fetchone()

                curr_source_dt = None
                curr_source_doc_id = None
                curr_dispo = None
                if curr:
                    # sqlite Row supports dict-like access
                    try:
                        curr_dispo = curr["disposition"]
                        curr_source_dt = curr["dispo_source_dt"]
                        curr_source_doc_id = curr["dispo_source_doc_id"]
                    except Exception:
                        curr_dispo = curr[0]
                        curr_source_dt = curr[1]
                        curr_source_doc_id = curr[2]

                curr_source_dt = normalize_sortable_dt(curr_source_dt)

                # Treat these as "no disposition mentioned" → do NOT overwrite an existing good value
                NO_DISPO_VALUES = {None, "", "unknown", "n/a", "na"}

                has_meaningful_dispo = (new_dispo is not None and str(new_dispo).strip().lower() not in NO_DISPO_VALUES)

                # Compare timestamps (ISO strings sort correctly)
                is_newer = False
                if eff_dt and not curr_source_dt:
                    is_newer = True
                elif eff_dt and curr_source_dt:
                    if eff_dt > curr_source_dt:
                        is_newer = True
                    elif eff_dt == curr_source_dt:
                        # tie-breaker: higher doc id wins
                        try:
                            is_newer = int(document_id) > int(curr_source_doc_id or 0)
                        except Exception:
                            is_newer = False

                # Update rules:
                # - If we DON'T have a meaningful new dispo, keep existing
                # - If we DO have a meaningful new dispo, only apply if this doc is newer
                if has_meaningful_dispo and is_newer:
                    cur.execute(
                        """
                        UPDATE hospital_discharges
                        SET disposition = ?,
                            dc_agency = ?,
                            dispo_source_dt = ?,
                            dispo_source_doc_id = ?,
                            updated_at = datetime('now')
                        WHERE visit_id = ?
                        """,
                        (
                            new_dispo,
                            (new_agency or None),
                            eff_dt,
                            int(document_id),
                            doc_payload["visit_id"],
                        ),
                    )


            # Store sections
            for i, s in enumerate(sections, start=1):
                execute_with_retry(
                    cur,
                    """
                    INSERT INTO hospital_document_sections (
                        document_id, section_key, section_title, section_order, section_text
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        document_id,
                        s["section_key"],
                        s.get("section_title"),
                        i,
                        s["section_text"],
                    ),
                )

            inserted += 1

            # ✅ Commit per document so we don't hold the write lock for the entire batch
            commit_with_retry(conn)

        # (no final conn.commit() here anymore)
    finally:
        conn.close()

    return {"ok": True, "inserted": inserted, "skipped": skipped, "errors": errors}



@app.post("/api/hospital-documents/ingest")
async def hospital_documents_ingest(payload: Dict[str, Any] = Body(...)):
    """
    Ingest ONE hospital document (Discharge Summary, MD Progress Note, etc.)
    as raw text, then auto-split into sections and store both.
    """
    hospital_name = (payload.get("hospital_name") or "").strip()
    document_type = (payload.get("document_type") or "").strip()
    source_text = (payload.get("source_text") or "").strip()

    if not hospital_name:
        raise HTTPException(status_code=400, detail="hospital_name is required")
    if not document_type:
        raise HTTPException(status_code=400, detail="document_type is required")
    if not source_text:
        raise HTTPException(status_code=400, detail="source_text is required")

    document_datetime = (payload.get("document_datetime") or "").strip() or None

    patient_mrn = (payload.get("patient_mrn") or "").strip() or None
    patient_name = (payload.get("patient_name") or "").strip() or None
    dob = (payload.get("dob") or "").strip() or None
    visit_id = (payload.get("visit_id") or "").strip() or None
    attending = (payload.get("attending") or "").strip() or None
    pcp = (payload.get("pcp") or "").strip() or None
    insurance = (payload.get("insurance") or "").strip() or None


    # ✅ NEW: admit/discharge dates (optional; PAD/API can send later)
    admit_date_raw = (payload.get("admit_date") or "").strip() or None
    dc_date_raw = (payload.get("dc_date") or "").strip() or None

    # Normalize to ISO so reports using date() work reliably
    admit_date = normalize_date_to_iso(admit_date_raw)
    dc_date = normalize_date_to_iso(dc_date_raw)

    source_system = (payload.get("source_system") or "EMR").strip() or "EMR"

    conn = get_db()
    try:
        profile_map = load_extraction_profile(conn, hospital_name, document_type)
        sections = split_document_into_sections(source_text, heading_map_override=profile_map)

        cur = conn.cursor()

        # 1) Dedup (hash + skip) like cm_notes_raw.note_hash
        doc_hash = compute_hospital_document_hash(
            hospital_name=hospital_name,
            document_type=document_type,
            visit_id=visit_id,
            document_datetime=document_datetime,
            source_text=source_text,
        )

        cur.execute("SELECT id FROM hospital_documents WHERE document_hash = ? LIMIT 1", (doc_hash,))
        existing = cur.fetchone()
        if existing:
            # Duplicate document already ingested — skip everything downstream
            existing_id = existing["id"] if isinstance(existing, sqlite3.Row) else existing[0]
            return {"ok": True, "skipped": True, "message": "Duplicate document (same hash) already exists.", "document_id": existing_id}

        # 2) Insert the document row
        cur.execute(
            """
            INSERT INTO hospital_documents (
                hospital_name, document_type, document_datetime,
                patient_mrn, patient_name, dob, visit_id,
                admit_date, dc_date,
                attending, pcp, insurance,
                source_text, source_system,
                document_hash
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                hospital_name,
                document_type,
                document_datetime,
                patient_mrn,
                patient_name,
                dob,
                visit_id,
                admit_date,
                dc_date,
                attending,
                pcp,
                insurance,
                source_text,
                source_system,
                doc_hash,
            ),
        )

        document_id = cur.lastrowid

        
        # ✅ NEW: upsert the discharge "hub" row by visit_id (if provided)
        if visit_id:
            cur.execute(
                """
                INSERT INTO hospital_discharges (
                    visit_id, patient_mrn, patient_name, hospital_name,
                    admit_date, dc_date,
                    attending, pcp, insurance,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                ON CONFLICT(visit_id) DO UPDATE SET
                    patient_mrn   = COALESCE(excluded.patient_mrn, hospital_discharges.patient_mrn),
                    patient_name  = COALESCE(excluded.patient_name, hospital_discharges.patient_name),
                    hospital_name = COALESCE(excluded.hospital_name, hospital_discharges.hospital_name),
                    admit_date    = COALESCE(excluded.admit_date, hospital_discharges.admit_date),
                    dc_date       = COALESCE(excluded.dc_date, hospital_discharges.dc_date),
                    attending     = COALESCE(excluded.attending, hospital_discharges.attending),
                    pcp           = COALESCE(excluded.pcp, hospital_discharges.pcp),
                    insurance     = COALESCE(excluded.insurance, hospital_discharges.insurance),
                    updated_at    = datetime('now')
                """
                ,
                (visit_id, patient_mrn, patient_name, hospital_name, admit_date, dc_date, attending, pcp, insurance),
            )

            # ✅ UPDATED: derive disposition + dc_agency, but only let the "latest doc" win
            dispo = analyze_discharge_disposition_with_llm(source_text)
            new_dispo = (dispo.get("disposition") if dispo else None)
            new_agency = (dispo.get("dc_agency") if dispo else None)

            # Determine this document's effective timestamp:
            # 1) document_datetime (from payload) if present
            # 2) else fallback to hospital_documents.created_at
            eff_dt = normalize_sortable_dt(document_datetime)

            if not eff_dt:
                cur.execute(
                    "SELECT created_at FROM hospital_documents WHERE id = ?",
                    (document_id,),
                )
                row = cur.fetchone()
                created_at_val = None
                if row:
                    try:
                        created_at_val = row["created_at"]
                    except Exception:
                        created_at_val = row[0]
                eff_dt = normalize_sortable_dt(created_at_val)

            # Fetch current "winning" disposition source for this visit_id
            cur.execute(
                "SELECT disposition, dispo_source_dt, dispo_source_doc_id FROM hospital_discharges WHERE visit_id = ?",
                (visit_id,),
            )
            curr = cur.fetchone()

            curr_source_dt = None
            curr_source_doc_id = None
            curr_dispo = None
            if curr:
                try:
                    curr_dispo = curr["disposition"]
                    curr_source_dt = curr["dispo_source_dt"]
                    curr_source_doc_id = curr["dispo_source_doc_id"]
                except Exception:
                    curr_dispo = curr[0]
                    curr_source_dt = curr[1]
                    curr_source_doc_id = curr[2]

            curr_source_dt = normalize_sortable_dt(curr_source_dt)

            # Treat these as "no disposition mentioned" → do NOT overwrite an existing good value
            NO_DISPO_VALUES = {None, "", "unknown", "n/a", "na"}

            has_meaningful_dispo = (
                new_dispo is not None and str(new_dispo).strip().lower() not in NO_DISPO_VALUES
            )

            # Compare timestamps (ISO strings sort correctly)
            is_newer = False
            if eff_dt and not curr_source_dt:
                is_newer = True
            elif eff_dt and curr_source_dt:
                if eff_dt > curr_source_dt:
                    is_newer = True
                elif eff_dt == curr_source_dt:
                    # tie-breaker: higher doc id wins
                    try:
                        is_newer = int(document_id) > int(curr_source_doc_id or 0)
                    except Exception:
                        is_newer = False

            # Update rules:
            # - If we DON'T have a meaningful new dispo, keep existing
            # - If we DO have a meaningful new dispo, only apply if this doc is newer
            if has_meaningful_dispo and is_newer:
                cur.execute(
                    """
                    UPDATE hospital_discharges
                    SET disposition = ?,
                        dc_agency = ?,
                        dispo_source_dt = ?,
                        dispo_source_doc_id = ?,
                        updated_at = datetime('now')
                    WHERE visit_id = ?
                    """,
                    (
                        new_dispo,
                        (new_agency or None),
                        eff_dt,
                        int(document_id),
                        visit_id,
                    ),
                )


        # 2) Insert each section row
        for s in sections:
            cur.execute(
                """
                INSERT INTO hospital_document_sections (
                    document_id, section_key, section_title, section_order, section_text
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    s["section_key"],
                    s.get("section_title"),
                    int(s["order"]),
                    s["section_text"],
                ),
            )

        conn.commit()

        return {
            "ok": True,
            "document_id": document_id,
            "sections_saved": len(sections),
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Dictionary & Facility helpers
# ---------------------------------------------------------------------------

def load_dictionary_maps(conn: sqlite3.Connection) -> Dict[str, Dict[str, Any]]:
    """
    Load basic maps for:
      - abbreviations  (kind='abbr')
      - synonyms       (kind='synonym')
      - facility_alias (kind='facility_alias', maps alias -> canonical name)
    Uses the existing dictionary table: (key, canonical, kind, notes, match_mode).
    """
    cur = conn.cursor()
    try:
        cur.execute("SELECT key, canonical, kind FROM dictionary")
        rows = cur.fetchall()
    except sqlite3.Error:
        return {"abbrev": {}, "synonym": {}, "facility_aliases": {}}

    abbrev_map: Dict[str, str] = {}
    synonym_map: Dict[str, str] = {}
    facility_aliases: Dict[str, str] = {}

    for r in rows:
        kind = (r["kind"] or "").strip().lower()
        key = (r["key"] or "").strip().lower()
        val = (r["canonical"] or "").strip()
        if not key or not val:
            continue

        if kind == "abbr":
            abbrev_map[key] = val
        elif kind in ("synonym", "alias"):
            # Treat plain "alias" rows as synonyms for normalization
            synonym_map[key] = val
        elif kind == "facility_alias":
            # Here we map alias text -> canonical facility *name*.
            facility_aliases[key] = val


    return {
        "abbrev": abbrev_map,
        "synonym": synonym_map,
        "facility_aliases": facility_aliases,
    }



def clear_dictionary_caches():
    """
    Placeholder cache clearer for dictionary lookups.
    If you later wrap load_dictionary_maps in @lru_cache, call cache_clear() here.
    """
    # Right now load_dictionary_maps is not cached; this is a no-op.
    # Left here for future optimization hooks.
    pass



def normalize_question_text(
    question: str,
    abbrev_map: Dict[str, str],
    synonym_map: Dict[str, str],
) -> str:
    """
    Normalize the user question so DB lookups and embeddings work better.

    Steps:
      - basic cleanup (lowercase, collapse whitespace)
      - remove common "fluff" prefixes ("can you please", "who is", etc.)
      - expand abbreviations from dictionary (kind='abbr')
      - apply synonyms from dictionary (kind='synonym')

    NOTE: This is only used for retrieval (search/embeddings), not for logging the
    original question back to the user.
    """
    if not question:
        return question

    text = question

    # Normalize whitespace / newlines
    text = re.sub(r"[\r\n\t]+", " ", text)

    # Lowercase for matching; we don't need case for retrieval
    text = text.lower().strip()

    # Remove very common polite prefixes / wrappers
    # ("can you please", "could you", "i was wondering", "what is", "who is", etc.)
    fluff_prefixes = [
        r"^(can you please|can you|could you please|could you|would you please|would you)\s+",
        r"^(please|hey|hi|hello)\s+",
        r"^(i was wondering( if)?|i wonder if)\s+",
        r"^(do you know|do we know)\s+",
        r"^(what is|what's|who is|who's|tell me|give me|show me)\s+",
    ]
    for pat in fluff_prefixes:
        text = re.sub(pat, "", text).strip()

    # Expand abbreviations (dictionary kind='abbr')
    # e.g. "evolv" -> "evolv health", "nsph ml" -> "nspire miami lakes" (if you set it)
    for abbr, full in abbrev_map.items():
        abbr_norm = abbr.lower().strip()
        full_norm = full.lower().strip()
        if not abbr_norm or not full_norm:
            continue

        pattern_abbr = rf"\b{re.escape(abbr_norm)}\b"
        text = re.sub(pattern_abbr, full_norm, text)

    # Apply synonyms (dictionary kind='synonym')
    # e.g. "ur" -> "utilization review", "hh" -> "home health"
    for src, canon in synonym_map.items():
        src_norm = src.lower().strip()
        canon_norm = canon.lower().strip()
        if not src_norm or not canon_norm:
            continue

        pattern_syn = rf"\b{re.escape(src_norm)}\b"
        text = re.sub(pattern_syn, canon_norm, text)

    # Collapse any extra spaces again
    text = re.sub(r"\s+", " ", text).strip()

    return text



def _normalize_for_facility_match(text: str) -> str:
    """
    Soft-normalize text for facility matching:
    - lowercase
    - remove punctuation / special chars
    - collapse whitespace
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text



def detect_facility_id(
    question: str, conn: sqlite3.Connection, facility_aliases: Dict[str, str]
) -> Optional[str]:
    """
    Try to infer the facility id from the question using:
      1) dictionary facility aliases (kind='facility_alias')
      2) direct facility_id matches (e.g. 'nsph-ml' in the text)
      3) facility_name from the facilities table

    Uses a soft-normalized text form so small punctuation differences
    ("St. Mary's" vs "st marys") don't break matching.
    """
    q_norm = _normalize_for_facility_match(question)

    # 1) Explicit aliases from dictionary (e.g. "nspire ml" -> "nsph-ml")
    for alias_text, fid in facility_aliases.items():
        if not alias_text or not fid:
            continue
        alias_norm = _normalize_for_facility_match(alias_text)
        if alias_norm and alias_norm in q_norm:
            return fid

    # 2) Direct facility_id mention in the question (e.g. "nsph-ml")
    try:
        cur = conn.cursor()
        cur.execute("SELECT facility_id FROM facilities")
        for row in cur.fetchall():
            fid = (row["facility_id"] or "").strip()
            if not fid:
                continue
            fid_norm = _normalize_for_facility_match(fid)
            if fid_norm and fid_norm in q_norm:
                return fid
    except sqlite3.Error:
        pass

    # 3) Substring match on facility_name
    try:
        cur = conn.cursor()
        cur.execute("SELECT facility_id, facility_name FROM facilities")
        for row in cur.fetchall():
            name = (row["facility_name"] or "").strip()
            if not name:
                continue
            name_norm = _normalize_for_facility_match(name)
            if name_norm and name_norm in q_norm:
                return row["facility_id"]
    except sqlite3.Error:
        return None

    return None


def map_snf_name_to_facility_id(
    snf_name: Optional[str],
    conn: sqlite3.Connection,
    facility_aliases: Dict[str, str],
) -> Optional[str]:
    """
    Map a free-text SNF name from notes (e.g. 'The Peaks SNF', 'Pks')
    to a facility_id using:
      1) dictionary facility_alias entries (kind='facility_alias')
      2) direct facility_name matches in facilities

    Returns facility_id or None if no good match.
    """
    if not snf_name:
        return None

    name_norm = _normalize_for_facility_match(snf_name)

    # 1) Try explicit aliases from dictionary (alias -> facility_id)
    for alias_text, fid in facility_aliases.items():
        if not alias_text or not fid:
            continue
        alias_norm = _normalize_for_facility_match(alias_text)
        if not alias_norm:
            continue

        # If either string contains the other, treat as a match
        if alias_norm in name_norm or name_norm in alias_norm:
            return fid

    # 2) Try matching facility_name from facilities table
    try:
        cur = conn.cursor()
        cur.execute("SELECT facility_id, facility_name FROM facilities")
        for row in cur.fetchall():
            fac_name = (row["facility_name"] or "").strip()
            if not fac_name:
                continue
            fac_norm = _normalize_for_facility_match(fac_name)
            if not fac_norm:
                continue
            if fac_norm in name_norm or name_norm in fac_norm:
                return row["facility_id"]
    except sqlite3.Error:
        return None

    return None



def get_facility_label(fid: Optional[str], conn: sqlite3.Connection) -> str:
    if not fid:
        return ""
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT facility_name, city, state FROM facilities WHERE facility_id=?",
            (fid,),
        )
        row = cur.fetchone()
        if not row:
            return ""
        name = row["facility_name"] or ""
        city = row["city"] or ""
        state = row["state"] or ""
        loc = ", ".join(p for p in (city, state) if p)
        return f"{name} ({loc})" if loc else name
    except sqlite3.Error:
        return ""



def _dictionary_rows_for_admin(conn: sqlite3.Connection) -> List[Dict[str, str]]:
    """
    Return dictionary rows in the shape expected by the Admin Dictionary tab:
      { key, canonical, kind, notes, match_mode }
    Backed by the existing dictionary schema:
      key, canonical, kind, notes, match_mode
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT key, canonical, kind, notes,
               COALESCE(NULLIF(match_mode, ''), 'exact') AS match_mode
        FROM dictionary
        ORDER BY LOWER(key)
        """
    )
    items: List[Dict[str, str]] = []
    for r in cur.fetchall():
        items.append(
            {
                "key": r["key"] or "",
                "canonical": r["canonical"] or "",
                "kind": r["kind"] or "abbr",
                "notes": r["notes"] or "",
                "match_mode": r["match_mode"] or "exact",
            }
        )
    return items



@app.get("/admin/dictionary/list")
async def admin_dictionary_list(request: Request, q: str = Query(default="")):
    """
    List dictionary rows for the Dictionary tab.
    Supports simple text filtering on key/canonical/kind/notes.
    """
    require_admin(request)
    conn = get_db()
    try:
        rows = _dictionary_rows_for_admin(conn)
        ql = (q or "").strip().lower()
        if ql:
            out = []
            for r in rows:
                haystack = " ".join(
                    [
                        r["key"].lower(),
                        r["canonical"].lower(),
                        r["kind"].lower(),
                        r["notes"].lower(),
                    ]
                )
                if ql in haystack:
                    out.append(r)
            rows = out
        return {"ok": True, "items": rows}
    finally:
        conn.close()


@app.post("/admin/dictionary/upsert")
async def admin_dictionary_upsert(
    request: Request,
    key: str = Form(...),
    canonical: str = Form(...),
    kind: str = Form("abbr"),
    notes: str = Form(""),
    match_mode: str = Form("exact"),
):
    """
    Insert or update a dictionary row using the existing schema:
      key PRIMARY KEY, canonical, kind, notes, match_mode
    """
    require_admin(request)
    k = (key or "").strip()
    v = (canonical or "").strip()
    kd = (kind or "abbr").strip()
    nt = (notes or "").strip()
    mm = (match_mode or "exact").strip() or "exact"

    if not k or not v:
        raise HTTPException(status_code=400, detail="key and canonical are required")

    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO dictionary (key, canonical, kind, notes, match_mode)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                canonical = excluded.canonical,
                kind      = excluded.kind,
                notes     = excluded.notes,
                match_mode= excluded.match_mode
            """,
            (k, v, kd, nt, mm),
        )
        conn.commit()
        clear_dictionary_caches()
        return {"ok": True}
    finally:
        conn.close()



@app.post("/admin/dictionary/delete")
async def admin_dictionary_delete(
    request: Request,
    key: str = Form(...),
):
    """
    Delete dictionary rows for this key.
    Uses the existing schema with key as PRIMARY KEY.
    """
    require_admin(request)
    k = (key or "").strip()
    if not k:
        raise HTTPException(status_code=400, detail="key is required")

    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM dictionary WHERE key = ?", (k,))
        conn.commit()
        clear_dictionary_caches()
        return {"ok": True}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Retrieval from DB (Q&A + Facility knowledge)
# ---------------------------------------------------------------------------

def search_qa_candidates(
    conn: sqlite3.Connection, question: str, section_hint: Optional[str], limit: int = 40
):
    kw = question.strip()
    if len(kw) > 120:
        kw = kw[:120]
    if not kw:
        return []

    like = f"%{kw}%"
    params = [like, like, like, like, like]
    sql = """
        SELECT id, section, question, answer, tags, topics
        FROM qa
        WHERE question LIKE ?
           OR answer  LIKE ?
           OR topics  LIKE ?
           OR section LIKE ?
           OR tags    LIKE ?
    """
    cur = conn.cursor()
    if section_hint:
        sql += " ORDER BY (topics LIKE ?) DESC, id DESC LIMIT ?"
        params.append(f"%{section_hint}%")
        params.append(limit)
    else:
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

    cur.execute(sql, params)
    return cur.fetchall()


def fetch_facility_knowledge(
    conn: sqlite3.Connection, facility_id: Optional[str], limit: int = 40
):
    """
    Gather facility-specific text snippets for retrieval:
      - fac_facts rows
      - facility_contacts (Administrator, UR, DON, etc.)
      - v_facility_knowledge view (services, partners, insurance, extra facts)
      - v_facility_summary (EMR, beds, avg dcs, etc.)
      - facilities table profile (legal name, corporate group, address, etc.)
      - client_info rows (key/value client metadata)
    """
    if not facility_id:
        return []

    snippets: List[Dict[str, Any]] = []
    cur = conn.cursor()

    # 1) fac_facts (explicit snippets)
    try:
        cur.execute(
            """
            SELECT fact_text, tags
            FROM fac_facts
            WHERE facility_id=?
            ORDER BY id DESC
            LIMIT ?
            """,
            (facility_id, limit),
        )
        for r in cur.fetchall():
            snippets.append(
                {
                    "text": r["fact_text"],
                    "source": "DB:facility_facts:fact",
                    "tags": r["tags"] or "",
                    "weight": 1.0,
                }
            )
    except sqlite3.Error:
        pass

    # 2) facility_contacts (Administrator, UR, DON, therapy, etc.)
    try:
        cur.execute(
            """
            SELECT type, name, email, phone, pref
            FROM facility_contacts
            WHERE facility_id = ?
            ORDER BY id ASC
            LIMIT ?
            """,
            (facility_id, limit),
        )
        for r in cur.fetchall():
            role = (r["type"] or "").strip()
            name = (r["name"] or "").strip()
            email = (r["email"] or "").strip()
            phone = (r["phone"] or "").strip()
            pref = (r["pref"] or "").strip()

            # Need at least role + name to be useful
            if not (role and name):
                continue

            parts = [f"{role}: {name}"]

            contact_bits = []
            if phone:
                contact_bits.append(f"phone {phone}")
            if email:
                contact_bits.append(f"email {email}")
            if contact_bits:
                parts.append("(" + ", ".join(contact_bits) + ")")

            if pref:
                parts.append(f"[prefers: {pref}]")

            text = " ".join(parts)

            snippets.append(
                {
                    "text": text,
                    "source": f"DB:facility_contacts:{role}",
                    "tags": "contacts",
                    # Boosted so "who is the administrator..." sees this first
                    "weight": 2.0,
                }
            )
    except sqlite3.Error:
        pass

    # 3) v_facility_knowledge (services, partners, insurance, extra facts)
    try:
        cur.execute(
            """
            SELECT kind, text, tags
            FROM v_facility_knowledge
            WHERE facility_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (facility_id, limit),
        )
        for r in cur.fetchall():
            kind = (r["kind"] or "data")
            tags = r["tags"] or ""
            # Slightly boost explicit "fact" rows if you want
            base_weight = 1.2 if kind == "fact" else 1.0
            snippets.append(
                {
                    "text": r["text"],
                    "source": f"DB:facility_{kind}",
                    "tags": tags,
                    "weight": base_weight,
                }
            )
    except sqlite3.Error:
        pass

    # 4) v_facility_summary (headline EMR, beds, avg discharges, etc.)
    try:
        cur.execute(
            """
            SELECT facility_name, city, state, zip,
                   emr, orders, orders_other,
                   outpatient_pt, short_beds, ltc_beds, avg_dcs
            FROM v_facility_summary
            WHERE facility_id = ?
            """,
            (facility_id,),
        )
        row = cur.fetchone()
        if row:
            parts: List[str] = []

            if row["emr"]:
                parts.append(f"EMR: {row['emr']}")

            orders_desc = " / ".join(
                p for p in [row["orders"], row["orders_other"]] if p
            )
            if orders_desc:
                parts.append(f"Orders: {orders_desc}")

            if row["outpatient_pt"]:
                parts.append(f"Outpatient PT: {row['outpatient_pt']}")

            beds: List[str] = []
            if row["short_beds"]:
                beds.append(f"short-term beds: {row['short_beds']}")
            if row["ltc_beds"]:
                beds.append(f"LTC beds: {row['ltc_beds']}")
            if beds:
                parts.append("Beds: " + ", ".join(beds))

            if row["avg_dcs"]:
                parts.append(f"Average daily discharges: {row['avg_dcs']}")

            summary_pieces = "; ".join(p for p in parts if p)
            if summary_pieces:
                loc = ", ".join(
                    p for p in [row["city"], row["state"], row["zip"]] if p
                )
                header = row["facility_name"] or ""
                if loc:
                    header = f"{header} ({loc})" if header else loc

                summary_text = (
                    f"{header}: {summary_pieces}" if header else summary_pieces
                )

                snippets.append(
                    {
                        "text": summary_text,
                        "source": "DB:facility_summary",
                        "tags": "summary",
                        # Boosted because this is high-value "profile" info
                        "weight": 2.0,
                    }
                )
    except sqlite3.Error:
        pass

    # 5) facilities table profile (legal name, corporate group, address, etc.)
    try:
        cur.execute(
            """
            SELECT facility_name, legal_name, corporate_group,
                   address_line1, address_line2, city, state, zip, county,
                   emr_other, pt_emr
            FROM facilities
            WHERE facility_id = ?
            """,
            (facility_id,),
        )
        row2 = cur.fetchone()
        if row2:
            parts2: List[str] = []

            if row2["legal_name"]:
                parts2.append(f"Legal name: {row2['legal_name']}")
            if row2["corporate_group"]:
                parts2.append(f"Corporate group: {row2['corporate_group']}")

            addr_bits: List[str] = []
            if row2["address_line1"]:
                addr_bits.append(row2["address_line1"])
            if row2["address_line2"]:
                addr_bits.append(row2["address_line2"])

            city_state_zip = ", ".join(
                p for p in [row2["city"], row2["state"], row2["zip"]] if p
            )
            if city_state_zip:
                addr_bits.append(city_state_zip)
            if row2["county"]:
                addr_bits.append(f"{row2['county']} County")

            if addr_bits:
                parts2.append("Address: " + "; ".join(addr_bits))

            if row2["emr_other"]:
                parts2.append(f"Other EMR details: {row2['emr_other']}")
            if row2["pt_emr"]:
                parts2.append(f"Therapy EMR: {row2['pt_emr']}")

            profile_text = "; ".join(parts2)
            if profile_text:
                header2 = row2["facility_name"] or ""
                profile_full = (
                    f"{header2} profile: {profile_text}" if header2 else profile_text
                )

                snippets.append(
                    {
                        "text": profile_full,
                        "source": "DB:facility_profile",
                        "tags": "profile",
                        "weight": 1.7,
                    }
                )
    except sqlite3.Error:
        pass

    # 6) client_info rows for this facility (if any)
    try:
        # First resolve the facility_name used in client_info
        cur.execute(
            """
            SELECT facility_name
            FROM facilities
            WHERE facility_id = ?
            """,
            (facility_id,),
        )
        name_row = cur.fetchone()
        fac_name_for_client = (name_row["facility_name"] or "").strip() if name_row else ""

        if fac_name_for_client:
            cur.execute(
                """
                SELECT field_name, value
                FROM client_info
                WHERE facility_name = ?
                ORDER BY updated_at DESC
                LIMIT 20
                """,
                (fac_name_for_client,),
            )
            ci_rows = cur.fetchall()
            if ci_rows:
                ci_lines: List[str] = []
                for r in ci_rows:
                    field = (r["field_name"] or "").strip()
                    val = (r["value"] or "").strip()
                    if not field or not val:
                        continue
                    ci_lines.append(f"{field}: {val}")

                if ci_lines:
                    ci_text = "Client info: " + "; ".join(ci_lines)
                    snippets.append(
                        {
                            "text": ci_text,
                            "source": "DB:client_info",
                            "tags": "client_info",
                            "weight": 1.3,
                        }
                    )
    except sqlite3.Error:
        pass

    return snippets


from fastapi import BackgroundTasks  # <-- put this with your other fastapi imports (near UploadFile/File/Form)

@app.post("/admin/census/upload")
async def admin_census_upload(
    background_tasks: BackgroundTasks,
    facility_name: str = Form(""),  # optional now (filename drives facility)
    facility_code: str = Form(""),
    files: list[UploadFile] = File(...),
    admin=Depends(require_admin),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    job_id = uuid.uuid4().hex

    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            INSERT INTO census_upload_jobs (job_id, status, total_files, processed_files, message)
            VALUES (?, 'queued', ?, 0, '')
            """,
            (job_id, len(files)),
        )
        conn.commit()
    finally:
        conn.close()

    # Save uploads to disk immediately (so request can return fast)
    job_dir = CENSUS_UPLOAD_TMP / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[str] = []
    for f in files:
        out_path = job_dir / (f.filename or f"upload_{uuid.uuid4().hex}.pdf")

        # ✅ stream to disk (avoid reading entire PDF into RAM)
        # Add a hard cap + yield occasionally so large PDFs don't monopolize CPU
        MAX_UPLOAD_BYTES = 75 * 1024 * 1024   # 75MB cap (tune if needed)
        CHUNK_SIZE = 256 * 1024               # 256KB chunks

        total = 0
        with open(out_path, "wb") as out_fp:
            while True:
                chunk = await f.read(CHUNK_SIZE)
                if not chunk:
                    break

                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"PDF too large ({total:,} bytes). Please split/re-export the report."
                    )

                out_fp.write(chunk)

                # yield to event loop occasionally (keeps server responsive)
                if total % (4 * 1024 * 1024) < CHUNK_SIZE:
                    await asyncio.sleep(0)

        # ✅ make sure the temp upload file handle is released ASAP
        try:
            await f.close()
        except Exception:
            pass

        saved_paths.append(str(out_path))

    logger.info("Census upload queued job_id=%s files=%s", job_id, len(saved_paths))

    background_tasks.add_task(
        _process_census_upload_job,
        job_id,
        saved_paths,
        (facility_name or "").strip(),
        (facility_code or "").strip(),
    )

    return {"ok": True, "job_id": job_id}


def _job_set(conn: sqlite3.Connection, job_id: str, status: str, processed: int | None = None, message: str | None = None):
    cur = conn.cursor()
    if processed is None and message is None:
        cur.execute(
            "UPDATE census_upload_jobs SET status=?, updated_at=datetime('now') WHERE job_id=?",
            (status, job_id),
        )
    elif processed is None:
        cur.execute(
            "UPDATE census_upload_jobs SET status=?, message=?, updated_at=datetime('now') WHERE job_id=?",
            (status, message or "", job_id),
        )
    elif message is None:
        cur.execute(
            "UPDATE census_upload_jobs SET status=?, processed_files=?, updated_at=datetime('now') WHERE job_id=?",
            (status, processed, job_id),
        )
    else:
        cur.execute(
            "UPDATE census_upload_jobs SET status=?, processed_files=?, message=?, updated_at=datetime('now') WHERE job_id=?",
            (status, processed, message or "", job_id),
        )


def _process_census_upload_job(job_id: str, file_paths: list[str], facility_name_override: str, facility_code_override: str):
    conn = get_db()
    cur = conn.cursor()

    processed = 0
    try:
        _job_set(conn, job_id, "running", processed=0, message="Starting…")
        conn.commit()

        # Enable tracemalloc only when PERF_LOG=1 (keeps overhead off by default)
        if PERF_LOG_ENABLED and not tracemalloc.is_tracing():
            tracemalloc.start(25)

        for p in file_paths:
            filename = Path(p).name
            try:
                _perf_log(job_id, filename, "start_file")

                t0 = time.perf_counter()
                sha = sha256_file(p)
                _perf_log(job_id, filename, "sha256_file", t0)

                # NOTE: Duplicate-PDF skipping is currently DISABLED.
                # (We used to skip if the same PDF was already uploaded by comparing SHA256
                # against census_runs.source_sha256. Keeping the code here commented-out
                # in case we want it again later.)

                # # skip if already uploaded
                # t0 = time.perf_counter()
                # exists = cur.execute(
                #     "SELECT id FROM census_runs WHERE source_sha256=? LIMIT 1",
                #     (sha,),
                # ).fetchone()
                # _perf_log(job_id, filename, "duplicate_check", t0)
                #
                # if exists:
                #     cur.execute(
                #         """
                #         INSERT INTO census_upload_job_files (job_id, filename, status, detail, run_id, rows_inserted)
                #         VALUES (?, ?, 'skipped', ?, NULL, 0)
                #         """,
                #         (job_id, filename, "Duplicate file (same SHA256)"),
                #     )
                #     processed += 1
                #     _job_set(conn, job_id, "running", processed=processed, message=f"Skipped duplicate: {filename}")
                #     conn.commit()
                #     continue

                t0 = time.perf_counter()
                parsed_iter = iter_pcc_admission_records_from_pdf_path(
                    p,
                    job_id=job_id,
                    filename=filename,
                )
                _perf_log(job_id, filename, "created_parsed_iter", t0)

                # Peek first row only (do NOT load whole PDF into memory)
                t0 = time.perf_counter()
                first_row = next(parsed_iter, None)
                _perf_log(job_id, filename, "parsed_first_row", t0)

                # Facility name: filename (AUTHORITATIVE) > UI override > PDF content
                fac_name = extract_facility_name_from_filename(filename)
                if not fac_name:
                    fac_name = facility_name_override or ((first_row or {}).get("facility_name") or "")

                # Facility code: UI override > filename (AUTHORITATIVE) > PDF content
                fac_code = (
                    facility_code_override
                    or extract_facility_code_from_filename(filename)
                    or ((first_row or {}).get("facility_code") or "")
                )

                report_dt = ((first_row or {}).get("report_dt") if first_row else None)

                cur.execute(
                    """
                    INSERT INTO census_runs (
                        facility_name, facility_code, report_dt,
                        source_filename, source_sha256
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (fac_name, fac_code, report_dt, filename, sha),
                )
                run_id = cur.lastrowid

                rows_inserted = 0

                def _insert_one(row: dict):
                    nonlocal rows_inserted

                    patient_key = (row.get("_patient_key") or "").strip()
                    if not patient_key:
                        patient_key = "|".join([
                            (fac_name or "").upper(),
                            (row.get("Last Name") or "").upper(),
                            (row.get("First Name") or "").upper(),
                            (row.get("DOB") or "").upper(),
                            (row.get("Admission Date") or "").upper(),
                        ]).strip("|")

                    cur.execute(
                        """
                        INSERT INTO census_run_patients (
                            run_id, facility_name, facility_code, patient_key,
                            first_name, last_name, dob, home_phone,
                            address, city, state, zip,
                            primary_ins, primary_number,
                            primary_care_phys, attending_phys,
                            admission_date, discharge_date,
                            room_number, reason_admission,
                            tag, raw_text
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            run_id,
                            fac_name,
                            fac_code,
                            patient_key,

                            (row.get("First Name") or "").strip(),
                            (row.get("Last Name") or "").strip(),
                            (row.get("DOB") or "").strip(),
                            (row.get("Home Phone") or "").strip(),

                            (row.get("Address") or "").strip(),
                            (row.get("City") or "").strip(),
                            (row.get("State") or "").strip(),
                            (row.get("Zip") or "").strip(),

                            (row.get("Primary Insurance") or "").strip(),
                            (row.get("Primary Number") or "").strip(),

                            (row.get("Primary Care Physician") or "").strip(),
                            (row.get("Attending Physician") or "").strip(),

                            (row.get("Admission Date") or "").strip(),
                            (row.get("Discharge Date") or "").strip(),

                            (row.get("Room Number") or "").strip(),
                            (row.get("Reason for admission") or "").strip(),

                            (row.get("Tag") or "").strip(),
                            (row.get("_raw") or ""),
                        ),
                    )
                    rows_inserted += 1

                # Insert first row, then stream the rest (NO giant list in memory)
                rows_seen = 0
                if first_row:
                    _insert_one(first_row)
                    rows_seen = 1

                t0_loop = time.perf_counter()

                for row in parsed_iter:
                    _insert_one(row)
                    rows_seen += 1

                    # Log every 200 rows to see whether parsing or inserts are ballooning RSS/CPU
                    if PERF_LOG_ENABLED and (rows_seen % 200 == 0):
                        _perf_log(job_id, filename, f"inserted_{rows_seen}", t0_loop)
                        t0_loop = time.perf_counter()



                cur.execute(
                    """
                    INSERT INTO census_upload_job_files (job_id, filename, status, detail, run_id, rows_inserted)
                    VALUES (?, ?, 'ok', ?, ?, ?)
                    """,
                    (job_id, filename, "Inserted", run_id, rows_inserted),
                )
                # ✅ Release cyclic refs (safe)
                gc.collect()

                processed += 1
                _job_set(conn, job_id, "running", processed=processed, message=f"Processed: {filename}")
                conn.commit()

            except ScannedPdfError as e:
                # ✅ Treat scanned/image PDFs as "skipped" (not an error)
                cur.execute(
                    """
                    INSERT INTO census_upload_job_files (job_id, filename, status, detail, run_id, rows_inserted)
                    VALUES (?, ?, 'skipped', ?, NULL, 0)
                    """,
                    (job_id, filename, str(e)),
                )
                processed += 1
                _job_set(conn, job_id, "running", processed=processed, message=f"Skipped scanned PDF: {filename}")
                conn.commit()

            except Exception as e:
                logger.exception("Census job file failed job_id=%s file=%s", job_id, filename)
                cur.execute(
                    """
                    INSERT INTO census_upload_job_files (job_id, filename, status, detail, run_id, rows_inserted)
                    VALUES (?, ?, 'error', ?, NULL, 0)
                    """,
                    (job_id, filename, str(e)),
                )
                processed += 1
                _job_set(conn, job_id, "running", processed=processed, message=f"Error: {filename}")
                conn.commit()
            finally:
                # ✅ ALWAYS delete the temp file for this specific 'p'
                try:
                    Path(p).unlink(missing_ok=True)
                except Exception:
                    logger.exception("Failed deleting tmp census PDF path=%s", p)

        _job_set(conn, job_id, "done", processed=processed, message="Complete")
        conn.commit()
        # ✅ cleanup the per-job tmp directory if it's empty
        try:
            job_dir = CENSUS_UPLOAD_TMP / job_id
            if job_dir.exists():
                # remove any leftover files just in case
                for child in job_dir.glob("*"):
                    try:
                        child.unlink(missing_ok=True)
                    except Exception:
                        logger.exception("Failed deleting leftover tmp file=%s", str(child))
                job_dir.rmdir()
        except Exception:
            logger.exception("Failed deleting tmp census job dir job_id=%s", job_id)

    except Exception as e:
        logger.exception("Census job failed job_id=%s", job_id)
        _job_set(conn, job_id, "error", processed=processed, message=str(e))
        conn.commit()
    finally:
        try:
            conn.close()
        except Exception:
            pass


@app.get("/admin/census/upload-status")
def admin_census_upload_status(job_id: str, lite: int = 0, admin=Depends(require_admin)):
    conn = get_db()
    cur = conn.cursor()
    try:
        job = cur.execute(
            "SELECT job_id, status, total_files, processed_files, message, created_at, updated_at FROM census_upload_jobs WHERE job_id=?",
            (job_id,),
        ).fetchone()
        if not job:
            raise HTTPException(status_code=404, detail="job not found")

        # ✅ during polling, skip the per-file rows to reduce DB + JSON churn
        if int(lite or 0) == 1:
            return {"ok": True, "job": dict(job), "files": []}

        files = cur.execute(
            """
            SELECT filename, status, detail, run_id, rows_inserted, created_at
            FROM census_upload_job_files
            WHERE job_id=?
            ORDER BY id ASC
            """,
            (job_id,),
        ).fetchall()

        return {
            "ok": True,
            "job": dict(job),
            "files": [dict(r) for r in files],
        }
    finally:
        conn.close()


@app.get("/admin/census/facilities")
def admin_census_facilities(admin=Depends(require_admin)):
    conn = get_db()
    cur = conn.cursor()
    try:
        rows = cur.execute(
            """
            SELECT DISTINCT facility_name
            FROM census_runs
            WHERE facility_name IS NOT NULL AND TRIM(facility_name) != ''
            ORDER BY facility_name ASC
            """
        ).fetchall()
        return {"ok": True, "facilities": [r["facility_name"] for r in rows]}
    finally:
        conn.close()

@app.get("/admin/census/view")
def admin_census_view(
    facility_name: str,
    view: str = "full",   # full | new | discharged
    admin=Depends(require_admin),
):
    conn = get_db()
    cur = conn.cursor()
    try:
        runs = cur.execute(
            """
            SELECT id
            FROM census_runs
            WHERE facility_name = ?
            ORDER BY id DESC
            LIMIT 2
            """,
            (facility_name,),
        ).fetchall()

        if not runs:
            return {"ok": True, "rows": [], "meta": {"facility_name": facility_name}}

        latest_id = runs[0]["id"]
        prev_id = runs[1]["id"] if len(runs) > 1 else None

        latest_rows = cur.execute(
            "SELECT * FROM census_run_patients WHERE run_id = ?",
            (latest_id,),
        ).fetchall()

        if view == "full" or not prev_id:
            out = [dict(r) for r in latest_rows]

            # If we have a previous run, mark which rows are "new admissions" (latest - previous)
            if prev_id:
                prev_keys = {r["patient_key"] for r in cur.execute(
                    "SELECT patient_key FROM census_run_patients WHERE run_id = ?",
                    (prev_id,),
                ).fetchall()}

                for r in out:
                    r["is_new"] = (r.get("patient_key") not in prev_keys)
            else:
                # No previous run to compare to, so we can't compute "new"
                for r in out:
                    r["is_new"] = False

            return {
                "ok": True,
                "rows": out,
                "meta": {"latest_run_id": latest_id, "prev_run_id": prev_id},
            }

        prev_keys = {r["patient_key"] for r in cur.execute(
            "SELECT patient_key FROM census_run_patients WHERE run_id = ?",
            (prev_id,),
        ).fetchall()}

        latest_keys = {r["patient_key"] for r in latest_rows}

        if view == "new":
            rows = [dict(r) for r in latest_rows if r["patient_key"] not in prev_keys]
        elif view == "discharged":
            prev_rows = cur.execute(
                "SELECT * FROM census_run_patients WHERE run_id = ?",
                (prev_id,),
            ).fetchall()
            rows = [dict(r) for r in prev_rows if r["patient_key"] not in latest_keys]
        else:
            raise HTTPException(status_code=400, detail="view must be full|new|discharged")

        return {"ok": True, "rows": rows, "meta": {"latest_run_id": latest_id, "prev_run_id": prev_id}}
    finally:
        conn.close()

@app.get("/admin/census/export_all_new_admissions.csv")
def admin_census_export_all_new_admissions_csv(admin=Depends(require_admin)):
    """
    Export NEW admissions across ALL facilities into a single CSV file.
    Uses each facility's latest 2 runs to compute "new" (latest - previous).
    """
    conn = get_db()
    cur = conn.cursor()
    try:
        facilities = cur.execute(
            """
            SELECT DISTINCT facility_name
            FROM census_runs
            WHERE facility_name IS NOT NULL AND TRIM(facility_name) != ''
            ORDER BY facility_name ASC
            """
        ).fetchall()

        headers = [
            "First Name","Last Name","DOB","Home Phone","Address","City","State","Zip",
            "Primary Ins","Primary Number","Primary Care Physician","Tag","Facility_Code",
            "Admission Date","Discharge Date","Room Number","Reason for admission","Attending Physician"
        ]

        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=headers)
        w.writeheader()

        # loop facilities and append "new" rows per facility
        for fr in facilities:
            facility_name = fr["facility_name"]

            runs = cur.execute(
                """
                SELECT id
                FROM census_runs
                WHERE facility_name = ?
                ORDER BY id DESC
                LIMIT 2
                """,
                (facility_name,),
            ).fetchall()

            if not runs:
                continue

            latest_id = runs[0]["id"]
            prev_id = runs[1]["id"] if len(runs) > 1 else None

            latest_rows = cur.execute(
                "SELECT * FROM census_run_patients WHERE run_id = ?",
                (latest_id,),
            ).fetchall()

            if not prev_id:
                new_rows = [dict(r) for r in latest_rows]
            else:
                prev_keys = {
                    r["patient_key"]
                    for r in cur.execute(
                        "SELECT patient_key FROM census_run_patients WHERE run_id = ?",
                        (prev_id,),
                    ).fetchall()
                }
                new_rows = [dict(r) for r in latest_rows if r["patient_key"] not in prev_keys]

            for r in new_rows:
                w.writerow({
                    "First Name": r.get("first_name",""),
                    "Last Name": r.get("last_name",""),
                    "DOB": r.get("dob",""),
                    "Home Phone": r.get("home_phone",""),
                    "Address": r.get("address",""),
                    "City": r.get("city",""),
                    "State": r.get("state",""),
                    "Zip": r.get("zip",""),
                    "Primary Ins": r.get("primary_ins",""),
                    "Primary Number": r.get("primary_number",""),
                    "Primary Care Physician": r.get("primary_care_phys",""),
                    "Tag": r.get("tag",""),
                    "Facility_Code": r.get("facility_code",""),
                    "Admission Date": r.get("admission_date",""),
                    "Discharge Date": r.get("discharge_date",""),
                    "Room Number": r.get("room_number",""),
                    "Reason for admission": r.get("reason_admission",""),
                    "Attending Physician": r.get("attending_phys",""),
                })

        filename = f"census_all_facilities_new_admissions_{dt.date.today().isoformat()}.csv"
        return Response(
            content=buf.getvalue().encode("utf-8"),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename=\"{filename}\"'},
        )

    finally:
        conn.close()


@app.get("/admin/census/export.csv")
def admin_census_export_csv(
    facility_name: str,
    view: str = "new",
    admin=Depends(require_admin),
):
    data = admin_census_view(facility_name=facility_name, view=view, admin=admin)
    rows = data["rows"]

    headers = [
        "First Name","Last Name","DOB","Home Phone","Address","City","State","Zip",
        "Primary Ins","Primary Number","Primary Care Physician","Tag","Facility_Code",
        "Admission Date","Discharge Date","Room Number","Reason for admission","Attending Physician"
    ]

    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=headers)
    w.writeheader()

    for r in rows:
        w.writerow({
            "First Name": r.get("first_name",""),
            "Last Name": r.get("last_name",""),
            "DOB": r.get("dob",""),
            "Home Phone": r.get("home_phone",""),
            "Address": r.get("address",""),
            "City": r.get("city",""),
            "State": r.get("state",""),
            "Zip": r.get("zip",""),
            "Primary Ins": r.get("primary_ins",""),
            "Primary Number": r.get("primary_number",""),
            "Primary Care Physician": r.get("primary_care_phys",""),
            "Tag": r.get("tag",""),
            "Facility_Code": r.get("facility_code",""),
            "Admission Date": r.get("admission_date",""),
            "Discharge Date": r.get("discharge_date",""),
            "Room Number": r.get("room_number",""),
            "Reason for admission": r.get("reason_admission",""),
            "Attending Physician": r.get("attending_phys",""),
        })

    return Response(
        content=buf.getvalue().encode("utf-8"),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="census_{facility_name}_{view}.csv"'},
    )

def _sensys_norm(s: str) -> str:
    s = (s or "").strip().upper()
    s = re.sub(r"[^A-Z0-9]+", " ", s).strip()
    s = re.sub(r"\s+", " ", s)
    return s

def _sensys_parse_date(val: str) -> str:
    """
    Sensys sample shows: 3/17/1935 0:00, 9/21/2023 0:00, etc.
    Return ISO date (YYYY-MM-DD) or "".
    """
    v = (val or "").strip()
    if not v:
        return ""
    # tolerate extra time portion
    v = v.replace(" 0:00", "").strip()
    for fmt in ("%m/%d/%Y", "%m/%d/%y"):
        try:
            d = dt.datetime.strptime(v, fmt).date()
            return d.isoformat()
        except Exception:
            pass
    return ""

def _sensys_key(first_name: str, last_name: str, dob_iso: str) -> str:
    return f"{_sensys_norm(last_name)}|{_sensys_norm(first_name)}|{dob_iso or ''}".strip("|")

def _sensys_name_similarity(a: str, b: str) -> float:
    aa = _sensys_norm(a)
    bb = _sensys_norm(b)
    if not aa or not bb:
        return 0.0
    return SequenceMatcher(None, aa, bb).ratio()


@app.post("/admin/census/sensys/process")
async def admin_census_sensys_process(
    file: UploadFile = File(...),
    admin=Depends(require_admin),
):
    fac = "SYSTEM"
    if not file or not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="CSV file is required")

    raw = await file.read()
    try:
        await file.close()
    except Exception:
        pass

    # parse CSV
    try:
        text = raw.decode("utf-8-sig", errors="replace")
    except Exception:
        text = raw.decode("latin-1", errors="replace")

    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise HTTPException(status_code=400, detail="CSV missing headers")

    # Expecting columns like your sample:
    # First Name, Last Name, Date of Birth, Admission Date, Discharge Date, Room Number, Primary Phone
    sensys_rows = []
    for r in reader:
        fn = (r.get("First Name") or r.get("First") or "").strip()
        ln = (r.get("Last Name") or r.get("Last") or "").strip()
        dob = _sensys_parse_date(r.get("Date of Birth") or r.get("DOB") or "")
        ad = _sensys_parse_date(r.get("Admission Date") or "")
        dd = _sensys_parse_date(r.get("Discharge Date") or "")
        room = (r.get("Room Number") or r.get("Room") or "").strip()
        phone = (r.get("Primary Phone") or r.get("Phone") or "").strip()

        if not (fn or ln or dob):
            continue

        sensys_rows.append(
            {
                "first_name": fn,
                "last_name": ln,
                "dob": dob,
                "admission_date": ad,
                "discharge_date": dd,
                "room_number": room,
                "primary_phone": phone,
                "key": _sensys_key(fn, ln, dob),
                "full_name": f"{fn} {ln}".strip(),
            }
        )

    # load latest census run PER facility, then union patients across all those runs
    conn = get_db()
    cur = conn.cursor()
    try:
        latest_runs = cur.execute(
            """
            SELECT facility_name, MAX(id) AS id
            FROM census_runs
            GROUP BY facility_name
            """
        ).fetchall()

        if not latest_runs:
            raise HTTPException(status_code=404, detail="No census runs found")

        run_ids = [r["id"] for r in latest_runs if r["id"] is not None]
        if not run_ids:
            raise HTTPException(status_code=404, detail="No latest run ids found")

        # ✅ define a "latest" id for the response payload (prevents NameError)
        # Since this endpoint unions the latest run for *each* facility, we use the newest run id overall.
        latest_id = max(run_ids)

        placeholders = ",".join(["?"] * len(run_ids))

        census = cur.execute(
            f"""
            SELECT
              p.first_name, p.last_name, p.dob, p.home_phone, p.address, p.city, p.state, p.zip,
              p.primary_ins, p.primary_number, p.primary_care_phys, p.tag,
              r.facility_code,
              p.admission_date, p.discharge_date, p.room_number, p.reason_admission, p.attending_phys
            FROM census_run_patients p
            JOIN census_runs r ON r.id = p.run_id
            WHERE p.run_id IN ({placeholders})
            """,
            tuple(run_ids),
        ).fetchall()

        census_rows = []
        for r in census:
            fn = (r["first_name"] or "").strip()
            ln = (r["last_name"] or "").strip()
            dob = _sensys_parse_date(r["dob"] or "")

            census_rows.append(
                {
                    "first_name": fn,
                    "last_name": ln,
                    "dob": dob,
                    "home_phone": (r["home_phone"] or "").strip(),
                    "address": (r["address"] or "").strip(),
                    "city": (r["city"] or "").strip(),
                    "state": (r["state"] or "").strip(),
                    "zip": (r["zip"] or "").strip(),
                    "primary_ins": (r["primary_ins"] or "").strip(),
                    "primary_number": (r["primary_number"] or "").strip(),
                    "primary_care_phys": (r["primary_care_phys"] or "").strip(),
                    "tag": (r["tag"] or "").strip(),
                    "facility_code": (r["facility_code"] or "").strip(),
                    "admission_date": (r["admission_date"] or "").strip(),
                    "discharge_date": (r["discharge_date"] or "").strip(),
                    "room_number": (r["room_number"] or "").strip(),
                    "reason_admission": (r["reason_admission"] or "").strip(),
                    "attending_phys": (r["attending_phys"] or "").strip(),

                    # keep these for matching + duplicates logic
                    "primary_phone": (r["home_phone"] or "").strip(),
                    "key": _sensys_key(fn, ln, dob),
                }
            )

        census_keys = {r["key"] for r in census_rows if r["key"]}
        sensys_keys = {r["key"] for r in sensys_rows if r["key"]}

        # 1) Missing Admissions: in latest census_run_patients but not in Sensys CSV
        missing_admissions = [r for r in census_rows if r["key"] and r["key"] not in sensys_keys]

        # 2) Missing Discharges: in Sensys CSV but not in latest census_run_patients
        missing_discharges = [r for r in sensys_rows if r["key"] and r["key"] not in census_keys]

        # 3) Duplicates on Sensys:
        #    - exact duplicates: same key appears > 1
        #    - possible duplicates: same DOB, similar name (SequenceMatcher), same facility (modal is per-fac)
        by_key = {}
        for r in sensys_rows:
            k = r["key"]
            if not k:
                continue
            by_key.setdefault(k, []).append(r)

        dup_rows = []
        for k, group in by_key.items():
            if len(group) > 1:
                for g in group:
                    dup_rows.append({**g, "note": "Exact duplicate (same First/Last/DOB)"})

        # possible duplicates (DOB match, similar names)
        by_dob = {}
        for r in sensys_rows:
            if r["dob"]:
                by_dob.setdefault(r["dob"], []).append(r)

        # avoid re-adding rows already in exact dup set
        exact_dup_keys = {r["key"] for r in dup_rows if r.get("key")}
        seen_possible = set()

        for dob, group in by_dob.items():
            if len(group) < 2:
                continue
            # compare pairs
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    a = group[i]
                    b = group[j]
                    if not a["full_name"] or not b["full_name"]:
                        continue
                    sim = _sensys_name_similarity(a["full_name"], b["full_name"])
                    if sim >= 0.85:
                        # mark both as possible dup (unless exact dup already)
                        if a["key"] not in exact_dup_keys and a["key"] not in seen_possible:
                            dup_rows.append({**a, "note": f"Possible duplicate (DOB match; similar name {sim:.2f})"})
                            seen_possible.add(a["key"])
                        if b["key"] not in exact_dup_keys and b["key"] not in seen_possible:
                            dup_rows.append({**b, "note": f"Possible duplicate (DOB match; similar name {sim:.2f})"})
                            seen_possible.add(b["key"])

        # Persist job + rows for export
        job_id = uuid.uuid4().hex
        cur.execute(
            "INSERT INTO sensys_census_jobs (job_id, facility_name) VALUES (?, ?)",
            (job_id, fac),
        )

        def _ins(kind: str, rr: dict, note: str = ""):
            cur.execute(
                """
                INSERT INTO sensys_census_job_rows
                (
                  job_id, kind,
                  first_name, last_name, dob,
                  admission_date, discharge_date, room_number, primary_phone, note,

                  home_phone, address, city, state, zip,
                  primary_ins, primary_number, primary_care_phys, tag, facility_code,
                  reason_admission, attending_phys
                )
                VALUES
                (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id, kind,
                    rr.get("first_name",""),
                    rr.get("last_name",""),
                    rr.get("dob",""),
                    rr.get("admission_date",""),
                    rr.get("discharge_date",""),
                    rr.get("room_number",""),
                    rr.get("primary_phone",""),
                    note or rr.get("note","") or "",

                    rr.get("home_phone",""),
                    rr.get("address",""),
                    rr.get("city",""),
                    rr.get("state",""),
                    rr.get("zip",""),
                    rr.get("primary_ins",""),
                    rr.get("primary_number",""),
                    rr.get("primary_care_phys",""),
                    rr.get("tag",""),
                    rr.get("facility_code",""),
                    rr.get("reason_admission",""),
                    rr.get("attending_phys",""),
                ),
            )

        for r in missing_admissions:
            _ins("missing_admissions", r, "Present in latest census run, missing from Sensys CSV")

        for r in missing_discharges:
            _ins("missing_discharges", r, "Present in Sensys CSV, missing from latest census run")

        for r in dup_rows:
            _ins("duplicates", r, r.get("note", ""))

        conn.commit()

        return {
            "ok": True,
            "job_id": job_id,
            "facility_name": fac,
            "latest_run_id": latest_id,
            "missing_admissions_count": len(missing_admissions),
            "missing_discharges_count": len(missing_discharges),
            "duplicates_count": len(dup_rows),
        }
    finally:
        conn.close()


@app.get("/admin/census/sensys/export.csv")
def admin_census_sensys_export_csv(
    job_id: str,
    kind: str = "missing_admissions",  # missing_admissions | missing_discharges | duplicates
    admin=Depends(require_admin),
):
    k = (kind or "").strip()
    if k not in ("missing_admissions", "missing_discharges", "duplicates"):
        raise HTTPException(status_code=400, detail="kind must be missing_admissions|missing_discharges|duplicates")

    conn = get_db()
    cur = conn.cursor()
    try:
        job = cur.execute("SELECT job_id, facility_name, created_at FROM sensys_census_jobs WHERE job_id=?", (job_id,)).fetchone()
        if not job:
            raise HTTPException(status_code=404, detail="job not found")

        if k == "missing_admissions":
            rows = cur.execute(
                """
                SELECT
                  first_name, last_name, dob,
                  home_phone, address, city, state, zip,
                  primary_ins, primary_number, primary_care_phys, tag, facility_code,
                  admission_date, discharge_date, room_number, reason_admission, attending_phys,
                  note
                FROM sensys_census_job_rows
                WHERE job_id=? AND kind=?
                ORDER BY facility_code ASC, last_name ASC, first_name ASC, dob ASC
                """,
                (job_id, k),
            ).fetchall()

            headers = [
                "First Name","Last Name","DOB","Home Phone","Address","City","State","Zip",
                "Primary Ins","Primary Number","Primary Care Physician","Tag","Facility_Code",
                "Admission Date","Discharge Date","Room Number","Reason for admission","Attending Physician"
            ]

            buf = io.StringIO()
            w = csv.DictWriter(buf, fieldnames=headers)
            w.writeheader()

            for r in rows:
                rr = dict(r)
                w.writerow({
                    "First Name": rr.get("first_name",""),
                    "Last Name": rr.get("last_name",""),
                    "DOB": rr.get("dob",""),
                    "Home Phone": rr.get("home_phone",""),
                    "Address": rr.get("address",""),
                    "City": rr.get("city",""),
                    "State": rr.get("state",""),
                    "Zip": rr.get("zip",""),
                    "Primary Ins": rr.get("primary_ins",""),
                    "Primary Number": rr.get("primary_number",""),
                    "Primary Care Physician": rr.get("primary_care_phys",""),
                    "Tag": rr.get("tag",""),
                    "Facility_Code": rr.get("facility_code",""),
                    "Admission Date": rr.get("admission_date",""),
                    "Discharge Date": rr.get("discharge_date",""),
                    "Room Number": rr.get("room_number",""),
                    "Reason for admission": rr.get("reason_admission",""),
                    "Attending Physician": rr.get("attending_phys",""),
                })

            return Response(
                content=buf.getvalue().encode("utf-8"),
                media_type="text/csv",
                headers={"Content-Disposition": f'attachment; filename="sensys_{k}_{job_id}.csv"'},
            )

        # default (missing_discharges / duplicates)
        rows = cur.execute(
            """
            SELECT first_name, last_name, dob, admission_date, discharge_date, room_number, primary_phone, note
            FROM sensys_census_job_rows
            WHERE job_id=? AND kind=?
            ORDER BY last_name ASC, first_name ASC, dob ASC
            """,
            (job_id, k),
        ).fetchall()

        headers = ["First Name","Last Name","DOB","Admission Date","Discharge Date","Room Number","Primary Phone","Note"]

        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=headers)
        w.writeheader()

        for r in rows:
            rr = dict(r)
            w.writerow({
                "First Name": rr.get("first_name",""),
                "Last Name": rr.get("last_name",""),
                "DOB": rr.get("dob",""),
                "Admission Date": rr.get("admission_date",""),
                "Discharge Date": rr.get("discharge_date",""),
                "Room Number": rr.get("room_number",""),
                "Primary Phone": rr.get("primary_phone",""),
                "Note": rr.get("note",""),
            })

        return Response(
            content=buf.getvalue().encode("utf-8"),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="sensys_{k}_{job_id}.csv"'},
        )
    finally:
        conn.close()

# ---------------------------------------------------------------------------
# Embeddings & ranking
# ---------------------------------------------------------------------------

def embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [item.embedding for item in resp.data]


def cosine_sim(u: List[float], v: List[float]) -> float:
    if not u or not v or len(u) != len(v):
        return 0.0
    dot = sum(a * b for a, b in zip(u, v))
    nu = math.sqrt(sum(a * a for a in u))
    nv = math.sqrt(sum(b * b for b in v))
    if nu == 0 or nv == 0:
        return 0.0
    return dot / (nu * nv)


def build_ranked_context(
    question: str,
    qa_rows,
    fac_snippets,
    top_k: int,
    section_hint: Optional[str] = None,
):
    """
    Build a ranked list of context chunks from:
      - Q&A rows
      - facility snippets

    Scoring:
      - base = cosine similarity between question and chunk
      - multiplied by a 'weight' term:
          * Q&A: boosted if section/topics/tags match section_hint
          * Facility snippets: use their pre-assigned .weight (contacts, summary, etc.)

    Returns:
      (top_chunks, sources, best_score)
    """
    candidates: List[Dict[str, Any]] = []

    sh = (section_hint or "").strip().lower()

    # ------------------------------
    # Q&A rows (default weight 1.0)
    # ------------------------------
    for r in qa_rows:
        section = (r.get("section") or "").strip()
        topics = (r.get("topics") or "").strip()
        tags = (r.get("tags") or "").strip()

        text = (
            f"Q: {r['question']}\n"
            f"A: {r['answer']}\n"
            f"Topics: {topics}\n"
            f"Tags: {tags}"
        )

        weight = 1.0
        if sh:
            # Exact section match gets a strong bump
            if section.lower() == sh:
                weight *= 1.5

            # If the hint appears in topics or tags, give a smaller bump
            topics_lower = topics.lower()
            tags_lower = tags.lower()
            if sh in topics_lower or sh in tags_lower:
                weight *= 1.2

        candidates.append(
            {
                "text": text,
                "source": f"Q&A:{r['id']}",
                "kind": "qa",
                "weight": weight,
            }
        )

    # --------------------------------------
    # Facility snippets (reuse their weight)
    # --------------------------------------
    for sn in fac_snippets:
        txt = sn["text"]
        tags = sn.get("tags") or ""
        text = f"{txt}\n(Tags: {tags})"
        candidates.append(
            {
                "text": text,
                "source": sn.get("source") or "DB:facility_data",
                "kind": "facility",
                "weight": float(sn.get("weight", 1.0) or 1.0),
            }
        )

    if not candidates:
        # no context found at all
        return [], [], 0.0

    # --------------------------------------
    # Embed and score with weight applied
    # --------------------------------------
    ctx_texts = [c["text"] for c in candidates]
    q_emb = embed_texts([question])[0]
    ctx_embs = embed_texts(ctx_texts)

    scored: List[tuple[float, Dict[str, Any]]] = []
    best_score = 0.0

    for c, emb in zip(candidates, ctx_embs):
        base_score = cosine_sim(q_emb, emb)
        w = float(c.get("weight", 1.0) or 1.0)
        final_score = base_score * w
        if final_score > best_score:
            best_score = final_score
        scored.append((final_score, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = [c for _, c in scored[: max(top_k, 1)]]
    sources = [c["source"] for c in top]
    return top, sources, best_score



# ---------------------------------------------------------------------------
# Analytics / listing helpers (Layer F)
# ---------------------------------------------------------------------------

def detect_analytics_intent(
    question: str,
    config: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Config-driven analytics intent detector, with a simple fallback.

    From config (ai_settings.analytics_config), expects:
      { "rules": [
          {
            "id": "list_facilities",
            "kind": "list",          # or "aggregate"
            "name": "List all facilities",
            "triggers": ["list all facilities", "list facilities"],
            "scope": "global",       # or "facility"
            "sql": "...",
            "answer_prefix": "...",
            "answer_suffix": "..."
          },
          ...
      ] }

    Returns:
      { "type": "list", "rule": <rule_dict> }
      { "type": "aggregate", "rule": <rule_dict> }
      or None
    """
    if not question:
        return None

    q = question.lower().strip()
    cfg = config or {}
    rules = cfg.get("rules") or []

    # 1) Config-driven rules (win first)
    for rule in rules:
        kind = (rule.get("kind") or "").strip().lower()
        if kind not in ("list", "aggregate"):
            continue

        triggers = rule.get("triggers") or []
        if isinstance(triggers, str):
            triggers = [triggers]

        for t in triggers:
            t_norm = (t or "").strip().lower()
            if not t_norm:
                continue
            # Very simple: substring match in the normalized question
            if t_norm in q:
                return {
                    "type": kind,
                    "rule": rule,
                }

    # 2) Fallback to your original simple heuristics
    # -------- LIST INTENT --------
    list_triggers = (
        "list ",
        "list all ",
        "show all ",
        "show me all ",
        "give me all ",
        "what are all ",
        "what are the ",
    )
    is_list = any(q.startswith(t) for t in list_triggers) or " list all " in q or " show all " in q

    if is_list:
        # Contacts
        if any(w in q for w in ("contact", "contacts", "administrator", "admin", "don", "director of nursing", "ur ")):
            return {"type": "list", "entity": "contacts"}

        # Facilities
        if any(w in q for w in ("facility", "facilities", "building", "buildings")):
            return {"type": "list", "entity": "facilities"}

    # -------- AGGREGATE INTENT (AVG) --------
    if "average" in q or "avg" in q:
        # short-term beds
        if any(w in q for w in ("short term beds", "short-term beds", "short beds", "short_beds")):
            return {"type": "aggregate", "operation": "avg", "metric": "short_beds"}

        # LTC beds
        if any(w in q for w in ("ltc beds", "long term beds", "long-term beds", "ltc_beds")):
            return {"type": "aggregate", "operation": "avg", "metric": "ltc_beds"}

    return None



def run_analytics_query(
    conn: sqlite3.Connection,
    intent: Dict[str, Any],
    facility_id: Optional[str],
    facility_label: str,
) -> Optional[tuple[str, List[str]]]:
    """
    Execute a simple analytics/listing query based on the parsed intent.

    Returns:
      (answer_text, sources)  or None if the intent can't be satisfied.

    This path intentionally does NOT call the LLM; it answers directly from
    the DB to avoid hallucinations for lists/totals.

    Two modes:
      1) Config-driven rules (ai_settings.analytics_config)
      2) Built-in fallback logic (facilities list, contacts list, avg beds)
    """
    itype = intent.get("type")

    # ------------------------------------------------
    # 1) Config-driven rule (if present in the intent)
    # ------------------------------------------------
    rule = intent.get("rule")
    if rule:
        kind = (rule.get("kind") or "").strip().lower()
        sql = (rule.get("sql") or "").strip()
        scope = (rule.get("scope") or "global").strip().lower()
        src = f"DB:analytics:{rule.get('id') or 'config'}"

        if not sql or kind not in ("list", "aggregate"):
            # Misconfigured rule; fall back to built-ins
            pass
        else:
            # Simple placeholder: {{facility_id}} → ?
            params: List[Any] = []
            if "{{facility_id}}" in sql:
                if not facility_id:
                    # Can't satisfy a facility-scoped rule without a detected facility
                    return None
                sql = sql.replace("{{facility_id}}", "?")
                params.append(facility_id)

            cur = conn.cursor()
            try:
                cur.execute(sql, params)
                rows = cur.fetchall()
                col_names = [d[0] for d in cur.description] if cur.description else []
            except sqlite3.Error as e:
                return (
                    f"I tried to run the configured analytics query but got a database error: {e}",
                    [src],
                    sql,
                )

            # ------- LIST RULES -------
            if kind == "list":
                if not rows:
                    msg = rule.get("empty_message") or "I couldn't find any results for that query in the database."
                    msg = msg.replace("{{facility_label}}", facility_label or (facility_id or "this facility"))
                    return (msg, [src])

                lines: List[str] = []
                for r in rows:
                    # If there's a column named 'line', prefer it
                    if "line" in col_names:
                        val = r[col_names.index("line")]
                    else:
                        # Otherwise join all non-null columns
                        vals = [str(r[i]) for i in range(len(r)) if r[i] is not None]
                        val = " ".join(vals)
                    val = str(val).strip()
                    if val:
                        lines.append(" - " + val)

                prefix = (rule.get("answer_prefix") or "Here are the results:\n").replace(
                    "{{facility_label}}",
                    facility_label or (facility_id or "this facility"),
                )
                answer_text = prefix + "\n" + "\n".join(lines)
                return (answer_text, [src], sql)

            # ------- AGGREGATE RULES -------
            if kind == "aggregate":
                if not rows:
                    return ("I couldn't compute that metric because the query returned no rows.", [src])

                row0 = rows[0]
                value = None

                # Common column names: value, val, avg_val, count, total
                for cname in ("value", "val", "avg_val", "count", "total"):
                    if cname in col_names:
                        value = row0[col_names.index(cname)]
                        break
                if value is None and col_names:
                    value = row0[0]

                vnum = value
                vround = value
                try:
                    vnum = float(value)
                    vround = round(vnum, 1)
                except Exception:
                    # leave as-is if not numeric
                    pass

                prefix = rule.get("answer_prefix") or f"The value for {rule.get('name') or 'this metric'} is"
                suffix = rule.get("answer_suffix") or ""
                text = prefix.replace("{value_rounded}", str(vround)).replace("{value}", str(vnum))
                if suffix:
                    text = f"{text} {suffix}"
                return (text, [src], sql)

    # ------------------------------------------------
    # 2) Built-in fallback logic (your original code)
    # ------------------------------------------------
    # ---------------- LIST QUERIES ----------------
    if itype == "list":
        entity = intent.get("entity")

        # List all facilities
        if entity == "facilities":
            cur = conn.cursor()
            cur.execute(
                """
                SELECT id, facility_name, attending, notes, notes2, aliases, facility_emails, pin_hash
                FROM snf_admission_facilities
                ORDER BY facility_name COLLATE NOCASE
                """
            )
            rows = cur.fetchall()
            if not rows:
                return ("There are no facilities in the database yet.", ["DB:analytics:facilities_list"])

            lines = []
            for r in rows:
                name = r["facility_name"] or "(no name)"
                city = r["city"] or ""
                state = r["state"] or ""
                fid = r["facility_id"] or ""
                corp = r["corporate_group"] or ""

                loc = ", ".join(p for p in (city, state) if p)
                pieces = [name]
                if loc:
                    pieces.append(f"({loc})")
                if corp:
                    pieces.append(f"– {corp}")
                if fid:
                    pieces.append(f"[id: {fid}]")

                lines.append(" - " + " ".join(pieces))

            answer = "Here are the facilities currently in the database:\n" + "\n".join(lines)
            return (answer, ["DB:analytics:facilities_list"])

        # List contacts (optionally scoped to a single facility)
        if entity == "contacts":
            cur = conn.cursor()
            if facility_id:
                # contacts for a specific facility
                cur.execute(
                    """
                    SELECT type, name, phone, email
                    FROM facility_contacts
                    WHERE facility_id = ?
                    ORDER BY type COLLATE NOCASE, name COLLATE NOCASE
                    """,
                    (facility_id,),
                )
                rows = cur.fetchall()
                if not rows:
                    label = facility_label or facility_id or "this facility"
                    return (
                        f"I couldn't find any contacts in the database for {label}.",
                        [f"DB:analytics:facility_contacts:{facility_id}"],
                    )

                label = facility_label or facility_id or "this facility"
                lines = []
                for r in rows:
                    role = r["type"] or ""
                    name = r["name"] or ""
                    phone = r["phone"] or ""
                    email = r["email"] or ""
                    bits = [role, ":", name]
                    contact = []
                    if phone:
                        contact.append(f"phone {phone}")
                    if email:
                        contact.append(f"email {email}")
                    if contact:
                        bits.append("(" + ", ".join(contact) + ")")
                    lines.append(" - " + " ".join(b for b in bits if b))

                answer = f"Here are the contacts we have on file for {label}:\n" + "\n".join(lines)
                return (answer, [f"DB:analytics:facility_contacts:{facility_id}"])

            else:
                # contacts across all facilities
                cur.execute(
                    """
                    SELECT c.facility_id, f.facility_name, c.type, c.name, c.phone, c.email
                    FROM facility_contacts c
                    LEFT JOIN facilities f ON f.facility_id = c.facility_id
                    ORDER BY f.facility_name COLLATE NOCASE, c.type COLLATE NOCASE, c.name COLLATE NOCASE
                    """
                )
                rows = cur.fetchall()
                if not rows:
                    return (
                        "There are no contacts in the facility_contacts table yet.",
                        ["DB:analytics:facility_contacts:all"],
                    )

                lines = []
                current_fac = None
                for r in rows:
                    fac_name = r["facility_name"] or r["facility_id"] or "(unknown facility)"
                    if fac_name != current_fac:
                        lines.append(f"\n{fac_name}:")
                        current_fac = fac_name

                    role = r["type"] or ""
                    name = r["name"] or ""
                    phone = r["phone"] or ""
                    email = r["email"] or ""
                    bits = [role, ":", name]
                    contact = []
                    if phone:
                        contact.append(f"phone {phone}")
                    if email:
                        contact.append(f"email {email}")
                    if contact:
                        bits.append("(" + ", ".join(contact) + ")")
                    lines.append("  - " + " ".join(b for b in bits if b))

                answer = "Here are the facility contacts currently in the database:\n" + "\n".join(lines)
                return (answer, ["DB:analytics:facility_contacts:all"])

        return None  # unsupported entity

    # ---------------- AGGREGATE QUERIES ----------------
    if itype == "aggregate":
        op = intent.get("operation")
        metric = intent.get("metric")

        if op == "avg" and metric in ("short_beds", "ltc_beds"):
            col = metric
            metric_label = "short-term beds per facility" if metric == "short_beds" else "long-term care beds per facility"

            cur = conn.cursor()
            cur.execute(
                f"""
                SELECT AVG({col}) AS avg_val,
                       COUNT(*) AS n_facilities,
                       SUM(CASE WHEN {col} IS NOT NULL THEN 1 ELSE 0 END) AS n_with_value
                FROM facilities
                """
            )
            row = cur.fetchone()
            if not row or row["n_facilities"] == 0:
                return (
                    "I couldn't compute an average because there are no facilities in the database yet.",
                    [f"DB:analytics:avg_{metric}"],
                )

            avg_val = row["avg_val"]
            n_fac = row["n_facilities"]
            n_with_val = row["n_with_value"] or 0

            if avg_val is None or n_with_val == 0:
                return (
                    f"I couldn't compute an average for {metric_label} because none of the facilities have a value in the database.",
                    [f"DB:analytics:avg_{metric}"],
                )

            avg_display = round(avg_val, 1)
            answer = (
                f"Across {n_fac} facilities in the database, the average {metric_label} "
                f"(based on {n_with_val} facilities with a non-blank value) is about {avg_display} beds."
            )
            return (answer, [f"DB:analytics:avg_{metric}"])

    # If we get here, the intent isn't supported yet
    return None


# ---------------------------------------------------------------------------
# Helper: listables "lexicon" so the LLM can see what's actually in the DB
# ---------------------------------------------------------------------------
def get_listables_lexicon(conn: sqlite3.Connection) -> Dict[str, List[str]]:
    """
    Pull distinct values from v_facility_listables so the LLM knows
    what is actually present in the data when writing SQL.

    Returns a dict with keys:
      - item_types
      - community_partner_types
      - additional_services
      - insurance_plans
    """
    lex: Dict[str, List[str]] = {
        "item_types": [],
        "community_partner_types": [],
        "additional_services": [],
        "insurance_plans": [],
    }
    cur = conn.cursor()

    try:
        # All item_type values currently in the view
        cur.execute(
            """
            SELECT DISTINCT item_type
            FROM v_facility_listables
            WHERE item_type IS NOT NULL AND item_type <> ''
            ORDER BY item_type COLLATE NOCASE
            """
        )
        lex["item_types"] = [
            str(r[0]).strip() for r in cur.fetchall() if r[0] is not None
        ]
    except sqlite3.Error:
        pass

    try:
        # Kinds of community partners (e.g. home health, hospice, DME)
        cur.execute(
            """
            SELECT DISTINCT LOWER(item_subtype) AS t
            FROM v_facility_listables
            WHERE item_type = 'community_partner'
              AND item_subtype IS NOT NULL
              AND item_subtype <> ''
            ORDER BY t
            """
        )
        lex["community_partner_types"] = [
            str(r[0]).strip() for r in cur.fetchall() if r[0] is not None
        ]
    except sqlite3.Error:
        pass

    try:
        # Additional services – where things like "wound care", "respiratory therapy"
        # live today
        cur.execute(
            """
            SELECT DISTINCT LOWER(display_name) AS s
            FROM v_facility_listables
            WHERE item_type = 'additional_service'
              AND display_name IS NOT NULL
              AND display_name <> ''
            ORDER BY s
            """
        )
        rows = [str(r[0]).strip() for r in cur.fetchall() if r[0] is not None]
        # Keep it to a reasonable length for the prompt
        lex["additional_services"] = rows[:40]
    except sqlite3.Error:
        pass

    try:
        # Insurance plans (also useful for list queries)
        cur.execute(
            """
            SELECT DISTINCT LOWER(display_name) AS p
            FROM v_facility_listables
            WHERE item_type = 'insurance_plan'
              AND display_name IS NOT NULL
              AND display_name <> ''
            ORDER BY p
            """
        )
        rows = [str(r[0]).strip() for r in cur.fetchall() if r[0] is not None]
        lex["insurance_plans"] = rows[:40]
    except sqlite3.Error:
        pass

    return lex


# ---------------------------------------------------------------------------
# Helper: make LLM-generated SQL on v_facility_listables case-insensitive
# ---------------------------------------------------------------------------
import re  # (already imported at top of file, but safe to reuse here)

def _normalize_listables_sql_case(sql: str) -> str:
    """
    Rewrite simple text filters to use LOWER(column) so that
    'home health' will match 'Home Health', 'HOME HEALTH', etc.

    We only touch a few known text columns:
      - item_type, item_subtype
      - state, corporate_group
      - display_name, details
    """
    # Equality on text columns: column = 'Value'
    for col in ("item_type", "item_subtype", "state", "corporate_group"):
        sql = re.sub(
            rf"({col}\s*=\s*)'([^']+)'",
            lambda m, col=col: f"LOWER({col}) = '{m.group(2).lower()}'",
            sql,
            flags=re.IGNORECASE,
        )

    # LIKE on text columns: column LIKE '%Value%'
    for col in ("display_name", "details", "item_subtype", "item_type"):
        sql = re.sub(
            rf"({col}\s+LIKE\s*)'([^']+)'",
            lambda m, col=col: f"LOWER({col}) LIKE '{m.group(2).lower()}'",
            sql,
            flags=re.IGNORECASE,
        )

    return sql

def normalize_sortable_dt(s: Optional[str]) -> Optional[str]:
    """
    Convert various incoming datetime strings to a sortable ISO-ish format:
      YYYY-MM-DDTHH:MM:SS
    Returns None if empty/unparseable.
    """
    if not s:
        return None
    raw = str(s).strip()
    if not raw:
        return None

    # Already ISO-like
    if "T" in raw and len(raw) >= 19:
        return raw[:19]

    # SQLite datetime('now') format: "YYYY-MM-DD HH:MM:SS"
    if len(raw) >= 19 and raw[4] == "-" and raw[7] == "-" and raw[10] == " ":
        return raw[:19].replace(" ", "T")

    fmts = [
        "%m/%d/%Y %I:%M:%S %p",   # 12/20/2025 2:32:00 PM
        "%m/%d/%Y %I:%M %p",      # 12/20/2025 2:32 PM
        "%m/%d/%Y %H:%M:%S",      # 12/20/2025 14:32:00
        "%m/%d/%Y %H:%M",         # 12/20/2025 14:32
        "%Y-%m-%d %H:%M:%S",      # 2025-12-20 14:32:00
        "%Y-%m-%d %H:%M",         # 2025-12-20 14:32
        "%m/%d/%Y",               # 12/20/2025
        "%Y-%m-%d",               # 2025-12-20
    ]

    for f in fmts:
        try:
            d = dt.datetime.strptime(raw, f)
            return d.replace(microsecond=0).isoformat()
        except Exception:
            continue

    return None


def try_llm_listables_query(
    conn: sqlite3.Connection,
    question: str,
    facility_id: Optional[str],
    facility_label: str,
) -> Optional[tuple[str, List[str]]]:
    """
    Hybrid path: let the LLM write a *safe* SELECT on v_facility_listables.
    
    Returns:
      (answer_text, sources, sql_used) or None.

    - Only used for "list" / "how many" / "average" style questions.
    - We constrain the model to:
        * SELECT-only
        * single table: v_facility_listables
        * no UPDATE/DELETE/INSERT/DROP/ALTER
        * no ';' or comments
    - If the SQL fails validation or errors, we return None and fall back
      to the normal embedding + LLM answer.
    """
    if not question:
        return None

    q_lower = question.lower()

    # Original triggers for list/aggregate style questions
    trigger_phrases = ["list ", "show ", "how many", "count ", "average", "avg "]
    base_trigger = any(tp in q_lower for tp in trigger_phrases)

    # NEW: also treat "Which facilities ..." / "What facilities ..." as list-intent
    which_facilities_pattern = (
        q_lower.startswith("which facilities")
        or "which facilities " in q_lower
        or q_lower.startswith("what facilities")
        or "what facilities " in q_lower
    )

    if not (base_trigger or which_facilities_pattern):
        return None

    # NEW: peek into v_facility_listables to see what is actually present
    lex = get_listables_lexicon(conn)

    schema_desc = (
        "You have a single SQLite view named v_facility_listables with columns:\n"
        "  - facility_id (TEXT)\n"
        "  - facility_name (TEXT)\n"
        "  - city (TEXT)\n"
        "  - state (TEXT)\n"
        "  - zip (TEXT)\n"
        "  - county (TEXT)\n"
        "  - corporate_group (TEXT)\n"
        "  - emr (TEXT)\n"
        "  - emr_other (TEXT)\n"
        "  - pt_emr (TEXT)\n"
        "  - orders (TEXT)\n"
        "  - orders_other (TEXT)\n"
        "  - outpatient_pt (TEXT)\n"
        "  - short_beds (INTEGER)\n"
        "  - ltc_beds (INTEGER)\n"
        "  - avg_dcs (REAL)\n"
        "  - item_type (TEXT)\n"
        "  - item_subtype (TEXT)\n"
        "  - display_name (TEXT)\n"
        "  - details (TEXT)\n\n"
        "Important rules:\n"
        "- For facility-level aggregates (beds, avg_dcs), ALWAYS filter item_type = 'facility'.\n"
        "- For lists of facilities, use item_type = 'facility'.\n"
        "- For lists of home health / hospice / DME, use item_type = 'community_partner' and the item_subtype for the kind of partner.\n"
        "- For lists of insurance plans, use item_type = 'insurance_plan'.\n"
        "- For lists of additional services (like wound care), use item_type = 'additional_service'.\n"
        "- When matching text (state, corporate_group, item_type, item_subtype, display_name, details), it is safest to use LOWER(column) and lowercase constants, e.g. LOWER(item_subtype) = 'home health' or LOWER(display_name) LIKE '%wound care%'.\n"
        "- For questions like 'Which facilities offer wound care?', filter rows where item_type = 'additional_service' and the service name in display_name mentions the requested service (for example, LOWER(display_name) LIKE '%wound care%'), then return DISTINCT facility_name (and optionally city/state).\n"
    )

    # NEW: append concrete examples from the actual data so the model
    # knows which strings (e.g. 'wound care', 'respiratory therapy',
    # 'home health') are present and where.
    extra_lines: List[str] = []

    if lex.get("item_types"):
        extra_lines.append(
            "Current item_type values in this database: "
            + ", ".join(sorted(lex["item_types"]))
        )

    if lex.get("community_partner_types"):
        extra_lines.append(
            "Known community_partner item_subtype values (types of community partners): "
            + "; ".join(lex["community_partner_types"][:20])
        )

    if lex.get("additional_services"):
        extra_lines.append(
            "Examples of additional services (item_type = 'additional_service', from display_name): "
            + "; ".join(lex["additional_services"][:20])
        )

    if lex.get("insurance_plans"):
        extra_lines.append(
            "Examples of insurance plans (item_type = 'insurance_plan', from display_name): "
            + "; ".join(lex["insurance_plans"][:20])
        )

    if extra_lines:
        schema_desc += (
            "\nHere are some example values currently present in v_facility_listables:\n"
            + "- " + "\n- ".join(extra_lines)
            + "\n"
        )



    system_msg = (
        "You are a read-only SQL generator for a SQLite database.\n"
        "Your ONLY job is to output a SINGLE SELECT statement that queries the view v_facility_listables.\n"
        "Rules:\n"
        "- Output ONLY raw SQL (no explanation, no comments, no backticks).\n"
        "- The SQL must start with SELECT and reference ONLY v_facility_listables.\n"
        "- Do NOT use UPDATE, INSERT, DELETE, DROP, ALTER, CREATE, PRAGMA or transactions.\n"
        "- Do NOT include multiple statements separated by ';'.\n"
        "- Prefer simple equality filters (e.g. state = 'FL', item_type = 'facility').\n"
        "- For facility-level aggregates, filter item_type = 'facility'.\n"
        "- For list questions, return columns like display_name and details.\n"
    )

    user_msg = (
        schema_desc
        + "\nUser question:\n"
        + question
        + "\n\n"
        "Write a single SELECT statement on v_facility_listables that best answers this question."
        " If the question is ambiguous or cannot be answered from these columns, still write the best safe SELECT you can.\n"
    )

    try:
        chat = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        )
        sql = (chat.choices[0].message.content or "").strip()
    except Exception:
        return None

    # Strip simple ```sql fences if present
    if sql.lower().startswith("```"):
        sql = sql.strip("`")
        lines = [ln for ln in sql.splitlines() if ln.strip()]
        for ln in lines:
            if ln.strip().lower().startswith("select"):
                sql = ln
                break

    sql_lower = sql.lower()

    # Safety checks
    if not sql_lower.startswith("select"):
        return None
    if " v_facility_listables" not in sql_lower and "from v_facility_listables" not in sql_lower:
        return None
    banned = ["update ", "insert ", "delete ", "drop ", "alter ", " create ", "pragma", ";", "--", "/*", "*/"]
    if any(b in sql_lower for b in banned):
        return None

    # Make text filters case-insensitive before running
    sql = _normalize_listables_sql_case(sql)

    # If the model forgot to include display_name in a simple list query,
    # rewrite the SELECT to return display_name and details so we show names
    # (home health agencies, facilities, etc.) instead of only "details".
    fixed_sql = sql
    sql_lc = sql.lower()
    is_aggregate = any(fn in sql_lc for fn in ["count(", "avg(", "sum(", "min(", "max("])

    if ("from v_facility_listables" in sql_lc) and ("display_name" not in sql_lc) and not is_aggregate:
        # Preserve DISTINCT if the model used it
        select_prefix = "SELECT DISTINCT" if sql_lc.strip().startswith("select distinct") else "SELECT"

        # Carry over the WHERE clause (if any) so we keep the same filters
        where_idx = sql_lc.find(" where ")
        where_clause = ""
        if where_idx != -1:
            # Use the original-casing slice for the WHERE part
            where_clause = sql[where_idx:]

        fixed_sql = f"{select_prefix} display_name, details FROM v_facility_listables{where_clause}"
        print("[listables-sql] Rewriting SELECT to prefer display_name:", fixed_sql)

    sql = fixed_sql

    cur = conn.cursor()
    try:
        cur.execute(sql)
        rows = cur.fetchall()
    except sqlite3.Error as e:
        print("[listables-sql] SQL error:", e, " SQL:", sql)
        return None


    # If the SQL ran but returned no rows, fall back to the normal pipeline
    # instead of giving a hard "no matching rows" message.
    if not rows:
        print("[listables-sql] No rows for SQL:", sql)
        return None


    col_names = [d[0] for d in cur.description] if cur.description else []
    src = "DB:listables"

    # Aggregate-style: 1 row, 1 column → just say "The result is X."
    if len(rows) == 1 and len(col_names) == 1:
        val = rows[0][0]
        return (f"The result is {val}.", [src], sql)

    # Otherwise treat it as a list
    lines_out: List[str] = []
    for r in rows:
        if "display_name" in col_names or "details" in col_names:
            dn = ""
            dt = ""
            if "display_name" in col_names:
                dn = str(r[col_names.index("display_name")] or "").strip()
            if "details" in col_names:
                dt = str(r[col_names.index("details")] or "").strip()
            if dn or dt:
                if dt:
                    lines_out.append(f" - {dn} ({dt})" if dn else f" - {dt}")
                else:
                    lines_out.append(f" - {dn}")
                continue

        vals = [str(v) for v in r if v is not None]
        if vals:
            lines_out.append(" - " + " | ".join(vals))

    label = facility_label or (facility_id or "the database")
    answer_text = f"Here are the results I found in v_facility_listables for '{question}' (scope: {label}):\n" + "\n".join(lines_out)
    return (answer_text, [src], sql)


def get_system_prompt(conn: sqlite3.Connection) -> str:
    cur = conn.cursor()

    try:
        cur.execute("SELECT value FROM ai_settings WHERE key='system_prompt'")
        row = cur.fetchone()
        if row and row["value"]:
            return row["value"]
    except sqlite3.Error:
        pass
    return (
        "You are Evolv's internal copilot. Use Evolv's database facts about "
        "facilities, processes, and tools first. If unsure, say so."
    )

# ---------------------------------------------------------------------------
# SNF admissions: AI extraction from CM notes
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# SNF admissions: AI extraction from CM notes
# ---------------------------------------------------------------------------

def _snf_latest_note_says_choice_is_not_final(latest_note_text: str) -> bool:
    """
    Returns True when the most recent note suggests the SNF choice is not finalized,
    is changing, or authorization isn't actually moving forward yet.
    This prevents stale facility names (ex: "The Luxe") from sticking as "probable"
    when the latest story is: undecided / changed mind / refused auth / etc.
    """
    t = (latest_note_text or "").lower()

    # "Choice not determined" / "will let me know" / "refused to give OK" signals uncertainty
    uncertainty_phrases = [
        "choice not determine",
        "choice not determined",
        "choice not finalize",
        "choice not final",
        "not determined by family",
        "not yet determined",
        "will let me know",
        "she will let me know",
        "he will let me know",
        "refused to give",
        "refused to give me the ok",
        "refused to give ok",
        "refused to authorize",
        "did not give consent",
        "no ok to request the auth",
        "no ok to request auth",
        "changed her mind",
        "changed his mind",
        "changed mind",
        "prefers",
        "now wants",
        "one more time",
    ]
    if any(p in t for p in uncertainty_phrases):
        return True

    return False


def _snf_note_text_has_negative_facility_outcome(note_text: str) -> bool:
    """
    True when the note indicates a facility is not viable (declined / no bed / not in network).
    """
    t = (note_text or "").lower()
    negative_phrases = [
        "declined",
        "no bed",
        "no beds",
        "not in network",
        "out of network",
        "denied the case",
        "denied case",
        "unable to accept",
        "cannot accept",
        "can't accept",
    ]
    return any(p in t for p in negative_phrases)


def _snf_should_clear_facility_name(notes: List[sqlite3.Row], snf_name: Optional[str]) -> Optional[str]:
    """
    If the latest note suggests the choice is not final, clear snf_name.
    Also clear snf_name if the most recent mention of that facility is negative (declined/no bed/out-of-network).
    Returns a short reason string if we cleared it, otherwise None.
    """
    if not snf_name or not notes:
        return None

    sorted_notes = sorted(notes, key=lambda r: (r["note_datetime"] or "", r["id"]))
    latest_text = (sorted_notes[-1]["note_text"] or "").strip()

    # 1) Latest note says choice isn't final / keeps changing
    if _snf_latest_note_says_choice_is_not_final(latest_text):
        return "Latest CM note indicates SNF choice is not finalized / is changing; cleared probable SNF name."

    # 2) If the latest *mention* of this facility is negative (declined / no bed / out-of-network), clear it
    snf_lower = snf_name.lower()
    for r in reversed(sorted_notes):
        txt = (r["note_text"] or "").lower()
        if snf_lower and snf_lower in txt:
            if _snf_note_text_has_negative_facility_outcome(txt):
                return "Most recent mention of that facility indicates it is not viable (declined/no bed/out-of-network); cleared probable SNF name."
            break

    return None


def analyze_patient_notes_with_llm(
    patient_mrn: str,
    notes: List[sqlite3.Row],
) -> Optional[Dict[str, Any]]:
    """
    Given all CM notes for a single patient (as rows from cm_notes_raw),
    call the LLM to decide:
      - is the patient going to a SNF?
      - which SNF name?
      - expected transfer date?
      - confidence (0–1)
    Returns a dict or None on failure.
    """
    if not notes:
        return None

    # Sort notes by note_datetime ascending for a chronological story
    sorted_notes = sorted(
        notes,
        key=lambda r: (r["note_datetime"] or "", r["id"]),
    )

    # Build a compact but clear representation of the notes
    note_chunks: List[str] = []
    for idx, r in enumerate(sorted_notes, start=1):
        dt_str = r["note_datetime"] or ""
        author = r["note_author"] or ""
        header_bits = [f"#{idx}", dt_str]
        if author:
            header_bits.append(f"by {author}")
        header = " ".join(b for b in header_bits if b)
        body = (r["note_text"] or "").strip()
        # Truncate extremely long notes to keep prompt bounded
        if len(body) > 4000:
            body = body[:4000] + " ...[truncated]"
        note_chunks.append(f"{header}:\n{body}")

    notes_block = "\n\n".join(note_chunks)

    system_msg = (
        "You are a discharge planning assistant reading hospital case management notes.\n"
        "Your job is to decide if the patient is being discharged or is likely to be discharged to a skilled nursing facility (SNF), "
        "and, if so, which facility and on what date.\n"
        "Treat terms like 'SNF', 'skilled nursing', 'skilled nursing facility', 'nursing home', 'inpatient SNF', "
        "'inpatient nursing', or generic 'rehab' when it clearly refers to a post-acute nursing facility as SNF-level care.\n"
        "If SNF is being actively considered, planned, or pending (choice letters, referrals, waiting on insurance), "
        "you should still treat the patient as an SNF candidate, but you can use a lower confidence.\n"
    )

    user_msg = (
        f"Patient MRN: {patient_mrn}\n\n"
        "Here are this patient's case management / discharge planning notes in chronological order:\n\n"
        f"{notes_block}\n\n"
        "From these notes, answer the following questions:\n"
        "1) Is there a clear plan for the patient to be discharged to a skilled nursing facility (SNF)?\n"
        "2) If yes, what is the name of the accepting SNF as written in the notes (even if abbreviated)?\n"
        "3) What is the expected transfer date (if clearly stated)?\n\n"
        "Return your answer as a single JSON object with EXACTLY these keys:\n"
        "{\n"
        '  \"is_snf_candidate\": true or false,\n'
        '  \"snf_name\": string or null,\n'
        '  \"expected_transfer_date\": \"YYYY-MM-DD\" or null,\n'
        '  \"confidence\": number between 0 and 1,\n'
        '  \"rationale\": string\n'
        "}\n"
        "Rules:\n"
        "- Read the notes in time order. Later notes override earlier notes.\n"
        "- If the patient is clearly accepted to a SNF, set is_snf_candidate to true.\n"
        "- If SNF is being pursued but the facility choice is NOT FINAL (family changing mind, 'choice not determined', waiting for family decision, refused to authorize),\n"
        "  set is_snf_candidate = TRUE but set snf_name = null and use lower confidence.\n"
        "- If multiple SNFs are mentioned over time, ONLY choose a facility name if the latest notes indicate that facility is the CURRENT selected/active plan\n"
        "  (examples: auth being started/submitted for that facility, confirmed placement, 'selected', 'accepted and proceeding').\n"
        "- Do NOT choose a facility as snf_name if notes indicate it was declined, has no bed, is out-of-network, or the family changed to another choice after it.\n"
        "\n"
        "- IMPORTANT: Insurance authorization language must be interpreted carefully:\n"
        "  • If notes say SNF authorization is PENDING / UNDER REVIEW / REQUEST SUBMITTED / REF# present / awaiting decision,\n"
        "    then is_snf_candidate should generally be TRUE (but confidence can be lower).\n"
        "  • If notes say SNF auth was DENIED but there is an APPEAL in progress OR a PEER-TO-PEER (P2P) is being offered/attempted\n"
        "    OR the denial is being contested (deadline, physician to call, expedited appeal), then keep is_snf_candidate = TRUE\n"
        "    until there is a clear FINAL outcome.\n"
        "  • If notes say the appeal was OVERTURNED / APPROVED / WON, treat SNF as continuing (is_snf_candidate = TRUE).\n"
        "  • If notes say the appeal was UPHELD / LOST / denial stands AND the plan changes to home / home health / other level of care,\n"
        "    then set is_snf_candidate = FALSE.\n"
        "\n"
        "- Only set is_snf_candidate = FALSE when the latest notes clearly indicate SNF is no longer the discharge plan\n"
        "  (examples: 'no longer going to SNF', 'plan changed to home with home health', 'family declined SNF', 'discharging home today').\n"
        "- If the date is not clear, use null for expected_transfer_date.\n"
        "- If the plan is uncertain but SNF is still being pursued (pending auth / appeal / P2P), you may set is_snf_candidate = TRUE\n"
        "  with confidence <= 0.5 and explain why in rationale.\n"
        "- Output ONLY the JSON object, with no extra text.\n"
    )


    try:
        chat = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        )
        raw = (chat.choices[0].message.content or "").strip()
    except Exception as e:
        print("[snf-llm] LLM error:", e)
        return None

    # Try to parse JSON
    try:
        data = json.loads(raw)
    except Exception:
        # Sometimes the model might wrap JSON in text; try a crude extract
        try:
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                data = json.loads(raw[start : end + 1])
            else:
                print("[snf-llm] Could not parse JSON from:", raw[:500])
                return None
        except Exception as e2:
            print("[snf-llm] JSON parse failed:", e2, " raw:", raw[:500])
            return None

    # Basic sanity checks / defaults
    is_snf_candidate = bool(data.get("is_snf_candidate"))
    snf_name = data.get("snf_name")
    if snf_name is not None:
        snf_name = str(snf_name).strip() or None

    expected_date = data.get("expected_transfer_date")
    if expected_date is not None:
        expected_date = str(expected_date).strip() or None

    try:
        conf = float(data.get("confidence", 0.0))
    except Exception:
        conf = 0.0
    conf = max(0.0, min(conf, 1.0))

    rationale = str(data.get("rationale", "") or "").strip()

    # Plan A: protect against stale facility names when the latest notes say the plan is changing/uncertain
    cleared_reason = _snf_should_clear_facility_name(notes, snf_name)
    if cleared_reason:
        snf_name = None
        # keep candidate true (SNF still being pursued), but reduce confidence
        if is_snf_candidate:
            conf = min(conf, 0.45)
        # append rationale so you can see why it cleared
        if rationale:
            rationale = f"{rationale} | {cleared_reason}"
        else:
            rationale = cleared_reason

    return {
        "is_snf_candidate": is_snf_candidate,
        "snf_name": snf_name,
        "expected_transfer_date": expected_date,
        "confidence": conf,
        "rationale": rationale,
    }


def _heuristic_disposition_from_text(source_text: str) -> Optional[str]:
    """
    Fast heuristic pass to avoid an LLM call when the disposition is obvious.
    Returns a normalized disposition string or None if unclear.
    """
    t = (source_text or "").lower()

    # strongest first
    if re.search(r"\bexpired\b|\bdeceased\b|\bdeath\b", t):
        return "Expired"
    if re.search(r"\bhospice\b|\bcomfort care\b", t):
        return "Hospice"
    if re.search(r"\bama\b|against medical advice", t):
        return "AMA"

    # IRF / inpatient rehab
    if re.search(r"\birf\b|inpatient rehab|inpatient rehabilitation|acute rehab", t):
        return "IRF"

    # SNF
    if re.search(r"\bsnf\b|skilled nursing|inpatient snf", t):
        return "SNF"

    # LTACH
    if re.search(r"\bltach\b|long[-\s]?term acute care", t):
        return "LTACH"

    # Home Health
    if re.search(r"home health|hha\b|homecare services|home care services", t):
        return "Home Health"

    # Home self-care
    if re.search(r"home[ /-]?self[ /-]?care|discharg(e|ed) home\b|\bhome\b.*self care", t):
        return "Home Self-care"

    # Generic rehab (when NOT clearly IRF)
    if re.search(r"\brehab\b|rehabilitation facility|subacute rehab|sar\b", t):
        return "Rehab"

    return None


def analyze_discharge_disposition_with_llm(source_text: str) -> Optional[Dict[str, Any]]:
    """
    Uses the LLM (with a heuristic shortcut) to determine discharge disposition AND discharge agency
    from raw hospital source_text.

    Returns:
      {
        "disposition": str,
        "dc_agency": str | null,
        "confidence": float,
        "evidence": str
      }
    """
    if not source_text or not source_text.strip():
        return None

    # 1) Cheap heuristic shortcut
    heuristic = _heuristic_disposition_from_text(source_text)
    if heuristic:
        return {"disposition": heuristic, "confidence": 0.85, "evidence": "heuristic"}

    # 2) LLM pass (trim to reduce token cost)
    text = source_text.strip()
    if len(text) > 12000:
        text = text[:12000]

    system_msg = (
        "You extract discharge disposition and discharge agency from messy hospital document text.\n"
        "Return ONLY a JSON object.\n"
        "\n"
        "1) Disposition:\n"
        "Pick exactly ONE disposition from this list:\n"
        "- Home Self-care\n"
        "- Home Health\n"
        "- SNF\n"
        "- IRF\n"
        "- Rehab\n"
        "- LTACH\n"
        "- Hospice\n"
        "- AMA\n"
        "- Expired\n"
        "- Other\n"
        "- Unknown\n"
        "\n"
        "2) dc_agency:\n"
        "- If disposition is SNF/IRF/Rehab/LTACH/Hospice, try to extract the destination facility name.\n"
        "- If disposition is Home Health, try to extract the home health agency name.\n"
        "- If the text contains multiple candidates, choose the one that is clearly tied to discharge planning.\n"
        "- If no agency/facility name is clearly present, set dc_agency to null.\n"
        "- Do NOT guess.\n"
        "\n"
        "Rules:\n"
        "- Prefer explicit discharge disposition / discharge destination lines if present.\n"
        "- If it says 'home with home health' => Home Health.\n"
        "- 'home' or 'home/self care' with no services => Home Self-care.\n"
        "- 'SNF', 'skilled nursing facility', 'inpatient SNF' => SNF.\n"
        "- 'IRF', 'inpatient rehab', 'acute rehab' => IRF.\n"
        "- If only 'rehab' with no IRF language => Rehab.\n"
        "- If unclear, use Unknown.\n"
        "- Include a short evidence quote (<= 140 chars) copied from the text.\n"
        "\n"
        "Output schema:\n"
        "  {\"disposition\":\".\",\"dc_agency\":null,\"confidence\":0.0,\"evidence\":\".\"}\n"
        "Notes:\n"
        "- dc_agency must be either a string or null.\n"
        "- confidence is 0.0-1.0.\n"
    )

    user_msg = f"Source text:\n{text}\n\nExtract discharge disposition and dc_agency."

    try:
        chat = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        )
        raw = (chat.choices[0].message.content or "").strip()
    except Exception as e:
        print("[dispo-llm] LLM error:", e)
        return None

    # Parse JSON (same robust pattern you used in snf-llm)
    try:
        data = json.loads(raw)
    except Exception:
        try:
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                data = json.loads(raw[start : end + 1])
            else:
                print("[dispo-llm] Could not parse JSON from:", raw[:500])
                return None
        except Exception as e2:
            print("[dispo-llm] JSON parse failed:", e2, " raw:", raw[:500])
            return None

    disp = str(data.get("disposition") or "").strip()
    evidence = str(data.get("evidence") or "").strip()

    try:
        conf = float(data.get("confidence", 0.0))
    except Exception:
        conf = 0.0
    conf = max(0.0, min(conf, 1.0))

    allowed = {
        "Home Self-care",
        "Home Health",
        "SNF",
        "IRF",
        "Rehab",
        "LTACH",
        "Hospice",
        "AMA",
        "Expired",
        "Other",
        "Unknown",
    }
    if disp not in allowed:
        disp = "Unknown"

    return {"disposition": disp, "confidence": conf, "evidence": evidence}




# ---------------------------------------------------------------------------
# Question → Answer pipeline
# ---------------------------------------------------------------------------

def run_answer_pipeline(
    question: str, section_hint: Optional[str], top_k: int
) -> AskResponse:
    conn = get_db()
    try:
        maps = load_dictionary_maps(conn)
        abbrev = maps["abbrev"]
        synonym_map = maps["synonym"]         
        facility_aliases = maps["facility_aliases"]

        normalized_q = normalize_question_text(question, abbrev, synonym_map)
        fac_id = detect_facility_id(normalized_q, conn, facility_aliases)
        fac_label = get_facility_label(fac_id, conn)

        # -------------------------------------------------
        # 1) Try analytics/listing path (Layer F)
        #    e.g. "list all contacts at The Pearl",
        #         "what is the average number of short term beds"
        # -------------------------------------------------
        analytics_cfg: Dict[str, Any] = {}
        try:
            cur = conn.cursor()
            cur.execute("SELECT value FROM ai_settings WHERE key='analytics_config'")
            row = cur.fetchone()
            if row and row["value"]:
                analytics_cfg = json.loads(row["value"])
        except Exception:
            analytics_cfg = {}

        analytics_intent = detect_analytics_intent(normalized_q, config=analytics_cfg)
        if analytics_intent:
            analytics_result = run_analytics_query(conn, analytics_intent, fac_id, fac_label)
            if analytics_result is not None:
                # run_analytics_query may return (answer, sources) or (answer, sources, sql_used)
                if len(analytics_result) == 3:
                    answer_text, sources, sql_used = analytics_result
                else:
                    answer_text, sources = analytics_result
                    sql_used = ""

                section_used = section_hint or ("Facility Details" if fac_id else "General")

                # Log to user_qa_log (same as normal path)
                cur = conn.cursor()
                cur.execute(
                    """
                    INSERT INTO user_qa_log (ts, section, q, a, promoted, a_quality, debug_sql)
                    VALUES (?, ?, ?, ?, 0, '', ?)
                    """,
                    (now_iso(), section_used, question, answer_text, sql_used),
                )
                log_id = cur.lastrowid
                conn.commit()

                return AskResponse(
                    answer=answer_text,
                    supporting_points=[],
                    sources=sources,
                    section_used=section_used,
                    log_id=log_id,
                )


        # -------------------------------------------------
        # 2) Hybrid LLM → SQL on v_facility_listables
        # -------------------------------------------------
        listables_result = try_llm_listables_query(conn, normalized_q, fac_id, fac_label)
        if listables_result is not None:
            # try_llm_listables_query returns (answer_text, sources, sql_used)
            if len(listables_result) == 3:
                answer_text, sources, sql_used = listables_result
            else:
                # backward-compat / safety
                answer_text, sources = listables_result
                sql_used = ""

            section_used = section_hint or ("Facility Details" if fac_id else "General")

            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO user_qa_log (ts, section, q, a, promoted, a_quality, debug_sql)
                VALUES (?, ?, ?, ?, 0, '', ?)
                """,
                (now_iso(), section_used, question, answer_text, sql_used),
            )
            log_id = cur.lastrowid
            conn.commit()

            return AskResponse(
                answer=answer_text,
                supporting_points=[],
                sources=sources,
                section_used=section_used,
                log_id=log_id,
            )


        # -------------------------------------------------
        # 2) Normal retrieval + LLM path
        # -------------------------------------------------
        qa_rows = search_qa_candidates(conn, normalized_q, section_hint, limit=40)
        fac_snippets = fetch_facility_knowledge(conn, fac_id, limit=40)

        ranked_chunks, sources, best_score = build_ranked_context(
            normalized_q,
            qa_rows,
            fac_snippets,
            top_k=top_k,
            section_hint=section_hint,
        )

        # Simple low-confidence heuristic:
        # - no chunks at all, or
        # - best weighted similarity is very low
        low_confidence = False
        if not ranked_chunks or best_score < 0.15:
            low_confidence = True

        context_lines = []
        for idx, c in enumerate(ranked_chunks, start=1):
            context_lines.append(f"[{idx}] {c['source']}\n{c['text']}\n")
        if context_lines:
            context_block = "\n".join(context_lines)
        else:
            context_block = "(no meaningful internal snippets were found for this question)"

        section_used = section_hint or ("Facility Details" if fac_id else "General")
        system_prompt = get_system_prompt(conn)

        confidence_label = "LOW" if low_confidence else "NORMAL"

        user_message = (
            f"User question:\n{question}\n\n"
            f"Detected facility: {fac_label or 'None'}\n"
            f"Context match confidence: {confidence_label}\n\n"
            f"Relevant internal snippets (Q&A + DB):\n{context_block}\n\n"
            "Instructions:\n"
            "- Use the internal snippets as your primary source of truth.\n"
            "- If snippets conflict, prefer the most specific / facility-specific ones.\n"
            "- You may use general medical knowledge only to fill small gaps.\n"
            "- If facility-specific details are present, be precise for that facility.\n"
        )

        if low_confidence:
            user_message += (
                "- The internal match is LOW confidence. If the snippets do not clearly "
                "answer the question, explicitly say that Evolv's database does not yet "
                "cover this, and avoid guessing. Suggest what information should be "
                "added to the Q&A or Facility Facts.\n"
            )
        else:
            user_message += (
                "- The internal match is NORMAL confidence. Still avoid making up "
                "facility-specific facts; rely on the snippets.\n"
            )

        user_message += (
            "- Begin with a concise, 1–3 sentence answer. Then, if useful, follow "
            "with bullet points for steps/nuances.\n"
        )


        chat = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        answer_text = (chat.choices[0].message.content or "").strip()

        # Log to user_qa_log
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO user_qa_log (ts, section, q, a, promoted, a_quality)
            VALUES (?, ?, ?, ?, 0, '')
            """,
            (now_iso(), section_used, question, answer_text),
        )
        log_id = cur.lastrowid
        conn.commit()


        return AskResponse(
            answer=answer_text,
            supporting_points=[],
            sources=sources,
            section_used=section_used,
            log_id=log_id,
        )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# /ask endpoint (used by TEST Ask.html)
# ---------------------------------------------------------------------------

@app.post("/ask", response_model=AskResponse)
async def ask_endpoint(payload: AskRequest):
    q = (payload.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="question is required")

    top_k = payload.top_k or 4
    top_k = max(1, min(top_k, 12))

    try:
        resp = run_answer_pipeline(q, payload.section_hint, top_k)
        return resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Feedback + Admin log endpoints
# ---------------------------------------------------------------------------

@app.post("/ulog/quality")
async def ulog_quality_public(
    id: int = Form(...),
    quality: str = Form(...),
):
    """Thumbs up/down endpoint used by Ask.html."""
    qv = (quality or "").strip().lower()
    if qv not in ("up", "down", ""):
        raise HTTPException(status_code=400, detail="quality must be 'up', 'down', or ''")

    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("SELECT id FROM user_qa_log WHERE id=?", (id,))
        if not cur.fetchone():
            raise HTTPException(status_code=404, detail="log row not found")
        cur.execute("UPDATE user_qa_log SET a_quality=? WHERE id=?", (qv, id))
        conn.commit()
        return {"ok": True, "id": id, "a_quality": qv}
    finally:
        conn.close()


async def _pad_read_json_body(request: Request) -> dict:
    """
    Robust JSON reader for PAD.
    - Accepts a normal JSON object
    - If PAD sends extra junk, tries to parse the first {...} block
    - If PAD sends invalid JSON (common when error_message has literal newlines),
      falls back to regex extraction for the key fields.
    """
    raw_bytes = await request.body()
    raw_text = raw_bytes.decode("utf-8", errors="ignore").strip()
    if not raw_text:
        return {}

    # 1) Try parsing as JSON
    try:
        payload = json.loads(raw_text)
        return payload if isinstance(payload, dict) else {"value": payload}
    except json.JSONDecodeError as e:
        print("[PAD JSON PARSE FAIL] primary json.loads:", e)

    # 2) Try extracting the first {...} block and parsing that
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            payload = json.loads(raw_text[start : end + 1])
            return payload if isinstance(payload, dict) else {"value": payload}
        except json.JSONDecodeError as e:
            print("[PAD JSON PARSE FAIL] extracted {...} json.loads:", e)

    # 3) LAST RESORT: regex extract key fields from invalid JSON
    # This specifically handles PAD error_message values that contain literal newlines.
    out: dict = {}

    m = re.search(r'"run_id"\s*:\s*"([^"]+)"', raw_text, flags=re.DOTALL)
    if m:
        out["run_id"] = m.group(1).strip()

    m = re.search(r'"flow_name"\s*:\s*"([^"]+)"', raw_text, flags=re.DOTALL)
    if m:
        out["flow_name"] = m.group(1).strip()

    m = re.search(r'"flow_key"\s*:\s*"([^"]+)"', raw_text, flags=re.DOTALL)
    if m:
        out["flow_key"] = m.group(1).strip()

    # error_message can span multiple lines; capture until the next key
    m = re.search(
        r'"error_message"\s*:\s*"(.+?)"\s*,\s*\n?\s*"run_id"\s*:',
        raw_text,
        flags=re.DOTALL,
    )
    if m:
        out["error_message"] = m.group(1).strip()

    return out

def _utc_now_text() -> str:
    return dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


@app.post("/api/pad/flow-log/start")
async def pad_flow_log_start(request: Request):
    """
    PAD calls this at flow start.
    Returns run_id which PAD should store and send on stop/error.
    """
    require_pad_api_key(request)
    raw_bytes = await request.body()
    raw_text = raw_bytes.decode("utf-8", errors="ignore").strip()
    payload = await _pad_read_json_body(request)
    log_pad_request_debug(request, raw_text, payload)


    flow_name = str(payload.get("flow_name") or "").strip()
    flow_key = str(payload.get("flow_key") or "").strip() or None

    if not flow_name:
        raise HTTPException(status_code=400, detail="flow_name is required")

    run_id = str(payload.get("run_id") or "").strip() or uuid.uuid4().hex

    # optional metadata
    flow_version = str(payload.get("flow_version") or "").strip() or None
    environment = str(payload.get("environment") or "").strip() or None
    machine_name = str(payload.get("machine_name") or "").strip() or None
    os_user = str(payload.get("os_user") or "").strip() or None
    triggered_by = str(payload.get("triggered_by") or "").strip() or None

    started_at = _utc_now_text()

    conn = get_db()
    try:
        cur = conn.cursor()

        # Insert run (idempotent: if PAD retries with same run_id, don't crash)
        cur.execute(
            """
            INSERT INTO pad_flow_runs (
                run_id, flow_name, flow_key, flow_version, environment,
                machine_name, os_user, triggered_by,
                started_at, status, start_payload_json, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'running', ?, datetime('now'))
            ON CONFLICT(run_id) DO UPDATE SET
                flow_name = excluded.flow_name,
                flow_key = excluded.flow_key,
                flow_version = excluded.flow_version,
                environment = excluded.environment,
                machine_name = excluded.machine_name,
                os_user = excluded.os_user,
                triggered_by = excluded.triggered_by,
                updated_at = datetime('now')
            """,
            (
                run_id, flow_name, flow_key, flow_version, environment,
                machine_name, os_user, triggered_by,
                started_at, json.dumps(payload),
            ),
        )

        # Start event
        cur.execute(
            """
            INSERT INTO pad_flow_events (run_id, event_type, event_ts, message, details_json)
            VALUES (?, 'start', ?, ?, ?)
            """,
            (run_id, started_at, "flow started", json.dumps(payload)),
        )

        conn.commit()
        return {"ok": True, "run_id": run_id, "started_at": started_at}
    finally:
        conn.close()


@app.post("/api/pad/flow-log/stop")
async def pad_flow_log_stop(request: Request):
    """
    PAD calls this at flow completion.
    Expects run_id from /start.
    """
    require_pad_api_key(request)
    payload = await _pad_read_json_body(request)

    run_id = str(payload.get("run_id") or "").strip()
    if not run_id:
        raise HTTPException(status_code=400, detail="run_id is required")

    ended_at = _utc_now_text()

    conn = get_db()
    try:
        cur = conn.cursor()

        # Update run to ok (compute duration_ms if started_at exists)
        cur.execute(
            """
            UPDATE pad_flow_runs
            SET
                ended_at = ?,
                status = 'ok',
                stop_payload_json = ?,
                duration_ms = CASE
                    WHEN started_at IS NOT NULL AND started_at <> ''
                    THEN CAST((julianday(?) - julianday(started_at)) * 86400000 AS INTEGER)
                    ELSE duration_ms
                END,
                updated_at = datetime('now')
            WHERE run_id = ?
            """,
            (ended_at, json.dumps(payload), ended_at, run_id),
        )

        # Stop event
        cur.execute(
            """
            INSERT INTO pad_flow_events (run_id, event_type, event_ts, message, details_json)
            VALUES (?, 'stop', ?, ?, ?)
            """,
            (run_id, ended_at, "flow stopped", json.dumps(payload)),
        )

        conn.commit()
        return {"ok": True, "run_id": run_id, "ended_at": ended_at}
    finally:
        conn.close()


@app.post("/api/pad/flow-log/error")
async def pad_flow_log_error(request: Request):
    require_pad_api_key(request)

    # --- RAW BODY READ (for logging) ---
    raw_bytes = await request.body()
    raw_text = raw_bytes.decode("utf-8", errors="ignore").strip()

    # --- PARSE USING PAD-SAFE PARSER ---
    payload = await _pad_read_json_body(request)

    # --- DEBUG LOG (Render will show this) ---
    log_pad_request_debug(request, raw_text, payload)

    # --- NORMAL VALIDATION ---
    run_id = str(payload.get("run_id") or "").strip()
    if not run_id:
        raise HTTPException(status_code=400, detail="run_id is required")

    error_message = str(payload.get("error_message") or "").strip() or "PAD flow error"
    error_details = payload.get("error_details")  # can be string or object
    step_name = str(payload.get("step_name") or "").strip() or None

    ended_at = _utc_now_text()

    conn = get_db()
    try:
        cur = conn.cursor()

        cur.execute(
            """
            UPDATE pad_flow_runs
            SET
                ended_at = COALESCE(ended_at, ?),
                status = 'error',
                error_message = ?,
                error_details_json = ?,
                duration_ms = CASE
                    WHEN started_at IS NOT NULL AND started_at <> ''
                    THEN CAST((julianday(?) - julianday(started_at)) * 86400000 AS INTEGER)
                    ELSE duration_ms
                END,
                updated_at = datetime('now')
            WHERE run_id = ?
            """,
            (
                ended_at,
                error_message,
                json.dumps(error_details) if error_details is not None else None,
                ended_at,
                run_id,
            ),
        )

        cur.execute(
            """
            INSERT INTO pad_flow_events (run_id, event_type, event_ts, step_name, message, details_json)
            VALUES (?, 'error', ?, ?, ?, ?)
            """,
            (
                run_id,
                ended_at,
                step_name,
                error_message,
                json.dumps(payload),
            ),
        )

        conn.commit()
        return {"ok": True, "run_id": run_id, "ended_at": ended_at, "error_message": error_message}
    finally:
        conn.close()


# ✅ NEW: PAD can ask what the last run was for a given flow_key (for resume logic)
@app.get("/api/pad/flow-log/last")
async def pad_flow_log_last(
    request: Request,
    flow_key: str = Query(..., description="PAD flow_key used in /api/pad/flow-log/start"),
):
    require_pad_api_key(request)

    fk = str(flow_key or "").strip()
    if not fk:
        raise HTTPException(status_code=400, detail="flow_key is required")

    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                run_id, flow_name, flow_key,
                status, error_message,
                started_at, ended_at, duration_ms
            FROM pad_flow_runs
            WHERE flow_key = ?
            ORDER BY started_at DESC
            LIMIT 1
            """,
            (fk,),
        )
        row = cur.fetchone()

        if not row:
            return {"ok": True, "found": False, "flow_key": fk}

        last_run = {
            "run_id": row["run_id"],
            "flow_name": row["flow_name"],
            "flow_key": row["flow_key"],
            "status": row["status"],
            "error_message": row["error_message"] or "",
            "started_at": row["started_at"] or "",
            "ended_at": row["ended_at"] or "",
            "duration_ms": row["duration_ms"],
        }

        return {"ok": True, "found": True, "flow_key": fk, "last_run": last_run}
    finally:
        conn.close()



def _is_snf_disposition(dispo: str) -> bool:
    """
    True if Hospital Discharge disposition indicates SNF / Skilled Nursing.
    """
    s = (dispo or "").strip().lower()
    if not s:
        return False
    return ("snf" in s) or ("skilled nursing" in s) or ("skilled" in s and "nursing" in s)


def _facility_display(final_display: str, effective_name: str, ai_raw: str) -> str:
    """
    Must match what the SNF Admissions Facility column shows:
      final_snf_name_display -> effective_facility_name -> ai_snf_name_raw
    """
    fd = (final_display or "").strip()
    if fd:
        return fd
    en = (effective_name or "").strip()
    if en:
        return en
    ar = (ai_raw or "").strip()
    if ar:
        return ar
    return "(Unknown)"


@app.get("/admin/ulog/list")
async def admin_ulog_list(
    request: Request,
    from_: Optional[str] = Query(None, alias="from"),
    to: Optional[str] = None,
    topics: Optional[str] = None,
    q: Optional[str] = None,
    quality: Optional[str] = None,
):
    require_admin(request)
    conn = get_db()
    try:
        where = []
        params: List[Any] = []
        if from_:
            where.append("ts >= ?")
            params.append(from_)
        if to:
            where.append("ts <= ?")
            params.append(to)
        if topics:
            # we don't have a topics column in the table; filter on section instead
            where.append("section LIKE ?")
            params.append(f"%{topics}%")
        if q:
            where.append("q LIKE ?")
            params.append(f"%{q}%")
        if quality:
            where.append("a_quality = ?")
            params.append(quality.lower())

        # NOTE: we no longer select 'topics' from the table
        sql = "SELECT id, ts, section, q, a, promoted, a_quality, debug_sql FROM user_qa_log"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY id DESC LIMIT 500"

        cur = conn.cursor()
        cur.execute(sql, params)
        rows = []
        for r in cur.fetchall():
            d = dict(r)
            # Admin UI expects a 'topics' field; synthesize from section
            d["topics"] = d.get("section", "") or ""
            rows.append(d)
        return {"rows": rows}
    finally:
        conn.close()




@app.post("/admin/ulog/delete")
async def admin_ulog_delete(request: Request, id: int = Form(...)):
    require_admin(request)
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM user_qa_log WHERE id=?", (id,))
        conn.commit()
        return {"ok": True}
    finally:
        conn.close()


@app.post("/admin/ulog/promote")
async def admin_ulog_promote(
    request: Request,
    id: int = Form(...),
    topics: str = Form(""),
    question: str = Form(""),
    answer: str = Form(""),
    tags: str = Form(""),
):
    """Promote a logged Q&A into the qa table."""
    require_admin(request)
    conn = get_db()
    try:
        cur = conn.cursor()
        # NOTE: we do NOT select a topics column from user_qa_log (it doesn't exist)
        cur.execute(
            "SELECT id, ts, section, q, a FROM user_qa_log WHERE id=?",
            (id,),
        )
        log_row = cur.fetchone()
        if not log_row:
            raise HTTPException(status_code=404, detail="log row not found")

        # Use the provided form topics if present; otherwise fall back to the log's section
        topics_f = (topics or log_row["section"] or "").strip()
        question_f = (question or log_row["q"] or "").strip()
        answer_f = (answer or log_row["a"] or "").strip()
        tags_f = (tags or "").strip()
        if not question_f or not answer_f:
            raise HTTPException(status_code=400, detail="question and answer are required")

        # Generate next QID (Q0001, Q0002, ...)
        cur.execute("SELECT id FROM qa")
        existing_ids = {str(r["id"]) for r in cur.fetchall() if r["id"] is not None}
        i = 1
        while True:
            candidate = f"Q{i:04d}"
            if candidate not in existing_ids:
                qid = candidate
                break
            i += 1

        cur.execute(
            """
            INSERT INTO qa (id, section, question, answer, tags, topics)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (qid, topics_f, question_f, answer_f, tags_f, topics_f),
        )
        cur.execute("UPDATE user_qa_log SET promoted=1 WHERE id=?", (id,))
        conn.commit()

        return {
            "ok": True,
            "added": {
                "id": qid,
                "section": topics_f,
                "question": question_f,
                "answer": answer_f,
                "tags": tags_f,
                "topics": topics_f,
            },
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Admin: SNF admissions extraction from CM notes
# ---------------------------------------------------------------------------

def notes_mention_possible_snf(notes: List[sqlite3.Row]) -> bool:
    """
    Heuristic: do any of these notes talk about SNF-like care at all,
    WITHOUT clearly cancelling or rejecting a SNF plan?

    Used to pull 'possible SNF' cases into the list even if the LLM is unsure.
    """
    # Work with all note text as a single lowercased string
    all_text = " ".join((r["note_text"] or "") for r in notes).lower()

    # If the notes explicitly say the SNF plan is cancelled / no longer happening,
    # do NOT treat this as a "possible SNF".
    negative_patterns = [
        "no longer going to snf",
        "no longer going to skilled nursing",
        "no longer going to a skilled nursing",
        "snf plan cancelled",
        "snf plan canceled",
        "snf placement cancelled",
        "snf placement canceled",
        "declined snf",
        "refused snf",
        "family declined snf",
        "no longer pursuing snf",
        "plan changed to home",
        "now going home",
        "now discharging home",
        "now going with home health instead of snf",
        "home with home health instead of snf",
    ]
    for phrase in negative_patterns:
        if phrase in all_text:
            return False

    # Positive SNF-ish keywords
    keywords = [
        "snf",
        "skilled nursing",
        "skilled nursing facility",
        "nursing home",
        "inpatient snf",
        "inpatient nursing",
        "rehab",
    ]
    for kw in keywords:
        if kw in all_text:
            return True

    return False

def denial_is_contested_or_pending(notes: List[sqlite3.Row]) -> bool:
    """
    Returns True when the notes indicate SNF auth was denied BUT there is still
    an active contest process (appeal / peer-to-peer / pending review), meaning
    we should NOT auto-remove the SNF prospect yet.
    """
    all_text = " ".join((r["note_text"] or "") for r in notes).lower()

    denied = any(p in all_text for p in [
        "denied", "denial", "auth has been denied", "authorization has been denied", "snf auth has been denied"
    ])
    if not denied:
        return False

    still_in_play = any(p in all_text for p in [
        "appeal", "appealing", "expedited appeal", "grievance",
        "peer to peer", "peer-to-peer", "p2p",
        "offered", "deadline", "call", "before",
        "pending", "under review", "still pending", "awaiting", "in review", "ref#"
    ])
    if not still_in_play:
        return False

    # If the notes clearly say they are going home / HH is arranged / pickup today,
    # treat that as SNF no longer the plan (even if earlier denial was contested).
    clearly_not_snf_now = any(p in all_text for p in [
        "discharging home", "going home", "daughter will be picking", "family picking",
        "home health", "hh referral", "no further cm needs"
    ])
    return not clearly_not_snf_now

def snf_upsert_from_notes_for_visit(conn: sqlite3.Connection, visit_id: str) -> bool:
    """
    Runs the same 'create/update SNF admission from notes + AI facility mapping' logic
    for a single visit_id (used by manual note add).
    Returns True if a SNF admission row exists/was updated, False if not a SNF candidate.
    """
    cur = conn.cursor()

    cur.execute(
        """
        SELECT *
        FROM cm_notes_raw
        WHERE visit_id = ?
          AND (ignored IS NULL OR ignored = 0)
        ORDER BY datetime(note_datetime) DESC, id DESC
        """,
        (visit_id,),
    )
    notes = cur.fetchall()
    if not notes:
        return False

    latest = notes[0]
    patient_mrn = latest["patient_mrn"]
    patient_name = latest["patient_name"]
    hospital_name = latest["hospital_name"]
    admit_date = latest["admit_date"] or latest["admission_date"]
    attending = latest["attending"] or latest["note_author"]

    # Build a combined note text similar to the pipeline
    note_blob = "\n\n".join([(n["note_text"] or "") for n in notes if (n["note_text"] or "").strip()])

    # Use your existing SNF detection helper(s)
    is_candidate = False
    try:
        # Prefer your helper if present in your file
        if "notes_mention_possible_snf" in globals():
            is_candidate = bool(notes_mention_possible_snf(note_blob))
        else:
            # Basic fallback: do not change pipeline behavior if helper exists
            is_candidate = "snf" in note_blob.lower()
    except Exception:
        is_candidate = False

    if not is_candidate:
        return False

    # Run your existing AI extraction (same function used in snf_run_extraction)
    ai = analyze_patient_notes_with_llm(note_blob)

    snf_name_raw = (ai.get("snf_name") or "").strip()
    transfer_date = ai.get("transfer_date") or None
    disposition = ai.get("disposition") or "SNF"
    status = ai.get("status") or "pending"
    note_summary = ai.get("summary") or None

    facility_id = None
    facility_label = None

    # Map facility name → facility_id using your existing mapping helper if present
    try:
        if "map_snf_name_to_facility_id" in globals():
            facility_id, facility_label = map_snf_name_to_facility_id(conn, snf_name_raw)
    except Exception:
        facility_id, facility_label = None, None

    # Upsert into snf_admissions (same pattern as snf_run_extraction)
    cur.execute(
        """
        INSERT INTO snf_admissions (
            visit_id,
            patient_mrn,
            patient_name,
            hospital_name,
            admit_date,
            attending,
            ai_disposition,
            ai_snf_name_raw,
            ai_transfer_date,
            ai_summary,
            facility_id,
            facility_label,
            status,
            created_at,
            updated_at,
            last_seen_active_date
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'), date('now'))
        ON CONFLICT(visit_id) DO UPDATE SET
            patient_mrn = excluded.patient_mrn,
            patient_name = excluded.patient_name,
            hospital_name = excluded.hospital_name,
            admit_date = excluded.admit_date,
            attending = excluded.attending,
            ai_disposition = excluded.ai_disposition,
            ai_snf_name_raw = excluded.ai_snf_name_raw,
            ai_transfer_date = excluded.ai_transfer_date,
            ai_summary = excluded.ai_summary,
            facility_id = excluded.facility_id,
            facility_label = excluded.facility_label,
            status = excluded.status,
            updated_at = datetime('now'),
            last_seen_active_date = date('now')
        """,
        (
            visit_id,
            patient_mrn,
            patient_name,
            hospital_name,
            admit_date,
            attending,
            disposition,
            snf_name_raw,
            transfer_date,
            note_summary,
            facility_id,
            facility_label,
            status,
        ),
    )

    return True


def snf_run_extraction(days_back: int = 3) -> Dict[str, Any]:
    """
    Internal helper to analyze recent CM notes and populate snf_admissions.
    Shared by the admin endpoint and the PAD ingest endpoint.
    """
    conn = get_db()
    try:
        cur = conn.cursor()

        # Load facility alias map (dictionary.kind='facility_alias')
        maps = load_dictionary_maps(conn)
        facility_aliases = maps["facility_aliases"]

        # Pull recent CM notes
        cur.execute(
            """
            SELECT *
            FROM cm_notes_raw
            WHERE created_at >= datetime('now', ?)
              AND (ignored IS NULL OR ignored = 0)
            ORDER BY patient_mrn, note_datetime, id
            """,
            (f"-{days_back} days",),
        )
        rows = cur.fetchall()
        if not rows:
            return {
                "ok": True,
                "message": f"No cm_notes_raw rows found in the last {days_back} days.",
                "patients_processed": 0,
                "snf_candidates": 0,
            }

        # Group by admission: prefer visit_id, fall back to patient_mrn
        notes_by_admission: Dict[str, List[sqlite3.Row]] = {}
        for r in rows:
            mrn = (r["patient_mrn"] or "").strip()
            visit_id = (r["visit_id"] or "").strip() if "visit_id" in r.keys() else ""
            key = visit_id or mrn
            if not key:
                continue
            notes_by_admission.setdefault(key, []).append(r)

        patients_processed = 0
        snf_candidates = 0

        for adm_key, notes in notes_by_admission.items():
            # Heuristic: do notes mention SNF-like terms at all
            # (ignoring explicit "no longer SNF" phrases)?
            has_snf_keywords = notes_mention_possible_snf(notes)

            # Derive MRN for the LLM from the first note in this admission group
            mrn_for_llm = (notes[0]["patient_mrn"] or "").strip()
            result = analyze_patient_notes_with_llm(mrn_for_llm, notes)

            # Always identify the latest note + admission identifiers up front
            latest_note = sorted(
                notes,
                key=lambda r: (r["note_datetime"] or "", r["id"]),
            )[-1]

            visit_id = ""
            if "visit_id" in latest_note.keys():
                visit_id = (latest_note["visit_id"] or "").strip()
            visit_id_db = visit_id or None

            patient_mrn = (latest_note["patient_mrn"] or "").strip()
            patient_name = latest_note["patient_name"] or ""
            hospital_name = latest_note["hospital_name"] or ""
            note_datetime = latest_note["note_datetime"] or ""

            # 1) If the LLM explicitly says "no SNF plan" for this admission,
            #    treat this as a cancellation of any prior SNF admission.
            #    BUT: do not auto-remove if auth was denied and is still being contested (appeal/P2P/pending).
            if result and not result.get("is_snf_candidate", False):
                if denial_is_contested_or_pending(notes):
                    # Keep it visible for review instead of removing the prospect prematurely
                    result = {
                        "is_snf_candidate": True,
                        "snf_name": None,
                        "expected_transfer_date": None,
                        "confidence": 0.35,
                        "rationale": (
                            "Insurance denial is mentioned, but notes indicate appeal/peer-to-peer/"
                            "pending review is still in progress; keep as SNF prospect until final outcome."
                        ),
                    }
                else:
                    if visit_id_db:
                        cur.execute(
                            """
                            UPDATE snf_admissions
                               SET ai_is_snf_candidate = 0,
                                   status = 'removed',
                                   review_comments = COALESCE(review_comments, '') ||
                                       ' [auto-removed based on later CM notes: plan no longer SNF]',
                                   last_seen_active_date = date('now')
                             WHERE visit_id = ?
                            """,
                            (visit_id_db,),
                        )
                    else:
                        # Fallback for legacy rows that may only have MRN
                        cur.execute(
                            """
                            UPDATE snf_admissions
                               SET ai_is_snf_candidate = 0,
                                   status = 'removed',
                                   review_comments = COALESCE(review_comments, '') ||
                                       ' [auto-removed based on later CM notes: plan no longer SNF]',
                                   last_seen_active_date = date('now')
                             WHERE visit_id IS NULL
                               AND patient_mrn = ?
                            """,
                            (patient_mrn,),
                        )

                    patients_processed += 1
                    # Nothing more to do for this admission
                    continue


            # 2) If the LLM gave nothing back AND there is no clear SNF-ish language, skip.
            if not result and not has_snf_keywords:
                continue

            patients_processed += 1

            # 3) If the LLM failed but we clearly see SNF-related language,
            #    treat as a low-confidence 'possible SNF' candidate so it still
            #    shows up for manual review in the UI.
            if not result and has_snf_keywords:
                result = {
                    "is_snf_candidate": True,
                    "snf_name": None,
                    "expected_transfer_date": None,
                    "confidence": 0.4,
                    "rationale": "Heuristic: CM notes mention SNF-related terms (rehab/skilled nursing/nursing home), but the LLM could not parse a clean plan.",
                }

            if not result or not result.get("is_snf_candidate", False):
                # Still not a candidate even after heuristic
                continue

            # At this point we have a positive SNF candidate
            snf_candidates += 1

            snf_name_raw = result.get("snf_name")
            expected_date = result.get("expected_transfer_date")
            conf = result.get("confidence", 0.0)

            # Map SNF name -> facility_id (if possible)
            ai_facility_id = map_snf_name_to_facility_id(snf_name_raw, conn, facility_aliases)

            # Upsert into snf_admissions: one row per visit_id (admission)
            cur.execute(
                """
                INSERT INTO snf_admissions (
                    raw_note_id,
                    visit_id,
                    patient_mrn,
                    patient_name,
                    dob,
                    attending,
                    admit_date,
                    hospital_name,
                    note_datetime,
                    ai_is_snf_candidate,
                    ai_snf_name_raw,
                    ai_snf_facility_id,
                    ai_expected_transfer_date,
                    ai_confidence,
                    status,
                    last_seen_active_date,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', date('now'), datetime('now'))
                ON CONFLICT(visit_id) DO UPDATE SET
                    raw_note_id               = excluded.raw_note_id,
                    patient_mrn               = excluded.patient_mrn,
                    patient_name              = excluded.patient_name,
                    dob                       = excluded.dob,
                    attending                 = excluded.attending,
                    admit_date                = excluded.admit_date,
                    hospital_name             = excluded.hospital_name,
                    note_datetime             = excluded.note_datetime,
                    ai_is_snf_candidate       = excluded.ai_is_snf_candidate,
                    ai_snf_name_raw           = excluded.ai_snf_name_raw,
                    ai_snf_facility_id        = excluded.ai_snf_facility_id,
                    ai_expected_transfer_date = excluded.ai_expected_transfer_date,
                    ai_confidence             = excluded.ai_confidence,
                    status = CASE
                               WHEN snf_admissions.status IN ('removed','confirmed','projected')
                                 THEN snf_admissions.status
                               ELSE 'pending'
                            END,
                    last_seen_active_date     = snf_admissions.last_seen_active_date
                ;
                """,
                (
                    latest_note["id"],
                    visit_id_db,
                    patient_mrn,
                    patient_name,

                    # NEW:
                    latest_note["dob"] if "dob" in latest_note.keys() else None,
                    (
                        (latest_note["attending"] if "attending" in latest_note.keys() else None)
                        or (latest_note["note_author"] if "note_author" in latest_note.keys() else None)
                        or ""
                    ),
                    (
                        (latest_note["admit_date"] if "admit_date" in latest_note.keys() else None)
                        or (latest_note["admission_date"] if "admission_date" in latest_note.keys() else None)
                        or ""
                    ),

                    hospital_name,
                    note_datetime,
                    1,
                    snf_name_raw,
                    ai_facility_id,
                    expected_date,
                    conf,
                ),

            )




        conn.commit()

        return {
            "ok": True,
            "days_back": days_back,
            "patients_processed": patients_processed,
            "snf_candidates": snf_candidates,
        }
    finally:
        conn.close()

@app.get("/admin/snf/cm-notes/hospitals")
def admin_cm_notes_hospitals(admin=Depends(require_admin)):
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT DISTINCT TRIM(hospital_name) AS hospital_name
            FROM cm_notes_raw
            WHERE hospital_name IS NOT NULL
              AND TRIM(hospital_name) <> ''
            ORDER BY TRIM(hospital_name) COLLATE NOCASE
            """
        )
        hospitals = [r["hospital_name"] for r in cur.fetchall() if r["hospital_name"]]
        return {"ok": True, "hospitals": hospitals}
    finally:
        conn.close()

@app.get("/admin/hospital-documents/hospitals")
def admin_hospital_documents_hospitals(admin=Depends(require_admin)):
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT hospital_name FROM (
              SELECT DISTINCT TRIM(hospital_name) AS hospital_name
              FROM hospital_documents
              WHERE hospital_name IS NOT NULL AND TRIM(hospital_name) <> ''
              UNION
              SELECT DISTINCT TRIM(hospital_name) AS hospital_name
              FROM hospital_discharges
              WHERE hospital_name IS NOT NULL AND TRIM(hospital_name) <> ''
            )
            ORDER BY hospital_name COLLATE NOCASE
            """
        )
        hospitals = [r[0] for r in cur.fetchall() if r and r[0]]
        return {"ok": True, "hospitals": hospitals}
    finally:
        conn.close()


@app.post("/admin/hospital-documents/manual-add")
async def admin_hospital_documents_manual_add(payload: ManualHospitalDocumentIn, admin=Depends(require_admin)):
    """
    Manual hospital document add for the Hospital Discharge HTML page.

    IMPORTANT: This intentionally calls the SAME ingest function used by the API:
      /api/hospital-documents/ingest
    so it triggers the same downstream behavior (section splitting, discharge upsert, etc.).
    """
    ingest_payload = {
        "hospital_name": (payload.hospital_name or "").strip(),
        "document_type": (payload.document_type or "").strip(),
        "source_text": (payload.source_text or "").strip(),
        "visit_id": (payload.visit_id or "").strip() or None,
        "document_datetime": (payload.document_datetime or "").strip() or None,
        "patient_mrn": (payload.patient_mrn or "").strip() or None,
        "patient_name": (payload.patient_name or "").strip() or None,
        "dob": (payload.dob or "").strip() or None,
        "admit_date": (payload.admit_date or "").strip() or None,
        "dc_date": (payload.dc_date or "").strip() or None,
        "attending": (payload.attending or "").strip() or None,
        "pcp": (payload.pcp or "").strip() or None,
        "insurance": (payload.insurance or "").strip() or None,
        "source_system": (payload.source_system or "EMR").strip() or "EMR",
    }

    # extra guardrails for Hospital Discharge page usability
    if not ingest_payload["visit_id"]:
        raise HTTPException(status_code=400, detail="visit_id is required for manual add from Hospital Discharge page")

    # Calls the real ingest logic (same as API path)
    return await hospital_documents_ingest(ingest_payload)


@app.post("/admin/snf/cm-notes/manual-add")
def admin_manual_add_cm_note(payload: ManualCmNoteIn, admin=Depends(require_admin)):
    conn = get_db()
    try:
        cur = conn.cursor()

        patient_mrn = (payload.patient_mrn or "").strip()
        visit_id = (payload.visit_id or "").strip()
        note_datetime = (payload.note_datetime or "").strip()
        note_text = payload.note_text or ""

        if not patient_mrn or not visit_id or not note_datetime or not note_text.strip():
            raise HTTPException(
                status_code=400,
                detail="patient_mrn, visit_id, note_datetime, and note_text are required",
            )

        # Normalize (match PAD behavior)
        note_hash = compute_note_hash(patient_mrn, note_datetime, note_text)

        # Skip if already exists
        cur.execute("SELECT 1 FROM cm_notes_raw WHERE note_hash = ?", (note_hash,))
        if cur.fetchone():
            return {"ok": True, "message": "Duplicate note (same hash) already exists. No changes made."}

        # Keep admissions marked active when a note arrives
        cur.execute(
            "UPDATE snf_admissions SET last_seen_active_date = date('now') WHERE visit_id = ?",
            (visit_id,),
        )

        cur.execute(
            """
            INSERT INTO cm_notes_raw (
                patient_mrn,
                patient_name,
                dob,
                visit_id,
                encounter_id,
                hospital_name,
                unit_name,
                admission_date,
                admit_date,
                attending,
                note_datetime,
                note_author,
                note_type,
                note_text,
                source_system,
                pad_run_id,
                ocr_confidence,
                note_hash,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """,
            (
                patient_mrn,
                payload.patient_name,
                payload.dob,
                visit_id or None,
                payload.encounter_id,
                payload.hospital_name,
                payload.unit_name,
                payload.admission_date,
                payload.admit_date or payload.admission_date,
                payload.attending or payload.note_author,
                note_datetime,
                payload.note_author,
                payload.note_type or "Case Management",
                note_text,
                payload.source_system or "EMR",
                payload.pad_run_id,
                payload.ocr_confidence,
                note_hash,
            ),
        )

        # Run the same SNF + AI logic for this visit_id
        updated = snf_upsert_from_notes_for_visit(conn, visit_id)

        conn.commit()

        if updated:
            return {"ok": True, "message": f"Manual CM note added. SNF admission AI updated for visit_id={visit_id}."}
        else:
            return {"ok": True, "message": f"Manual CM note added. Visit {visit_id} not flagged as SNF candidate by AI logic."}

    finally:
        conn.close()


@app.post("/admin/snf/run-extraction")
async def admin_snf_run_extraction(
    request: Request,
    days_back: int = Query(0, ge=1, le=30),
):
    """
    Admin-only endpoint to analyze recent CM notes and populate snf_admissions.

    - Looks at cm_notes_raw.created_at within the last `days_back` days.
    - Groups notes by patient_mrn.
    - For each patient, calls the LLM to decide SNF status / facility / date.
    - Inserts or updates snf_admissions rows (status='pending').
    """
    require_admin(request)
    return snf_run_extraction(days_back=days_back)

# ===================== NEW: recompute SNF AI for one admission =====================

def snf_recompute_for_admission(visit_id: str = "", patient_mrn: str = "") -> Dict[str, Any]:
    """
    Recompute SNF AI for exactly one admission group (visit_id preferred, else patient_mrn).
    Uses ONLY non-ignored notes.
    """
    visit_id = (visit_id or "").strip()
    patient_mrn = (patient_mrn or "").strip()
    if not visit_id and not patient_mrn:
        return {"ok": False, "message": "visit_id or patient_mrn is required"}

    conn = get_db()
    try:
        cur = conn.cursor()

        # Load facility alias map (dictionary.kind='facility_alias')
        maps = load_dictionary_maps(conn)
        facility_aliases = maps["facility_aliases"]

        if visit_id:
            cur.execute(
                """
                SELECT *
                FROM cm_notes_raw
                WHERE visit_id = ?
                  AND (ignored IS NULL OR ignored = 0)
                ORDER BY note_datetime, id
                """,
                (visit_id,),
            )
        else:
            cur.execute(
                """
                SELECT *
                FROM cm_notes_raw
                WHERE patient_mrn = ?
                  AND (ignored IS NULL OR ignored = 0)
                ORDER BY note_datetime, id
                """,
                (patient_mrn,),
            )

        notes = cur.fetchall()
        if not notes:
            # Nothing left → mark any matching snf_admissions row removed
            if visit_id:
                cur.execute(
                    """
                    UPDATE snf_admissions
                    SET ai_is_snf_candidate = 0,
                        status = 'removed',
                        updated_at = datetime('now')
                    WHERE visit_id = ?
                    """,
                    (visit_id,),
                )
            else:
                cur.execute(
                    """
                    UPDATE snf_admissions
                    SET ai_is_snf_candidate = 0,
                        status = 'removed',
                        updated_at = datetime('now')
                    WHERE visit_id IS NULL
                      AND patient_mrn = ?
                    """,
                    (patient_mrn,),
                )
            conn.commit()
            return {"ok": True, "message": "No notes left; SNF row marked removed."}

        has_snf_keywords = notes_mention_possible_snf(notes)
        mrn_for_llm = (notes[0]["patient_mrn"] or "").strip()
        result = analyze_patient_notes_with_llm(mrn_for_llm, notes)

        # Identify latest note for identifying fields + linking
        latest_note = sorted(notes, key=lambda r: (r["note_datetime"] or "", r["id"]))[-1]
        visit_id_eff = (latest_note["visit_id"] or "").strip() if "visit_id" in latest_note.keys() else ""
        visit_id_db = visit_id_eff or (visit_id or None)
        mrn_eff = (latest_note["patient_mrn"] or "").strip() or patient_mrn
        patient_name = latest_note["patient_name"] or ""
        hospital_name = latest_note["hospital_name"] or ""
        note_datetime = latest_note["note_datetime"] or ""

        # If LLM says "not SNF", treat as removal unless denial is still contested
        if result and not result.get("is_snf_candidate", False):
            if denial_is_contested_or_pending(notes):
                result = {
                    "is_snf_candidate": True,
                    "snf_name": None,
                    "expected_transfer_date": None,
                    "confidence": 0.35,
                    "rationale": (
                        "Insurance denial mentioned, but appeal/peer-to-peer/pending review is still in progress."
                    ),
                }
            else:
                # mark not candidate
                if visit_id_db:
                    cur.execute(
                        """
                        UPDATE snf_admissions
                        SET raw_note_id = ?,
                            note_datetime = ?,
                            ai_is_snf_candidate = 0,
                            status = CASE WHEN status='confirmed' THEN status ELSE 'removed' END,
                            updated_at = datetime('now')
                        WHERE visit_id = ?
                        """,
                        (latest_note["id"], note_datetime, visit_id_db),
                    )
                else:
                    cur.execute(
                        """
                        UPDATE snf_admissions
                        SET raw_note_id = ?,
                            note_datetime = ?,
                            ai_is_snf_candidate = 0,
                            status = CASE WHEN status='confirmed' THEN status ELSE 'removed' END,
                            updated_at = datetime('now')
                        WHERE visit_id IS NULL
                          AND patient_mrn = ?
                        """,
                        (latest_note["id"], note_datetime, mrn_eff),
                    )
                conn.commit()
                return {"ok": True, "message": "Recomputed: no longer SNF candidate."}

        # If LLM gave nothing and there are no SNF-ish keywords, do nothing (but still update latest raw_note_id)
        if not result and not has_snf_keywords:
            # keep row as-is but ensure it points to latest note
            if visit_id_db:
                cur.execute(
                    """
                    UPDATE snf_admissions
                    SET raw_note_id = ?,
                        note_datetime = COALESCE(?, note_datetime),
                        updated_at = datetime('now')
                    WHERE visit_id = ?
                    """,
                    (latest_note["id"], note_datetime, visit_id_db),
                )
            else:
                cur.execute(
                    """
                    UPDATE snf_admissions
                    SET raw_note_id = ?,
                        note_datetime = COALESCE(?, note_datetime),
                        updated_at = datetime('now')
                    WHERE visit_id IS NULL
                      AND patient_mrn = ?
                    """,
                    (latest_note["id"], note_datetime, mrn_eff),
                )
            conn.commit()
            return {"ok": True, "message": "Recomputed: no SNF signal found."}

        # If LLM failed but keywords exist, force low-confidence candidate
        if not result and has_snf_keywords:
            result = {
                "is_snf_candidate": True,
                "snf_name": None,
                "expected_transfer_date": None,
                "confidence": 0.4,
                "rationale": "Heuristic: SNF-ish terms present, but LLM did not return a structured plan.",
            }

        if not result or not result.get("is_snf_candidate", False):
            # same “no longer candidate” treatment
            if visit_id_db:
                cur.execute(
                    """
                    UPDATE snf_admissions
                    SET raw_note_id = ?,
                        note_datetime = ?,
                        ai_is_snf_candidate = 0,
                        status = CASE WHEN status='confirmed' THEN status ELSE 'removed' END,
                        updated_at = datetime('now')
                    WHERE visit_id = ?
                    """,
                    (latest_note["id"], note_datetime, visit_id_db),
                )
            else:
                cur.execute(
                    """
                    UPDATE snf_admissions
                    SET raw_note_id = ?,
                        note_datetime = ?,
                        ai_is_snf_candidate = 0,
                        status = CASE WHEN status='confirmed' THEN status ELSE 'removed' END,
                        updated_at = datetime('now')
                    WHERE visit_id IS NULL
                      AND patient_mrn = ?
                    """,
                    (latest_note["id"], note_datetime, mrn_eff),
                )
            conn.commit()
            return {"ok": True, "message": "Recomputed: not a SNF candidate."}

        snf_name_raw = result.get("snf_name")
        expected_date = result.get("expected_transfer_date")
        conf = result.get("confidence", 0.0)
        ai_facility_id = map_snf_name_to_facility_id(snf_name_raw, conn, facility_aliases)

        # Update the existing snf_admissions row for this admission group
        if visit_id_db:
            cur.execute(
                """
                UPDATE snf_admissions
                SET raw_note_id = ?,
                    note_datetime = ?,
                    ai_is_snf_candidate = 1,
                    ai_snf_name_raw = ?,
                    ai_snf_facility_id = ?,
                    ai_expected_transfer_date = ?,
                    ai_confidence = ?,
                    status = CASE WHEN status IN ('confirmed','projected') THEN status ELSE 'pending' END,
                    updated_at = datetime('now')
                WHERE visit_id = ?
                """,
                (latest_note["id"], note_datetime, snf_name_raw, ai_facility_id, expected_date, conf, visit_id_db),
            )
        else:
            cur.execute(
                """
                UPDATE snf_admissions
                SET raw_note_id = ?,
                    note_datetime = ?,
                    ai_is_snf_candidate = 1,
                    ai_snf_name_raw = ?,
                    ai_snf_facility_id = ?,
                    ai_expected_transfer_date = ?,
                    ai_confidence = ?,
                    status = CASE WHEN status IN ('confirmed','projected') THEN status ELSE 'pending' END,
                    updated_at = datetime('now')
                WHERE visit_id IS NULL
                  AND patient_mrn = ?
                """,
                (latest_note["id"], note_datetime, snf_name_raw, ai_facility_id, expected_date, conf, mrn_eff),
            )

        conn.commit()
        return {"ok": True, "message": "Recomputed: SNF candidate updated."}

    finally:
        conn.close()


@app.post("/admin/snf/admission/recompute")
async def admin_snf_admission_recompute(request: Request, payload: Dict[str, Any] = Body(...)):
    """
    Recompute SNF AI for one admission after note ignore.
    Payload: { visit_id: "...", patient_mrn: "..." }
    """
    require_admin(request)
    visit_id = (payload.get("visit_id") or "").strip()
    patient_mrn = (payload.get("patient_mrn") or "").strip()
    return snf_recompute_for_admission(visit_id=visit_id, patient_mrn=patient_mrn)


# ===================== NEW: recompute SNF AI for ALL admissions =====================

@app.post("/admin/snf/recompute-all")
async def admin_snf_recompute_all(request: Request):
    """
    Recompute SNF AI for all admissions (admin-only).
    - Uses ONLY non-ignored notes (snf_recompute_for_admission does that).
    - Also covers admissions that now have 0 remaining notes (marks removed).
    """
    require_admin(request)

    conn = get_db()
    try:
        cur = conn.cursor()

        # Build a set of "admission groups" to recompute.
        # Prefer visit_id when present; otherwise fall back to patient_mrn.
        groups: set[tuple[str, str]] = set()

        # 1) From SNF table (so we recompute rows that already exist)
        cur.execute("SELECT DISTINCT COALESCE(visit_id, '') AS visit_id, COALESCE(patient_mrn, '') AS patient_mrn FROM snf_admissions")
        for r in cur.fetchall():
            v = (r["visit_id"] or "").strip()
            m = (r["patient_mrn"] or "").strip()
            if v or m:
                groups.add((v, m))

        # 2) From notes (so we also recompute any note-groups that might not yet have a row)
        cur.execute(
            """
            SELECT DISTINCT
                COALESCE(visit_id, '') AS visit_id,
                COALESCE(patient_mrn, '') AS patient_mrn
            FROM cm_notes_raw
            WHERE (ignored IS NULL OR ignored = 0)
            """
        )
        for r in cur.fetchall():
            v = (r["visit_id"] or "").strip()
            m = (r["patient_mrn"] or "").strip()
            if v or m:
                groups.add((v, m))

        total = len(groups)
        ok_count = 0
        fail_count = 0
        failures: list[dict[str, str]] = []

        # IMPORTANT: snf_recompute_for_admission opens its own DB connection internally.
        # So we keep this outer connection only for collecting groups, then recompute per group.
    finally:
        conn.close()

    for (visit_id, patient_mrn) in sorted(groups):
        try:
            res = snf_recompute_for_admission(visit_id=visit_id, patient_mrn=patient_mrn)
            if res and res.get("ok"):
                ok_count += 1
            else:
                fail_count += 1
                failures.append(
                    {
                        "visit_id": visit_id,
                        "patient_mrn": patient_mrn,
                        "error": (res.get("message") if isinstance(res, dict) else "ok=false"),
                    }
                )
        except Exception as e:
            fail_count += 1
            failures.append(
                {"visit_id": visit_id, "patient_mrn": patient_mrn, "error": str(e)}
            )

    # Keep response small but useful
    return {
        "ok": True,
        "total_groups": total,
        "recomputed_ok": ok_count,
        "recomputed_failed": fail_count,
        "failures_sample": failures[:20],
    }


@app.post("/admin/snf/clear")
async def admin_snf_clear(
    request: Request,
    include_cm_notes: bool = Query(
        False,
        description="If true, also delete all rows from cm_notes_raw.",
    ),
):

    """
    Admin-only endpoint to wipe SNF-related test data.

    WARNING: This permanently deletes data from snf_admissions
    (and optionally cm_notes_raw). Intended for test/cleanup only.
    """
    require_admin(request)

    conn = get_db()
    try:
        cur = conn.cursor()

        # Delete all SNF admissions
        cur.execute("DELETE FROM snf_admissions")
        deleted_snf = cur.rowcount

        deleted_notes = 0
        if include_cm_notes:
            cur.execute("DELETE FROM cm_notes_raw")
            deleted_notes = cur.rowcount

        conn.commit()
    finally:
        conn.close()

    return {
        "ok": True,
        "deleted_snf_admissions": deleted_snf,
        "deleted_cm_notes_raw": deleted_notes,
    }

@app.get("/admin/snf/clear-browser")
async def admin_snf_clear_browser(
    request: Request,
    admin_token: str = Query(..., description="Your admin token"),
    include_cm_notes: bool = Query(False),
):
    """
    Browser-friendly version of /admin/snf/clear.
    Allows cleanup directly from the browser address bar
    by passing the admin token in the query string.
    """
    # Validate admin token
    if admin_token != os.getenv("ADMIN_TOKEN"):
        raise HTTPException(status_code=403, detail="Invalid admin token")

    conn = get_db()
    try:
        cur = conn.cursor()

        cur.execute("DELETE FROM snf_admissions")
        deleted_snf = cur.rowcount

        deleted_notes = 0
        if include_cm_notes:
            cur.execute("DELETE FROM cm_notes_raw")
            deleted_notes = cur.rowcount

        conn.commit()
    finally:
        conn.close()

    return {
        "ok": True,
        "deleted_snf_admissions": deleted_snf,
        "deleted_cm_notes_raw": deleted_notes,
    }


@app.get("/admin/snf/pad-runs")
async def admin_snf_pad_runs(request: Request, limit: int = 10):
    require_admin(request)

    limit = max(1, min(50, int(limit or 10)))

    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, endpoint, received_at, pad_run_id, rows_received, inserted, skipped, error_count, status, message
            FROM pad_api_runs
            ORDER BY received_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cur.fetchall()
        return {"ok": True, "runs": [dict(r) for r in rows]}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Admin: Q&A list/add/update/delete  (used by TEST Admin.html "Q&A" tab)
# ---------------------------------------------------------------------------


@app.get("/admin/list")
async def admin_list(request: Request):
    """
    Return all QA rows for the Admin Q&A tab.
    Matches the JSON shape expected by TEST Admin.html:
      { "rows": [ {id, question, answer, tags, topics, section}, ... ],
        "count": N }
    """
    require_admin(request)
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, section, question, answer, tags, topics FROM qa "
            "ORDER BY id"
        )
        rows = []
        for r in cur.fetchall():
            rows.append(
                {
                    "id": r["id"],
                    "section": r["section"] or "",
                    "question": r["question"] or "",
                    "answer": r["answer"] or "",
                    "tags": r["tags"] or "",
                    "topics": r["topics"] or (r["section"] or ""),
                }
            )
        return {"rows": rows, "count": len(rows)}
    finally:
        conn.close()


@app.post("/admin/add")
async def admin_add(
    request: Request,
    question: str = Form(...),
    answer: str = Form(...),
    tags: str = Form(""),
    section: str = Form(""),
    topics: str = Form(""),
):
    """
    Add a new QA row from the Admin Q&A tab.
    IDs are generated on the server (Q0001, Q0002, ...).
    """
    require_admin(request)

    q = (question or "").strip()
    a = (answer or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question is required")
    if not a:
        raise HTTPException(status_code=400, detail="Answer is required")

    section_s = (section or "").strip()
    topics_s = (topics or section_s).strip()
    tags_s = (tags or "").strip()

    conn = get_db()
    try:
        qid = next_qa_id(conn)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO qa (id, section, question, answer, tags, topics)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (qid, section_s, q, a, tags_s, topics_s),
        )
        conn.commit()

        return {
            "ok": True,
            "row": {
                "id": qid,
                "section": section_s,
                "question": q,
                "answer": a,
                "tags": tags_s,
                "topics": topics_s,
            },
        }
    finally:
        conn.close()


@app.post("/admin/update")
async def admin_update(
    request: Request,
    id: str = Form(...),
    question: str = Form(...),
    answer: str = Form(...),
    tags: str = Form(""),
    section: str = Form(""),
    topics: str = Form(""),
):
    """
    Update an existing QA row (called from the Edit modal).
    """
    require_admin(request)

    qid = (id or "").strip()
    if not qid:
        raise HTTPException(status_code=400, detail="id is required")

    q = (question or "").strip()
    a = (answer or "").strip()
    if not q or not a:
        raise HTTPException(status_code=400, detail="Question and Answer are required")

    section_s = (section or "").strip()
    topics_s = (topics or section_s).strip()
    tags_s = (tags or "").strip()

    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE qa
               SET section = ?,
                   question = ?,
                   answer   = ?,
                   tags     = ?,
                   topics   = ?
             WHERE id = ?
            """,
            (section_s, q, a, tags_s, topics_s, qid),
        )
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="QA row not found")
        conn.commit()
        return {"ok": True}
    finally:
        conn.close()


@app.post("/admin/delete")
async def admin_delete(
    request: Request,
    id: str = Form(...),
):
    """
    Delete a QA row (delete button in Q&A tab).
    """
    require_admin(request)
    qid = (id or "").strip()
    if not qid:
        raise HTTPException(status_code=400, detail="id is required")

    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM qa WHERE id=?", (qid,))
        conn.commit()
        return {"ok": True}
    finally:
        conn.close()



@app.get("/admin/top_tags")
async def admin_top_tags(request: Request, limit: int = 5):
    """
    Return the most frequent tags across QA rows.
    Used by TEST Admin.html to render recommended tag pills.
    """
    require_admin(request)
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("SELECT tags FROM qa")
        freq: Dict[str, int] = {}
        import re

        for row in cur.fetchall():
            raw = row["tags"] or ""
            if not raw:
                continue
            parts = re.split(r"[,\|;]", raw)
            for t in parts:
                tag = (t or "").strip()
                if not tag:
                    continue
                # Skip facility-specific tags in popularity counts
                if tag.lower().startswith("f-"):
                    continue
                key = tag.lower()
                freq[key] = freq.get(key, 0) + 1

        top = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:limit]
        return {"tags": [t for t, _ in top]}
    finally:
        conn.close()



# ---------------------------------------------------------------------------
# Admin: Facilities list/create + Facts/Aliases
# ---------------------------------------------------------------------------

def build_facility_url_full(facility_id: str, intake_token: str) -> str:
    """
    Build a public Facility Details URL using FACILITY_FORM_BASE_URL (?f=&t=).
    Admin HTML uses this to construct both Wix and Render URLs.
    """
    if FACILITY_FORM_BASE_URL and facility_id and intake_token:
        return f"{FACILITY_FORM_BASE_URL}?f={facility_id}&t={intake_token}"
    return ""


@app.get("/admin/facilities")
async def admin_facilities_list(
    request: Request,
    q: Optional[str] = Query(None),
):
    """
    List facilities for the Admin Facility tab.
    Returns: { items: [ { facility_id, facility_name, corporate_group, url_full, ... } ] }
    """
    require_admin(request)
    conn = get_db()
    try:
        cur = conn.cursor()
        where = []
        params: List[Any] = []

        if q:
            like = f"%{q}%"
            where.append("(facility_name LIKE ? OR corporate_group LIKE ?)")
            params.extend([like, like])

        sql = """
            SELECT facility_id, facility_name, corporate_group,
                   intake_token, intake_status, created_at, updated_at
            FROM facilities
        """
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY created_at DESC"

        cur.execute(sql, params)
        items = []
        for row in cur.fetchall():
            fid = row["facility_id"]
            token = row["intake_token"] or ""
            items.append(
                {
                    "facility_id": fid,
                    "facility_name": row["facility_name"] or "",
                    "corporate_group": row["corporate_group"] or "",
                    "intake_status": row["intake_status"] or "",
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "url_full": build_facility_url_full(fid, token),
                }
            )
        return {"ok": True, "items": items}
    finally:
        conn.close()


@app.post("/admin/facilities/create")
async def admin_facilities_create(
    request: Request,
    facility_name: str = Form(...),
    facility_id: str = Form(""),
):
    """
    Create a new facility + intake token.
    Called by the 'Create New Facility' form in TEST Admin.html.
    """
    require_admin(request)

    name = (facility_name or "").strip()
    fid = (facility_id or "").strip()

    if not name:
        raise HTTPException(status_code=400, detail="facility_name is required")

    if not fid:
        # slugify from name
        fid = re.sub(r"[^a-z0-9-]+", "-", name.lower()).strip("-") or "facility"

    if not re.fullmatch(r"[a-z0-9\-]+", fid):
        raise HTTPException(status_code=400, detail="facility_id must match [a-z0-9-]+")

    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM facilities WHERE facility_id=?", (fid,))
        if cur.fetchone():
            # Admin UI checks for 409 to show "ID already exists" message
            raise HTTPException(status_code=409, detail="Facility ID already exists")

        # generate intake token
        token = secrets.token_urlsafe(16)
        now = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"

        cur.execute(
            """
            INSERT INTO facilities(
                facility_id, facility_name, corporate_group,
                intake_token, intake_status, created_at, updated_at
            )
            VALUES (?, ?, '', ?, 'not-started', ?, ?)
            """,
            (fid, name, token, now, now),
        )
        conn.commit()

        url_full = build_facility_url_full(fid, token)
        return {
            "ok": True,
            "facility": {
                "facility_id": fid,
                "facility_name": name,
                "corporate_group": "",
                "intake_status": "not-started",
                "created_at": now,
                "updated_at": now,
                "url_full": url_full,
            },
        }
    finally:
        conn.close()


@app.get("/admin/facilities/{facility_id}/facts")
async def admin_facilities_facts_list(
    request: Request,
    facility_id: str,
):
    """
    List facility facts for the 'Facts' modal in Admin.
    Returns: { items: [ { id, fact_text, tags, created_at } ] }
    """
    require_admin(request)
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, fact_text, tags, created_at
            FROM fac_facts
            WHERE facility_id=?
            ORDER BY created_at DESC, id DESC
            """,
            (facility_id,),
        )
        items = []
        for r in cur.fetchall():
            items.append(
                {
                    "id": r["id"],
                    "fact_text": r["fact_text"] or "",
                    "tags": r["tags"] or "",
                    "created_at": r["created_at"],
                }
            )

        return {"ok": True, "items": items}
    finally:
        conn.close()



@app.post("/admin/facilities/facts/add")
async def admin_facilities_facts_add(
    request: Request,
    facility_id: str = Form(...),
    fact_text: str = Form(...),
    tags: str = Form(""),
    token: str = Form(""),
):
    """
    Add a new fact to fac_facts.
    Form fields come from facAddFact() in TEST Admin.html.
    """
    require_admin(request)
    text = (fact_text or "").strip()
    if not facility_id or not text:
        raise HTTPException(status_code=400, detail="facility_id and fact_text are required")

    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO fac_facts (facility_id, fact_text, tags, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (facility_id, text, (tags or "").strip(), now_iso()),
        )

        new_id = cur.lastrowid
        conn.commit()
        return {"ok": True, "id": new_id}
    finally:
        conn.close()


@app.post("/admin/facilities/facts/delete")
async def admin_facilities_facts_delete(
    request: Request,
    id: int = Form(...),
    token: str = Form(""),
):
    """Delete a facility fact (called by facDeleteFact)."""
    require_admin(request)
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM fac_facts WHERE id=?", (id,))
        conn.commit()
        return {"ok": True}
    finally:
        conn.close()


@app.get("/admin/facilities/{facility_id}/aliases")
async def admin_facilities_aliases_list(
    request: Request,
    facility_id: str,
):
    """
    List facility_alias entries for this facility.
    Returns: { ok: True, items: [ { id, key, canonical } ] }
    """
    require_admin(request)
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT d.key,
                   d.canonical,
                   f.facility_name
            FROM dictionary d
            LEFT JOIN facilities f ON f.facility_id = d.canonical
            WHERE d.kind = 'facility_alias'
              AND d.canonical = ?
            ORDER BY d.key COLLATE NOCASE
            """,
            (facility_id,),
        )
        items = []
        for r in cur.fetchall():
            items.append(
                {
                    "id": r["key"],
                    "key": r["key"] or "",
                    "canonical": r["facility_name"] or (r["canonical"] or ""),
                }
            )
        return {"ok": True, "items": items}
    finally:
        conn.close()




@app.post("/admin/facilities/aliases/add")
async def admin_facilities_aliases_add(
    request: Request,
    facility_id: str = Form(...),
    alias: str = Form(...),
    token: str = Form(""),
):
    """
    Add a new facility_alias dictionary row.
    Form fields come from facAddAlias() in TEST Admin.html.
    """
    require_admin(request)
    key = (alias or "").strip()
    if not facility_id or not key:
        raise HTTPException(status_code=400, detail="facility_id and alias are required")

    conn = get_db()
    try:
        cur = conn.cursor()
        # Avoid duplicates: same facility_id (canonical) + alias key
        cur.execute(
            """
            SELECT 1
            FROM dictionary
            WHERE kind = 'facility_alias'
              AND canonical = ?
              AND key = ?
            """,
            (facility_id, key),
        )
        if not cur.fetchone():
            cur.execute(
                """
                INSERT INTO dictionary (key, canonical, kind, notes, match_mode)
                VALUES (?, ?, 'facility_alias', '', 'exact')
                """,
                (key, facility_id),
            )
            conn.commit()
        return {"ok": True}
    finally:
        conn.close()



@app.post("/admin/facilities/aliases/delete")
async def admin_facilities_aliases_delete(
    request: Request,
    facility_id: str = Form(...),
    alias: str = Form(...),
    token: str = Form(""),
):
    """
    Delete a facility_alias.
    Called by facDeleteAlias() in TEST Admin.html.
    """
    require_admin(request)
    key = (alias or "").strip()
    if not facility_id or not key:
        raise HTTPException(status_code=400, detail="facility_id and alias are required")

    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            DELETE FROM dictionary
            WHERE kind = 'facility_alias'
              AND canonical = ?
              AND key = ?
            """,
            (facility_id, key),
        )
        conn.commit()
        return {"ok": True}
    finally:
        conn.close()





# ---------------------------------------------------------------------------
# Facility Intake: preload + submit + email
# ---------------------------------------------------------------------------

@app.get("/intake/f/{facility_id}")
async def intake_get_facility(
    request: Request,
    facility_id: str,
    t: str = Query(default=""),
):
    """
    Preload facility info for TEST Facility_Details.html.
    Validates intake token (?t=...).
    """
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT facility_id, facility_name, corporate_group,
                   intake_token, intake_status,
                   created_at, updated_at, extras,
                   legal_name, address_line1, address_line2, city, state, zip, county,
                   avg_dcs, short_beds, ltc_beds, outpatient_pt,
                   emr, emr_other, pt_emr, orders, orders_other,
                   raw_json
            FROM facilities
            WHERE facility_id = ?
            LIMIT 1
            """,
            (facility_id,),
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Facility not found")

        if not row["intake_token"] or not t or t != row["intake_token"]:
            raise HTTPException(status_code=401, detail="Invalid intake token")

        base = {
            "facility_id": row["facility_id"],
            "facility_name": row["facility_name"] or "",
            "corporate_group": row["corporate_group"] or "",
            "intake_status": row["intake_status"] or "not-started",
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "extras": row["extras"] or "",
            "legal_name": row["legal_name"] or "",
            "address_line1": row["address_line1"] or "",
            "address_line2": row["address_line2"] or "",
            "city": row["city"] or "",
            "state": row["state"] or "",
            "zip": row["zip"] or "",
            "county": row["county"] or "",
            "avg_dcs": row["avg_dcs"] or "",
            "short_beds": row["short_beds"] or "",
            "ltc_beds": row["ltc_beds"] or "",
            "outpatient_pt": row["outpatient_pt"] or "",
            "emr": row["emr"] or "",
            "emr_other": row["emr_other"] or "",
            "pt_emr": row["pt_emr"] or "",
            "orders": row["orders"] or "",
            "orders_other": row["orders_other"] or "",
        }

        cur.execute(
            "SELECT type, name, email, phone, pref FROM facility_contacts WHERE facility_id = ? ORDER BY id ASC",
            (facility_id,),
        )
        contacts = [
            {
                "type": r["type"] or "",
                "name": r["name"] or "",
                "email": r["email"] or "",
                "phone": r["phone"] or "",
                "pref": r["pref"] or "",
            }
            for r in cur.fetchall()
        ]

        cur.execute(
            "SELECT type, name, ins_only, insurance FROM facility_partners WHERE facility_id = ? ORDER BY id ASC",
            (facility_id,),
        )
        partners = [
            {
                "type": r["type"] or "",
                "name": r["name"] or "",
                "ins_only": r["ins_only"] or "",
                "insurance": r["insurance"] or "",
            }
            for r in cur.fetchall()
        ]

        return {"ok": True, "facility": base, "contacts": contacts, "community_partners": partners}
    finally:
        conn.close()


def upsert_facility_and_children(payload: dict) -> str:
    """
    Insert or update facility + child tables based on Facility Details payload.
    """
    conn = get_db()
    cur = conn.cursor()

    facility_name = (payload.get("facility_name") or payload.get("legal_name") or "").strip()
    if not facility_name:
        raise HTTPException(status_code=400, detail="facility_name (or legal_name) is required")

    facility_id = (payload.get("facility_id") or "").strip()
    if not facility_id:
        s = re.sub(r"[^a-z0-9-]+", "-", facility_name.lower().strip())
        s = re.sub(r"-+", "-", s).strip("-") or "facility"
        facility_id = s

    row = {
        "facility_id": facility_id,
        "facility_name": facility_name,
        "legal_name": payload.get("legal_name", ""),
        "corporate_group": payload.get("corporate_group", ""),
        "address_line1": payload.get("address_line1", ""),
        "address_line2": payload.get("address_line2", ""),
        "city": payload.get("city", ""),
        "state": payload.get("state", ""),
        "zip": payload.get("zip", ""),
        "county": payload.get("county", ""),
        "avg_dcs": payload.get("avg_dcs", ""),
        "short_beds": payload.get("short_beds", ""),
        "ltc_beds": payload.get("ltc_beds", ""),
        "outpatient_pt": payload.get("outpatient_pt", ""),
        "emr": payload.get("emr", ""),
        "emr_other": payload.get("emr_other", ""),
        "pt_emr": payload.get("pt_emr", ""),
        "orders": payload.get("orders", ""),
        "orders_other": payload.get("orders_other", ""),
        "raw_json": json.dumps(payload, ensure_ascii=False),
    }

    cur.execute(
        """
        INSERT INTO facilities(
          facility_id, facility_name, legal_name, corporate_group,
          address_line1, address_line2, city, state, zip, county,
          avg_dcs, short_beds, ltc_beds, outpatient_pt,
          emr, emr_other, pt_emr, orders, orders_other,
          raw_json, created_at, updated_at
        )
        VALUES (
          :facility_id, :facility_name, :legal_name, :corporate_group,
          :address_line1, :address_line2, :city, :state, :zip, :county,
          :avg_dcs, :short_beds, :ltc_beds, :outpatient_pt,
          :emr, :emr_other, :pt_emr, :orders, :orders_other,
          :raw_json, datetime('now'), datetime('now')
        )
        ON CONFLICT(facility_id) DO UPDATE SET
          facility_name=excluded.facility_name,
          legal_name=excluded.legal_name,
          corporate_group=excluded.corporate_group,
          address_line1=excluded.address_line1,
          address_line2=excluded.address_line2,
          city=excluded.city,
          state=excluded.state,
          zip=excluded.zip,
          county=excluded.county,
          avg_dcs=excluded.avg_dcs,
          short_beds=excluded.short_beds,
          ltc_beds=excluded.ltc_beds,
          outpatient_pt=excluded.outpatient_pt,
          emr=excluded.emr,
          emr_other=excluded.emr_other,
          pt_emr=excluded.pt_emr,
          orders=excluded.orders,
          orders_other=excluded.orders_other,
          raw_json=excluded.raw_json,
          updated_at=datetime('now');
        """,
        row,
    )

    addl = payload.get("additional_services") or []
    plans = payload.get("insurance_plans") or []
    contacts = payload.get("contacts") or []
    partners = payload.get("community_partners") or []

    cur.execute("DELETE FROM facility_additional_services WHERE facility_id = ?", (facility_id,))
    cur.execute("DELETE FROM facility_insurance_plans   WHERE facility_id = ?", (facility_id,))
    cur.execute("DELETE FROM facility_contacts          WHERE facility_id = ?", (facility_id,))
    cur.execute("DELETE FROM facility_partners          WHERE facility_id = ?", (facility_id,))

    if addl:
        cur.executemany(
            "INSERT INTO facility_additional_services(facility_id, service) VALUES (?, ?)",
            [(facility_id, str(s).strip()) for s in addl if str(s).strip()],
        )
    if plans:
        cur.executemany(
            "INSERT INTO facility_insurance_plans(facility_id, plan) VALUES (?, ?)",
            [(facility_id, str(p).strip()) for p in plans if str(p).strip()],
        )
    if contacts:
        cur.executemany(
            "INSERT INTO facility_contacts(facility_id, type, name, email, phone, pref) VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    facility_id,
                    str(c.get("type", "")).strip(),
                    str(c.get("name", "")).strip(),
                    str(c.get("email", "")).strip(),
                    str(c.get("phone", "")).strip(),
                    str(c.get("pref", "")).strip(),
                )
                for c in contacts
                if c and str(c.get("type", "")).strip() and str(c.get("name", "")).strip()
            ],
        )
    if partners:
        cur.executemany(
            "INSERT INTO facility_partners(facility_id, type, name, ins_only, insurance) VALUES (?, ?, ?, ?, ?)",
            [
                (
                    facility_id,
                    str(p.get("type", "")).strip(),
                    str(p.get("name", "")).strip(),
                    str(p.get("ins_only", "")).strip(),
                    str(p.get("insurance", "")).strip(),
                )
                for p in partners
                if p and str(p.get("type", "")).strip() and str(p.get("name", "")).strip()
            ],
        )

    conn.commit()
    conn.close()
    return facility_id


# ---------------------------------------------------------------------------
# Admin: SNF admissions APIs used by TEST SNF_Admissions.html
# ---------------------------------------------------------------------------

SNF_HIGHLIGHT_KEYWORDS_KEY = "snf_highlight_keywords"

def get_setting(cur: sqlite3.Cursor, key: str) -> str:
    cur.execute("SELECT value FROM ai_settings WHERE key = ?", (key,))
    row = cur.fetchone()
    return (row["value"] if row else "") or ""

def set_setting(conn: sqlite3.Connection, cur: sqlite3.Cursor, key: str, value: str) -> None:
    cur.execute(
        """
        INSERT INTO ai_settings(key, value)
        VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value
        """,
        (key, value or ""),
    )
    conn.commit()

@app.get("/admin/snf/highlight-keywords/get")
async def admin_snf_highlight_keywords_get(request: Request):
    require_admin(request)
    conn = get_db()
    try:
        cur = conn.cursor()
        raw = get_setting(cur, SNF_HIGHLIGHT_KEYWORDS_KEY)
        return {"ok": True, "keywords": raw}
    finally:
        conn.close()

@app.post("/admin/snf/highlight-keywords/set")
async def admin_snf_highlight_keywords_set(
    request: Request,
    payload: Dict[str, Any] = Body(...),
):
    require_admin(request)
    keywords = (payload.get("keywords") or "").strip()

    # light safety limits (prevents huge blobs)
    if len(keywords) > 5000:
        raise HTTPException(status_code=400, detail="keywords too long")

    conn = get_db()
    try:
        cur = conn.cursor()
        set_setting(conn, cur, SNF_HIGHLIGHT_KEYWORDS_KEY, keywords)
        return {"ok": True}
    finally:
        conn.close()


@app.get("/admin/snf-facilities/list")
async def admin_snf_facilities_list(request: Request):
    """
    List SNF Admission Facilities for dropdowns and simple editing.
    Returns:
      { ok: True, items: [ { id, facility_name, attending, notes, notes2, aliases, facility_emails }, ... ] }
    """
    require_admin(request)
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, facility_name, facility_phone, attending, notes, notes2, aliases, facility_emails, pin_hash
            FROM snf_admission_facilities
            ORDER BY facility_name COLLATE NOCASE
            """
        )

        items = []
        for r in cur.fetchall():
            d = dict(r)
            d["pin_set"] = bool((d.get("pin_hash") or "").strip())
            d.pop("pin_hash", None)  # don't send hashes to the browser
            items.append(d)

        return {"ok": True, "items": items}
    finally:
        conn.close()


@app.post("/admin/snf-facilities/upsert")
async def admin_snf_facilities_upsert(
    request: Request,
    payload: Dict[str, Any] = Body(...),
):
    require_admin(request)

    name = (payload.get("facility_name") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="facility_name is required")

    fac_id = payload.get("id")
    facility_phone = (payload.get("facility_phone") or "").strip()
    attending = (payload.get("attending") or "").strip()
    notes = (payload.get("notes") or "").strip()
    notes2 = (payload.get("notes2") or "").strip()
    aliases = (payload.get("aliases") or "").strip()
    facility_emails = (payload.get("facility_emails") or "").strip()

    # NEW: facility PIN controls
    facility_pin = (payload.get("facility_pin") or "").strip()
    clear_pin = bool(payload.get("clear_pin") or False)

    # new_pin_hash meanings:
    #   ""    => clear pin_hash
    #   None  => leave unchanged
    #   "..." => set to new hash
    if clear_pin:
        new_pin_hash = ""
    elif facility_pin:
        new_pin_hash = hash_pin(facility_pin)
    else:
        new_pin_hash = None

    conn = get_db()
    try:
        cur = conn.cursor()

        if fac_id:
            # UPDATE
            if new_pin_hash is None:
                cur.execute(
                    """
                    UPDATE snf_admission_facilities
                       SET facility_name   = ?,
                           facility_phone  = ?,
                           attending       = ?,
                           notes           = ?,
                           notes2          = ?,
                           aliases         = ?,
                           facility_emails = ?
                     WHERE id = ?
                    """,
                    (name, facility_phone, attending, notes, notes2, aliases, facility_emails, fac_id)
                )
            else:
                cur.execute(
                    """
                    UPDATE snf_admission_facilities
                       SET facility_name   = ?,
                           facility_phone  = ?,
                           attending       = ?,
                           notes           = ?,
                           notes2          = ?,
                           aliases         = ?,
                           facility_emails = ?,
                           pin_hash        = ?
                     WHERE id = ?
                    """,
                    (name, facility_phone, attending, notes, notes2, aliases, facility_emails, new_pin_hash, fac_id)
                )

            if cur.rowcount == 0:
                raise HTTPException(status_code=404, detail="SNF facility not found")

            new_id = fac_id

        else:
            # INSERT
            if new_pin_hash is None:
                new_pin_hash = ""  # on insert, blank means "no facility PIN set"

            cur.execute(
                """
                INSERT INTO snf_admission_facilities
                  (facility_name, facility_phone, attending, notes, notes2, aliases, facility_emails, pin_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (name, facility_phone, attending, notes, notes2, aliases, facility_emails, new_pin_hash)
            )
            new_id = cur.lastrowid

        conn.commit()
        return {"ok": True, "id": new_id}
    finally:
        conn.close()




@app.get("/admin/snf/list")
async def admin_snf_list(
    request: Request,
    status: str = Query("pending"),          # 'pending', 'confirmed', 'corrected', 'rejected', 'all'
    for_date: Optional[str] = Query(None),   # 'YYYY-MM-DD'
    days_ahead: int = Query(0, ge=0, le=30),
    notified_only: int = Query(0),
    email_at: Optional[str] = Query(None),  # YYYY-MM-DD; filters by DATE(s.emailed_at)
    email_days_ahead: int = Query(0, ge=0, le=30),
):
    """
    List SNF admissions for the SNF Admissions page.

    Filters:
      - status: filter by snf_admissions.status (or 'all' for no filter)
      - for_date: base date for transfer date filtering (optional)
      - days_ahead: include records up to N days after for_date

    Returns: { ok: True, items: [ ... ] }
    """
    require_admin(request)

    conn = get_db()
    try:
        cur = conn.cursor()

        where = ["s.ai_is_snf_candidate = 1"]
        params: List[Any] = []

        # Status filter
        status_s = (status or "").strip().lower()
        if status_s and status_s != "all":
            where.append("s.status = ?")
            params.append(status_s)
            
        # NEW: Notified-only filter
        if int(notified_only or 0) == 1:
            where.append("COALESCE(s.notified_by_hospital, 0) = 1")
            
        # NEW: Email-at date filter (matches the day, regardless of time)
        # If email_days_ahead > 0, treat email_at as the start of a window [email_at, email_at + email_days_ahead]
        email_at_s = (email_at or "").strip()
        if email_at_s:
            if not re.match(r"^\d{4}-\d{2}-\d{2}$", email_at_s):
                raise HTTPException(status_code=400, detail="email_at must be YYYY-MM-DD")

            d = int(email_days_ahead or 0)
            if d > 0:
                where.append("s.emailed_at IS NOT NULL AND date(s.emailed_at) >= ? AND date(s.emailed_at) <= date(?, ? || ' days')")
                params.extend([email_at_s, email_at_s, d])
            else:
                where.append("s.emailed_at IS NOT NULL AND date(s.emailed_at) = ?")
                params.append(email_at_s)

        # Date filter: use last_seen_active_date so the list shows "active admissions"
        # - If days_ahead = 0: show only that exact date
        # - If days_ahead > 0: show a window [for_date, for_date + days_ahead]
        if for_date:
            where.append("s.last_seen_active_date IS NOT NULL")

            if days_ahead > 0:
                # Range: for_date <= last_seen_active_date <= for_date + days_ahead
                where.append("s.last_seen_active_date >= ?")
                params.append(for_date)

                where.append("s.last_seen_active_date <= date(?, ? || ' days')")
                params.append(for_date)
                params.append(days_ahead)
            else:
                # Exact match: only rows from this batch date
                where.append("s.last_seen_active_date = ?")
                params.append(for_date)


        sql = f"""
            SELECT
                s.*,
                COALESCE(s.final_expected_transfer_date, s.ai_expected_transfer_date) AS effective_date,
                COALESCE(s.final_snf_facility_id, s.ai_snf_facility_id) AS effective_facility_id,
                f.facility_name AS effective_facility_name,
                f.city AS effective_facility_city,
                f.state AS effective_facility_state
            FROM snf_admissions s
            LEFT JOIN facilities f
              ON f.facility_id = COALESCE(s.final_snf_facility_id, s.ai_snf_facility_id)
            WHERE {" AND ".join(where) if where else "1=1"}
            ORDER BY effective_date IS NULL, effective_date, s.patient_name COLLATE NOCASE
        """

        cur.execute(sql, params)
        items: List[Dict[str, Any]] = []
        for r in cur.fetchall():
            items.append(
                {
                    "id": r["id"],
                    "visit_id": r["visit_id"],
                    "patient_mrn": r["patient_mrn"],
                    "patient_name": r["patient_name"],
                    "hospital_name": r["hospital_name"],

                    # Notification fields
                    "notified_by_hospital": int(r["notified_by_hospital"] or 0),
                    "notified_by": r["notified_by"] or "",
                    "notification_dt": r["notification_dt"] or "",
                    "hospital_reported_facility": r["hospital_reported_facility"] or "",
                    "notification_details": r["notification_details"] or "",

                    # SNF Physician Assignment (NEW)
                    "assignment_confirmation": r["assignment_confirmation"] or "Unknown",
                    "billing_confirmed": int(r["billing_confirmed"] or 0),
                    "confirmation_call_dt": r["confirmation_call_dt"] or "",
                    "snf_staff_name": r["snf_staff_name"] or "",
                    "physician_assigned": r["physician_assigned"] or "",
                    "assignment_notes": r["assignment_notes"] or "",

                    "note_datetime": r["note_datetime"],
                    "note_datetime_et": utc_text_to_eastern_display(r["note_datetime"]),

                    "ai_snf_name_raw": r["ai_snf_name_raw"],
                    "ai_expected_transfer_date": r["ai_expected_transfer_date"],
                    "ai_confidence": r["ai_confidence"],
                    "status": r["status"],
                    "final_snf_facility_id": r["final_snf_facility_id"],
                    "final_snf_name_display": r["final_snf_name_display"],
                    "final_expected_transfer_date": r["final_expected_transfer_date"],
                    "review_comments": r["review_comments"],

                    "emailed_at": r["emailed_at"],
                    "emailed_at_et": utc_text_to_eastern_display(r["emailed_at"]),

                    "email_run_id": r["email_run_id"],
                    "effective_date": r["effective_date"],
                    "effective_facility_id": r["effective_facility_id"],
                    "effective_facility_name": r["effective_facility_name"],
                    "effective_facility_city": r["effective_facility_city"],
                    "effective_facility_state": r["effective_facility_state"],

                    "last_seen_active_date": r["last_seen_active_date"],
                    "last_seen_active_date_et": utc_text_to_eastern_display(r["last_seen_active_date"]),

                    "dob": r["dob"],
                    "attending": r["attending"],
                    "admit_date": r["admit_date"],
                }
            )
        return {"ok": True, "items": items}
    finally:
        conn.close()


@app.get("/admin/snf/note/{snf_id}")
async def admin_snf_get_note(
    request: Request,
    snf_id: int,
):
    """
    Fetch the SNF admission row + the underlying raw CM note.
    Used when the user clicks 'View Note' on the SNF page.
    """
    require_admin(request)

    conn = get_db()
    try:
        cur = conn.cursor()

        cur.execute("SELECT * FROM snf_admissions WHERE id = ?", (snf_id,))
        adm = cur.fetchone()
        if not adm:
            raise HTTPException(status_code=404, detail="SNF admission not found")

        raw_note_id = adm["raw_note_id"]

        cur.execute("SELECT * FROM cm_notes_raw WHERE id = ?", (raw_note_id,))
        note = cur.fetchone()
        if not note:
            raise HTTPException(status_code=404, detail="Raw note not found")

        admission = {
            "id": adm["id"],
            "visit_id": adm["visit_id"],
            "patient_mrn": adm["patient_mrn"],
            "patient_name": adm["patient_name"],
            "hospital_name": adm["hospital_name"],
            "note_datetime": adm["note_datetime"],
            "ai_snf_name_raw": adm["ai_snf_name_raw"],
            "ai_expected_transfer_date": adm["ai_expected_transfer_date"],
            "ai_confidence": adm["ai_confidence"],
            "status": adm["status"],
            "final_snf_facility_id": adm["final_snf_facility_id"],
            "final_snf_name_display": adm["final_snf_name_display"],
            "final_expected_transfer_date": adm["final_expected_transfer_date"],
            "review_comments": adm["review_comments"],
            "disposition": adm["disposition"],
            "facility_free_text": adm["facility_free_text"],
            "emailed_at": adm["emailed_at"],
            "email_run_id": adm["email_run_id"],
            
            # NEW: show admit_date in UI near hospital name
            "admit_date": adm["admit_date"],

            # NEW: notification fields
            "notified_by_hospital": int(adm["notified_by_hospital"] or 0),
            "notified_by": adm["notified_by"] or "",
            "notification_dt": adm["notification_dt"] or "",
            "hospital_reported_facility": adm["hospital_reported_facility"] or "",
            "notification_details": adm["notification_details"] or "",

            # SNF Physician Assignment (NEW)
            "assignment_confirmation": adm["assignment_confirmation"] or "Unknown",
            "billing_confirmed": int(adm["billing_confirmed"] or 0),
            "confirmation_call_dt": adm["confirmation_call_dt"] or "",
            "snf_staff_name": adm["snf_staff_name"] or "",
            "physician_assigned": adm["physician_assigned"] or "",
            "assignment_notes": adm["assignment_notes"] or "",
        }

        note_data = {
            "id": note["id"],
            "visit_id": note["visit_id"],
            "patient_mrn": note["patient_mrn"],
            "patient_name": note["patient_name"],
            "dob": note["dob"],
            "encounter_id": note["encounter_id"],
            "hospital_name": note["hospital_name"],
            "unit_name": note["unit_name"],
            "admission_date": note["admission_date"],
            "note_datetime": note["note_datetime"],
            "note_author": note["note_author"],
            "note_type": note["note_type"],
            "note_text": note["note_text"],
            "source_system": note["source_system"],
            "pad_run_id": note["pad_run_id"],
            "created_at_et": utc_text_to_eastern_display(note["created_at"]),
            "ignored": int(note["ignored"] or 0) if "ignored" in note.keys() else 0,
        }

        # Also return all notes for this admission (visit_id if present, otherwise MRN)
        notes_list: List[Dict[str, Any]] = []
        try:
            if adm["visit_id"]:
                cur.execute(
                    """
                    SELECT *
                    FROM cm_notes_raw
                    WHERE visit_id = ?
                      AND (ignored IS NULL OR ignored = 0)
                    ORDER BY note_datetime, id
                    """,
                    (adm["visit_id"],),
                )
            else:
                cur.execute(
                    """
                    SELECT *
                    FROM cm_notes_raw
                    WHERE patient_mrn = ?
                    ORDER BY note_datetime, id
                    """,
                    (adm["patient_mrn"],),
                )
            all_notes = cur.fetchall()
            for n in all_notes:
                notes_list.append(
                    {
                        "id": n["id"],
                        "visit_id": n["visit_id"],
                        "patient_mrn": n["patient_mrn"],
                        "patient_name": n["patient_name"],
                        "dob": n["dob"],
                        "encounter_id": n["encounter_id"],
                        "hospital_name": n["hospital_name"],
                        "unit_name": n["unit_name"],
                        "admission_date": n["admission_date"],
                        "note_datetime": n["note_datetime"],
                        "note_author": n["note_author"],
                        "note_type": n["note_type"],
                        "note_text": n["note_text"],
                        "source_system": n["source_system"],
                        "pad_run_id": n["pad_run_id"],
                        "created_at": n["created_at"],
                        "created_at_et": utc_text_to_eastern_display(n["created_at"]),
                    }
                )
        except sqlite3.Error:
            notes_list = [note_data]

        return {"ok": True, "admission": admission, "note": note_data, "notes": notes_list}

    finally:
        conn.close()

class IgnoreCmNoteRequest(BaseModel):
    note_id: int


@app.get("/admin/snf/last-processed")
async def admin_snf_last_processed(request: Request):
    """
    Return the latest PAD API received timestamp (from pad_api_runs).
    Fallback to latest cm_notes_raw insert timestamp if the log table isn't present yet.
    """
    require_admin(request)

    conn = get_db()
    try:
        cur = conn.cursor()

        last_api = None

        # Preferred: latest API run received_at (even if 0 inserts)
        try:
            SNF_PAD_ENDPOINT = "/api/pad/cm-notes/bulk"

            cur.execute(
                """
                SELECT received_at
                FROM pad_api_runs
                WHERE endpoint = ?
                ORDER BY received_at DESC
                LIMIT 1
                """,
                (SNF_PAD_ENDPOINT,),
            )
            r = cur.fetchone()
            last_api = (r["received_at"] if r else None)
        except Exception:
            # table doesn't exist yet, or older DB
            last_api = None

        # Fallback: legacy behavior (latest inserted cm note)
        if not last_api:
            cur.execute("SELECT MAX(created_at) AS last_processed FROM cm_notes_raw")
            row = cur.fetchone()
            last_api = (row["last_processed"] if row else None)

        return {
            "ok": True,
            "last_processed": last_api,
            "last_processed_et": utc_text_to_eastern_display(last_api),
        }
    finally:
        conn.close()


@app.post("/admin/snf/cm-note/ignore")
async def admin_snf_ignore_cm_note(
    request: Request,
    payload: IgnoreCmNoteRequest,
):
    """Soft-delete a CM note so it is hidden in the UI and ignored by the AI."""
    require_admin(request)

    note_id = int(payload.note_id or 0)
    if not note_id:
        raise HTTPException(status_code=400, detail="note_id is required")

    conn = get_db()
    try:
        cur = conn.cursor()

        cur.execute("SELECT * FROM cm_notes_raw WHERE id = ?", (note_id,))
        note = cur.fetchone()
        if not note:
            raise HTTPException(status_code=404, detail="Note not found")

        # Mark ignored
        cur.execute(
            "UPDATE cm_notes_raw SET ignored = 1, ignored_at = datetime('now') WHERE id = ?",
            (note_id,),
        )

        # If any SNF admission currently points to this raw note, repoint it to the latest remaining note
        visit_id = (note["visit_id"] or "").strip() if "visit_id" in note.keys() else ""
        mrn = (note["patient_mrn"] or "").strip()

        latest_keep = None
        if visit_id:
            cur.execute(
                """
                SELECT *
                FROM cm_notes_raw
                WHERE visit_id = ?
                  AND (ignored IS NULL OR ignored = 0)
                ORDER BY note_datetime DESC, id DESC
                LIMIT 1
                """,
                (visit_id,),
            )
            latest_keep = cur.fetchone()
        elif mrn:
            cur.execute(
                """
                SELECT *
                FROM cm_notes_raw
                WHERE patient_mrn = ?
                  AND (ignored IS NULL OR ignored = 0)
                ORDER BY note_datetime DESC, id DESC
                LIMIT 1
                """,
                (mrn,),
            )
            latest_keep = cur.fetchone()

        if latest_keep:
            cur.execute(
                """
                UPDATE snf_admissions
                SET raw_note_id = ?,
                    note_datetime = COALESCE(?, note_datetime),
                    status = CASE WHEN status IN ('confirmed','projected') THEN status ELSE 'pending' END,
                    updated_at = datetime('now')
                WHERE raw_note_id = ?
                """,
                (latest_keep["id"], latest_keep["note_datetime"], note_id),
            )
        else:
            # No notes left for that admission; mark any linked admission as removed
            cur.execute(
                """
                UPDATE snf_admissions
                SET status = 'removed',
                    updated_at = datetime('now')
                WHERE raw_note_id = ?
                """,
                (note_id,),
            )

        conn.commit()

        # NEW: immediately recompute AI for this admission group so the UI updates instantly
        try:
            snf_recompute_for_admission(visit_id=visit_id, patient_mrn=mrn)
        except Exception as e:
            # Do not fail the ignore action if recompute has an issue
            print("[snf-ignore] recompute failed:", e)

        return {"ok": True}

    finally:
        conn.close()


@app.post("/admin/snf/update")
async def admin_snf_update(
    request: Request,
    payload: Dict[str, Any] = Body(...),
):
    """
    Update a single SNF admission row from the SNF Admissions page.

    Payload:
      {
        "id": 123,
        "status": "confirmed" | "corrected" | "rejected" | "pending",
        "final_expected_transfer_date": "YYYY-MM-DD" or null,
        "review_comments": "..."
      }

    (For now we treat facility selection as read-only; later we can add edits.)
    """
    require_admin(request)

    snf_id = int(payload.get("id") or 0)
    if not snf_id:
        raise HTTPException(status_code=400, detail="id is required")

    status = (payload.get("status") or "").strip().lower()
    if status not in ("pending", "projected", "confirmed", "corrected", "rejected", "removed"):
        raise HTTPException(status_code=400, detail="Invalid status")

    final_date = payload.get("final_expected_transfer_date")
    if final_date:
        final_date = str(final_date).strip()
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", final_date):
            raise HTTPException(status_code=400, detail="final_expected_transfer_date must be YYYY-MM-DD")
    else:
        final_date = None

    review_comments = (payload.get("review_comments") or "").strip()
    reviewer = (payload.get("reviewed_by") or "admin").strip()

    # New: disposition + manual facility free text
    raw_disp = (payload.get("disposition") or "").strip()
    disposition = ""
    if raw_disp:
        disp_norm = raw_disp.lower()
        allowed = {
            "snf": "SNF",
            "home health": "Home Health",
            "home self-care": "Home Self-care",
            "home self care": "Home Self-care",
            "irf": "IRF",
            "ltach": "LTACH",
            "other": "Other",
            "unknown": "Unknown",
        }
        if disp_norm in allowed:
            disposition = allowed[disp_norm]
        else:
            raise HTTPException(status_code=400, detail="Invalid disposition value")
    facility_free_text = (payload.get("facility_free_text") or "").strip()

    conn = get_db()
    try:
        cur = conn.cursor()

        # Load current row so we can update SNF facility/name intelligently
        cur.execute("SELECT * FROM snf_admissions WHERE id = ?", (snf_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="SNF admission not found")
        # IMPORTANT: prevent unintended resets.
        # Only overwrite these fields if they were included in the payload.
        # Otherwise keep what is already in the database row.

        # --- Notification fields ---
        if "notified_by_hospital" in payload:
            notified_by_hospital = 1 if payload.get("notified_by_hospital") else 0
        else:
            try:
                notified_by_hospital = int(row["notified_by_hospital"] or 0)
            except Exception:
                notified_by_hospital = 0

        if "notified_by" in payload:
            notified_by = (payload.get("notified_by") or "").strip()
        else:
            notified_by = (row["notified_by"] or "").strip()

        if "notification_dt" in payload:
            notification_dt = (payload.get("notification_dt") or "").strip()
        else:
            notification_dt = (row["notification_dt"] or "").strip()

        if "hospital_reported_facility" in payload:
            hospital_reported_facility = (payload.get("hospital_reported_facility") or "").strip()
        else:
            hospital_reported_facility = (row["hospital_reported_facility"] or "").strip()

        if "notification_details" in payload:
            notification_details = (payload.get("notification_details") or "").strip()
        else:
            notification_details = (row["notification_details"] or "").strip()

        # --- SNF Physician Assignment fields ---
        if "assignment_confirmation" in payload:
            assignment_confirmation = (payload.get("assignment_confirmation") or "Unknown").strip() or "Unknown"
        else:
            assignment_confirmation = (row["assignment_confirmation"] or "Unknown").strip() or "Unknown"

        if assignment_confirmation not in ("Unknown", "Assigned", "Assigned Out"):
            assignment_confirmation = "Unknown"

        if "billing_confirmed" in payload:
            billing_confirmed = 1 if payload.get("billing_confirmed") else 0
        else:
            try:
                billing_confirmed = int(row["billing_confirmed"] or 0)
            except Exception:
                billing_confirmed = 0

        if "confirmation_call_dt" in payload:
            confirmation_call_dt = (payload.get("confirmation_call_dt") or "").strip()
        else:
            confirmation_call_dt = (row["confirmation_call_dt"] or "").strip()

        if "snf_staff_name" in payload:
            snf_staff_name = (payload.get("snf_staff_name") or "").strip()
        else:
            snf_staff_name = (row["snf_staff_name"] or "").strip()

        if "physician_assigned" in payload:
            physician_assigned = (payload.get("physician_assigned") or "").strip()
        else:
            physician_assigned = (row["physician_assigned"] or "").strip()

        if "assignment_notes" in payload:
            assignment_notes = (payload.get("assignment_notes") or "").strip()
        else:
            assignment_notes = (row["assignment_notes"] or "").strip()

        existing_final_facility_id = row["final_snf_facility_id"]
        existing_final_name_display = row["final_snf_name_display"]
        existing_ai_is_candidate = row["ai_is_snf_candidate"]
        ai_snf_name_raw = row["ai_snf_name_raw"]

        new_final_facility_id = existing_final_facility_id
        new_final_name_display = existing_final_name_display
        new_ai_is_candidate = existing_ai_is_candidate

        # If user selected "(none)" in the Facility dropdown, always clear any prior final override.
        # This ensures the UI falls back to AI (or Unknown) instead of snapping back to a previous manual choice.
        if not facility_free_text:
            new_final_facility_id = None
            new_final_name_display = None

        # If the reviewer explicitly sets disposition = SNF,
        # treat this as an authoritative override:
        #  - ensure it is marked as a SNF candidate
        #  - prefer the manual Facility free text as the SNF name
        if disposition == "SNF":
            new_ai_is_candidate = 1

            if facility_free_text:
                # Manual override ON
                maps = load_dictionary_maps(conn)
                facility_aliases = maps["facility_aliases"]
                mapped_id = map_snf_name_to_facility_id(facility_free_text, conn, facility_aliases)
                new_final_facility_id = mapped_id
                new_final_name_display = facility_free_text
            else:
                # Manual override OFF (user selected "(none)")
                # Clear the final override so UI falls back to AI (or Unknown if AI is unknown).
                new_final_facility_id = None
                new_final_name_display = None

        # NOTE: for non-SNF dispositions we leave ai_is_snf_candidate alone,
        # but still record disposition and facility_free_text for audit.

        # NEW: Notification fields
        notified_by_hospital = 1 if payload.get("notified_by_hospital") else 0
        notified_by = (payload.get("notified_by") or "").strip()
        notification_dt = (payload.get("notification_dt") or "").strip()
        hospital_reported_facility = (payload.get("hospital_reported_facility") or "").strip()
        notification_details = (payload.get("notification_details") or "").strip()

        cur.execute(
            """
            UPDATE snf_admissions
               SET status = ?,
                   final_expected_transfer_date = ?,
                   review_comments = ?,
                   reviewed_by = ?,
                   reviewed_at = datetime('now'),
                   disposition = ?,
                   facility_free_text = ?,
                   ai_is_snf_candidate = ?,
                   final_snf_facility_id = ?,
                   final_snf_name_display = ?,
                   notified_by_hospital = ?,
                   notified_by = ?,
                   notification_dt = ?,
                   hospital_reported_facility = ?,
                   notification_details = ?,

                   assignment_confirmation = ?,
                   billing_confirmed = ?,
                   confirmation_call_dt = ?,
                   snf_staff_name = ?,
                   physician_assigned = ?,
                   assignment_notes = ?
             WHERE id = ?
            """,
            (
                status,
                final_date,
                review_comments,
                reviewer,
                disposition,
                facility_free_text,
                new_ai_is_candidate,
                new_final_facility_id,
                new_final_name_display,
                notified_by_hospital,
                notified_by,
                notification_dt,
                hospital_reported_facility,
                notification_details,

                assignment_confirmation,
                billing_confirmed,
                confirmation_call_dt,
                snf_staff_name,
                physician_assigned,
                assignment_notes,

                snf_id,
            ),
        )

        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="SNF admission not found")
        conn.commit()
        return {"ok": True}
    finally:
        conn.close()


@app.post("/admin/snf/send-emails")
async def admin_snf_send_emails(
    request: Request,
    payload: Dict[str, Any] = Body(...),
):
    """
    Send upcoming SNF admission emails grouped by facility.

    Payload:
      {
        "for_date": "YYYY-MM-DD",        # required
        "test_only": true/false,         # if true, do not mark as emailed
        "test_email_to": "me@domain.com" # optional override recipient in test mode
      }

    Logic:
      - Find snf_admissions where:
          status IN ('confirmed','corrected')
          effective_date = for_date
          emailed_at IS NULL   (to avoid double-sending)
      - Group by effective facility_id (final_snf_facility_id or ai_snf_facility_id)
      - For each facility, look up snf_notification_targets for email_to/email_cc
      - Send an email with a simple text list of upcoming admissions
      - If not test_only, mark rows as emailed with emailed_at + email_run_id
    """
    require_admin(request)

    for_date = (payload.get("for_date") or "").strip()
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", for_date):
        raise HTTPException(status_code=400, detail="for_date must be YYYY-MM-DD")

    test_only = bool(payload.get("test_only"))
    test_email_to = (payload.get("test_email_to") or "").strip()

    if test_only and not test_email_to:
        raise HTTPException(status_code=400, detail="test_email_to is required when test_only=true")

    if not (SMTP_HOST and SMTP_USER and SMTP_PASSWORD):
        return {"ok": False, "message": "SMTP not configured; skipping send."}

    conn = get_db()
    try:
        cur = conn.cursor()

        # Pull rows to email
        cur.execute(
            """
            SELECT
                s.*,
                COALESCE(s.final_snf_facility_id, s.ai_snf_facility_id) AS effective_facility_id,
                COALESCE(s.final_expected_transfer_date, s.ai_expected_transfer_date) AS effective_date
            FROM snf_admissions s
            WHERE s.status IN ('confirmed','corrected')
              AND COALESCE(s.final_expected_transfer_date, s.ai_expected_transfer_date) = ?
              AND (s.emailed_at IS NULL OR s.emailed_at = '')
            """,
            (for_date,),
        )
        rows = cur.fetchall()
        if not rows:
            return {"ok": True, "message": f"No confirmed/corrected admissions for {for_date} needing email."}

        # Group by facility
        by_fac: Dict[str, List[sqlite3.Row]] = {}
        skipped_no_facility = 0
        for r in rows:
            fid = (r["effective_facility_id"] or "").strip()
            if not fid:
                skipped_no_facility += 1
                continue
            by_fac.setdefault(fid, []).append(r)

        email_run_id = secrets.token_hex(8)
        sent = 0
        skipped_no_targets = 0

        # Preload facility names
        fac_names: Dict[str, str] = {}
        cur.execute("SELECT facility_id, facility_name FROM facilities")
        for fr in cur.fetchall():
            fac_names[fr["facility_id"]] = fr["facility_name"] or ""

        # Preload notification targets
        notif: Dict[str, List[Dict[str, str]]] = {}
        cur.execute(
            """
            SELECT facility_id, email_to, email_cc
            FROM snf_notification_targets
            WHERE active = 1
            """
        )
        for nr in cur.fetchall():
            fid = (nr["facility_id"] or "").strip()
            if not fid:
                continue
            notif.setdefault(fid, []).append(
                {
                    "email_to": nr["email_to"] or "",
                    "email_cc": nr["email_cc"] or "",
                }
            )

        # Send emails
        for fid, items in by_fac.items():
            if test_only:
                # Override recipients in test mode
                target_to = test_email_to
                target_cc = ""
            else:
                targets = notif.get(fid) or []
                if not targets:
                    skipped_no_targets += len(items)
                    continue
                # Combine all email_to into one comma-separated TO, same for CC
                tos = sorted({t["email_to"].strip() for t in targets if t["email_to"].strip()})
                ccs = sorted({t["email_cc"].strip() for t in targets if t["email_cc"].strip()})
                if not tos:
                    skipped_no_targets += len(items)
                    continue
                target_to = ", ".join(tos)
                target_cc = ", ".join(ccs)

            fac_name = fac_names.get(fid, fid)
            subject = f"Upcoming SNF admissions for {for_date} – {fac_name}"
            if test_only:
                subject = "[TEST] " + subject

            lines = [
                f"Upcoming SNF admissions for {fac_name} on {for_date}",
                "",
            ]
            for r in items:
                pname = r["patient_name"] or ""
                mrn = r["patient_mrn"] or ""
                hosp = r["hospital_name"] or ""
                eff_date = r["effective_date"] or for_date
                ai_name = r["ai_snf_name_raw"] or ""
                final_name = r["final_snf_name_display"] or ai_name
                status = r["status"] or ""
                line = f"- {pname} (MRN {mrn}) from {hosp}, transfer {eff_date}, facility '{final_name}' [status: {status}]"
                lines.append(line)

            body = "\n".join(lines)

            msg = EmailMessage()
            msg["Subject"] = subject
            msg["From"] = INTAKE_EMAIL_FROM or SMTP_USER
            msg["To"] = target_to
            if target_cc:
                msg["Cc"] = target_cc
            msg.set_content(body)

            try:
                context = ssl.create_default_context()
                with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
                    server.starttls(context=context)
                    server.login(SMTP_USER, SMTP_PASSWORD)
                    server.send_message(msg)
                sent += 1

                # Mark rows emailed if not test
                if not test_only:
                    ids = [r["id"] for r in items]
                    cur.execute(
                        f"""
                        UPDATE snf_admissions
                           SET emailed_at = datetime('now'),
                               email_run_id = ?
                         WHERE id IN ({",".join("?" for _ in ids)})
                        """,
                        (email_run_id, *ids),
                    )
            except Exception as e:
                print("[snf-email] Failed for facility", fid, ":", e)
                # Don't raise; continue other facilities

        if not test_only:
            conn.commit()

        return {
            "ok": True,
            "for_date": for_date,
            "email_run_id": email_run_id,
            "emails_sent": sent,
            "skipped_no_facility": skipped_no_facility,
            "skipped_no_targets": skipped_no_targets,
            "test_only": test_only,
        }
    finally:
        conn.close()


SNF_UNIVERSAL_PIN_KEY = "snf_universal_pin_hash"

def get_snf_universal_pin_hash(cur) -> str:
    cur.execute("SELECT value FROM ai_settings WHERE key = ?", (SNF_UNIVERSAL_PIN_KEY,))
    row = cur.fetchone()
    return ((row["value"] if row else "") or "").strip()

@app.get("/admin/snf/universal-pin/get")
async def admin_snf_universal_pin_get(request: Request):
    require_admin(request)
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("SELECT value FROM ai_settings WHERE key = ?", (SNF_UNIVERSAL_PIN_KEY,))
        row = cur.fetchone()
        pin_hash = (row["value"] if row else "") or ""
        return {"ok": True, "pin_set": bool(pin_hash.strip())}
    finally:
        conn.close()

@app.post("/admin/snf/universal-pin/set")
async def admin_snf_universal_pin_set(request: Request, payload: Dict[str, Any] = Body(...)):
    require_admin(request)
    pin = (payload.get("pin") or "").strip()
    if not pin:
        raise HTTPException(status_code=400, detail="pin is required")

    pin_hash = hash_pin(pin)

    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO ai_settings (key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value
            """,
            (SNF_UNIVERSAL_PIN_KEY, pin_hash),
        )
        conn.commit()
        return {"ok": True}
    finally:
        conn.close()

@app.post("/admin/snf/universal-pin/clear")
async def admin_snf_universal_pin_clear(request: Request):
    require_admin(request)
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO ai_settings (key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value
            """,
            (SNF_UNIVERSAL_PIN_KEY, ""),
        )
        conn.commit()
        return {"ok": True}
    finally:
        conn.close()

@app.get("/admin/snf/email-log/list")
async def admin_snf_email_log_list(request: Request):
    """
    Returns sent email records based on snf_secure_links that were successfully sent.
    (No PHI is returned; only metadata: facility, count, sent time, recipients.)
    """
    require_admin(request)
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                l.id AS secure_link_id,
                l.snf_facility_id,
                l.for_date,
                l.sent_at,
                l.sent_to,
                l.admission_ids,
                f.facility_name AS snf_facility_name,
                COALESCE(o.facility_open_count, 0)  AS facility_open_count,
                COALESCE(o.universal_open_count, 0) AS universal_open_count,
                COALESCE(o.open_count, 0)           AS open_count
            FROM snf_secure_links l
            LEFT JOIN snf_admission_facilities f
              ON f.id = l.snf_facility_id
            LEFT JOIN (
                SELECT
                    secure_link_id,
                    SUM(CASE WHEN LOWER(COALESCE(pin_type,'')) = 'facility' THEN 1 ELSE 0 END)  AS facility_open_count,
                    SUM(CASE WHEN LOWER(COALESCE(pin_type,'')) = 'universal' THEN 1 ELSE 0 END) AS universal_open_count,
                    COUNT(*) AS open_count
                FROM snf_secure_link_access_log
                GROUP BY secure_link_id
            ) o
              ON o.secure_link_id = l.id
            WHERE l.sent_at IS NOT NULL AND l.sent_at <> ''
            ORDER BY l.sent_at DESC
            LIMIT 250
            """
        )
        rows = cur.fetchall()

        items = []
        for r in rows:
            try:
                ids = json.loads(r["admission_ids"] or "[]")
                patient_count = len(ids) if isinstance(ids, list) else 0
            except Exception:
                patient_count = 0

            sent_at = (r["sent_at"] or "").strip()

            items.append(
                {
                    "secure_link_id": int(r["secure_link_id"]),
                    "snf_facility_id": int(r["snf_facility_id"]) if r["snf_facility_id"] is not None else None,
                    "snf_facility_name": (r["snf_facility_name"] or "").strip(),
                    "for_date": (r["for_date"] or "").strip(),
                    "sent_at": sent_at,
                    "sent_at_et": utc_text_to_eastern_display(sent_at),
                    "sent_to": (r["sent_to"] or "").strip(),
                    "patient_count": patient_count,
                    "facility_open_count": int(r["facility_open_count"] or 0),
                    "universal_open_count": int(r["universal_open_count"] or 0),
                    "open_count": int(r["open_count"] or 0),  # keep for compatibility
                }
            )


        return {"ok": True, "items": items}
    finally:
        conn.close()


@app.get("/admin/snf/email-log/opens/{secure_link_id}")
async def admin_snf_email_log_opens(request: Request, secure_link_id: int):
    """
    Returns all access log rows for a specific secure link, including pin_type.
    """
    require_admin(request)
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT pin_type, accessed_at, ip, user_agent
            FROM snf_secure_link_access_log
            WHERE secure_link_id = ?
            ORDER BY accessed_at DESC
            """,
            (int(secure_link_id),),
        )
        rows = cur.fetchall()
        items = []
        for r in rows:
            accessed_at = (r["accessed_at"] or "").strip()
            items.append(
                {
                    "pin_type": (r["pin_type"] or "").strip(),
                    "accessed_at": accessed_at,
                    "accessed_at_et": utc_text_to_eastern_display(accessed_at),
                    "ip": (r["ip"] or "").strip(),
                    "user_agent": (r["user_agent"] or "").strip(),
                }
            )
        return {"ok": True, "items": items}
    finally:
        conn.close()

@app.post("/admin/snf/reports/run")
async def admin_snf_reports_run(request: Request, payload: Dict[str, Any] = Body(...)):
    """
    Runs report calculations for the SNF Admissions reporting modal.

    Current reports:
      - snf_admission_summary

    Filters:
      - dc_from (YYYY-MM-DD) : Hospital discharge dc_date start
      - dc_to   (YYYY-MM-DD) : Hospital discharge dc_date end
    """
    require_admin(request)

    report_key = (payload.get("report_key") or "").strip()
    dc_from = (payload.get("dc_from") or "").strip()
    dc_to = (payload.get("dc_to") or "").strip()

    if report_key != "snf_admission_summary":
        raise HTTPException(status_code=400, detail="Unknown report_key")

    if not dc_from or not dc_to:
        raise HTTPException(status_code=400, detail="dc_from and dc_to are required (YYYY-MM-DD)")

    conn = get_db()
    try:
        cur = conn.cursor()

        # Cohort = hospital discharges in date range, joined to SNF admissions by visit_id
        cur.execute(
            """
            SELECT
                h.visit_id,
                h.dc_date,
                h.disposition AS dc_disposition,

                s.id AS snf_id,
                s.ai_is_snf_candidate,
                s.notified_by_hospital,
                s.assignment_confirmation,

                s.final_snf_name_display,
                s.ai_snf_name_raw,
                f.facility_name AS effective_facility_name,
                f.attending AS facility_attending,

                COALESCE(s.final_snf_facility_id, s.ai_snf_facility_id) AS effective_facility_id
            FROM hospital_discharges h
            LEFT JOIN snf_admissions s
              ON s.visit_id = h.visit_id
            LEFT JOIN snf_admission_facilities f
              ON f.id = COALESCE(s.final_snf_facility_id, s.ai_snf_facility_id)
            WHERE h.dc_date IS NOT NULL
              AND date(h.dc_date) >= date(?)
              AND date(h.dc_date) <= date(?)
            """,
            (dc_from, dc_to),
        )
        rows = cur.fetchall()

        # Metrics should be computed only where we actually have an SNF admission row
        snf_rows = [r for r in rows if r["snf_id"] is not None]

        total_joined = len(rows)
        total_snf = len(snf_rows)

        # -----------------------------
        # Facility volume (top 10)
        # -----------------------------
        volume_map: Dict[str, Dict[str, Any]] = {}
        for r in snf_rows:
            fac_label = _facility_display(r["final_snf_name_display"], r["effective_facility_name"], r["ai_snf_name_raw"])
            is_starred = bool((r["facility_attending"] or "").strip())  # matches your dropdown star logic
            if fac_label not in volume_map:
                volume_map[fac_label] = {"facility": fac_label, "count": 0, "is_starred": False}
            volume_map[fac_label]["count"] += 1
            if is_starred:
                volume_map[fac_label]["is_starred"] = True

        top_facilities = sorted(volume_map.values(), key=lambda x: x["count"], reverse=True)[:10]

        # -----------------------------
        # Notified by Hospital
        # -----------------------------
        notified_count = sum(1 for r in snf_rows if int(r["notified_by_hospital"] or 0) == 1)
        notified_pct = (notified_count / total_snf * 100.0) if total_snf else 0.0

        # -----------------------------
        # Assignment confirmation distribution
        # -----------------------------
        assign_counts = {"Assigned": 0, "Assigned Out": 0, "Unknown": 0}
        for r in snf_rows:
            v = (r["assignment_confirmation"] or "Unknown").strip()
            if v not in assign_counts:
                v = "Unknown"
            assign_counts[v] += 1

        assign_pct = {
            k: (assign_counts[k] / total_snf * 100.0) if total_snf else 0.0
            for k in assign_counts
        }

        # -----------------------------
        # AI candidate accuracy vs discharge disposition (SNF vs not SNF)
        # -----------------------------
        # Only score where discharge disposition exists and we have an ai_is_snf_candidate value
        scored = 0
        correct = 0
        for r in snf_rows:
            dc_dispo = (r["dc_disposition"] or "").strip()
            if not dc_dispo:
                continue
            scored += 1
            actual_is_snf = _is_snf_disposition(dc_dispo)
            predicted_is_snf = bool(int(r["ai_is_snf_candidate"] or 0) == 1)
            if actual_is_snf == predicted_is_snf:
                correct += 1

        ai_accuracy_pct = (correct / scored * 100.0) if scored else 0.0

        # -----------------------------
        # Percent of emails opened with FACILITY PIN (de-duped to 1 open per email)
        # Cohort emails = secure links whose admission_ids include a cohort snf_id
        # -----------------------------
        cohort_admission_ids = set(int(r["snf_id"]) for r in snf_rows if r["snf_id"] is not None)

        cur.execute(
            """
            SELECT id, admission_ids
            FROM snf_secure_links
            WHERE sent_at IS NOT NULL AND sent_at <> ''
            ORDER BY sent_at DESC
            """
        )
        link_rows = cur.fetchall()

        cohort_link_ids: List[int] = []
        for lr in link_rows:
            try:
                ids = json.loads(lr["admission_ids"] or "[]")
                if not isinstance(ids, list):
                    continue
                if any(int(x) in cohort_admission_ids for x in ids if str(x).isdigit()):
                    cohort_link_ids.append(int(lr["id"]))
            except Exception:
                continue

        total_emails = len(cohort_link_ids)

        opened_email_ids: set[int] = set()
        if cohort_link_ids:
            placeholders = ",".join("?" for _ in cohort_link_ids)
            cur.execute(
                f"""
                SELECT DISTINCT secure_link_id
                FROM snf_secure_link_access_log
                WHERE LOWER(COALESCE(pin_type,'')) = 'facility'
                  AND secure_link_id IN ({placeholders})
                """,
                cohort_link_ids,
            )
            opened_email_ids = set(int(x[0]) for x in cur.fetchall())

        opened_count = len(opened_email_ids)
        opened_pct = (opened_count / total_emails * 100.0) if total_emails else 0.0
        unopened_pct = 100.0 - opened_pct if total_emails else 0.0

        return {
            "ok": True,
            "report_key": report_key,
            "filters": {"dc_from": dc_from, "dc_to": dc_to},

            "totals": {
                "joined_rows": total_joined,
                "snf_rows": total_snf,
                "scored_ai_rows": scored,
                "emails_in_cohort": total_emails,
            },

            "top_facilities": top_facilities,

            "metrics": {
                "notified_count": notified_count,
                "notified_pct": round(notified_pct, 1),

                "facility_pin_opened_pct": round(opened_pct, 1),
                "facility_pin_unopened_pct": round(unopened_pct, 1),

                "assignment_counts": assign_counts,
                "assignment_pct": {k: round(assign_pct[k], 1) for k in assign_pct},

                "ai_accuracy_correct": correct,
                "ai_accuracy_total": scored,
                "ai_accuracy_pct": round(ai_accuracy_pct, 1),
            },
        }
    finally:
        conn.close()

@app.get("/admin/snf/email-log/pdf/{secure_link_id}")
async def admin_snf_email_log_pdf(request: Request, secure_link_id: int):
    """
    Admin-only: regenerate the PDF for a previously sent secure link and return it as a download.
    """
    require_admin(request)

    if not HAVE_WEASYPRINT:
        raise HTTPException(status_code=500, detail="PDF support is not installed (weasyprint).")

    conn = get_db()
    try:
        cur = conn.cursor()

        cur.execute(
            """
            SELECT *
            FROM snf_secure_links
            WHERE id = ?
            """,
            (int(secure_link_id),),
        )
        link = cur.fetchone()
        if not link:
            raise HTTPException(status_code=404, detail="Secure link not found")

        cur.execute(
            """
            SELECT *
            FROM snf_admission_facilities
            WHERE id = ?
            """,
            (int(link["snf_facility_id"]),),
        )
        fac = cur.fetchone()
        fac_name = (fac["facility_name"] if fac else "") or "Receiving Facility"
        attending = (fac["attending"] if fac else "") or ""

        try:
            admission_ids = json.loads(link["admission_ids"] or "[]")
        except Exception:
            admission_ids = []

        if not admission_ids:
            raise HTTPException(status_code=400, detail="No admissions found for this link")

        placeholders = ",".join("?" for _ in admission_ids)
        sql = f"""
        SELECT
            s.id,
            s.patient_name,
            s.hospital_name,
            COALESCE(s.dob, raw.dob) AS dob,
            COALESCE(s.attending, raw.attending, raw.note_author) AS hospitalist
        FROM snf_admissions s
        LEFT JOIN cm_notes_raw raw
          ON raw.id = s.raw_note_id
        WHERE s.id IN ({placeholders})
        ORDER BY s.patient_name COLLATE NOCASE
        """
        cur.execute(sql, admission_ids)
        rows = cur.fetchall()
        if not rows:
            raise HTTPException(status_code=400, detail="Admissions no longer available")

        for_date = (link["for_date"] or "").strip() or dt.date.today().isoformat()

        html_doc = build_snf_pdf_html(fac_name, for_date, rows, attending)
        pdf_bytes = WEASY_HTML(string=html_doc).write_pdf()

        safe_fac = re.sub(r"[^A-Za-z0-9]+", "_", fac_name) or "SNF"
        filename = f"SNF_Referrals_{safe_fac}_{for_date}.pdf"

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    finally:
        conn.close()


def build_snf_pdf_html(
    facility_name: str,
    for_date: str,
    rows: List[sqlite3.Row],
    attending: str = "",
) -> str:
    """
    Build the HTML for the SNF admissions PDF, styled like the
    Example PDF Export v2.1, but with slightly smaller fonts and
    no Source pill.
    """
    safe_fac = html.escape(facility_name or "Receiving Facility")
    safe_attending = html.escape(attending) if attending else ""

    provider_callout = ""
    if safe_attending:
        provider_callout = f"""
      <div class="provider-callout">
        <span class="provider-label">The patients below should be assigned to the following First Docs provider:</span>
        <strong class="provider-name">{safe_attending}</strong>
      </div>
    """


    patient_count = len(rows)
    patient_word = "patient" if patient_count == 1 else "patients"

    # Build table body rows
    body_rows: List[str] = []
    for r in rows:
        name = html.escape(r["patient_name"] or "")
        dob = html.escape(r["dob"] or "")
        hosp = html.escape(r["hospital_name"] or "")
        md = html.escape(r["hospitalist"] or "")

        body_rows.append(
            f"""
            <tr>
              <td class="col-patient">
                <strong>{name}</strong>
              </td>
              <td>{dob}</td>
              <td class="col-hospital">{hosp}</td>
              <td class="col-md">{md}</td>
            </tr>
            """
        )

    body_html = "\n".join(body_rows).strip() or """
        <tr>
          <td colspan="4" style="padding: 16px; text-align: center; color: #6b7280;">
            No patients found for this date.
          </td>
        </tr>
    """.strip()

    PAGE_MARGIN_PX = 16

    # NOTE: all CSS/HTML braces that are not Python variables are doubled {{ }}
    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Upcoming SNF Admissions</title>
  <style>
    /* Page margins so content fills more of the page */
    @page {{
      size: letter;
      margin: {PAGE_MARGIN_PX}px;
    }}

    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}

    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #f5f5f7;
      color: #111827;
      line-height: 1.5;
    }}

    .report-shell {{
      padding: 8px;
      display: flex;
      justify-content: center;
      align-items: flex-start;
    }}

    .report {{
      width: 100%;
      max-width: 100%;
      background: #ffffff;
      border-radius: 20px;
      padding: 28px 32px 18px;
      display: flex;
      flex-direction: column;
      gap: 24px;
    }}

    .report-header {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 16px;
      border-bottom: 1px solid #e5e7eb;
      padding-bottom: 16px;
    }}

    /* ✅ Key fix: make the left side expand to full available width */
    .header-main {{
      flex: 1;
      min-width: 0;
      width: 100%;
      display: flex;
      flex-direction: column;
      gap: 4px;
    }}

    /* ✅ Ensure the “top sections” actually render full-width */
    .facility-line,
    .header-description,
    .provider-callout {{
      width: 100%;
      box-sizing: border-box;
    }}


    .report-kicker {{
      font-size: 10px;
      letter-spacing: 0.15em;
      text-transform: uppercase;
      color: #6b7280;
    }}

    .report-title {{
      font-size: 20px;
      font-weight: 700;
      color: #111827;
    }}

    .facility-line {{
      font-size: 13px;
      color: #4b5563;
    }}

    .facility-chip {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 10px;
      border-radius: 999px;
      background: #f3f4ff;
      font-size: 11px;
      color: #3730a3;
      margin-top: 4px;
    }}

    .legend-dot {{
      width: 7px;
      height: 7px;
      border-radius: 999px;
      background: #4f46e5;
    }}

    .header-side {{
      text-align: right;
      font-size: 11px;
      color: #6b7280;
    }}

    .header-side-label {{
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-size: 9px;
      color: #9ca3af;
    }}

    .header-side-value {{
      margin-top: 2px;
    }}

    .header-description {{
      font-size: 12px;
      color: #4b5563;
      margin-top: 8px;
    }}

    .provider-callout {{
      margin-top: 10px;
      padding: 10px 10px;
      border-left: 4px solid #4f46e5;          /* slightly stronger accent */
      background: #eef2ff;                     /* subtle indigo tint */
      border: 1px solid #e0e7ff;               /* soft border */
      border-radius: 12px;
      
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;      
      
      font-size: 12px;                         /* slightly larger than line above */
      font-weight: 500;                        /* slightly bolder */
      line-height: 1.3;
      color: #111827;
      box-shadow: 0 1px 0 rgba(17, 24, 39, 0.04); /* subtle depth, stays on-brand */
    }}

    .provider-callout .provider-label {{
      color: #374151;
      margin-top: 2px;
      margin-bottom: 2px;
    }}

    .provider-callout .provider-name {{
      font-weight: 700; /* still bold, but not heavy */
      color: #111827;
    }}


    .summary-bar {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      padding: 10px 12px;
      background: #f9fafb;
      border-radius: 12px;
      border: 1px solid #e5e7eb;
      font-size: 11px;
      color: #4b5563;
    }}

    .summary-count {{
      font-weight: 600;
      color: #111827;
    }}

    .summary-tags {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      justify-content: flex-end;
    }}

    /* plain text status (no pill) */
    .summary-status {{
      font-size: 10px;
      color: #374151;
    }}

    .table-wrapper {{
      border-radius: 16px;
      border: 1px solid #e5e7eb;
      overflow: hidden;
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }}

    thead {{
      background: #f3f4ff;
    }}

    th,
    td {{
      padding: 10px 14px;
      text-align: left;
    }}

    th {{
      font-size: 10px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.09em;
      color: #4b5563;
      border-bottom: 1px solid #e5e7eb;
      white-space: nowrap;
    }}

    tbody tr:nth-child(even) {{
      background: #f9fafb;
    }}

    tbody tr:hover {{
      background: #eef2ff;
    }}

    td {{
      border-bottom: 1px solid #e5e7eb;
      vertical-align: middle;
      color: #111827;
    }}

    tbody tr:last-child td {{
      border-bottom: none;
    }}

    .col-patient strong {{
      font-weight: 600;
    }}

    .col-patient span {{
      display: block;
      font-size: 10px;
      color: #6b7280;
      margin-top: 2px;
    }}

    .col-hospital {{
      font-size: 12px;
      color: #111827;
    }}

    .col-md {{
      font-size: 12px;
      color: #374151;
    }}

    .report-footer {{
      padding-top: 10px;
      margin-top: 4px;
      border-top: 1px solid #f3f4f6;
      text-align: center;
    }}

    .pdf-disclaimer {{
      margin-top: 12px;
      margin-bottom: 18px;
      font-size: 10.5px;
      color: #6b7280;
      line-height: 1.4;
      text-align: left;
    }}


    .footer-brand {{
      font-size: 10px;
      color: #000000;
    }}
  </style>
</head>
<body>
  <div class="report-shell">
    <div class="report">
      <!-- Header -->
      <header class="report-header">
        <div class="header-main">
          <div class="report-kicker">Upcoming SNF Admissions</div>
          <div class="report-title">First Docs</div>
          <div class="facility-line">
            Receiving Facility:
            <span class="facility-chip">
              <span class="legend-dot"></span>
              {safe_fac}
            </span>
          </div>
          <p class="header-description">
            Our Hospitalists at HCA Florida JFK Hospital have identified the following patients as expected discharges to your facility. Please contact Doug Neal (Doug.Neal@medrina.com) or Stephanie Sellers (ssellers@startevolv.com) if you have questions or would like to add additional recipients to future emails.
          </p>

{provider_callout}
        </div>
        <div class="header-side">
          <div class="header-side-label">Disposition:</div>
          <div class="header-side-value">SNF</div>
        </div>
      </header>

      <!-- Summary Bar -->
      <section class="summary-bar">
        <div>
          <span class="summary-count">{patient_count} {patient_word}</span>
          currently flagged as expected SNF admissions.
        </div>
        <div class="summary-tags">
          <!-- Source pill removed on purpose -->
          <span class="summary-status">Status: Upcoming Discharges</span>
        </div>
      </section>

      <!-- Patient Table -->
      <section class="table-wrapper">
        <table>
          <thead>
            <tr>
              <th>Patient</th>
              <th>DOB</th>
              <th>Hospital</th>
              <th>Hospitalist</th>
            </tr>
          </thead>
          <tbody>
            {body_html}
          </tbody>
        </table>
      </section>

      <!-- Disclaimer -->
      <div class="pdf-disclaimer">
        Disclaimer: The information provided reflects anticipated or potential patient referrals and is shared for planning purposes only. Actual admissions are not guaranteed and may change at any time.
      </div>

      <!-- Footer -->
      <footer class="report-footer">
        <div class="footer-brand">Powered by Evolv Health</div>
      </footer>
    </div>
  </div>
</body>
</html>
"""
    return html_doc





# ============================
# SNF Secure Link routes
# ============================

def _parse_utc_iso(s: str) -> Optional[dt.datetime]:
    if not s:
        return None
    try:
        # accept "2025-12-19T04:11:09Z" or without Z
        s2 = s.replace("Z", "")
        return dt.datetime.fromisoformat(s2)
    except Exception:
        return None

@app.head("/snf/secure/{token}")
async def snf_secure_link_head(token: str, request: Request):
    """
    Email clients/security scanners often send HEAD requests to links.
    Return a fast 200 so they don't log 405s.
    """
    return Response(status_code=200, headers={"Content-Type": "text/html; charset=utf-8"})

@app.get("/snf/secure/{token}", response_class=HTMLResponse)
async def snf_secure_link_get(token: str, request: Request):
    """Shows a simple PIN entry page."""
    token_hash = sha256_hex(token)
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, snf_facility_id, for_date, expires_at
            FROM snf_secure_links
            WHERE token_hash = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (token_hash,),
        )
        row = cur.fetchone()
        if not row:
            return HTMLResponse("<h2>Link not found</h2><p>This secure link is invalid.</p>", status_code=404)

        exp = _parse_utc_iso(row["expires_at"])
        if not exp or exp <= dt.datetime.utcnow():
            return HTMLResponse("<h2>Link expired</h2><p>This secure link has expired.</p>", status_code=410)

        # light facility name for the page header
        cur.execute("SELECT facility_name FROM snf_admission_facilities WHERE id = ?", (row["snf_facility_id"],))
        fac = cur.fetchone()
        fac_name = html.escape((fac["facility_name"] if fac else "your facility") or "your facility")

        page = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Secure SNF List</title>
  <style>
    /* Keep layout stable so the PIN field never overflows/crops */
    *{{box-sizing:border-box;}}

    body{{margin:0;padding:0;background:#F5F7FA;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;color:#111827;}}
    .wrap{{max-width:520px;margin:40px auto;padding:0 14px;}}
    .card{{background:#fff;border:1px solid #e5e7eb;border-radius:16px;box-shadow:0 10px 28px rgba(0,0,0,.08);overflow:hidden;}}
    .topbar{{background:#0D3B66;padding:18px 22px;color:#fff;font-weight:700;}}
    .mintbar{{height:4px;background:#A8E6CF;}} /* subtle mint accent */
    .content{{padding:22px;}}
    h1{{margin:0 0 6px 0;font-size:18px;color:#0D3B66;}}
    p{{margin:0 0 14px 0;font-size:13px;color:#374151;line-height:1.5;}}
    label{{display:block;font-size:12px;color:#374151;margin-bottom:6px;font-weight:600;}}
    input{{width:100%;max-width:100%;padding:12px 12px;border-radius:12px;border:1px solid #d1d5db;font-size:14px;}}
    input:focus{{outline:none;border-color:#A8E6CF;box-shadow:0 0 0 3px rgba(168,230,207,.45);}}

    /* Match the email button styling */
    .btn{{margin-top:12px;display:inline-block;background:#0D3B66;color:#ffffff;border:none;padding:12px 18px;border-radius:10px;font-weight:800;font-size:14px;cursor:pointer;box-shadow:0 8px 18px rgba(13,59,102,.18);}}
    .btn:hover{{background:#0b3357;}}

    .fine{{margin-top:14px;font-size:12px;color:#6b7280;}}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <div class="topbar">First Docs • Secure List</div>
      <div class="mintbar" aria-hidden="true"></div>
      <div class="content">
        <h1>{fac_name}</h1>
        <p>Enter your facility password / PIN to view the secure list. This link expires in 48hrs.</p>
        <form method="post">
          <label for="pin">Facility PIN</label>
          <input id="pin" name="pin" type="password" autocomplete="one-time-code" required />
          <button class="btn" type="submit">View List</button>
        </form>
        <div class="fine">Powered by Evolv Health</div>
      </div>
    </div>
  </div>
</body>
</html>"""
        return HTMLResponse(page)

    finally:
        conn.close()

# ---------------------------------------------------------------------------
# Hospital extraction profiles CRUD (Admin)
# ---------------------------------------------------------------------------

@app.get("/api/hospital-extraction-profiles/list")
def api_hex_profiles_list():
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, hospital_name, document_type, profile_json, active, updated_at
            FROM hospital_extraction_profiles
            ORDER BY hospital_name ASC, document_type ASC
            """
        )
        rows = cur.fetchall()
        items = []
        for r in rows:
            rr = dict(r)
            # profile_json stays as string for the UI to parse safely
            items.append(rr)
        return {"ok": True, "items": items}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/hospital-extraction-profiles/upsert")
def api_hex_profiles_upsert(payload: dict):
    """
    payload:
      {
        hospital_name: str,
        document_type: str,
        active: 0|1,
        profile_json: dict   # heading(lowercased) -> section_key
      }
    """
    try:
        hospital_name = (payload.get("hospital_name") or "").strip()
        document_type = (payload.get("document_type") or "").strip()
        active = 1 if int(payload.get("active", 1) or 1) else 0
        profile_obj = payload.get("profile_json") or {}

        if not hospital_name or not document_type:
            return {"ok": False, "error": "hospital_name and document_type are required"}

        if not isinstance(profile_obj, dict) or len(profile_obj) == 0:
            return {"ok": False, "error": "profile_json must be a non-empty object"}

        # Normalize keys to lowercase for consistent matching
        normalized = {}
        for k, v in profile_obj.items():
            kk = (str(k) or "").strip().lower()
            vv = (str(v) or "").strip()
            if kk and vv:
                normalized[kk] = vv

        if len(normalized) == 0:
            return {"ok": False, "error": "profile_json produced no valid mappings"}

        conn = get_db()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO hospital_extraction_profiles (hospital_name, document_type, profile_json, active, updated_at)
            VALUES (?, ?, ?, ?, datetime('now'))
            ON CONFLICT(hospital_name, document_type)
            DO UPDATE SET
              profile_json = excluded.profile_json,
              active = excluded.active,
              updated_at = datetime('now')
            """,
            (hospital_name, document_type, json.dumps(normalized), active),
        )
        conn.commit()
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/api/hospital-discharges/list")
async def hospital_discharges_list():
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                visit_id,
                patient_name,
                patient_mrn,
                hospital_name,
                admit_date,
                dc_date,
                attending,
                pcp,
                insurance,
                disposition,
                dc_agency,
                updated_at
            FROM hospital_discharges
            ORDER BY COALESCE(dc_date, admit_date, updated_at) DESC, updated_at DESC
            LIMIT 500
            """
        )
        return {"ok": True, "items": [dict(r) for r in cur.fetchall()]}
    finally:
        conn.close()

@app.get("/api/hospital-discharges/visit/{visit_id}")
async def hospital_discharge_visit_details(visit_id: str):
    conn = get_db()
    try:
        cur = conn.cursor()

        cur.execute(
            """
            SELECT
                visit_id, patient_name, patient_mrn, hospital_name,
                admit_date, dc_date,
                attending, pcp, insurance,
                disposition, dc_agency, updated_at
            FROM hospital_discharges
            WHERE visit_id = ?
            """,
            (visit_id,),
        )
        visit = cur.fetchone()
        if not visit:
            raise HTTPException(status_code=404, detail="visit_id not found")

        cur.execute(
            """
            SELECT
                id,
                document_type,
                document_datetime,
                hospital_name,
                created_at
            FROM hospital_documents
            WHERE visit_id = ?
            ORDER BY COALESCE(document_datetime, created_at) DESC
            """,
            (visit_id,),
        )
        docs = [dict(r) for r in cur.fetchall()]

        # Optionally include sections for the newest doc by default
        sections_by_doc = {}
        if docs:
            first_doc_id = docs[0]["id"]
            cur.execute(
                """
                SELECT section_key, section_title, section_order, section_text
                FROM hospital_document_sections
                WHERE document_id = ?
                ORDER BY section_order ASC
                """,
                (first_doc_id,),
            )
            sections_by_doc[str(first_doc_id)] = [dict(r) for r in cur.fetchall()]

        return {
            "ok": True,
            "visit": dict(visit),
            "documents": docs,
            "sections_by_doc": sections_by_doc,
        }
    finally:
        conn.close()

@app.get("/api/hospital-documents/{document_id}/sections")
async def hospital_document_sections_get(document_id: int):
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT section_key, section_title, section_order, section_text
            FROM hospital_document_sections
            WHERE document_id = ?
            ORDER BY section_order ASC
            """,
            (document_id,),
        )
        return {"ok": True, "items": [dict(r) for r in cur.fetchall()]}
    finally:
        conn.close()

@app.post("/snf/secure/{token}")
async def snf_secure_link_post(token: str, request: Request, pin: Optional[str] = Form(None)):
    """Validates PIN and returns the PDF for this secure link."""
    if not HAVE_WEASYPRINT:
        raise HTTPException(status_code=500, detail="PDF support is not installed (weasyprint).")

    token_hash = sha256_hex(token)
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, snf_facility_id, admission_ids, for_date, expires_at
            FROM snf_secure_links
            WHERE token_hash = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (token_hash,),
        )
        link = cur.fetchone()
        if not link:
            raise HTTPException(status_code=404, detail="Invalid link")

        exp = _parse_utc_iso(link["expires_at"])
        if not exp or exp <= dt.datetime.utcnow():
            raise HTTPException(status_code=410, detail="Link expired")

        snf_facility_id = int(link["snf_facility_id"] or 0)

        # Load facility + PIN hash
        cur.execute(
            "SELECT facility_name, attending, pin_hash FROM snf_admission_facilities WHERE id = ?",
            (snf_facility_id,),
        )
        fac = cur.fetchone()
        if not fac:
            raise HTTPException(status_code=404, detail="Facility not found")

        fac_name = fac["facility_name"] or "SNF facility"

        # Render the PIN form page (same page, with optional error message)
        def _render_form(error_msg: str = "") -> str:
            err_html = f'<div class="err">{error_msg}</div>' if error_msg else ""
            return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Secure SNF List</title>
  <style>
    *{{box-sizing:border-box;}}
    body{{margin:0;padding:0;background:#F5F7FA;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;color:#111827;}}
    .wrap{{max-width:520px;margin:40px auto;padding:0 14px;}}
    .card{{background:#fff;border:1px solid #e5e7eb;border-radius:16px;box-shadow:0 10px 28px rgba(0,0,0,.08);overflow:hidden;}}
    .topbar{{background:#0D3B66;padding:16px 22px;color:#fff;font-weight:700;}}
    .mintbar{{height:4px;background:#A8E6CF;}} /* subtle mint accent */
    .content{{padding:22px;}}
    h1{{margin:0 0 6px 0;font-size:18px;color:#0D3B66;}}
    p{{margin:0 0 14px 0;font-size:13px;color:#374151;line-height:1.5;}}
    label{{display:block;font-size:12px;color:#374151;margin-bottom:6px;font-weight:600;}}
    form{{margin:0;}}
    input{{display:block;width:100%;max-width:100%;min-width:0;padding:12px 12px;border-radius:12px;border:1px solid #d1d5db;font-size:14px;}}
    input:focus{{outline:none;border-color:#A8E6CF;box-shadow:0 0 0 3px rgba(168,230,207,.45);}}

    /* Match email button styling */
    .btn{{margin-top:12px;display:inline-block;background:#0D3B66;color:#ffffff;border:none;padding:12px 18px;border-radius:10px;font-weight:800;font-size:14px;cursor:pointer;box-shadow:0 8px 18px rgba(13,59,102,.18);}}
    .btn:hover{{background:#0b3357;}}

    .fine{{margin-top:14px;font-size:12px;color:#6b7280;text-align:center;}}
    .err{{margin:0 0 12px 0;padding:10px 12px;border-radius:12px;border:1px solid #fca5a5;background:#fef2f2;color:#991b1b;font-size:13px;}}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <div class="topbar">First Docs • Secure List</div>
      <div class="mintbar" aria-hidden="true"></div>
      <div class="content">
        <h1>{fac_name}</h1>
        <p>Enter your facility password / PIN to view the secure list. This link expires automatically.</p>
        {err_html}
        <form method="post">
          <label for="pin">Facility PIN</label>
          <input id="pin" name="pin" type="password" autocomplete="one-time-code" />
          <button class="btn" type="submit">View List</button>
        </form>
        <div class="fine">Powered by Evolv Health</div>
      </div>
    </div>
  </div>
</body>
</html>"""

        # Normalize PIN (prevents FastAPI from returning raw 422 JSON)
        pin = (pin or "").strip()
        if not pin:
            return HTMLResponse(_render_form("Please enter your Facility PIN."), status_code=401)

        # Accept Facility PIN OR Universal PIN OR SNF_DEFAULT_PIN
        facility_hash = (fac["pin_hash"] or "").strip()
        universal_hash = get_snf_universal_pin_hash(cur)  # stored in ai_settings
        default_hash = hash_pin(SNF_DEFAULT_PIN) if SNF_DEFAULT_PIN else ""

        candidate_hashes = [h for h in (facility_hash, universal_hash, default_hash) if h]

        if not candidate_hashes:
            return HTMLResponse(
                _render_form("No facility PIN is configured. Please contact Evolv."),
                status_code=500
            )

        # NEW: determine WHICH PIN matched (so we can log pin_type)
        pin_type = None
        if facility_hash and verify_pin(pin, facility_hash):
            pin_type = "facility"
        elif universal_hash and verify_pin(pin, universal_hash):
            pin_type = "universal"
        elif default_hash and verify_pin(pin, default_hash):
            pin_type = "default"

        if not pin_type:
            return HTMLResponse(_render_form("Incorrect PIN. Please try again."), status_code=401)

        # NEW: write an access log row (timestamp + pin type + IP/UA)
        ip = (request.client.host if request.client else None)
        ua = request.headers.get("user-agent")

        try:
            cur.execute(
                """
                INSERT INTO snf_secure_link_access_log (secure_link_id, snf_facility_id, pin_type, ip, user_agent)
                VALUES (?, ?, ?, ?, ?)
                """,
                (int(link["id"]), int(link["snf_facility_id"]), pin_type, ip, ua),
            )
        except Exception as e:
            # Don't block PDF if logging fails
            print("[snf-secure-link] access log insert failed:", repr(e))

        # Parse admission ids
        try:
            admission_ids = json.loads(link["admission_ids"] or "[]")
        except Exception:
            admission_ids = []

        if not admission_ids:
            raise HTTPException(status_code=400, detail="No admissions found for this link")

        # Fetch admissions rows
        placeholders = ",".join("?" for _ in admission_ids)
        sql = f"""
        SELECT
            s.id,
            s.patient_name,
            s.hospital_name,
            COALESCE(s.dob, raw.dob) AS dob,
            COALESCE(s.attending, raw.attending, raw.note_author) AS hospitalist
        FROM snf_admissions s
        LEFT JOIN cm_notes_raw raw
          ON raw.id = s.raw_note_id
        WHERE s.id IN ({placeholders})
        ORDER BY s.patient_name COLLATE NOCASE
        """
        cur.execute(sql, admission_ids)
        rows = cur.fetchall()
        if not rows:
            raise HTTPException(status_code=400, detail="Admissions no longer available")

        for_date = (link["for_date"] or "").strip() or dt.date.today().isoformat()
        facility_name = fac["facility_name"] or "SNF facility"
        attending = fac["attending"] or ""

        html_doc = build_snf_pdf_html(facility_name, for_date, rows, attending)
        pdf_bytes = WEASY_HTML(string=html_doc).write_pdf()

        # mark used_at (audit)
        cur.execute("UPDATE snf_secure_links SET used_at = datetime('now') WHERE id = ?", (link["id"],))
        conn.commit()

        safe_fac = re.sub(r"[^A-Za-z0-9]+", "_", facility_name) or "SNF"
        filename = f"SNF_Referrals_{safe_fac}_{for_date}.pdf"

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'inline; filename="{filename}"'},
        )
    finally:
        conn.close()




@app.post("/admin/snf/email-pdf")
async def admin_snf_email_pdf(
    request: Request,
    payload: Dict[str, Any] = Body(...),
):
    """
    Send a single SNF email with a nicely formatted PDF attachment
    for the CURRENT filtered list of admissions for one SNF facility.

    Uses HTML → PDF via WeasyPrint so the design exactly matches
    Example PDF Export v2.1.html.
    """
    require_admin(request)

    if not HAVE_WEASYPRINT:
        raise HTTPException(
            status_code=500,
            detail="PDF support is not installed (weasyprint). Please add 'weasyprint' to your environment."
        )

    snf_facility_id = int(payload.get("snf_facility_id") or 0)
    admission_ids = payload.get("admission_ids") or []
    for_date = (payload.get("for_date") or "").strip()
    test_only = bool(payload.get("test_only"))
    test_email_to = (payload.get("test_email_to") or "").strip()

    if not snf_facility_id:
        raise HTTPException(status_code=400, detail="snf_facility_id is required")
    if not admission_ids:
        raise HTTPException(status_code=400, detail="admission_ids must be a non-empty list")

    if test_only and not test_email_to:
        raise HTTPException(
            status_code=400,
            detail="test_email_to is required when test_only=true"
        )

    # Validate date format lightly (optional)
    if for_date and not re.match(r"^\d{4}-\d{2}-\d{2}$", for_date):
        raise HTTPException(status_code=400, detail="for_date must be YYYY-MM-DD")

    if not (SMTP_HOST and SMTP_USER and SMTP_PASSWORD):
        return {"ok": False, "message": "SMTP not configured; skipping send."}

    conn = get_db()
    try:
        cur = conn.cursor()

        # Load SNF facility master row
        cur.execute(
            """
            SELECT id, facility_name, attending, notes, notes2, aliases, facility_emails
            FROM snf_admission_facilities
            WHERE id = ?
            """,
            (snf_facility_id,),
        )
        fac = cur.fetchone()
        if not fac:
            raise HTTPException(status_code=404, detail="SNF Admission Facility not found")

        facility_name = fac["facility_name"] or "SNF facility"
        attending = fac["attending"] or ""
        facility_emails = normalize_email_list(fac["facility_emails"] or "")

        if not test_only and not facility_emails:
            raise HTTPException(
                status_code=400,
                detail="No facility_emails configured for this SNF Admission Facility."
            )

        # ✅ NEW: determine the recipient list we will send to (and log in DB)
        to_addr = test_email_to if test_only else facility_emails

        # Look up the admissions (and DOB from raw notes)
        # Look up the admissions (DOB + Hospitalist)
        placeholders = ",".join("?" for _ in admission_ids)

        sql = f"""
        SELECT
            s.id,
            s.patient_name,
            s.hospital_name,
            COALESCE(s.dob, raw.dob) AS dob,
            COALESCE(s.attending, raw.attending, raw.note_author) AS hospitalist
        FROM snf_admissions s
        LEFT JOIN cm_notes_raw raw
          ON raw.id = s.raw_note_id
        WHERE s.id IN ({placeholders})
        ORDER BY s.patient_name COLLATE NOCASE
        """

        try:
            cur.execute(sql, admission_ids)
        except Exception as e:
            print("[snf-email-pdf] SQL failed:", repr(e))
            print("[snf-email-pdf] SQL text was:\n", sql)
            print("[snf-email-pdf] admission_ids:", admission_ids)
            raise


        rows = cur.fetchall()
        if not rows:
            raise HTTPException(status_code=400, detail="No admissions found for the provided IDs.")

        
        # ------------------------
        # Build secure expiring link (instead of attaching PHI)
        # ------------------------
        if not for_date:
            for_date = dt.date.today().isoformat()

        # Require a public base URL (Render env var) so links work for recipients
        base_url = get_public_base_url(request)
        if not base_url:
            raise HTTPException(status_code=500, detail="PUBLIC_APP_BASE_URL is not configured.")

        # Create a one-time random token, but only store its HASH in the DB
        raw_token = secrets.token_urlsafe(32)
        token_hash = sha256_hex(raw_token)

        expires_at = (dt.datetime.utcnow() + dt.timedelta(hours=SNF_LINK_TTL_HOURS)).replace(microsecond=0).isoformat() + "Z"

        # NEW: generate email_run_id BEFORE we insert the secure link row
        email_run_id = secrets.token_hex(8)

        cur.execute(
            """
            INSERT INTO snf_secure_links (token_hash, snf_facility_id, admission_ids, for_date, expires_at, email_run_id, sent_to)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (token_hash, snf_facility_id, json.dumps(admission_ids), for_date, expires_at, email_run_id, to_addr),
        )
        conn.commit()

        secure_url = f"{base_url}/snf/secure/{raw_token}"

        # ✅ NEW: build the outbound email (HTML + plain text)
        subject = f"Pending SNF Admissions for {facility_name} – {for_date} PLEASE REVIEW"
        if test_only:
            subject = "[TEST] " + subject

        plain_body = (
            f"Secure SNF admissions list link (expires in {SNF_LINK_TTL_HOURS} hours):\n\n"
            f"{secure_url}\n"
        )
        html_body = build_snf_secure_link_email_html(secure_url, SNF_LINK_TTL_HOURS)

        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = INTAKE_EMAIL_FROM or SMTP_USER
        msg["To"] = to_addr
        msg.set_content(plain_body)
        msg.add_alternative(html_body, subtype="html")

        try:
            context = ssl.create_default_context()
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
                server.starttls(context=context)
                server.login(SMTP_USER, SMTP_PASSWORD)
                server.send_message(msg)
                
            # NEW: mark link "sent_at" only after successful SMTP send
            cur.execute(
                "UPDATE snf_secure_links SET sent_at = datetime('now') WHERE token_hash = ? AND email_run_id = ?",
                (token_hash, email_run_id),
            )
            conn.commit()

            # Mark those admissions as emailed if this is a real send
            if not test_only:
                cur.execute(
                    f"""
                    UPDATE snf_admissions
                       SET emailed_at = datetime('now'),
                           email_run_id = ?
                     WHERE id IN ({placeholders})
                    """,
                    (email_run_id, *admission_ids),
                )
                conn.commit()

            return {
                "ok": True,
                "emails_sent": 1,
                "email_run_id": email_run_id if not test_only else None,
                "admissions_tagged": 0 if test_only else len(admission_ids),
                "test_only": test_only,
            }
        except Exception as e:
            print("[snf-email-pdf] SMTP failed:", repr(e))
            raise HTTPException(status_code=500, detail=f"Failed to send email: {e}")
    finally:
        conn.close()


def send_intake_email(facility_id: str, payload: dict) -> None:
    """
    Fire-and-forget email notification for Facility Details submissions.
    Uses Google Workspace SMTP credentials from environment variables.
    """
    # If any critical piece is missing, just log and skip email (don't break intake)
    if not (SMTP_HOST and SMTP_USER and SMTP_PASSWORD and INTAKE_EMAIL_TO):
        print("[intake-email] Missing SMTP config; skipping email.")
        return

    facility_name = (payload.get("facility_name") or "").strip() or "(blank)"
    legal_name    = (payload.get("legal_name") or "").strip() or "(blank)"
    city          = (payload.get("city") or "").strip()
    state         = (payload.get("state") or "").strip()

    # Pull the token that was used for this intake (JS sends it in the payload)
    token = (payload.get("token") or "").strip()

    # Build the public Facility Details URL using env + facility_id + token
    facility_link = ""
    if FACILITY_FORM_BASE_URL and facility_id and token:
        facility_link = f"{FACILITY_FORM_BASE_URL}?f={facility_id}&t={token}"

    location = ""
    if city and state:
        location = f"{city}, {state}"
    elif city:
        location = city
    elif state:
        location = state

    subject = f"New Facility Details submission: {facility_name}"
    lines = [
        "A Facility Details form has been submitted.",
        "",
        f"Facility ID: {facility_id or '(unknown)'}",
        f"Facility Name: {facility_name}",
        f"Legal Name: {legal_name}",
    ]
    if location:
        lines.append(f"Location: {location}")

    # Optional: include a hint that they can view details in Admin
    lines.extend([
        "",
        "You can review this facility in the Admin portal under Facilities.",
        "https://ask.startevolv.com/admin#fac",
    ])

    # NEW: include direct link back to the Facility Details page
    if facility_link:
        lines.extend([
            "",
            "Open this Facility Details form:",
            facility_link,
        ])

    body = "\n".join(lines)

    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = INTAKE_EMAIL_FROM or SMTP_USER
        msg["To"] = INTAKE_EMAIL_TO
        msg.set_content(body)

        context = ssl.create_default_context()
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls(context=context)
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)

        print("[intake-email] Notification sent.")
    except Exception as e:
        # Log but do not raise; never break the intake flow for email issues
        print("[intake-email] Failed:", e)



@app.post("/intake/submit")
async def intake_submit(payload: dict = Body(...)):
    """
    Public facility intake submit.
    Always writes into SQLite; also triggers an email notification.
    """
    facility_name = (payload.get("facility_name") or "").strip()
    legal_name = (payload.get("legal_name") or "").strip()
    if not facility_name and not legal_name:
        raise HTTPException(status_code=400, detail="facility_name or legal_name is required")

    # Upsert facility + children
    try:
        fac_id = upsert_facility_and_children(payload)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

    # Fire-and-forget email (doesn't block the response)
    try:
        threading.Thread(
            target=send_intake_email,
            args=(fac_id, payload),
            daemon=True,
        ).start()
    except Exception as e:
        print("[intake-email] thread spawn failed:", e)

    return {"ok": True, "facility_id": fac_id}


# ---------------------------------------------------------------------------
# HTML routes & health check
# ---------------------------------------------------------------------------

def read_html(path: Path) -> str:
    """
    Small helper to load an HTML file from disk.
    If the file doesn't exist, return a simple 404-style HTML snippet.
    """
    try:
        if not path.exists():
            # This avoids a crash if the file is missing and gives you a clear message.
            return f"<html><body><h1>Missing file: {path.name}</h1></body></html>"
        return path.read_text(encoding="utf-8")
    except Exception as e:
        # In case of any unexpected file I/O error, don't crash the app.
        return f"<html><body><h1>Error loading {path.name}</h1><pre>{e}</pre></body></html>"


@app.get("/", response_class=HTMLResponse)
async def root():
    # Default Ask page
    return HTMLResponse(content=read_html(ASK_HTML))


@app.get("/ask-ui", response_class=HTMLResponse)
async def ask_ui():
    return HTMLResponse(content=read_html(ASK_HTML))


# Wix and your older links hit /ask-page, so keep this alias
@app.get("/ask-page", response_class=HTMLResponse)
async def ask_page():
    return HTMLResponse(content=read_html(ASK_HTML))


@app.get("/admin", response_class=HTMLResponse)
async def admin_ui():
    return HTMLResponse(content=read_html(ADMIN_HTML))


@app.get("/snf-admissions", response_class=HTMLResponse)
async def snf_admissions_ui():
    # New SNF admissions audit page
    return HTMLResponse(content=read_html(SNF_HTML))
    
@app.get("/census", response_class=HTMLResponse)
async def census_ui():
    return HTMLResponse(content=read_html(CENSUS_HTML))

from fastapi.responses import FileResponse

@app.get("/hospital-discharges", response_class=HTMLResponse)
def hospital_discharges_page():
    return read_html(HOSPITAL_DISCHARGE_HTML)

@app.get("/facility", response_class=HTMLResponse)
async def facility_ui():
    return HTMLResponse(content=read_html(FACILITY_HTML))


@app.get("/health")
async def health():
    return {"ok": True, "db_path": DB_PATH}


@app.get("/admin/download-db")
async def download_db(token: str):
    """
    Download the SQLite DB used by this app.
    Usage: GET /admin/download-db?token=YOUR_ADMIN_TOKEN
    """
    # Security: require your admin token
    if token != os.getenv("ADMIN_TOKEN"):
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Use the same DB file the app uses everywhere else
    db_path = Path(DB_PATH)

    if not db_path.exists():
        raise HTTPException(status_code=500, detail=f"DB not found at {db_path}")

    return FileResponse(
        path=str(db_path),
        filename="evolv.db",
        media_type="application/octet-stream",
    )


from fastapi.responses import HTMLResponse, Response

@app.get("/snf/secure/{token}", response_class=HTMLResponse)
async def snf_secure_token_page(token: str):
    # Simple PIN entry page (no PHI displayed here)
    return f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Secure SNF List</title>
  <style>
    body{{font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;background:#f3f4f6;margin:0;padding:28px;}}
    .card{{max-width:520px;margin:0 auto;background:#fff;border:1px solid #e5e7eb;border-radius:14px;box-shadow:0 10px 28px rgba(0,0,0,.08);overflow:hidden;}}
    .top{{background:#0b2ea5;color:#fff;padding:16px 18px;font-weight:700;}}
    .content{{padding:18px;}}
    label{{display:block;font-size:13px;color:#374151;margin-bottom:6px;}}
    input{{width:100%;padding:12px 12px;border-radius:10px;border:1px solid #d1d5db;font-size:14px;}}
    button{{margin-top:12px;background:#0b2ea5;color:#fff;border:0;border-radius:10px;padding:12px 14px;font-weight:700;width:100%;font-size:14px;cursor:pointer;}}
    .fine{{margin-top:12px;color:#6b7280;font-size:12px;line-height:1.4;}}
  </style>
</head>
<body>
  <div class="card">
    <div class="top">Secure patient referral list</div>
    <div class="content">
      <p style="margin:0 0 12px 0;color:#374151;font-size:14px;line-height:1.5;">
        Enter your facility PIN to view the secure PDF. This link expires in 48hrs.
      </p>
      <form method="POST" action="/snf/secure/{token}">
        <label>Facility PIN</label>
        <input type="password" name="pin" autocomplete="current-password" required />
        <button type="submit">View PDF</button>
      </form>
      <div class="fine">
        If you received this email by mistake, you can ignore it. Do not forward outside your organization.
      </div>
    </div>
  </div>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Entrypoint for Render
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
