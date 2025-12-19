import os
import math
import sqlite3
import datetime as dt
import json
import re
import threading
import smtplib
import ssl
import secrets
import hashlib
import io
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

from fastapi import FastAPI, HTTPException, Request, Form, Query, Body
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
    SNF_LINK_TTL_HOURS = int(os.getenv("SNF_LINK_TTL_HOURS", "24"))
except Exception:
    SNF_LINK_TTL_HOURS = 24



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

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def now_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


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



# ============================
# SNF Secure Link + PIN helpers
# ============================

def sha256_hex(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()


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
    ttl_hours = int(ttl_hours or 24)

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
        <h1>Accountable Care Hospitalist Group (ACHG)</h1>
        <p>
          Our Hospitalists have identified upcoming patients expected to discharge to your facility.
          To protect patient information, the list is available through a secure link (PIN required).
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
          Please contact Anthony Aguirre (Anthony.Aguirre@accountablecarehg.com) or Stephanie Sellers (ssellers@startevolv.com).
        </p>
      </div>

      <div class="btn-row">
        <a class="btn" href="{safe_url}" target="_blank" rel="noopener">
          View secure list
        </a>
      </div>

      <div class="fine">
        If you did not expect this email, you can ignore it. For security, do not forward this message outside of your organization.
      </div>

      <div class="divider"></div>

      <div class="footer">
        <div><strong>Powered by Evolv Health</strong></div>
        <div style="margin-top:6px;">
          Tip: If your browser asks for a password/PIN, enter your facility PIN.
          If you don’t know it, ask your facility admin or reply to your contact at Evolv.
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

            created_at                  TEXT DEFAULT (datetime('now'))
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

    inserted = 0
    skipped = 0
    errors: List[Dict[str, Any]] = []

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
    finally:
        conn.close()

    # Kick off SNF extraction in the background so that the SNF Admissions
    # page is populated without requiring a manual 'run extraction' click.
    try:
        threading.Thread(
            target=snf_run_extraction,
            kwargs={"days_back": 3},
            daemon=True,
        ).start()
    except Exception as e:
        print("[pad_cm_notes_bulk] failed to trigger SNF extraction:", e)

    return {
        "ok": True,
        "inserted": inserted,
        "skipped": skipped,
        "errors": errors,
    }





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
                SELECT facility_id, facility_name, city, state, corporate_group
                FROM facilities
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
        "- If multiple SNFs are mentioned over time, choose the FINAL accepted facility or the latest facility based on the latest note "
        "  (for example: if an early note says 'SNF Morse Life' but a later note says 'Confirmed with family "
        "  patient is now going to Signature Healthcare SNF', you must use 'Signature Healthcare SNF' as snf_name).\n"
        "- If the notes say the SNF plan is cancelled or changed to a different level of care "
        "  (for example: 'no longer going to SNF', 'plan changed to home with home health', "
        "  'family declined SNF and now wants home'), then set is_snf_candidate = false and use "
        "  snf_name = null and expected_transfer_date = null, even if earlier notes talked about SNF.\n"
        "- If the date is not clear, use null for expected_transfer_date.\n"
        "- If the plan is uncertain or only mentioned as a vague possibility without a clear SNF discharge plan, "
        "  set is_snf_candidate to false, snf_name = null, expected_transfer_date = null, confidence <= 0.4.\n"
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

    return {
        "is_snf_candidate": is_snf_candidate,
        "snf_name": snf_name,
        "expected_transfer_date": expected_date,
        "confidence": conf,
        "rationale": rationale,
    }



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
    Optional guard for PAD → API calls.
    If PAD_API_KEY is set in env, require header X-PAD-API-Key to match.
    If PAD_API_KEY is not set, this is a no-op.
    """
    api_key = os.getenv("PAD_API_KEY")
    if not api_key:
        return  # no key configured; allow all
    header = request.headers.get("x-pad-api-key") or ""
    if header.strip() != api_key.strip():
        raise HTTPException(status_code=401, detail="invalid PAD api key")


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
            if result and not result.get("is_snf_candidate", False):
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
                    status                    = CASE
                                                   WHEN snf_admissions.status = 'removed'
                                                       THEN snf_admissions.status
                                                   ELSE 'pending'
                                                END,
                    last_seen_active_date     = snf_admissions.last_seen_active_date,
                    emailed_at                = NULL,
                    email_run_id              = NULL
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
            SELECT id, facility_name, attending, notes, notes2, aliases, facility_emails
            FROM snf_admission_facilities
            ORDER BY facility_name COLLATE NOCASE
            """
        )
        items = [dict(r) for r in cur.fetchall()]
        return {"ok": True, "items": items}
    finally:
        conn.close()


@app.post("/admin/snf-facilities/upsert")
async def admin_snf_facilities_upsert(
    request: Request,
    payload: Dict[str, Any] = Body(...),
):
    """
    Insert or update a SNF Admission Facility.

    Payload (JSON):
      {
        "id": optional integer (omit/null for insert),
        "facility_name": "The Peaks",
        "attending": "Dr. Smith",
        "notes": "...",
        "notes2": "...",
        "aliases": "Peaks SNF, The Peaks Rehab",
        "facility_emails": "referrals@snf.com, admissions@snf.com"
      }
    """
    require_admin(request)

    name = (payload.get("facility_name") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="facility_name is required")

    fac_id = payload.get("id")
    attending = (payload.get("attending") or "").strip()
    notes = (payload.get("notes") or "").strip()
    notes2 = (payload.get("notes2") or "").strip()
    aliases = (payload.get("aliases") or "").strip()
    facility_emails = (payload.get("facility_emails") or "").strip()

    conn = get_db()
    try:
        cur = conn.cursor()
        if fac_id:
            cur.execute(
                """
                UPDATE snf_admission_facilities
                   SET facility_name = ?,
                       attending     = ?,
                       notes         = ?,
                       notes2        = ?,
                       aliases       = ?,
                       facility_emails = ?
                 WHERE id = ?
                """,
                (name, attending, notes, notes2, aliases, facility_emails, fac_id),
            )
            if cur.rowcount == 0:
                raise HTTPException(status_code=404, detail="SNF facility not found")
            new_id = fac_id
        else:
            cur.execute(
                """
                INSERT INTO snf_admission_facilities
                  (facility_name, attending, notes, notes2, aliases, facility_emails)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (name, attending, notes, notes2, aliases, facility_emails),
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
                    "note_datetime": r["note_datetime"],
                    "ai_snf_name_raw": r["ai_snf_name_raw"],
                    "ai_expected_transfer_date": r["ai_expected_transfer_date"],
                    "ai_confidence": r["ai_confidence"],
                    "status": r["status"],
                    "final_snf_facility_id": r["final_snf_facility_id"],
                    "final_snf_name_display": r["final_snf_name_display"],
                    "final_expected_transfer_date": r["final_expected_transfer_date"],
                    "review_comments": r["review_comments"],
                    "emailed_at": r["emailed_at"],
                    "email_run_id": r["email_run_id"],
                    "effective_date": r["effective_date"],
                    "effective_facility_id": r["effective_facility_id"],
                    "effective_facility_name": r["effective_facility_name"],
                    "effective_facility_city": r["effective_facility_city"],
                    "effective_facility_state": r["effective_facility_state"],
                    "last_seen_active_date": r["last_seen_active_date"],
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
                    }
                )
        except sqlite3.Error:
            notes_list = [note_data]

        return {"ok": True, "admission": admission, "note": note_data, "notes": notes_list}

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
    if status not in ("pending", "confirmed", "corrected", "rejected", "removed"):
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

        existing_final_facility_id = row["final_snf_facility_id"]
        existing_final_name_display = row["final_snf_name_display"]
        existing_ai_is_candidate = row["ai_is_snf_candidate"]
        ai_snf_name_raw = row["ai_snf_name_raw"]

        new_final_facility_id = existing_final_facility_id
        new_final_name_display = existing_final_name_display
        new_ai_is_candidate = existing_ai_is_candidate

        # If the reviewer explicitly sets disposition = SNF,
        # treat this as an authoritative override:
        #  - ensure it is marked as a SNF candidate
        #  - prefer the manual Facility free text as the SNF name
        if disposition == "SNF":
            new_ai_is_candidate = 1

            # Prefer the manually-entered Facility free text
            if facility_free_text:
                maps = load_dictionary_maps(conn)
                facility_aliases = maps["facility_aliases"]
                mapped_id = map_snf_name_to_facility_id(facility_free_text, conn, facility_aliases)
                new_final_facility_id = mapped_id
                new_final_name_display = facility_free_text
            else:
                # No manual name; at least confirm the AI name as the final display
                if not new_final_name_display and ai_snf_name_raw:
                    new_final_name_display = ai_snf_name_raw

        # NOTE: for non-SNF dispositions we leave ai_is_snf_candidate alone,
        # but still record disposition and facility_free_text for audit.

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
                   final_snf_name_display = ?
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
        <span class="provider-label">The patients below should be assigned to the following ACHG provider:</span>
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
      min-height: 100vh;
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
      box-shadow: 0 16px 40px rgba(15, 23, 42, 0.1);
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
          <div class="report-title">Accountable Care Hospitalist Group</div>
          <div class="facility-line">
            Receiving Facility:
            <span class="facility-chip">
              <span class="legend-dot"></span>
              {safe_fac}
            </span>
          </div>
          <p class="header-description">
            Our Hospitalists have identified the following patients as expected discharges to your facility. Please contact Anthony Aguirre (Anthony.Aguirre@accountablecarehg.com ) or Stephanie Sellers (ssellers@startevolv.com) if you have questions or would like to add additional recipients to future emails.
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
      <div class="topbar">ACHG • Secure List</div>
      <div class="mintbar" aria-hidden="true"></div>
      <div class="content">
        <h1>{fac_name}</h1>
        <p>Enter your facility password / PIN to view the secure list. This link expires automatically.</p>
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


@app.post("/snf/secure/{token}")
async def snf_secure_link_post(token: str, request: Request, pin: str = Form(...)):
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

        expected_hash = (fac["pin_hash"] or "").strip()
        if not expected_hash and SNF_DEFAULT_PIN:
            expected_hash = hash_pin(SNF_DEFAULT_PIN)

        if not expected_hash:
            raise HTTPException(status_code=500, detail="No facility PIN is configured.")

        if not verify_pin(pin, expected_hash):
            # Keep error generic (don't leak whether a facility exists)
            return HTMLResponse(
                "<h2>Incorrect PIN</h2><p>Please go back and try again.</p>",
                status_code=401,
            )

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

        cur.execute(
            """
            INSERT INTO snf_secure_links (token_hash, snf_facility_id, admission_ids, for_date, expires_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (token_hash, snf_facility_id, json.dumps(admission_ids), for_date, expires_at),
        )
        conn.commit()

        secure_url = f"{base_url}/snf/secure/{raw_token}"

        # ------------------------
        # Build and send the email
        # ------------------------
        if test_only:
            to_addr = test_email_to
            cc_addr = ""
        else:
            to_addr = facility_emails
            cc_addr = ""

        subject = f"Upcoming SNF referrals for {facility_name} on {for_date}"
        if test_only:
            subject = "[TEST] " + subject

        # Secure link was created above and stored in snf_secure_links

        # IMPORTANT: keep PHI out of email body
        body_lines = [
          "Secure patient referral list",
          "",
          "To protect patient information, the referral list is available through a secure link (PIN required).",
          f"View secure list: {secure_url}",
          "",
          f"This link expires in {SNF_LINK_TTL_HOURS} hours.",
          "",
          "Powered by Evolv Health",
        ]
        body = "\n".join(body_lines)

        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = INTAKE_EMAIL_FROM or SMTP_USER
        msg["To"] = to_addr
        if cc_addr:
            msg["Cc"] = cc_addr

        # Plain text + HTML email
        msg.set_content(body)
        msg.add_alternative(build_snf_secure_link_email_html(secure_url, SNF_LINK_TTL_HOURS), subtype="html")

        email_run_id = secrets.token_hex(8)

        try:
            context = ssl.create_default_context()
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
                server.starttls(context=context)
                server.login(SMTP_USER, SMTP_PASSWORD)
                server.send_message(msg)

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
        Enter your facility PIN to view the secure PDF. This link expires automatically.
      </p>
      <form method="POST" action="/snf/secure/{token}/pdf">
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

@app.post("/snf/secure/{token}/pdf")
async def snf_secure_token_pdf(token: str, pin: str = Form(...)):
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT snf_facility_id, for_date, admission_ids_json, expires_at FROM snf_pdf_links WHERE token = ?",
            (token,),
        )
        link = cur.fetchone()
        if not link:
            raise HTTPException(status_code=404, detail="Link not found")

        # Expiration check (UTC-based, consistent)
        expires_at = link["expires_at"] or ""
        if expires_at and dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S") > expires_at:
            raise HTTPException(status_code=410, detail="Link expired")

        # PIN check (simple default PIN for now)
        if not SNF_DEFAULT_PIN or pin.strip() != SNF_DEFAULT_PIN:
            raise HTTPException(status_code=401, detail="Invalid PIN")

        snf_facility_id = int(link["snf_facility_id"])
        for_date = link["for_date"]
        admission_ids = json.loads(link["admission_ids_json"] or "[]")

        # Reuse your existing email-pdf logic to load facility + rows and build PDF
        cur.execute("""
          SELECT id, facility_name, attending
          FROM snf_admission_facilities
          WHERE id = ?
        """, (snf_facility_id,))
        fac = cur.fetchone()
        if not fac:
            raise HTTPException(status_code=404, detail="Facility not found")

        facility_name = fac["facility_name"] or "Receiving Facility"
        attending = fac["attending"] or ""

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
            raise HTTPException(status_code=400, detail="No admissions found for this link")

        html_doc = build_snf_pdf_html(facility_name, for_date, rows, attending)
        pdf_bytes = WEASY_HTML(string=html_doc).write_pdf()

        safe_fac = re.sub(r"[^A-Za-z0-9]+", "_", facility_name) or "SNF"
        filename = f"SNF_Referrals_{safe_fac}_{for_date}.pdf"

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'inline; filename="{filename}"'},
        )
    finally:
        conn.close()

# ---------------------------------------------------------------------------
# Entrypoint for Render
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
