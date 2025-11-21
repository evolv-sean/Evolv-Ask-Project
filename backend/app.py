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
from email.message import EmailMessage
from pathlib import Path
from typing import List, Dict, Any, Optional

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
            a_quality  TEXT
        )
        """
    )


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
        elif kind == "synonym":
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



def normalize_question_text(question: str, abbrev_map: Dict[str, str]) -> str:
    """
    Expand abbreviations in the user question using the dictionary's abbrev_map.
    Uses word boundaries so 'evolv?' still matches 'evolv'.
    """
    if not question:
        return question

    for abbr, full in abbrev_map.items():
        pattern_abbr = rf"\b{re.escape(abbr)}\b"
        if not re.search(pattern_abbr, question, flags=re.IGNORECASE):
            continue

        pattern_full = rf"\b{re.escape(full)}\b"
        if re.search(pattern_full, question, flags=re.IGNORECASE):
            continue

        question = re.sub(pattern_abbr, full, question, flags=re.IGNORECASE)

    return question


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
      1) dictionary facility aliases
      2) facility_name from the facilities table

    Uses a soft-normalized text form so small punctuation differences
    ('St. Mary's' vs 'st marys') don't break matching.
    """
    q_norm = _normalize_for_facility_match(question)

    # 1) Explicit aliases from dictionary
    for alias_text, fid in facility_aliases.items():
        if not alias_text:
            continue
        alias_norm = _normalize_for_facility_match(alias_text)
        if alias_norm and alias_norm in q_norm:
            return fid

    # 2) Substring match on facility_name
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


# ---------------------------------------------------------------------------
# Admin: Dictionary (abbr/synonym/payer/etc.) used by TEST Admin.html
# ---------------------------------------------------------------------------

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
    if not facility_id:
        return []

    snippets: List[Dict[str, Any]] = []
    cur = conn.cursor()

    # fac_facts (explicit snippets)
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

    # NEW: facility_contacts — key roles like Administrator, Therapy, etc.
    try:
        cur.execute(
            """
            SELECT type, name, phone, email, pref
            FROM facility_contacts
            WHERE facility_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (facility_id, limit),
        )
        for r in cur.fetchall():
            role = (r[0] or "").strip()   # type
            name = (r[1] or "").strip()   # name
            phone = (r[2] or "").strip()  # phone
            email = (r[3] or "").strip()  # email
            pref  = (r[4] or "").strip()  # pref

            # Need at least a role + name to be useful
            if not (role and name):
                continue

            bits = [f"{role}: {name}"]

            contact_bits = []
            if phone:
                contact_bits.append(f"phone {phone}")
            if email:
                contact_bits.append(f"email {email}")
            if contact_bits:
                bits.append("(" + ", ".join(contact_bits) + ")")

            if pref:
                bits.append(f"[prefers: {pref}]")

            text = " ".join(bits)

            snippets.append(
                {
                    "text": text,
                    "source": f"DB:facility_contacts:{role}",
                    "tags": "contacts",
                    "weight": 2.0,  # slightly boosted vs generic data
                }
            )

    except sqlite3.Error:
        pass

    # Optional views from your existing schema, if present
    try:
        cur.execute(
            """
            SELECT kind, text, tags
            FROM v_facility_knowledge
            WHERE facility_id = ?
            LIMIT ?
            """,
            (facility_id, limit),
        )
        for r in cur.fetchall():
            kind = (r["kind"] or "data")
            snippets.append(
                {
                    "text": r["text"],
                    "source": f"DB:facility_{kind}",
                    "tags": r["tags"] or "",
                    "weight": 1.0,
                }
            )
    except sqlite3.Error:
        pass

    # Facility summary (beds, EMR, etc.)
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
            parts = []

            if row["emr"]:
                parts.append(f"EMR: {row['emr']}")

            orders_desc = " / ".join(
                p for p in [row["orders"], row["orders_other"]] if p
            )
            if orders_desc:
                parts.append(f"Orders: {orders_desc}")

            if row["outpatient_pt"]:
                parts.append(f"Outpatient PT: {row['outpatient_pt']}")

            beds = []
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
                        "weight": 2.0,
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


def build_ranked_context(question: str, qa_rows, fac_snippets, top_k: int):
    candidates: List[Dict[str, Any]] = []

    # Q&A rows (default weight 1.0)
    for r in qa_rows:
        text = (
            f"Q: {r['question']}\n"
            f"A: {r['answer']}\n"
            f"Topics: {r['topics'] or ''}\n"
            f"Tags: {r['tags'] or ''}"
        )
        candidates.append(
            {
                "text": text,
                "source": f"Q&A:{r['id']}",
                "kind": "qa",
                "weight": 1.0,
            }
        )

    # Facility snippets (reuse their weight, default 1.0)
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
        return [], []

    ctx_texts = [c["text"] for c in candidates]
    q_emb = embed_texts([question])[0]
    ctx_embs = embed_texts(ctx_texts)

    scored = []
    for c, emb in zip(candidates, ctx_embs):
        base_score = cosine_sim(q_emb, emb)
        w = float(c.get("weight", 1.0) or 1.0)
        scored.append((base_score * w, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = [c for _, c in scored[: max(top_k, 1)]]
    sources = [c["source"] for c in top]
    return top, sources



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
# Question → Answer pipeline
# ---------------------------------------------------------------------------

def run_answer_pipeline(
    question: str, section_hint: Optional[str], top_k: int
) -> AskResponse:
    conn = get_db()
    try:
        maps = load_dictionary_maps(conn)
        abbrev = maps["abbrev"]
        facility_aliases = maps["facility_aliases"]

        normalized_q = normalize_question_text(question, abbrev)
        fac_id = detect_facility_id(normalized_q, conn, facility_aliases)
        fac_label = get_facility_label(fac_id, conn)

        qa_rows = search_qa_candidates(conn, normalized_q, section_hint, limit=40)
        fac_snippets = fetch_facility_knowledge(conn, fac_id, limit=40)

        ranked_chunks, sources = build_ranked_context(
            normalized_q, qa_rows, fac_snippets, top_k=top_k
        )

        context_lines = []
        for idx, c in enumerate(ranked_chunks, start=1):
            context_lines.append(f"[{idx}] {c['source']}\n{c['text']}\n")
        context_block = "\n".join(context_lines) if context_lines else "(no DB snippets found)"

        section_used = section_hint or ("Facility Details" if fac_id else "General")
        system_prompt = get_system_prompt(conn)

        user_message = (
            f"User question:\n{question}\n\n"
            f"Section hint (topics): {section_hint or 'n/a'}\n"
            f"Detected facility: {fac_label or 'none'}\n\n"
            f"Internal knowledge snippets (if any):\n{context_block}\n\n"
            "Instructions:\n"
            "- Answer as Evolv's AI assistant.\n"
            "- Prioritize the internal snippets over general knowledge.\n"
            "- If facility-specific details are present, be precise for that facility.\n"
            "- If the DB doesn't cover something, say what you *can* infer and what "
            "is unknown instead of guessing.\n"
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
        sql = "SELECT id, ts, section, q, a, promoted, a_quality FROM user_qa_log"
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



@app.get("/facility", response_class=HTMLResponse)
async def facility_ui():
    return HTMLResponse(content=read_html(FACILITY_HTML))


@app.get("/health")
async def health():
    return {"ok": True, "db_path": DB_PATH}

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


# ---------------------------------------------------------------------------
# Entrypoint for Render
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)


# ---------------------------------------------------------------------------
# Entrypoint for Render
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
