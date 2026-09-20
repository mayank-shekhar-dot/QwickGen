"""
mentor_api.py  -  AI Business Mentor backend (Gemini version, same style as your AI Tool Hub app)

Two ways to use it:
  A) Standalone:   python mentor_api.py
  B) Inside your existing app.py (2 lines):
         from mentor_api import mentor_bp
         app.register_blueprint(mentor_bp)

ENV VARIABLES
  GOOGLE_API_KEY    required  (same key you already use)
  DATA_SOURCE       "db" (default)  -> business data is read from PostgreSQL (DATABASE_URL)
                    "client"        -> business data comes in the request body
  DATABASE_URL      PostgreSQL connection string (needed for DATA_SOURCE=db)
  JWT_SECRET        the SAME secret your app uses to sign login tokens
  JWT_ALGORITHM     default HS256
  JWT_USER_CLAIM    claim that holds the user id, default "sub" (falls back to "user_id")
  REQUIRE_AUTH      forced ON in db mode
  ALLOWED_ORIGIN    your site, e.g. https://quickgenai.in   (default: *)
  MENTOR_MODEL      default gemini-2.5-flash
"""

import logging
import os
import time
from collections import defaultdict, deque
from datetime import date, datetime
from decimal import Decimal
from functools import wraps

import requests
from dotenv import load_dotenv
from flask import Blueprint, Flask, g, jsonify, request
from flask_cors import CORS

from prompt_builder import build_system_prompt

load_dotenv()                       # .env in the app folder (local testing / Render secret file)
load_dotenv("/etc/secrets/.env")    # Render "Secret Files" location
logging.basicConfig(level=logging.INFO)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL = os.getenv("MENTOR_MODEL", "gemini-2.5-flash")
DATA_SOURCE = os.getenv("DATA_SOURCE", "db").lower()
REQUIRE_AUTH = os.getenv("REQUIRE_AUTH", "false").lower() == "true" or DATA_SOURCE == "db"
JWT_SECRET = os.getenv("JWT_SECRET", "")
DATABASE_URL = os.getenv("DATABASE_URL", "")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_USER_CLAIM = os.getenv("JWT_USER_CLAIM", "sub")

GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"

MAX_QUESTION_CHARS = 1000
MAX_HISTORY_TURNS = 6
RATE_LIMIT, RATE_WINDOW = 15, 60          # 15 questions per minute
_hits = defaultdict(deque)

mentor_bp = Blueprint("mentor", __name__)


# ---------------------------------------------------------------------------
# Auth (only used when REQUIRE_AUTH is on). Adapt to your login system.
# ---------------------------------------------------------------------------
def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        g.user_id = None
        if not REQUIRE_AUTH:
            return fn(*args, **kwargs)
        import jwt
        header = request.headers.get("Authorization", "")
        if not header.startswith("Bearer ") or not JWT_SECRET:
            return jsonify(success=False, error="Please log in to use the mentor."), 401
        try:
            payload = jwt.decode(header[7:], JWT_SECRET, algorithms=[JWT_ALGORITHM])
            g.user_id = int(payload.get(JWT_USER_CLAIM) or payload["user_id"])
        except Exception:
            return jsonify(success=False, error="Your session has expired. Please log in again."), 401
        return fn(*args, **kwargs)
    return wrapper


def rate_limited(key) -> bool:
    now = time.time()
    q = _hits[key]
    while q and now - q[0] > RATE_WINDOW:
        q.popleft()
    if len(q) >= RATE_LIMIT:
        return True
    q.append(now)
    return False


# ---------------------------------------------------------------------------
# Business data
# ---------------------------------------------------------------------------
def _plain(v):
    """Make PostgreSQL values JSON-friendly (Decimal -> float, dates -> text)."""
    if isinstance(v, Decimal):
        return float(v)
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    return v


def _rows(sql, params):
    import psycopg2
    import psycopg2.extras

    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not set")
    conn = psycopg2.connect(DATABASE_URL, connect_timeout=10)
    try:
        conn.set_session(readonly=True)                      # the mentor can never write
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SET statement_timeout = 10000")     # 10 seconds max
            cur.execute(sql, params)
            return [{k: _plain(v) for k, v in row.items()} for row in cur.fetchall()]
    finally:
        conn.close()


# Only these columns are ever read. password_hash, emails, phones and addresses
# are never selected, so they cannot reach the AI.
def load_from_db(user_id):
    user = _rows(
        """SELECT name, business_name, business_city, business_state,
                  gst_registered, gst_registration_type, gst_state
           FROM users WHERE id = %s""",
        (user_id,),
    )
    products = _rows(
        """SELECT name, sku, unit_price, purchase_price, stock_quantity,
                  hsn_sac, gst_rate, unit, updated_at
           FROM products WHERE user_id = %s ORDER BY name""",
        (user_id,),
    )
    invoices = _rows(
        """SELECT invoice_number, customer_name, status, invoice_type, due_date,
                  customer_state, place_of_supply, subtotal, subtotal_discount,
                  taxable_value, cgst, sgst, igst, total, amount_paid,
                  balance_due, created_at
           FROM invoices WHERE user_id = %s
           ORDER BY created_at DESC LIMIT 200""",
        (user_id,),
    )
    items = _rows(
        """SELECT i.invoice_number, ii.product_name, ii.hsn_sac, ii.quantity, ii.unit,
                  ii.purchase_price, ii.unit_price, ii.discount, ii.taxable_value,
                  ii.gst_rate, ii.cgst, ii.sgst, ii.igst, ii.line_total
           FROM invoice_items ii
           JOIN invoices i ON i.id = ii.invoice_id
           WHERE i.user_id = %s
           ORDER BY i.created_at DESC, ii.id LIMIT 800""",
        (user_id,),
    )
    return (user[0] if user else {}), products, invoices, items


def load_from_request(payload):
    """Client mode: your business app's page sends its own data (user, products, invoices, invoice_items)."""
    d = payload.get("business_data") or {}
    as_list = lambda v: v if isinstance(v, list) else []
    user = d.get("user") if isinstance(d.get("user"), dict) else {}
    return user, as_list(d.get("products")), as_list(d.get("invoices")), as_list(d.get("invoice_items"))


# ---------------------------------------------------------------------------
# Gemini call
# ---------------------------------------------------------------------------
def call_gemini(system_prompt: str, history: list, question: str) -> str:
    contents = []
    for turn in history:
        role = "model" if turn["role"] == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": turn["content"]}]})
    contents.append({"role": "user", "parts": [{"text": question}]})

    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": contents,
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 1500},
    }

    try:
        resp = requests.post(
            GEMINI_URL,
            json=payload,
            headers={"x-goog-api-key": GOOGLE_API_KEY},
            timeout=60,
        )
        data = resp.json()
    except requests.exceptions.Timeout:
        return None, "The AI took too long to answer. Please try again."
    except Exception:
        logging.exception("Gemini request failed")
        return None, "Could not reach the AI service. Please try again."

    if resp.status_code != 200:
        logging.error("Gemini error %s: %s", resp.status_code, data.get("error"))
        return None, "The AI service returned an error. Please try again shortly."

    candidates = data.get("candidates") or []
    if candidates:
        parts = (candidates[0].get("content") or {}).get("parts") or []
        text = "\n".join(p["text"] for p in parts if "text" in p).strip()
        if text:
            return text, None
    return None, "The AI could not produce an answer for that question. Try rephrasing it."


def clean_history(raw):
    out = []
    for m in (raw or [])[-MAX_HISTORY_TURNS:]:
        if not isinstance(m, dict):
            continue
        role, text = m.get("role"), str(m.get("content", ""))[: MAX_QUESTION_CHARS * 3]
        if role in ("user", "assistant") and text.strip():
            out.append({"role": role, "content": text})
    while out and out[0]["role"] != "user":       # Gemini needs to start with a user turn
        out.pop(0)
    return out


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@mentor_bp.route("/api/mentor", methods=["POST"])
@login_required
def mentor():
    if missing_settings():
        logging.error("Mentor called but settings are missing: %s", missing_settings())
        return jsonify(success=False, error="The mentor is not set up on the server yet."), 503

    data = request.get_json(silent=True) or {}
    question = str(data.get("question", "")).strip()

    if not question:
        return jsonify(success=False, error="Please type a question."), 400
    if len(question) > MAX_QUESTION_CHARS:
        return jsonify(success=False, error=f"Please keep your question under {MAX_QUESTION_CHARS} characters."), 400

    limit_key = g.user_id or request.headers.get("X-Forwarded-For", request.remote_addr)
    if rate_limited(limit_key):
        return jsonify(success=False, error="Too many questions. Please wait a minute and try again."), 429

    try:
        if DATA_SOURCE == "db":
            user, products, invoices, items = load_from_db(g.user_id)
        else:
            user, products, invoices, items = load_from_request(data)

        system_prompt = build_system_prompt(user, products, invoices, items)
        answer, error = call_gemini(system_prompt, clean_history(data.get("history")), question)

        if error:
            return jsonify(success=False, error=error), 502
        return jsonify(success=True, answer=answer)

    except Exception:
        logging.exception("Mentor request failed")
        return jsonify(success=False, error="The mentor is unavailable right now. Please try again shortly."), 500


@mentor_bp.route("/api/mentor/health", methods=["GET"])
def mentor_health():
    missing = missing_settings()
    return jsonify(status="healthy" if not missing else "needs_setup",
                   service="AI Business Mentor", data_source=DATA_SOURCE, missing_settings=missing)


# ---------------------------------------------------------------------------
# Standalone app
# ---------------------------------------------------------------------------
def missing_settings():
    """Names (never values) of required settings that are not set."""
    missing = []
    if not os.getenv("GOOGLE_API_KEY"):
        missing.append("GOOGLE_API_KEY")
    if DATA_SOURCE == "db" and not DATABASE_URL:
        missing.append("DATABASE_URL")
    if REQUIRE_AUTH and not JWT_SECRET:
        missing.append("JWT_SECRET")
    return missing


def create_app():
    app = Flask(__name__)
    origins = [o.strip() for o in os.getenv("ALLOWED_ORIGIN", "*").split(",") if o.strip()]
    CORS(app, origins=origins)   # ALLOWED_ORIGIN can hold several sites, comma separated
    app.register_blueprint(mentor_bp)
    if missing_settings():
        logging.error("MISSING SETTINGS: %s (add them in Render > Environment or Secret Files)", ", ".join(missing_settings()))
    return app


app = create_app()      # always defined, so `gunicorn mentor_api:app` always starts

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
