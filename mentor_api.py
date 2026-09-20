"""
mentor_ai.py  -  AI Mentor for QuickGENBiz (runs INSIDE your existing Flask app)

No separate server, no JWT, no CORS. It uses your app's normal login (session),
so the AI only ever sees the data of the user who is logged in right now.

Needs these environment variables in your app (Render > Environment):
    GOOGLE_API_KEY   Gemini key
    DATABASE_URL     PostgreSQL URL (or SQLALCHEMY_DATABASE_URI, whichever your app already has)
Optional:
    MENTOR_MODEL     default gemini-2.5-flash
"""

import logging
import os
import time
from collections import defaultdict, deque
from datetime import date, datetime
from decimal import Decimal

import requests
from flask import Blueprint, jsonify, request, session

from prompt_builder import build_system_prompt

MODEL = os.getenv("MENTOR_MODEL", "gemini-2.5-flash")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"

MAX_QUESTION_CHARS = 1000
MAX_HISTORY_TURNS = 6
RATE_LIMIT, RATE_WINDOW = 15, 60            # 15 questions per minute per user
_hits = defaultdict(deque)

mentor_ai_bp = Blueprint("mentor_ai", __name__)


# ---------------------------------------------------------------------------
# Who is logged in?  (Flask-Login first, then plain session keys)
# If your app stores the user id under another session key, add it to the tuple.
# ---------------------------------------------------------------------------
def current_user_id():
    try:
        from flask_login import current_user
        if current_user and current_user.is_authenticated:
            return int(current_user.get_id())
    except Exception:
        pass
    for key in ("user_id", "uid", "_user_id"):
        if session.get(key):
            try:
                return int(session[key])
            except (TypeError, ValueError):
                pass
    return None


# ---------------------------------------------------------------------------
# Database (read-only, explicit columns: password_hash / emails / phones never selected)
# ---------------------------------------------------------------------------
def _db_url():
    return os.getenv("DATABASE_URL") or os.getenv("SQLALCHEMY_DATABASE_URI") or ""


def _plain(v):
    if isinstance(v, Decimal):
        return float(v)
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    return v


def _rows(sql, params):
    import psycopg2
    import psycopg2.extras

    url = _db_url()
    if not url:
        raise RuntimeError("DATABASE_URL is not set")
    conn = psycopg2.connect(url, connect_timeout=10)
    try:
        conn.set_session(readonly=True)
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SET statement_timeout = 10000")
            cur.execute(sql, params)
            return [{k: _plain(v) for k, v in row.items()} for row in cur.fetchall()]
    finally:
        conn.close()


def load_business_data(user_id):
    user = _rows(
        """SELECT name, business_name, business_city, business_state,
                  gst_registered, gst_registration_type, gst_state
           FROM users WHERE id = %s""", (user_id,))
    products = _rows(
        """SELECT name, sku, unit_price, purchase_price, stock_quantity,
                  hsn_sac, gst_rate, unit, updated_at
           FROM products WHERE user_id = %s ORDER BY name""", (user_id,))
    invoices = _rows(
        """SELECT invoice_number, customer_name, status, invoice_type, due_date,
                  customer_state, place_of_supply, subtotal, subtotal_discount,
                  taxable_value, cgst, sgst, igst, total, amount_paid,
                  balance_due, created_at
           FROM invoices WHERE user_id = %s
           ORDER BY created_at DESC LIMIT 200""", (user_id,))
    items = _rows(
        """SELECT i.invoice_number, ii.product_name, ii.hsn_sac, ii.quantity, ii.unit,
                  ii.purchase_price, ii.unit_price, ii.discount, ii.taxable_value,
                  ii.gst_rate, ii.cgst, ii.sgst, ii.igst, ii.line_total
           FROM invoice_items ii JOIN invoices i ON i.id = ii.invoice_id
           WHERE i.user_id = %s
           ORDER BY i.created_at DESC, ii.id LIMIT 800""", (user_id,))
    return (user[0] if user else {}), products, invoices, items


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------
def call_gemini(system_prompt, history, question):
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        return None, "The mentor is not set up on the server yet."

    contents = [{"role": "model" if t["role"] == "assistant" else "user",
                 "parts": [{"text": t["content"]}]} for t in history]
    contents.append({"role": "user", "parts": [{"text": question}]})

    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": contents,
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 1500},
    }
    try:
        resp = requests.post(GEMINI_URL, json=payload, headers={"x-goog-api-key": key}, timeout=60)
        data = resp.json()
    except requests.exceptions.Timeout:
        return None, "The AI took too long to answer. Please try again."
    except Exception:
        logging.exception("Gemini request failed")
        return None, "Could not reach the AI service. Please try again."

    if resp.status_code != 200:
        logging.error("Gemini error %s: %s", resp.status_code, data.get("error"))
        return None, "The AI service returned an error. Please try again shortly."

    cands = data.get("candidates") or []
    if cands:
        parts = (cands[0].get("content") or {}).get("parts") or []
        text = "\n".join(p["text"] for p in parts if "text" in p).strip()
        if text:
            return text, None
    return None, "The AI could not produce an answer for that. Try rephrasing your question."


def _clean_history(raw):
    out = []
    for m in (raw or [])[-MAX_HISTORY_TURNS:]:
        if isinstance(m, dict) and m.get("role") in ("user", "assistant"):
            text = str(m.get("content", ""))[: MAX_QUESTION_CHARS * 3]
            if text.strip():
                out.append({"role": m["role"], "content": text})
    while out and out[0]["role"] != "user":
        out.pop(0)
    return out


def _rate_limited(uid):
    now, q = time.time(), _hits[uid]
    while q and now - q[0] > RATE_WINDOW:
        q.popleft()
    if len(q) >= RATE_LIMIT:
        return True
    q.append(now)
    return False


# ---------------------------------------------------------------------------
# Route:  POST /mentor/ask   (same site, uses the logged-in session)
# ---------------------------------------------------------------------------
@mentor_ai_bp.route("/mentor/ask", methods=["POST"])
def ask():
    uid = current_user_id()
    if not uid:
        return jsonify(success=False, error="Please log in to use the AI Mentor."), 401

    data = request.get_json(silent=True) or {}
    question = str(data.get("question", "")).strip()
    if not question:
        return jsonify(success=False, error="Please type a question."), 400
    if len(question) > MAX_QUESTION_CHARS:
        return jsonify(success=False, error=f"Please keep your question under {MAX_QUESTION_CHARS} characters."), 400
    if _rate_limited(uid):
        return jsonify(success=False, error="Too many questions. Please wait a minute and try again."), 429

    try:
        user, products, invoices, items = load_business_data(uid)
        system_prompt = build_system_prompt(user, products, invoices, items)
        answer, error = call_gemini(system_prompt, _clean_history(data.get("history")), question)
        if error:
            return jsonify(success=False, error=error), 502
        return jsonify(success=True, answer=answer)
    except Exception:
        logging.exception("Mentor request failed")
        return jsonify(success=False, error="The mentor is unavailable right now. Please try again shortly."), 500
