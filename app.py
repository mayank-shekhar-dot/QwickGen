"""
QuickGenAI Backend
-------------------
Single central Flask API serving every tool page.

Endpoints:
  GET  /api/health    -> liveness check
  POST /api/generate  -> single-shot generation tools (hook, script, blog, ...)
  POST /api/chat      -> conversational tools (chat_general, chat_emotional, ...)

Response contract (always this shape, never anything else):
  success: { "success": true,  "content": "...", "tool": "<tool_id>" }
  error:   { "success": false, "error": "Human-readable message" }

Required environment variable:
  GEMINI_API_KEY   - the Gemini API key. NEVER hardcode this. NEVER send it
                      to the frontend. It is only read server-side.

Optional environment variables:
  ALLOWED_ORIGINS  - comma-separated list of extra origins to allow via CORS,
                      e.g. "https://staging.quickgenai.in"
  GEMINI_MODEL     - defaults to "gemini-1.5-flash"
  REQUEST_TIMEOUT  - seconds, defaults to 30
"""

import os
import logging
import traceback
from typing import Any, Dict, List, Optional

import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

# ---------------------------------------------------------------------------
# Logging (never log the API key or full request bodies containing user PII
# beyond what's needed to debug; we only log tool id + error class)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("quickgenai-backend")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash").strip()
REQUEST_TIMEOUT = float(os.environ.get("REQUEST_TIMEOUT", "30"))

GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)

# Production + local dev origins. Add your real domain here once you have it,
# e.g. "https://quickgenai.in" and "https://www.quickgenai.in".
DEFAULT_ALLOWED_ORIGINS = [
    "https://quickgenai.in",
    "https://www.quickgenai.in",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
]
extra_origins = [
    o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()
]
ALLOWED_ORIGINS = list(dict.fromkeys(DEFAULT_ALLOWED_ORIGINS + extra_origins))

app = Flask(__name__)
CORS(
    app,
    resources={r"/api/*": {"origins": ALLOWED_ORIGINS}},
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

# ---------------------------------------------------------------------------
# Tool prompt definitions
# All prompt-construction logic lives here on the backend, not the frontend.
# Each function receives (user_input, extra) and returns the full prompt
# string sent to the model.
# ---------------------------------------------------------------------------

def _hook(i: str, extra: str) -> str:
    return (
        "Generate 5 powerful, scroll-stopping content hooks for the "
        f"following topic. Return each hook on its own line, no numbering, "
        f"no extra commentary.\n\nTopic: {i}"
    )

def _script(i: str, extra: str) -> str:
    return (
        "Write a short-form video script (30-60 seconds) for the following "
        f"topic. Include clear spoken lines only, no scene direction unless "
        f"essential.\n\nTopic: {i}"
    )

def _blog(i: str, extra: str) -> str:
    return (
        "Write a complete, well-structured blog post draft on the following "
        f"topic. Use clear paragraphs.\n\nTopic: {i}"
    )

def _email(i: str, extra: str) -> str:
    return (
        "Write 2 professional email templates for the following purpose. "
        f"Separate the two options clearly.\n\nPurpose: {i}"
    )

def _idea(i: str, extra: str) -> str:
    return (
        "Generate 10 creative, actionable content ideas for the following "
        f"topic. Return each idea on its own line, no numbering.\n\nTopic: {i}"
    )

def _tweet(i: str, extra: str) -> str:
    return (
        "Generate 5 high-engagement Twitter/X hook variations for the "
        f"following topic. Return each on its own line, no numbering.\n\n"
        f"Topic: {i}"
    )

def _keyword(i: str, extra: str) -> str:
    return (
        "Generate a list of 10 relevant keyword ideas for the following "
        f"seed topic. Return each keyword on its own line, no numbering.\n\n"
        f"Seed topic: {i}"
    )

def _title(i: str, extra: str) -> str:
    return (
        "Generate 10 high-CTR title options for the following topic. "
        f"Return each title on its own line, no numbering.\n\nTopic: {i}"
    )

def _meta(i: str, extra: str) -> str:
    return (
        "Generate 3 high-CTR meta descriptions (under 160 characters each) "
        f"for the following page/topic. Return each on its own line.\n\n"
        f"Topic: {i}"
    )

def _faq(i: str, extra: str) -> str:
    return (
        "Generate 8 SEO-optimized FAQ questions and concise answers for the "
        f"following topic. Format each as 'Q: ...' then 'A: ...' on the "
        f"next line.\n\nTopic: {i}"
    )

def _yt_title(i: str, extra: str) -> str:
    return (
        "Generate 10 high-CTR YouTube title options for the following video "
        f"topic. Return each on its own line, no numbering.\n\nTopic: {i}"
    )

def _debug(i: str, extra: str) -> str:
    lang = f" written in {extra}" if extra else ""
    return (
        f"Review the following code{lang} and identify any bugs or issues. "
        f"Explain each issue clearly and suggest a fix.\n\nCode:\n{i}"
    )

def _code_gen(i: str, extra: str) -> str:
    lang = extra or "a suitable general-purpose language"
    return (
        f"Generate production-ready code in {lang} for the following "
        f"description. Include only the code and brief inline comments "
        f"where useful.\n\nDescription: {i}"
    )

def _explain(i: str, extra: str) -> str:
    lang = f" ({extra})" if extra else ""
    return (
        f"Explain the following code{lang} in plain language, describing "
        f"what it does step by step.\n\nCode:\n{i}"
    )

def _convert(i: str, extra: str) -> str:
    target = extra or "Python"
    return (
        f"Convert the following code to {target}. Preserve behavior. "
        f"Return only the converted code.\n\nCode:\n{i}"
    )

def _sql(i: str, extra: str) -> str:
    dialect = f" for {extra}" if extra else ""
    return (
        f"Generate a SQL query{dialect} for the following request. Return "
        f"only the SQL.\n\nRequest: {i}"
    )

GENERATE_TOOLS = {
    "hook": _hook,
    "script": _script,
    "blog": _blog,
    "email": _email,
    "idea": _idea,
    "tweet": _tweet,
    "keyword": _keyword,
    "title": _title,
    "meta": _meta,
    "faq": _faq,
    "yt_title": _yt_title,
    "debug": _debug,
    "code_gen": _code_gen,
    "explain": _explain,
    "convert": _convert,
    "sql": _sql,
}

CHAT_PERSONAS = {
    "chat_general": (
        "You are a helpful, general-purpose assistant. Answer clearly and "
        "concisely."
    ),
    "chat_emotional": (
        "You are a warm, supportive conversational assistant. You are not a "
        "therapist and must not provide diagnosis, treatment, or crisis "
        "intervention. If the user expresses intent to harm themselves or "
        "others, gently encourage them to contact a crisis line or "
        "emergency services in their area. Otherwise, listen and respond "
        "supportively."
    ),
    "chat_career": (
        "You are a general career-conversation assistant. You do not "
        "guarantee any employment, salary, promotion, or hiring outcome. "
        "Discuss job searching, interviews, and workplace topics helpfully "
        "and realistically."
    ),
    "chat_mindfulness": (
        "You are a general mindfulness and reflection conversational guide. "
        "You are not a medical or therapeutic service. Offer general "
        "prompts such as breathing exercises or reflective questions."
    ),
}

VALID_CHAT_TOOLS = set(CHAT_PERSONAS.keys())

# ---------------------------------------------------------------------------
# Model call
# ---------------------------------------------------------------------------

class ModelError(Exception):
    pass


def call_gemini(prompt: str) -> str:
    if not GEMINI_API_KEY:
        raise ModelError("Server is not configured with an API key.")

    payload = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }

    try:
        resp = requests.post(
            GEMINI_URL,
            params={"key": GEMINI_API_KEY},
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
    except requests.exceptions.Timeout:
        raise ModelError("The request to the AI service timed out.")
    except requests.exceptions.RequestException:
        raise ModelError("Unable to reach the AI service.")

    if resp.status_code != 200:
        logger.error("Gemini API returned status %s", resp.status_code)
        raise ModelError("The AI service returned an error.")

    try:
        data = resp.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, ValueError):
        logger.error("Unexpected Gemini response shape")
        raise ModelError("The AI service returned an unexpected response.")

    return text.strip()


def call_gemini_chat(history: List[Dict[str, str]], persona: str, message: str) -> str:
    if not GEMINI_API_KEY:
        raise ModelError("Server is not configured with an API key.")

    contents = []
    if persona:
        # Seed the conversation with a system-style instruction as the first
        # user turn followed by a short model acknowledgement, since the
        # Gemini generateContent API has no dedicated system role here.
        contents.append({"role": "user", "parts": [{"text": persona}]})
        contents.append({"role": "model", "parts": [{"text": "Understood."}]})

    for turn in history:
        role = turn.get("role")
        text = turn.get("content") or turn.get("text") or ""
        if role not in ("user", "model") or not text:
            continue
        contents.append({"role": role, "parts": [{"text": text}]})

    contents.append({"role": "user", "parts": [{"text": message}]})

    payload = {"contents": contents}

    try:
        resp = requests.post(
            GEMINI_URL,
            params={"key": GEMINI_API_KEY},
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
    except requests.exceptions.Timeout:
        raise ModelError("The request to the AI service timed out.")
    except requests.exceptions.RequestException:
        raise ModelError("Unable to reach the AI service.")

    if resp.status_code != 200:
        logger.error("Gemini API returned status %s", resp.status_code)
        raise ModelError("The AI service returned an error.")

    try:
        data = resp.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, ValueError):
        logger.error("Unexpected Gemini response shape")
        raise ModelError("The AI service returned an unexpected response.")

    return text.strip()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def error_response(message: str, status: int = 400):
    return jsonify({"success": False, "error": message}), status


def get_json_body() -> Optional[Dict[str, Any]]:
    """Returns parsed JSON body, or None if the body is missing/malformed."""
    try:
        body = request.get_json(force=False, silent=True)
    except Exception:
        return None
    if body is None or not isinstance(body, dict):
        return None
    return body


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200


@app.route("/api/generate", methods=["POST"])
def generate():
    body = get_json_body()
    if body is None:
        return error_response("Request body must be valid JSON.", 400)

    tool = (body.get("tool") or "").strip()
    user_input = (body.get("input") or "").strip()
    extra = (body.get("extra") or "").strip()

    if not tool:
        return error_response("Missing required field: tool.", 400)

    if tool not in GENERATE_TOOLS:
        return error_response(f"Unknown tool id: '{tool}'.", 400)

    if not user_input:
        return error_response("Missing required field: input.", 400)

    prompt_builder = GENERATE_TOOLS[tool]
    prompt = prompt_builder(user_input, extra)

    try:
        content = call_gemini(prompt)
    except ModelError as e:
        logger.error("Generate failed for tool=%s: %s", tool, e)
        return error_response(str(e), 502)
    except Exception:
        logger.error("Unhandled error in /api/generate for tool=%s:\n%s",
                      tool, traceback.format_exc())
        return error_response("An unexpected server error occurred.", 500)

    if not content:
        return error_response("The AI service returned an empty response.", 502)

    return jsonify({"success": True, "content": content, "tool": tool}), 200


@app.route("/api/chat", methods=["POST"])
def chat():
    body = get_json_body()
    if body is None:
        return error_response("Request body must be valid JSON.", 400)

    tool = (body.get("tool") or "").strip()
    message = (body.get("message") or "").strip()
    history = body.get("history")
    persona_override = (body.get("persona") or "").strip()

    if not tool:
        return error_response("Missing required field: tool.", 400)

    if tool not in VALID_CHAT_TOOLS:
        return error_response(f"Unknown chat tool id: '{tool}'.", 400)

    if not message:
        return error_response("Missing required field: message.", 400)

    if history is None:
        history = []
    if not isinstance(history, list):
        return error_response("Field 'history' must be a list.", 400)

    persona = persona_override or CHAT_PERSONAS[tool]

    try:
        content = call_gemini_chat(history, persona, message)
    except ModelError as e:
        logger.error("Chat failed for tool=%s: %s", tool, e)
        return error_response(str(e), 502)
    except Exception:
        logger.error("Unhandled error in /api/chat for tool=%s:\n%s",
                      tool, traceback.format_exc())
        return error_response("An unexpected server error occurred.", 500)

    if not content:
        return error_response("The AI service returned an empty response.", 502)

    return jsonify({"success": True, "content": content, "tool": tool}), 200


@app.errorhandler(404)
def not_found(e):
    return error_response("Not found.", 404)


@app.errorhandler(405)
def method_not_allowed(e):
    return error_response("Method not allowed.", 405)


@app.errorhandler(500)
def server_error(e):
    return error_response("An unexpected server error occurred.", 500)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
