"""
QuickGenAI - Central AI Backend
--------------------------------
One Flask backend for all 20 QuickGenAI tool pages.

Generate tools:
  hook, script, blog, email, idea, tweet, keyword, title,
  meta, faq, yt_title, debug, code_gen, explain, convert, sql

Chat tools:
  chat_general, chat_emotional, chat_career, chat_mindfulness

Required Render environment variable:
  GEMINI_API_KEY

Optional:
  GEMINI_MODEL=gemini-2.5-flash
  REQUEST_TIMEOUT=60
  ALLOWED_ORIGINS=https://example.com,https://another.com

The frontend pages stay on quickgenai.in.
This backend only handles AI API requests.
"""

import os
import logging
import traceback
from typing import Any, Dict, List, Optional

import requests
from flask import Flask, request, jsonify
from flask_cors import CORS


# ============================================================
# CONFIG
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("quickgenai-backend")

# Read environment variables at runtime.
# Do NOT put the Gemini key in any HTML/JavaScript file.
def get_gemini_key() -> str:
    return os.environ.get("GEMINI_API_KEY", "").strip()


GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip()
REQUEST_TIMEOUT = float(os.environ.get("REQUEST_TIMEOUT", "60"))

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)

DEFAULT_ALLOWED_ORIGINS = [
    "https://quickgenai.in",
    "https://www.quickgenai.in",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
]

extra_origins = [
    origin.strip()
    for origin in os.environ.get("ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
]

ALLOWED_ORIGINS = list(
    dict.fromkeys(DEFAULT_ALLOWED_ORIGINS + extra_origins)
)

app = Flask(__name__)

CORS(
    app,
    resources={
        r"/api/*": {
            "origins": ALLOWED_ORIGINS
        }
    },
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


# ============================================================
# PROMPT BUILDERS - GENERATION TOOLS
# ============================================================

def _hook(user_input: str, extra: str) -> str:
    return f"""
Create 5 strong, scroll-stopping content hooks for this topic.

Requirements:
- Each hook should be concise and engaging.
- Avoid fake claims.
- Return exactly 5 hooks.
- Number them 1 to 5.

Topic:
{user_input}
""".strip()


def _script(user_input: str, extra: str) -> str:
    return f"""
Write a short-form video script about the topic below.

Requirements:
- Around 30-60 seconds.
- Natural spoken language.
- Strong opening hook.
- Clear flow.
- Useful ending/call to action.
- Do not invent facts.

Topic:
{user_input}
""".strip()


def _blog(user_input: str, extra: str) -> str:
    return f"""
Write a useful, original, well-structured blog article about the topic below.

Requirements:
- Clear introduction.
- Use meaningful headings and subheadings.
- Explain the topic in depth.
- Give practical examples where appropriate.
- Keep the writing natural and useful.
- Do not add unsupported statistics or fake claims.
- End with a concise conclusion.

Topic:
{user_input}
""".strip()


def _email(user_input: str, extra: str) -> str:
    return f"""
Write 2 professional email versions for the purpose below.

Requirements:
- Include a suitable subject line for each.
- Keep the wording natural and professional.
- Make the two versions meaningfully different.
- Do not invent personal details.

Purpose:
{user_input}
""".strip()


def _idea(user_input: str, extra: str) -> str:
    return f"""
Generate 10 useful and practical ideas related to this topic.

Requirements:
- Make every idea distinct.
- Avoid generic repetition.
- Each idea should be understandable on its own.
- Number the ideas 1 to 10.

Topic:
{user_input}
""".strip()


def _tweet(user_input: str, extra: str) -> str:
    return f"""
Generate 5 strong X/Twitter hooks for the following topic.

Requirements:
- Short and attention-grabbing.
- Natural language.
- No fake claims.
- Make each variation different.
- Number them 1 to 5.

Topic:
{user_input}
""".strip()


def _keyword(user_input: str, extra: str) -> str:
    return f"""
Generate 15 relevant keyword ideas for the following topic.

Requirements:
- Include a useful mix of broad and specific keywords.
- Avoid unrelated keywords.
- Return one keyword per line.
- Do not add explanations.

Topic:
{user_input}
""".strip()


def _title(user_input: str, extra: str) -> str:
    return f"""
Generate 10 useful title options for this topic.

Requirements:
- Clear and relevant.
- Interesting without misleading clickbait.
- Suitable for a webpage or article.
- Number them 1 to 10.

Topic:
{user_input}
""".strip()


def _meta(user_input: str, extra: str) -> str:
    return f"""
Create 5 SEO meta description options for the following topic.

Requirements:
- Clear and descriptive.
- Keep each option reasonably concise and suitable for a search result.
- Accurately represent the topic.
- Avoid keyword stuffing.
- Number them 1 to 5.

Topic:
{user_input}
""".strip()


def _faq(user_input: str, extra: str) -> str:
    return f"""
Create 8 useful FAQ questions and answers about this topic.

Requirements:
- Questions should reflect realistic user searches.
- Answers should be concise but helpful.
- Do not invent unsupported facts.
- Format:
  Q: ...
  A: ...

Topic:
{user_input}
""".strip()


def _yt_title(user_input: str, extra: str) -> str:
    return f"""
Generate 10 YouTube title options for this video topic.

Requirements:
- Interesting but not misleading.
- Easy to understand.
- Different from each other.
- Number them 1 to 10.

Topic:
{user_input}
""".strip()


def _debug(user_input: str, extra: str) -> str:
    language = extra if extra else "the language shown in the code"

    return f"""
Review the following {language} code for bugs and problems.

Provide:
1. Problems found.
2. Why each problem occurs.
3. How to fix it.
4. A corrected version when useful.

Do not claim a problem exists if it does not.

Code:
{user_input}
""".strip()


def _code_gen(user_input: str, extra: str) -> str:
    language = extra if extra else "the most appropriate programming language"

    return f"""
Generate working {language} code for the following requirement.

Requirements:
- Follow the user's requirement exactly.
- Keep the code practical and readable.
- Include necessary comments only where helpful.
- Do not include fake libraries or nonexistent APIs.
- Return the code first, followed by a short explanation.

Requirement:
{user_input}
""".strip()


def _explain(user_input: str, extra: str) -> str:
    language = f" in {extra}" if extra else ""

    return f"""
Explain the following code{language} in simple language.

Include:
- What the code does.
- How it works step by step.
- Important functions or sections.
- Any obvious issues or limitations.

Code:
{user_input}
""".strip()


def _convert(user_input: str, extra: str) -> str:
    target = extra if extra else "Python"

    return f"""
Convert the following code to {target}.

Requirements:
- Preserve the original behavior as closely as possible.
- Use idiomatic {target} syntax.
- Do not remove important functionality.
- Return the converted code.
- Then briefly mention any unavoidable differences.

Original code:
{user_input}
""".strip()


def _sql(user_input: str, extra: str) -> str:
    dialect = extra if extra else "standard SQL"

    return f"""
Create a {dialect} SQL query for the following requirement.

Requirements:
- Return a practical query.
- Use clear formatting.
- Do not invent table or column names without stating assumptions.
- Briefly explain the assumptions after the SQL if necessary.

Requirement:
{user_input}
""".strip()


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


# ============================================================
# CHAT TOOLS
# ============================================================

CHAT_PERSONAS = {
    "chat_general": """
You are QuickGenAI General AI Chat.
Be helpful, clear, accurate, and practical.
If you are uncertain, say so instead of inventing information.
""".strip(),

    "chat_emotional": """
You are QuickGenAI Emotional Support Chat.
Be warm, respectful, and supportive.
You are not a doctor or therapist and must not diagnose or provide
professional treatment.
For an immediate danger or self-harm situation, encourage the user
to contact local emergency services or a crisis service and reach
out to a trusted person nearby.
Otherwise provide general supportive conversation.
""".strip(),

    "chat_career": """
You are QuickGenAI Career Coach Chat.
Help users with career planning, resumes, interviews, skills,
job-search strategy, workplace communication, and professional
development.
Do not guarantee employment, salary, promotion, or hiring outcomes.
""".strip(),

    "chat_mindfulness": """
You are QuickGenAI Mindfulness Guide.
Provide general mindfulness, breathing, reflection, and
relaxation exercises.
You are not a medical or mental-health treatment service.
Keep exercises practical and easy to follow.
""".strip(),
}

VALID_CHAT_TOOLS = set(CHAT_PERSONAS.keys())


# ============================================================
# GEMINI API
# ============================================================

class ModelError(Exception):
    pass


def extract_gemini_text(data: Dict[str, Any]) -> str:
    try:
        candidates = data.get("candidates", [])

        if not candidates:
            raise ValueError("No candidates")

        parts = candidates[0].get("content", {}).get("parts", [])

        texts = []
        for part in parts:
            if isinstance(part, dict) and part.get("text"):
                texts.append(part["text"])

        result = "\n".join(texts).strip()

        if not result:
            raise ValueError("Empty response")

        return result

    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        raise ModelError("The AI service returned an unexpected response.")


def gemini_post(payload: Dict[str, Any]) -> Dict[str, Any]:
    api_key = get_gemini_key()

    if not api_key:
        raise ModelError(
            "Server is not configured with an API key. "
            "Set GEMINI_API_KEY in Render Environment Variables."
        )

    try:
        response = requests.post(
            GEMINI_URL,
            params={"key": api_key},
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
    except requests.exceptions.Timeout:
        raise ModelError("The AI service request timed out.")
    except requests.exceptions.RequestException:
        raise ModelError("Unable to reach the AI service.")

    if response.status_code != 200:
        # Keep Google's detailed API response out of the frontend.
        # Log it server-side for debugging without logging the API key.
        logger.error(
            "Gemini API error: status=%s body=%s",
            response.status_code,
            response.text[:1000],
        )

        if response.status_code in (401, 403):
            raise ModelError(
                "Gemini rejected the API key. Check GEMINI_API_KEY "
                "and its Google AI API permissions."
            )

        if response.status_code == 404:
            raise ModelError(
                f"Gemini model '{GEMINI_MODEL}' was not found. "
                "Check GEMINI_MODEL in Render."
            )

        if response.status_code == 429:
            raise ModelError(
                "The AI service rate limit was reached. Please try again later."
            )

        raise ModelError("The AI service returned an error.")

    try:
        return response.json()
    except ValueError:
        raise ModelError("The AI service returned invalid JSON.")


def call_gemini(prompt: str) -> str:
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt}
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0.7,
        },
    }

    data = gemini_post(payload)
    return extract_gemini_text(data)


def call_gemini_chat(
    history: List[Dict[str, str]],
    persona: str,
    message: str,
) -> str:

    contents = [
        {
            "role": "user",
            "parts": [
                {"text": persona}
            ],
        },
        {
            "role": "model",
            "parts": [
                {"text": "Understood. I will follow these instructions."}
            ],
        },
    ]

    # Keep only valid conversation turns.
    for turn in history:
        if not isinstance(turn, dict):
            continue

        role = turn.get("role")
        text = turn.get("content") or turn.get("text") or ""

        if role not in ("user", "model"):
            continue

        if not isinstance(text, str) or not text.strip():
            continue

        contents.append(
            {
                "role": role,
                "parts": [
                    {"text": text.strip()}
                ],
            }
        )

    contents.append(
        {
            "role": "user",
            "parts": [
                {"text": message}
            ],
        }
    )

    payload = {
        "contents": contents,
        "generationConfig": {
            "temperature": 0.7,
        },
    }

    data = gemini_post(payload)
    return extract_gemini_text(data)


# ============================================================
# HELPERS
# ============================================================

def error_response(message: str, status: int = 400):
    return jsonify({
        "success": False,
        "error": message,
    }), status


def get_json_body() -> Optional[Dict[str, Any]]:
    body = request.get_json(silent=True)

    if not isinstance(body, dict):
        return None

    return body


# ============================================================
# ROUTES
# ============================================================

@app.route("/api/health", methods=["GET"])
def health():
    # Liveness endpoint.
    # It intentionally does not fail when the API key is missing.
    return jsonify({
        "status": "healthy",
        "service": "quickgenai-backend",
        "model": GEMINI_MODEL,
    }), 200


@app.route("/api/generate", methods=["POST"])
def generate():
    body = get_json_body()

    if body is None:
        return error_response(
            "Request body must be valid JSON.",
            400,
        )

    tool = str(body.get("tool") or "").strip()
    user_input = str(body.get("input") or "").strip()
    extra = str(body.get("extra") or "").strip()

    if not tool:
        return error_response(
            "Missing required field: tool.",
            400,
        )

    if tool not in GENERATE_TOOLS:
        return error_response(
            f"Unknown tool id: '{tool}'.",
            400,
        )

    if not user_input:
        return error_response(
            "Missing required field: input.",
            400,
        )

    try:
        prompt_builder = GENERATE_TOOLS[tool]
        prompt = prompt_builder(user_input, extra)

        content = call_gemini(prompt)

    except ModelError as exc:
        logger.error(
            "Generation failed: tool=%s error=%s",
            tool,
            exc,
        )

        return error_response(
            str(exc),
            502,
        )

    except Exception:
        logger.error(
            "Unhandled generate error: tool=%s\n%s",
            tool,
            traceback.format_exc(),
        )

        return error_response(
            "An unexpected server error occurred.",
            500,
        )

    return jsonify({
        "success": True,
        "content": content,
        "tool": tool,
    }), 200


@app.route("/api/chat", methods=["POST"])
def chat():
    body = get_json_body()

    if body is None:
        return error_response(
            "Request body must be valid JSON.",
            400,
        )

    tool = str(body.get("tool") or "").strip()
    message = str(body.get("message") or "").strip()
    history = body.get("history", [])
    persona_override = str(body.get("persona") or "").strip()

    if not tool:
        return error_response(
            "Missing required field: tool.",
            400,
        )

    if tool not in VALID_CHAT_TOOLS:
        return error_response(
            f"Unknown chat tool id: '{tool}'.",
            400,
        )

    if not message:
        return error_response(
            "Missing required field: message.",
            400,
        )

    if not isinstance(history, list):
        return error_response(
            "Field 'history' must be a list.",
            400,
        )

    persona = persona_override or CHAT_PERSONAS[tool]

    try:
        content = call_gemini_chat(
            history=history,
            persona=persona,
            message=message,
        )

    except ModelError as exc:
        logger.error(
            "Chat failed: tool=%s error=%s",
            tool,
            exc,
        )

        return error_response(
            str(exc),
            502,
        )

    except Exception:
        logger.error(
            "Unhandled chat error: tool=%s\n%s",
            tool,
            traceback.format_exc(),
        )

        return error_response(
            "An unexpected server error occurred.",
            500,
        )

    return jsonify({
        "success": True,
        "content": content,
        "tool": tool,
    }), 200


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(404)
def not_found(error):
    return error_response("Not found.", 404)


@app.errorhandler(405)
def method_not_allowed(error):
    return error_response("Method not allowed.", 405)


@app.errorhandler(500)
def internal_server_error(error):
    return error_response(
        "An unexpected server error occurred.",
        500,
    )


# ============================================================
# LOCAL RUN
# ============================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
    )
