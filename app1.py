import logging
import os
import json
import requests
from flask import Flask, request, jsonify, send_from_directory, session
from flask_cors import CORS

logging.basicConfig(level=logging.INFO)

app = Flask(__name__, static_folder=".")
app.secret_key = os.getenv("SESSION_SECRET", "dev-secret-key-change-in-prod")
CORS(app, supports_credentials=True)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"gemini-2.5-flash:generateContent?key={GOOGLE_API_KEY}"
)

# ─────────────────────────────────────────────
# AI Personality System Prompts
# ─────────────────────────────────────────────
PERSONALITY_PROMPTS = {
    "professional": (
        "You are a highly professional AI workspace assistant. "
        "Respond in a formal, structured, and concise manner. "
        "Use proper business language, avoid slang or casual expressions, "
        "organize your responses with clear sections, headings, and bullet points when appropriate. "
        "Be precise, authoritative, and thorough."
    ),
    "friendly": (
        "You are a warm and friendly AI assistant. "
        "Respond conversationally and approachably, as if talking to a good friend. "
        "Use a casual, upbeat tone, feel free to use light humor, "
        "and make the user feel comfortable and understood."
    ),
    "technical": (
        "You are a highly technical AI assistant with deep expertise across software engineering, "
        "data science, mathematics, and computer science. "
        "Provide detailed, precise, and technically accurate responses. "
        "Include code examples, algorithms, and in-depth technical explanations whenever relevant. "
        "Assume the user is technically proficient."
    ),
    "creative": (
        "You are a creative and imaginative AI assistant. "
        "Think outside the box, provide innovative and original responses, "
        "use vivid and expressive language, explore unconventional angles, "
        "and inspire the user with fresh perspectives and ideas."
    ),
}

WORKSPACE_SYSTEM = """
You are an advanced AI workspace assistant. Always produce highly structured, well-formatted output using Markdown.

Structure your responses with:
- A clear **Title** (H1 or H2)
- An **Executive Summary** or intro paragraph
- Organized **H2 Sections** and **H3 Subsections**
- **Tables** where data comparison is useful
- **Bullet lists** for enumerable items
- **Numbered steps** for processes or instructions
- **Code blocks** with language tags for any code

For reports: Executive Summary → Key Findings → Recommendations
For analysis: Problem Statement → Analysis → Conclusions → Next Steps
For technical topics: Overview → Implementation → Examples → Notes
For creative content: Feel free to use creative formatting that enhances the presentation.

Always aim for clarity, depth, and professional quality.
"""

# ─────────────────────────────────────────────
# In-memory storage (per session)
# ─────────────────────────────────────────────
chat_histories = {}  # session_id -> list of {role, content}
user_settings = {}   # session_id -> settings dict


def get_session_id():
    if "sid" not in session:
        import uuid
        session["sid"] = str(uuid.uuid4())
    return session["sid"]


def get_settings(sid):
    return user_settings.get(sid, {
        "personality": "professional",
        "customInstructions": "",
        "aboutMe": "",
        "theme": "dark",
        "fontSize": "medium",
    })


# ─────────────────────────────────────────────
# Gemini API helper
# ─────────────────────────────────────────────
def call_gemini(user_prompt: str, system_prompt: str, history: list = None) -> str:
    contents = []

    # Inject history turns
    if history:
        for turn in history[-12:]:
            role = "user" if turn["role"] == "user" else "model"
            contents.append({
                "role": role,
                "parts": [{"text": turn["content"]}]
            })

    # Current user message (prepend system into first user message if no history)
    if not contents:
        contents.append({
            "role": "user",
            "parts": [{"text": f"{system_prompt}\n\n---\n\n{user_prompt}"}]
        })
    else:
        contents.append({
            "role": "user",
            "parts": [{"text": user_prompt}]
        })

    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": contents,
        "generationConfig": {
            "temperature": 0.8,
            "maxOutputTokens": 8192,
        }
    }

    try:
        resp = requests.post(GEMINI_URL, json=payload, timeout=90)
        data = resp.json()

        if "candidates" in data and data["candidates"]:
            candidate = data["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                return candidate["content"]["parts"][0]["text"]

        if "error" in data:
            err = data["error"]
            logging.error("Gemini API error: %s", err)
            return f"⚠️ API Error: {err.get('message', 'Unknown error')}"

        logging.error("Unexpected Gemini response: %s", data)
        return "⚠️ Unexpected response from AI. Please try again."

    except requests.exceptions.Timeout:
        return "⚠️ Request timed out. The AI took too long to respond. Please try again."
    except Exception as e:
        logging.exception("Gemini call failed")
        return f"⚠️ AI service error: {str(e)}"


# ─────────────────────────────────────────────
# Deep Research helper
# ─────────────────────────────────────────────
def deep_research(topic: str, personality_prompt: str) -> dict:
    """
    Multi-step Gemini-powered deep research:
    Step 1 — generate targeted sub-questions
    Step 2 — research each sub-question
    Step 3 — synthesize into a final structured report
    """
    steps = []

    # Step 1: Break topic into research questions
    q_prompt = (
        f"You are a research analyst. Given the research topic below, "
        f"generate 5 focused sub-questions that would collectively give a thorough understanding of the topic.\n\n"
        f"Topic: {topic}\n\n"
        f"Return ONLY a numbered list of 5 questions, nothing else."
    )
    questions_raw = call_gemini(q_prompt, "You are a research planning assistant.")
    steps.append({"label": "🔍 Identifying Research Questions", "content": questions_raw})

    # Step 2: Research each question
    research_parts = []
    questions = [
        line.strip().lstrip("0123456789.)- ").strip()
        for line in questions_raw.strip().split("\n")
        if line.strip() and any(c.isalpha() for c in line)
    ][:5]

    for i, q in enumerate(questions):
        r_prompt = (
            f"Research the following question thoroughly and provide a detailed, factual answer:\n\n"
            f"Question: {q}\n\n"
            f"Provide a clear, well-structured answer with specific facts, data, and insights."
        )
        answer = call_gemini(r_prompt, "You are a thorough research analyst.")
        research_parts.append(f"### {q}\n\n{answer}")
        steps.append({"label": f"📖 Researching: {q[:60]}...", "content": answer})

    # Step 3: Synthesize final report
    combined = "\n\n---\n\n".join(research_parts)
    synthesis_prompt = (
        f"You have conducted research on the topic: **{topic}**\n\n"
        f"Here are the research findings:\n\n{combined}\n\n"
        f"Now synthesize these findings into a comprehensive, professional research report. "
        f"Structure it with:\n"
        f"1. Executive Summary\n"
        f"2. Key Findings (organized by theme)\n"
        f"3. Detailed Analysis\n"
        f"4. Conclusions\n"
        f"5. Recommendations\n\n"
        f"Use proper Markdown formatting with headers, tables, and bullet points."
    )

    system = f"{personality_prompt}\n\n{WORKSPACE_SYSTEM}"
    final_report = call_gemini(synthesis_prompt, system)
    steps.append({"label": "✅ Synthesizing Final Report", "content": "Report generated."})

    return {
        "report": final_report,
        "steps": steps,
        "questions": questions,
    }


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(".", "zyra.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "service": "AI Workspace"})


# Settings
@app.route("/api/settings", methods=["GET"])
def get_settings_route():
    sid = get_session_id()
    return jsonify(get_settings(sid))


@app.route("/api/settings", methods=["POST"])
def save_settings_route():
    sid = get_session_id()
    data = request.get_json(force=True)
    current = get_settings(sid)
    current.update({
        k: v for k, v in data.items()
        if k in ("personality", "customInstructions", "aboutMe", "theme", "fontSize")
    })
    user_settings[sid] = current
    return jsonify({"success": True, "settings": current})


# Chat history
@app.route("/api/history", methods=["GET"])
def get_history():
    sid = get_session_id()
    return jsonify({"history": chat_histories.get(sid, [])})


@app.route("/api/history/clear", methods=["POST"])
def clear_history():
    sid = get_session_id()
    chat_histories[sid] = []
    return jsonify({"success": True})


# Main Chat endpoint
@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        sid = get_session_id()
        data = request.get_json(force=True)
        message = (data.get("message") or "").strip()

        if not message:
            return jsonify({"success": False, "error": "Empty message"}), 400

        settings = get_settings(sid)
        personality = settings.get("personality", "professional")
        custom_instructions = settings.get("customInstructions", "")
        about_me = settings.get("aboutMe", "")

        # Build system prompt
        personality_prompt = PERSONALITY_PROMPTS.get(personality, PERSONALITY_PROMPTS["professional"])
        system_parts = [personality_prompt, WORKSPACE_SYSTEM]

        if about_me:
            system_parts.append(f"\n## About the User\n{about_me}")
        if custom_instructions:
            system_parts.append(f"\n## Custom Instructions\n{custom_instructions}")

        system_prompt = "\n\n".join(system_parts)

        # Get or create history
        history = chat_histories.get(sid, [])

        # Call Gemini
        response_text = call_gemini(message, system_prompt, history)

        # Save to history
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response_text})
        chat_histories[sid] = history[-40:]  # keep last 20 turns

        return jsonify({
            "success": True,
            "response": response_text,
            "personality": personality,
        })

    except Exception as e:
        logging.exception("Chat failed")
        return jsonify({"success": False, "error": str(e)}), 500


# Deep Research endpoint
@app.route("/api/research", methods=["POST"])
def research():
    try:
        sid = get_session_id()
        data = request.get_json(force=True)
        topic = (data.get("topic") or "").strip()

        if not topic:
            return jsonify({"success": False, "error": "Please provide a research topic"}), 400

        settings = get_settings(sid)
        personality = settings.get("personality", "professional")
        personality_prompt = PERSONALITY_PROMPTS.get(personality, PERSONALITY_PROMPTS["professional"])

        result = deep_research(topic, personality_prompt)

        # Save to chat history
        history = chat_histories.get(sid, [])
        history.append({"role": "user", "content": f"[Deep Research] {topic}"})
        history.append({"role": "assistant", "content": result["report"]})
        chat_histories[sid] = history[-40:]

        return jsonify({
            "success": True,
            "report": result["report"],
            "steps": result["steps"],
            "questions": result["questions"],
        })

    except Exception as e:
        logging.exception("Research failed")
        return jsonify({"success": False, "error": str(e)}), 500


# Image generation (via Pollinations AI)
@app.route("/api/image", methods=["POST"])
def generate_image():
    try:
        data = request.get_json(force=True)
        prompt = (data.get("prompt") or "").strip()
        aspect = data.get("aspect", "1:1")  # "1:1", "16:9", "9:16"
        style = data.get("style", "")

        if not prompt:
            return jsonify({"success": False, "error": "Please provide an image prompt"}), 400

        # Map aspect ratios to dimensions
        dimensions = {
            "1:1": (1024, 1024),
            "16:9": (1344, 768),
            "9:16": (768, 1344),
            "4:3": (1024, 768),
            "3:4": (768, 1024),
        }
        w, h = dimensions.get(aspect, (1024, 1024))

        full_prompt = f"{prompt}, {style}" if style else prompt
        import urllib.parse
        encoded_prompt = urllib.parse.quote(full_prompt)

        image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width={w}&height={h}&nologo=true&enhance=true"

        return jsonify({
            "success": True,
            "url": image_url,
            "prompt": full_prompt,
            "width": w,
            "height": h,
        })

    except Exception as e:
        logging.exception("Image generation failed")
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    logging.info("Starting AI Workspace on port %s", port)
    app.run(host="0.0.0.0", port=port, debug=False)
