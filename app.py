import logging
import os
import requests
from flask import Flask, request, jsonify, redirect
from flask_cors import CORS
from time import sleep

# ----------------------------
# Logging (Production Ready)
# ----------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# ----------------------------
# App Init
# ----------------------------
app = Flask(__name__, static_folder='.')
app.secret_key = os.getenv("SESSION_SECRET", "prod-secret-key")

CORS(app, resources={r"/api/*": {"origins": "*"}})

# ----------------------------
# API KEY
# ----------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY missing")

# ----------------------------
# Gemini URLs (Primary + Fallback)
# ----------------------------
PRIMARY_MODEL = "gemini-2.5-flash"
FALLBACK_MODEL = "gemini-1.5-flash"

def get_url(model):
    return f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent?key={GOOGLE_API_KEY}"

# ----------------------------
# Prompt Templates (Shortened example)
# ----------------------------
PROMPT_TEMPLATES = {
    "hook": {
        "system": "You are a viral content strategist.",
        "template": "Write 8 hooks about: {input}"
    },
    "blog": {
        "system": "You are an SEO blog writer.",
        "template": "Write blog about: {input}"
    },
    "chat_general": {
        "system": "You are a helpful AI assistant.",
        "template": "{input}"
    }
}

# ----------------------------
# Gemini Call (RETRY + FALLBACK)
# ----------------------------
def call_gemini(prompt, system_message, retries=3):
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": f"{system_message}\n\nUser: {prompt}"}]
            }
        ]
    }

    # Try primary model
    for attempt in range(retries):
        try:
            response = requests.post(
                get_url(PRIMARY_MODEL),
                json=payload,
                timeout=20
            )

            if response.status_code == 200:
                data = response.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]

            logging.warning(f"Primary model failed: {response.text}")

        except Exception as e:
            logging.error(f"Attempt {attempt+1} failed: {e}")

        sleep(1)  # small delay before retry

    # Fallback model
    try:
        logging.info("Switching to fallback model...")
        response = requests.post(
            get_url(FALLBACK_MODEL),
            json=payload,
            timeout=20
        )

        if response.status_code == 200:
            data = response.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]

    except Exception as e:
        logging.error(f"Fallback failed: {e}")

    # Final safe response
    return "⚠️ AI service is busy. Please try again in a moment."

# ----------------------------
# Routes
# ----------------------------

@app.route("/")
def home():
    return "✅ AI Backend Running"

@app.route("/ai-tools")
def ai_tools():
    return redirect("https://quickgenai.in/ai-tools")

# ----------------------------
# GENERATE TOOL
# ----------------------------
@app.route("/api/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json(force=True)

        tool_id = data.get("tool", "")
        user_input = (data.get("input") or "").strip()

        if tool_id not in PROMPT_TEMPLATES:
            return jsonify({"success": False, "error": "Invalid tool"}), 400

        if not user_input:
            return jsonify({"success": False, "error": "Empty input"}), 400

        tpl = PROMPT_TEMPLATES[tool_id]
        prompt = tpl["template"].format(input=user_input)

        result = call_gemini(prompt, tpl["system"])

        return jsonify({
            "success": True,
            "content": result
        })

    except Exception as e:
        logging.exception("Generate error")
        return jsonify({
            "success": False,
            "error": "Server error"
        }), 500

# ----------------------------
# CHAT
# ----------------------------
@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(force=True)

        message = (data.get("message") or "").strip()
        history = data.get("history", [])

        if not message:
            return jsonify({"success": False, "error": "Empty message"}), 400

        conversation = ""
        for turn in history[-10:]:
            conversation += f"User: {turn.get('user','')}\nAssistant: {turn.get('assistant','')}\n"

        conversation += f"User: {message}"

        result = call_gemini(conversation, "You are a helpful assistant.")

        return jsonify({
            "success": True,
            "response": result
        })

    except Exception as e:
        logging.exception("Chat error")
        return jsonify({
            "success": False,
            "error": "Server error"
        }), 500

# ----------------------------
# HEALTH CHECK (IMPORTANT)
# ----------------------------
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "service": "AI Backend",
        "version": "production-1.0"
    })

# ----------------------------
# RUN
# ----------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
