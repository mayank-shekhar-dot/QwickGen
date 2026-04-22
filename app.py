import logging
import os
import requests
from flask import Flask, request, jsonify, send_from_directory, redirect
from flask_cors import CORS

logging.basicConfig(level=logging.INFO)

app = Flask(__name__, static_folder='.')
app.secret_key = os.getenv("SESSION_SECRET", "dev-secret-key")
CORS(app)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1/models/"
    f"gemini-1.5-flash:generateContent?key={GOOGLE_API_KEY}"
)


# ----------------------------
# Prompt Templates
# ----------------------------
PROMPT_TEMPLATES = {
    # Creator Tools
    "hook": {
        "system": "You are a viral content strategist who writes scroll-stopping hooks.",
        "template": "Write 8 attention-grabbing hooks for content about: {input}\n\nReturn them as a numbered list. Each hook should be punchy, curiosity-driven, and under 20 words.",
    },
    "script": {
        "system": "You are a professional scriptwriter for short-form video content.",
        "template": "Write a complete short-form video script (60-90 seconds) on the topic: {input}\n\nStructure it with:\n- HOOK (first 3 seconds)\n- BODY (main points with visual cues)\n- CTA (call to action)\n\nKeep it engaging, conversational, and tight.",
    },
    "blog": {
        "system": "You are an expert SEO blog writer.",
        "template": "Write a complete, well-structured blog post about: {input}\n\nInclude:\n- A compelling title\n- An engaging introduction\n- 4-6 H2 sections with clear explanations\n- A conclusion with a takeaway\n\nAim for around 800 words. Use markdown formatting.",
    },
    "email": {
        "system": "You are a professional copywriter who writes high-converting business emails.",
        "template": "Write a professional email for the following purpose: {input}\n\nInclude a clear subject line, a friendly greeting, a focused body, and a strong call to action. Keep it concise.",
    },
    "idea": {
        "system": "You are a creative strategist who generates fresh, actionable ideas.",
        "template": "Generate 10 unique and creative ideas for: {input}\n\nFor each idea, give a short title and a 1-2 sentence explanation of why it works. Format as a numbered list.",
    },
    "tweet": {
        "system": "You are a Twitter/X growth expert known for viral tweet hooks.",
        "template": "Write 10 viral tweet hooks (under 280 characters each) about: {input}\n\nMix formats: bold claims, contrarian takes, listicles, questions, and stories. Number each one.",
    },

    # SEO Tools
    "keyword": {
        "system": "You are an SEO expert specializing in keyword research.",
        "template": "Generate a comprehensive keyword list for the topic: {input}\n\nOrganize the output into:\n- 10 Primary keywords (high volume)\n- 10 Long-tail keywords (low competition)\n- 5 Question-based keywords\n\nFormat as clean lists.",
    },
    "title": {
        "system": "You are an SEO copywriter who crafts click-worthy, ranking titles.",
        "template": "Generate 10 SEO-optimized titles for an article about: {input}\n\nEach title should be under 60 characters, include the focus keyword naturally, and feel compelling. Number the list.",
    },
    "meta": {
        "system": "You are an SEO specialist who writes meta descriptions that drive clicks.",
        "template": "Write 5 SEO meta descriptions for: {input}\n\nEach must be 140-160 characters, include a clear value proposition and a soft CTA. Number them.",
    },
    "faq": {
        "system": "You are an SEO content strategist who writes FAQ sections optimized for featured snippets.",
        "template": "Generate 8 frequently asked questions and concise, accurate answers about: {input}\n\nFormat each as:\nQ: [question]\nA: [2-3 sentence answer]",
    },
    "yt_title": {
        "system": "You are a YouTube growth expert who crafts high-CTR video titles.",
        "template": "Generate 10 high-CTR YouTube video titles for content about: {input}\n\nEach title should be under 70 characters, use power words, and trigger curiosity. Mix listicles, how-tos, and bold claims. Number the list.",
    },

    # Developer Tools
    "debug": {
        "system": "You are a senior software engineer and expert code debugger.",
        "template": "Analyze the following code, identify any bugs or issues, explain them clearly, and provide a corrected version:\n\n{input}",
    },
    "code_gen": {
        "system": "You are an expert software engineer who writes clean, production-ready code.",
        "template": "Write code for the following requirement:\n\n{input}\n\nReturn the complete, runnable code with brief inline comments where helpful.",
    },
    "explain": {
        "system": "You are a programming teacher who explains code in clear, simple terms.",
        "template": "Explain the following code step by step. Cover what it does, how it works, and any important concepts:\n\n{input}",
    },
    "convert": {
        "system": "You are an expert polyglot programmer.",
        "template": "Convert the following code to {target}. Preserve the logic exactly and follow idiomatic conventions of the target language:\n\n{input}",
    },
    "sql": {
        "system": "You are a database expert who writes efficient, well-structured SQL queries.",
        "template": "Write an SQL query for the following request:\n\n{input}\n\nReturn the query in a code block, then briefly explain what it does.",
    },

    # Chatbots
    "chat_general": {
        "system": "You are a helpful, friendly, and knowledgeable AI assistant. Give clear, accurate, and concise responses.",
        "template": "{input}",
    },
    "chat_emotional": {
        "system": "You are a deeply empathetic and emotionally intelligent companion. Listen carefully, validate feelings, and respond with warmth and understanding. Never judge.",
        "template": "{input}",
    },
    "chat_career": {
        "system": "You are an experienced career coach. Give practical, actionable career guidance with structure. Ask thoughtful follow-up questions when needed.",
        "template": "{input}",
    },
    "chat_mindfulness": {
        "system": "You are a calm, grounded mindfulness guide. Speak gently, focus on the present moment, and offer simple breathing or grounding techniques when relevant.",
        "template": "{input}",
    },
}


# ----------------------------
# Gemini helper
# ----------------------------
def call_gemini(prompt: str, system_message: str = "You are a helpful AI assistant.") -> str:
    try:
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": f"{system_message}\n\nUser: {prompt}"}],
                }
            ]
        }
        response = requests.post(GEMINI_URL, json=payload, timeout=60)
        data = response.json()
        if "candidates" in data:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        logging.error("Unexpected Gemini response: %s", data)
        return "Sorry, I couldn't generate a response. Please try again."
    except Exception as e:
        logging.exception("Gemini API error")
        return f"AI service temporarily unavailable: {e}"


# ----------------------------
# Routes
# ----------------------------
@app.route("/")
def index():
    return send_from_directory(".", "tool10.html")


@app.route("/ai-tools")
def ai_tools():
    return redirect("https://quickgenai.in/ai-tools")


@app.route("/api/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json(force=True)
        tool_id = data.get("tool", "")
        user_input = (data.get("input") or "").strip()
        target = data.get("target", "Python")

        if tool_id not in PROMPT_TEMPLATES:
            return jsonify({"success": False, "error": "Unknown tool"}), 400
        if not user_input:
            return jsonify({"success": False, "error": "Please provide some input."}), 400

        tpl = PROMPT_TEMPLATES[tool_id]
        prompt = tpl["template"].format(input=user_input, target=target)
        result = call_gemini(prompt, tpl["system"])
        return jsonify({"success": True, "content": result})
    except Exception as e:
        logging.exception("Generation failed")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(force=True)
        tool_id = data.get("tool", "chat_general")
        message = (data.get("message") or "").strip()
        history = data.get("history", [])

        if tool_id not in PROMPT_TEMPLATES:
            return jsonify({"success": False, "error": "Unknown chatbot"}), 400
        if not message:
            return jsonify({"success": False, "error": "Empty message"}), 400

        system_prompt = PROMPT_TEMPLATES[tool_id]["system"]

        conversation = ""
        for turn in history[-10:]:
            conversation += f"User: {turn.get('user', '')}\nAssistant: {turn.get('assistant', '')}\n"
        conversation += f"User: {message}"

        result = call_gemini(conversation, system_prompt)
        return jsonify({"success": True, "response": result})
    except Exception as e:
        logging.exception("Chat failed")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "service": "AI Tool Hub"})


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
