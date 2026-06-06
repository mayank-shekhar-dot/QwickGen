import logging
import os
import requests
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

logging.basicConfig(level=logging.INFO)

app = Flask(__name__, static_folder=".")
app.secret_key = os.getenv("SESSION_SECRET", "dev-secret-key")
CORS(app)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"gemini-2.0-flash:generateContent?key={GOOGLE_API_KEY}"
)

PROMPT_TEMPLATES = {
    # Student Hub
    "notes": {
        "system": "You are an expert academic note-taker and summarizer.",
        "template": "Create detailed, well-structured study notes on: {input}\n\nInclude:\n- Key concepts with definitions\n- Important points in bullet form\n- Examples where helpful\n- Summary at the end\n\nUse markdown formatting with headers and bullet points.",
    },
    "quiz": {
        "system": "You are an expert educator who creates effective quiz questions.",
        "template": "Create a comprehensive quiz with 10 questions on: {input}\n\nFor each question provide:\n- The question\n- 4 multiple choice options (A, B, C, D)\n- The correct answer\n- Brief explanation\n\nFormat clearly with markdown.",
    },
    "flashcard": {
        "system": "You are an expert educator who creates effective flashcards for learning.",
        "template": "Create 15 flashcards for studying: {input}\n\nFormat each as:\n**Card N:**\n**Front:** [Question/Term]\n**Back:** [Answer/Definition]\n\nMake them concise and memorable.",
    },
    "planner": {
        "system": "You are an expert academic advisor and study planner.",
        "template": "Create a detailed study plan for: {input}\n\nInclude:\n- Weekly schedule breakdown\n- Daily goals and milestones\n- Study techniques for each topic\n- Resources needed\n- Review checkpoints\n\nFormat as a structured plan with markdown.",
    },
    "assignment": {
        "system": "You are an expert academic assistant who helps students with assignments.",
        "template": "Help with the following assignment: {input}\n\nProvide:\n- Detailed approach and methodology\n- Key points to cover\n- Well-structured response\n- References to consider\n\nUse clear markdown formatting.",
    },
    "research": {
        "system": "You are an expert research analyst and academic summarizer.",
        "template": "Summarize and analyze the following research topic: {input}\n\nProvide:\n- Executive summary\n- Key findings and insights\n- Main arguments or theories\n- Critical analysis\n- Conclusions\n\nUse academic markdown formatting.",
    },
    # Developer Hub
    "debug": {
        "system": "You are a senior software engineer and expert code debugger.",
        "template": "Analyze the following code, identify any bugs or issues, explain them clearly, and provide a corrected version:\n\n{input}",
    },
    "code_gen": {
        "system": "You are an expert software engineer who writes clean, production-ready code.",
        "template": "Write code for the following requirement:\n\n{input}\n\nReturn the complete, runnable code with brief inline comments where helpful. Use markdown code blocks.",
    },
    "explain": {
        "system": "You are a programming teacher who explains code in clear, simple terms.",
        "template": "Explain the following code step by step. Cover what it does, how it works, and any important concepts:\n\n{input}",
    },
    "sql": {
        "system": "You are a database expert who writes efficient, well-structured SQL queries.",
        "template": "Write an SQL query for the following request:\n\n{input}\n\nReturn the query in a code block, then briefly explain what it does.",
    },
    "api_builder": {
        "system": "You are an expert API architect and backend developer.",
        "template": "Design and write a complete REST API for: {input}\n\nInclude:\n- Endpoint definitions with methods\n- Request/response schemas\n- Authentication approach\n- Example code\n- Error handling\n\nUse markdown with code blocks.",
    },
    "docs": {
        "system": "You are a technical writer who creates clear, comprehensive documentation.",
        "template": "Write complete technical documentation for: {input}\n\nInclude:\n- Overview and purpose\n- Installation/setup\n- Usage examples\n- API reference\n- Troubleshooting\n\nFormat in clean markdown.",
    },
    # Creator Hub
    "hook": {
        "system": "You are a viral content strategist who writes scroll-stopping hooks.",
        "template": "Write 8 attention-grabbing hooks for content about: {input}\n\nReturn them as a numbered list. Each hook should be punchy, curiosity-driven, and under 20 words.",
    },
    "script": {
        "system": "You are a professional scriptwriter for short-form video content.",
        "template": "Write a complete short-form video script (60-90 seconds) on the topic: {input}\n\nStructure it with:\n- HOOK (first 3 seconds)\n- BODY (main points with visual cues)\n- CTA (call to action)\n\nKeep it engaging, conversational, and tight.",
    },
    "yt_idea": {
        "system": "You are a YouTube growth strategist with deep knowledge of viral content.",
        "template": "Generate 10 unique YouTube video ideas for: {input}\n\nFor each idea provide:\n- Title (under 70 chars, high CTR)\n- Hook concept\n- Main angle/value\n- Target audience\n\nFormat as a numbered list.",
    },
    "thumbnail": {
        "system": "You are an expert YouTube thumbnail designer and visual strategist.",
        "template": "Generate 8 thumbnail concept ideas for a video about: {input}\n\nFor each concept describe:\n- Visual composition\n- Text overlay\n- Color scheme\n- Emotional trigger\n- Why it works\n\nFormat as a numbered list.",
    },
    "seo": {
        "system": "You are an SEO expert specializing in content optimization.",
        "template": "Generate a complete SEO strategy for: {input}\n\nInclude:\n- 10 primary keywords\n- 10 long-tail keywords\n- 5 question-based keywords\n- Meta title and description\n- Content structure recommendations\n\nFormat cleanly with markdown.",
    },
    "caption": {
        "system": "You are a social media expert who writes high-engagement captions.",
        "template": "Write 8 engaging social media captions for: {input}\n\nInclude variations for:\n- Instagram\n- Twitter/X\n- LinkedIn\n- TikTok\n\nEach with relevant hashtags. Format as a numbered list.",
    },
    # Chat modes
    "chat_general": {
        "system": "You are GYRA, a helpful, friendly, and knowledgeable AI assistant. Give clear, accurate, and concise responses. Format with markdown when helpful.",
        "template": "{input}",
    },
    "chat_student": {
        "system": "You are GYRA, an expert academic tutor and study companion. Explain concepts clearly, use examples, and help students understand and learn effectively.",
        "template": "{input}",
    },
    "chat_dev": {
        "system": "You are GYRA, an expert software engineer and coding assistant. Provide clean, production-ready code with explanations. Use markdown code blocks.",
        "template": "{input}",
    },
    "chat_creator": {
        "system": "You are GYRA, a creative strategist and content expert. Help creators with ideas, scripts, content strategy, and growth.",
        "template": "{input}",
    },
}


def call_gemini(prompt: str, system_message: str = "You are a helpful AI assistant.") -> str:
    try:
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": f"{system_message}\n\nUser: {prompt}"}],
                }
            ],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 8192,
            }
        }
        response = requests.post(GEMINI_URL, json=payload, timeout=60)
        data = response.json()

        logging.info("Gemini status: %s", response.status_code)

        if "candidates" in data:
            return data["candidates"][0]["content"]["parts"][0]["text"]

        if "error" in data:
            return f"API Error: {data['error'].get('message', 'Unknown error')}"

        logging.error("Unexpected Gemini response: %s", data)
        return "Sorry, I couldn't generate a response. Please try again."
    except Exception as e:
        logging.exception("Gemini API error")
        return f"AI service temporarily unavailable: {e}"


@app.route("/")
def index():
    return send_from_directory(".", "zyra.html")


@app.route("/api/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json(force=True)
        tool_id = data.get("tool", "")
        user_input = (data.get("input") or "").strip()
        target = data.get("target", "Python")

        if tool_id not in PROMPT_TEMPLATES:
            return jsonify({"success": False, "error": f"Unknown tool: {tool_id}"}), 400
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
    return jsonify({"status": "healthy", "service": "GYRA AI Workspace"})


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
