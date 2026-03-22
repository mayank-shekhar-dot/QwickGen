import logging
import os
import requests
from flask import Flask, request, jsonify, redirect
from flask_cors import CORS

# ----------------------------
# Configure logging
# ----------------------------
logging.basicConfig(level=logging.DEBUG)

# ----------------------------
# Initialize Flask app
# ----------------------------
app = Flask(__name__, static_folder='.')
app.secret_key = "qwikgen-secret-key-2025"
CORS(app)

# ----------------------------
# Load API Key (SECURE)
# ----------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY not found in environment variables")

# ----------------------------
# Gemini API URL
# ----------------------------
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent?key={GOOGLE_API_KEY}"

# ----------------------------
# Redirect to www
# ----------------------------
@app.before_request
def force_www():
    if request.host == "quickgenai.in":
        return redirect("https://www.quickgenai.in" + request.full_path, code=301)

# ----------------------------
# Helper function (Gemini)
# ----------------------------
def call_gemini(prompt, system_message="You are a helpful AI assistant."):
    try:
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": f"{system_message}\n\nUser: {prompt}"}
                    ]
                }
            ]
        }

        response = requests.post(GEMINI_URL, json=payload)
        data = response.json()

        # Debug log
        logging.debug(f"Gemini Response: {data}")

        if "candidates" in data:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        else:
            return str(data)

    except Exception as e:
        logging.error(f"Gemini API error: {str(e)}")
        return f"AI service temporarily unavailable: {str(e)}"

# ----------------------------
# Text Generation
# ----------------------------
@app.route('/api/generate-text', methods=['POST'])
def generate_text():
    try:
        data = request.get_json(force=True)
        prompt = data.get('prompt', '')
        tool_type = data.get('type', 'general')

        if tool_type == 'blog':
            system_prompt = "Write a professional blog post."
        elif tool_type == 'email':
            system_prompt = "Write a professional email."
        elif tool_type == 'startup':
            system_prompt = "Give practical startup ideas."
        else:
            system_prompt = "Give accurate and helpful responses."

        result = call_gemini(prompt, system_prompt)

        return jsonify({
            'success': True,
            'content': result,
            'type': tool_type
        })

    except Exception as e:
        logging.exception("Text generation failed")
        return jsonify({'success': False, 'error': str(e)}), 500

# ----------------------------
# Chat
# ----------------------------
@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json(force=True)
        message = data.get('message', '')
        history = data.get('history', [])

        system_prompt = "You are a helpful and friendly chatbot."

        conversation = ""
        for turn in history[-10:]:
            conversation += f"User: {turn.get('user','')}\nAssistant: {turn.get('assistant','')}\n"

        conversation += f"User: {message}"

        result = call_gemini(conversation, system_prompt)

        return jsonify({
            "success": True,
            "response": result
        })

    except Exception as e:
        logging.exception("Chat failed")
        return jsonify({"success": False, 'error': str(e)}), 500

# ----------------------------
# Code Generation
# ----------------------------
@app.route('/api/generate-code', methods=['POST'])
def generate_code():
    try:
        data = request.get_json(force=True)
        prompt = data.get('prompt', '')
        language = data.get('language', 'python')

        system_prompt = f"You are an expert {language} developer. Return only code."

        result = call_gemini(prompt, system_prompt)

        return jsonify({
            "success": True,
            "content": result,
            "language": language
        })

    except Exception as e:
        logging.error(f"Code generation error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

# ----------------------------
# Summarization
# ----------------------------
@app.route('/api/summarize', methods=['POST'])
def summarize():
    try:
        data = request.get_json(force=True)
        text = data.get('text', '')

        prompt = f"Summarize this:\n\n{text}"

        result = call_gemini(prompt, "You are an expert summarizer.")

        return jsonify({'success': True, 'summary': result})

    except Exception as e:
        logging.exception("Summarization failed")
        return jsonify({'success': False, 'error': str(e)}), 500

# ----------------------------
# Translation
# ----------------------------
@app.route('/api/translate', methods=['POST'])
def translate():
    try:
        data = request.get_json(force=True)
        text = data.get('text', '')
        target_language = data.get('target_language', 'Hindi')
        source_language = data.get('source_language', 'English')

        prompt = f"Translate from {source_language} to {target_language}:\n\n{text}"

        result = call_gemini(prompt, "You are a professional translator.")

        return jsonify({
            'success': True,
            'translation': result
        })

    except Exception as e:
        logging.exception("Translation failed")
        return jsonify({'success': False, 'error': str(e)}), 500

# ----------------------------
# Health Check
# ----------------------------
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'QwikGen API',
        'version': '2.0.0 (Gemini)'
    })

# ----------------------------
# Run App
# ----------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
