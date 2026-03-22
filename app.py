import logging
from flask import Flask, request, jsonify, redirect
from flask_cors import CORS
import requests
import os

# ----------------------------
# Configure logging
# ----------------------------
logging.basicConfig(level=logging.DEBUG)

# ----------------------------
# Initialize Flask app
# ----------------------------
app = Flask(__name__, static_folder='.')
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "default-secret-key")
CORS(app)

# ----------------------------
# Google AI Studio API config (Gemini)
# ----------------------------
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not set!")

GOOGLE_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GOOGLE_API_KEY}"

# ----------------------------
# Redirect to www (optional)
# ----------------------------
@app.before_request
def force_www():
    if request.host == "quickgenai.in":
        return redirect(
            "https://www.quickgenai.in" + request.full_path,
            code=301
        )

# ----------------------------
# Helper function (Google AI)
# ----------------------------
def call_google_ai(prompt):
    try:
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ]
        }

        response = requests.post(GOOGLE_API_URL, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()

        return data["candidates"][0]["content"]["parts"][0]["text"].strip()

    except Exception as e:
        logging.error(f"Google AI error: {str(e)}")
        return "AI service temporarily unavailable."

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
            system_prompt = "Write a high-quality blog."
        elif tool_type == 'email':
            system_prompt = "Write a professional email."
        elif tool_type == 'startup':
            system_prompt = "Give startup ideas."
        else:
            system_prompt = "Give helpful and accurate response."

        final_prompt = f"{system_prompt}\n\n{prompt}"
        generated_text = call_google_ai(final_prompt)

        return jsonify({'success': True, 'content': generated_text, 'type': tool_type})

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

        conversation = "You are a helpful AI assistant.\n\n"

        for turn in history[-10:]:
            conversation += f"User: {turn.get('user','')}\nAssistant: {turn.get('assistant','')}\n"

        conversation += f"User: {message}\nAssistant:"

        ai_response = call_google_ai(conversation)

        return jsonify({"success": True, "response": ai_response})

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
        prompt = data.get('prompt','')
        language = data.get('language','python')

        system_prompt = f"Write clean {language} code only. No explanation."

        final_prompt = f"{system_prompt}\n\n{prompt}"
        generated_code = call_google_ai(final_prompt)

        return jsonify({"success": True, "content": generated_code, "language": language})

    except Exception as e:
        logging.error(f"Code generation error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

# ----------------------------
# Summarization
# ----------------------------
@app.route('/api/summarize', methods=['POST'])
def summarize():
    try:
        text = request.get_json(force=True).get('text', '')

        prompt = f"Summarize this:\n\n{text}"
        summary = call_google_ai(prompt)

        return jsonify({'success': True, 'summary': summary})

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
        source = data.get('source_language', 'English')
        target = data.get('target_language', 'Hindi')

        prompt = f"Translate from {source} to {target}:\n\n{text}"
        translation = call_google_ai(prompt)

        return jsonify({
            'success': True,
            'translation': translation,
            'source_language': source,
            'target_language': target
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
        'service': 'QwikGen API (Google AI)',
        'version': '2.0.0'
    })

# ----------------------------
# Run App
# ----------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
