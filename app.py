import os
import logging
import json
from flask import Flask, request, jsonify, redirect
from flask_cors import CORS
import requests

# ----------------------------
# Configure logging
# ----------------------------
logging.basicConfig(level=logging.DEBUG)

# ----------------------------
# Initialize Flask app
# ----------------------------
app = Flask(__name__, static_folder='.')
app.secret_key = os.environ.get("SESSION_SECRET", "qwikgen-secret-key-2025")
CORS(app)

# ----------------------------
# GEMINI AI API configuration
# ----------------------------
GEMINI_API_KEY = os.environ.get("AIzaSyCkIDmZCMSnL6ecJR1SDyaslk0n0MBcgYM")

GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"


@app.before_request
def force_www():
    if request.host == "quickgenai.in":
        return redirect(
            "https://www.quickgenai.in" + request.full_path,
            code=301
        )

# ----------------------------
# Helper function to call Together AI API
# ----------------------------
def call_gemini(prompt, system_message="You are a helpful AI assistant."):
    try:
        headers = {
            "Content-Type": "application/json"
        }

        data = {
            "contents": [
                {
                    "parts": [
                        {"text": f"{system_message}\n\n{prompt}"}
                    ]
                }
            ]
        }

        response = requests.post(
            GEMINI_API_URL,
            headers=headers,
            json=data,
            timeout=15
        )

        # 🔥 DEBUG (VERY IMPORTANT)
        print("STATUS:", response.status_code)
        print("RESPONSE:", response.text)

        if response.status_code != 200:
            return f"API ERROR: {response.text}"

        result = response.json()

        if "candidates" in result and result["candidates"]:
            parts = result["candidates"][0].get("content", {}).get("parts", [])
            for part in parts:
                if "text" in part:
                    return part["text"]

        return "No response from AI"

    except Exception as e:
        print("ERROR:", str(e))
        return "AI service unavailable"
# ----------------------------
# Serve static files / index.html
# ----------------------------




# ----------------------------
# Text Generation
# ----------------------------
@app.route('/api/generate-text', methods=['POST'])
def generate_text():
    try:
        data = request.get_json(force=True) or {}
        prompt = data.get('prompt', '').strip()
        tool_type = data.get('type', 'general')

        if not prompt:
            return jsonify({'success': False, 'error': 'Prompt is required'}), 400

        system_prompt = {
            'blog': "You are a professional blog writer. Write clear, engaging blogs.",
            'email': "You are an expert at writing professional emails.",
            'startup': "You are a startup advisor. Give practical startup ideas."
        }.get(tool_type, "You are a helpful AI assistant. Give accurate, concise responses.")

        result = call_gemini(prompt, system_message=system_prompt)

        return jsonify({
            'success': True,
            'content': result,
            'type': tool_type
        })

    except Exception as e:
        logging.exception("Text generation failed")
        return jsonify({'success': False, 'error': 'Internal server error'}), 500


# ----------------------------
# Chat
# ----------------------------
@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json(force=True) or {}
        message = data.get('message', '').strip()
        history = data.get('history', [])

        if not message:
            return jsonify({'success': False, 'error': 'Message is required'}), 400

        system_prompt = (
            "You are a helpful, friendly AI assistant. "
            "Give clear and useful responses."
        )

        conversation = f"{system_prompt}\n\n"
        for turn in history[-10:]:
            conversation += f"User: {turn.get('user','')}\nAssistant: {turn.get('assistant','')}\n"
        conversation += f"User: {message}\nAssistant:"

        result = call_gemini(conversation)

        return jsonify({"success": True, "response": result})

    except Exception as e:
        logging.exception("Chat failed")
        return jsonify({"success": False, 'error': 'Internal server error'}), 500


# ----------------------------
# Code Generation
# ----------------------------
@app.route('/api/generate-code', methods=['POST'])
def generate_code():
    try:
        data = request.get_json(force=True) or {}
        prompt = data.get('prompt','').strip()
        language = data.get('language','python')
        history = data.get('history', [])

        if not prompt:
            return jsonify({'success': False, 'error': 'Prompt is required'}), 400

        system_prompt = (
            f"You are an expert {language} developer. "
            "Return only clean, production-ready code without explanation."
        )

        conversation = f"{system_prompt}\n\n"
        for turn in history[-10:]:
            conversation += f"User: {turn.get('user','')}\nAssistant: {turn.get('assistant','')}\n"
        conversation += f"User: {prompt}\nAssistant:"

        result = call_gemini(conversation)

        return jsonify({"success": True, "content": result, "language": language})

    except Exception as e:
        logging.error(f"Code generation error: {str(e)}")
        return jsonify({"success": False, "error": 'Internal server error'}), 500


# ----------------------------
# Summarization
# ----------------------------
@app.route('/api/summarize', methods=['POST'])
def summarize():
    try:
        data = request.get_json(force=True) or {}
        text = data.get('text', '').strip()

        if not text:
            return jsonify({'success': False, 'error': 'Text is required'}), 400

        prompt = f"Summarize this:\n\n{text}"

        result = call_gemini(prompt)

        return jsonify({'success': True, 'summary': result})

    except Exception as e:
        logging.exception("Summarization failed")
        return jsonify({'success': False, 'error': 'Internal server error'}), 500


# ----------------------------
# Translation
# ----------------------------
@app.route('/api/translate', methods=['POST'])
def translate():
    try:
        data = request.get_json(force=True) or {}
        text = data.get('text', '').strip()
        target_language = data.get('target_language', 'Hindi')
        source_language = data.get('source_language', 'English')

        if not text:
            return jsonify({'success': False, 'error': 'Text is required'}), 400

        prompt = f"Translate from {source_language} to {target_language}:\n\n{text}"

        result = call_gemini(prompt)

        return jsonify({
            'success': True,
            'translation': result,
            'source_language': source_language,
            'target_language': target_language
        })

    except Exception as e:
        logging.exception("Translation failed")
        return jsonify({'success': False, 'error': 'Internal server error'}), 500

# ----------------------------
# Run App
# ----------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
















