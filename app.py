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
# Together AI API Key (from environment variable)
# ----------------------------
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")
TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"  # Together AI chat endpoint
TOGETHER_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"  # Suitable Mistral model

# ----------------------------
# Redirect to www (optional)
# ----------------------------
@app.before_request
def force_www():
    if request.method == "GET" and request.host == "quickgenai.in":
        return redirect(
            "https://www.quickgenai.in" + request.full_path,
            code=302
        )

# ----------------------------
# Helper function for Together AI
# ----------------------------
def call_together_ai(messages, max_tokens=1000, temperature=0.7):
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": TOGETHER_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    try:
        response = requests.post(TOGETHER_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        # Together AI returns messages in data['choices'][0]['message']['content']
        return data['choices'][0]['message']['content'].strip()
    except requests.exceptions.RequestException as e:
        logging.error(f"Together AI API error: {e}")
        return "AI service temporarily unavailable."

# ----------------------------
# Routes
# ----------------------------
@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/api/generate-text', methods=['POST'])
def generate_text():
    try:
        data = request.get_json(force=True)
        prompt = data.get('prompt', '')
        tool_type = data.get('type', 'general')

        if tool_type == 'blog':
            system_prompt = "You are a professional blog writer. Write clear, engaging blogs."
        elif tool_type == 'email':
            system_prompt = "You are an expert at writing professional emails."
        elif tool_type == 'startup':
            system_prompt = "You are a startup advisor. Give practical startup ideas."
        else:
            system_prompt = "You are a helpful AI assistant. Give accurate, concise responses."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        generated_text = call_together_ai(messages)
        return jsonify({'success': True, 'content': generated_text, 'type': tool_type})
    except Exception as e:
        logging.exception("Text generation failed")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json(force=True)
        message = data.get('message', '')
        history = data.get('history', [])

        messages = [{"role": "system", "content": "You are a helpful, friendly AI assistant."}]
        for turn in history[-10:]:
            if "user" in turn:
                messages.append({"role": "user", "content": turn["user"]})
            if "assistant" in turn:
                messages.append({"role": "assistant", "content": turn["assistant"]})
        messages.append({"role": "user", "content": message})

        ai_response = call_together_ai(messages)
        return jsonify({"success": True, "response": ai_response})
    except Exception as e:
        logging.exception("Chat failed")
        return jsonify({"success": False, 'error': str(e)}), 500

@app.route('/api/generate-code', methods=['POST'])
def generate_code():
    try:
        data = request.get_json(force=True)
        prompt = data.get('prompt','')
        language = data.get('language','python')
        history = data.get('history', [])

        messages = [{"role": "system", "content": f"You are Ghostwriter, an expert {language} developer. Return only clean, production-ready code, no explanations."}]
        for turn in history[-10:]:
            if "user" in turn:
                messages.append({"role": "user", "content": turn["user"]})
            if "assistant" in turn:
                messages.append({"role": "assistant", "content": turn["assistant"]})
        messages.append({"role": "user", "content": prompt})

        generated_code = call_together_ai(messages)
        return jsonify({"success": True, "content": generated_code, "language": language})
    except Exception as e:
        logging.error(f"Code generation error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/summarize', methods=['POST'])
def summarize():
    try:
        data = request.get_json(force=True)
        text = data.get('text', '')
        messages = [
            {"role": "system", "content": "You are an expert at summarizing text."},
            {"role": "user", "content": text}
        ]
        summary = call_together_ai(messages)
        return jsonify({'success': True, 'summary': summary})
    except Exception as e:
        logging.exception("Summarization failed")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/translate', methods=['POST'])
def translate():
    try:
        data = request.get_json(force=True)
        text = data.get('text', '')
        source_language = data.get('source_language', 'English')
        target_language = data.get('target_language', 'Hindi')
        messages = [
            {"role": "system", "content": f"You are a professional translator. Translate from {source_language} to {target_language}."},
            {"role": "user", "content": text}
        ]
        translation = call_together_ai(messages)
        return jsonify({
            'success': True,
            'translation': translation,
            'source_language': source_language,
            'target_language': target_language
        })
    except Exception as e:
        logging.exception("Translation failed")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'service': 'QwikGen API (Together AI)', 'version': '1.0.0'})

# ----------------------------
# Run app
# ----------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
