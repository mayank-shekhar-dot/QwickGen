import os
import logging
import json
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests

# ----------------------------
# Setup
# ----------------------------
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__, static_folder='docs', static_url_path='')
app.secret_key = os.environ.get("SESSION_SECRET", "qwikgen-secret-key-2025")
CORS(app)

# ----------------------------
# Together AI Configuration
# ----------------------------
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY", "21c340b3fdc58cf97d62c7c111a4b599c0824e335b5f7a9268460581cb719ba1")
TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"

TOGETHER_MODELS = {
    "text": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "code": "meta-llama/Llama-3-8b-chat-hf",
    "chat": "mistralai/Mixtral-8x7B-Instruct-v0.1"
}

# ----------------------------
# Together API Helper
# ----------------------------
def call_together_ai(prompt, model="mistralai/Mixtral-8x7B-Instruct-v0.1", system_message="You are a helpful AI assistant."):
    try:
        headers = {
            "Authorization": f"Bearer {TOGETHER_API_KEY}",
            "Content-Type": "application/json",
        }
        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1000,
            "temperature": 0.7,
            "top_p": 0.7
        }
        response = requests.post(TOGETHER_API_URL, headers=headers, json=data, timeout=20)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        logging.error(f"Together AI API error: {str(e)}")
        return f"AI service temporarily unavailable: {str(e)}"

# ----------------------------
# Routes
# ----------------------------
@app.route('/')
def serve_index():
    """Serve the frontend index.html"""
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    """Serve all other frontend assets (JS, CSS, images)"""
    return send_from_directory(app.static_folder, path)


@app.route('/api/generate-text', methods=['POST'])
def generate_text():
    data = request.get_json(force=True)
    prompt = data.get('prompt', '')
    tool_type = data.get('type', 'general')

    system_prompt = {
        'blog': "You are a professional blog writer. Write clear, engaging blogs.",
        'email': "You are an expert at writing professional emails.",
        'startup': "You are a startup advisor. Give practical startup ideas."
    }.get(tool_type, "You are a helpful AI assistant. Give accurate, concise responses.")

    generated_text = call_together_ai(prompt, model=TOGETHER_MODELS["text"], system_message=system_prompt)
    return jsonify({'success': True, 'content': generated_text})


@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json(force=True)
    message = data.get('message', '')
    history = data.get('history', [])

    system_prompt = "You are ChatGPT, a helpful, friendly, and conversational AI assistant."

    conversation = f"System: {system_prompt}\n\n"
    for turn in history[-10:]:
        conversation += f"User: {turn.get('user', '')}\nAssistant: {turn.get('assistant', '')}\n"
    conversation += f"User: {message}\nAssistant:"

    ai_response = call_together_ai(conversation, model=TOGETHER_MODELS["chat"], system_message=system_prompt)
    return jsonify({"success": True, "response": ai_response})


@app.route('/api/health')
def health():
    return jsonify({'status': 'healthy', 'version': '1.0.0'})

# ----------------------------
# Run the server
# ----------------------------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
