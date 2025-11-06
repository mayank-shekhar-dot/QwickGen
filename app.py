import os
import logging
import json
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Define BASE_DIR so Flask can find files in root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "qwikgen-secret-key-2025")
CORS(app)

# Together AI API configuration
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY", "21c340b3fdc58cf97d62c7c111a4b599c0824e335b5f7a9268460581cb719ba1")
TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"

TOGETHER_MODELS = {
    "text": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "code": "meta-llama/Llama-3-8b-chat-hf",
    "chat": "mistralai/Mixtral-8x7B-Instruct-v0.1"
}

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
        response = requests.post(TOGETHER_API_URL, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        logging.error(f"Together AI API error: {str(e)}")
        return f"AI service temporarily unavailable: {str(e)}"


# ----------------------------
# Frontend
# ----------------------------
@app.route('/')
def serve_index():
    return send_from_directory('docs', 'index.html')


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
            system_prompt = "You are a professional blog writer. Write clear, engaging blogs."
        elif tool_type == 'email':
            system_prompt = "You are an expert at writing professional emails."
        elif tool_type == 'startup':
            system_prompt = "You are a startup advisor. Give practical startup ideas."
        else:
            system_prompt = "You are a helpful AI assistant. Give accurate, concise responses."

        generated_text = call_together_ai(prompt, model=TOGETHER_MODELS["text"], system_message=system_prompt)

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

        system_prompt = (
            "You are ChatGPT, a helpful, friendly, and conversational AI assistant. "
            "Answer clearly, explain step by step if useful."
        )

        conversation = f"System: {system_prompt}\n\n"
        for turn in history[-10:]:
            conversation += f"User: {turn.get('user', '')}\nAssistant: {turn.get('assistant', '')}\n"
        conversation += f"User: {message}\nAssistant:"

        ai_response = call_together_ai(conversation, model=TOGETHER_MODELS["chat"], system_message=system_prompt)

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
        data = request.json
        prompt = data.get('prompt','')
        language = data.get('language','python')
        history = data.get('history', [])

        system_prompt = (
            f"You are Ghostwriter, an expert {language} developer and web designer. "
            "Always return only code in the best possible format without extra explanation."
        )

        conversation = f"System: {system_prompt}\n\n"
        for turn in history[-10:]:
            conversation += f"User: {turn.get('user','')}\nAssistant: {turn.get('assistant','')}\n"
        conversation += f"User: {prompt}\nAssistant:"

        generated_code = call_together_ai(conversation, model="mistralai/Mixtral-8x7B-Instruct-v0.1", system_message=system_prompt)

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
        data = request.get_json(force=True)
        text = data.get('text', '')

        prompt = f"Please summarize the following text:\n\n{text}\n\nSummary:"
        summary = call_together_ai(prompt, model=TOGETHER_MODELS["text"], system_message="You are an expert at summarizing text.")
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
        target_language = data.get('target_language', 'Hindi')
        source_language = data.get('source_language', 'English')

        prompt = f"Translate from {source_language} to {target_language}:\n\n{text}\n\nTranslation:"
        translation = call_together_ai(prompt, model=TOGETHER_MODELS["text"], system_message="You are a professional translator.")

        return jsonify({
            'success': True,
            'translation': translation,
            'source_language': source_language,
            'target_language': target_language
        })

    except Exception as e:
        logging.exception("Translation failed")
        return jsonify({'success': False, 'error': str(e)}), 500


# ----------------------------
# Health Check
# ----------------------------
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'service': 'QwikGen API', 'version': '1.0.0'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)


