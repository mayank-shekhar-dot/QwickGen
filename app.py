import logging
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
app.secret_key = "qwikgen-secret-key-2025"
CORS(app)

# ----------------------------
# Hugging Face API configuration (hardcoded key)
# ----------------------------
HF_API_KEY = "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"

# Default models
HF_MODELS = {
    "text": "gpt2",
    "chat": "gpt2",
    "code": "codeparrot/codeparrot-small",
}

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
# Helper function to call Hugging Face API
# ----------------------------
def call_huggingface(prompt, model="gpt2"):
    try:
        headers = {"Authorization": f"Bearer {HF_API_KEY}"}
        data = {"inputs": prompt}
        url = f"https://api-inference.huggingface.co/models/{model}"

        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()

        # Most models return a list with generated_text
        if isinstance(result, list) and "generated_text" in result[0]:
            return result[0]["generated_text"]
        return str(result)

    except Exception as e:
        logging.error(f"Hugging Face API error: {str(e)}")
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

        generated_text = call_huggingface(prompt, model=HF_MODELS["text"])

        return jsonify({
            'success': True,
            'content': generated_text,
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

        conversation = ""
        for turn in history[-10:]:
            conversation += f"User: {turn.get('user','')}\nAssistant: {turn.get('assistant','')}\n"
        conversation += f"User: {message}\nAssistant:"

        ai_response = call_huggingface(conversation, model=HF_MODELS["chat"])

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
        history = data.get('history', [])

        conversation = ""
        for turn in history[-10:]:
            conversation += f"User: {turn.get('user','')}\nAssistant: {turn.get('assistant','')}\n"
        conversation += f"User: {prompt}\nAssistant:"

        generated_code = call_huggingface(conversation, model=HF_MODELS["code"])

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
        summary = call_huggingface(f"Summarize this:\n{text}", model=HF_MODELS["text"])
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

        translation = call_huggingface(f"Translate from {source_language} to {target_language}:\n{text}", model=HF_MODELS["text"])

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

# ----------------------------
# Run App
# ----------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
