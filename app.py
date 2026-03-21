import logging
from flask import Flask, request, jsonify, redirect
from flask_cors import CORS
import requests

# ----------------------------
# Configure logging
# ----------------------------
logging.basicConfig(level=logging.INFO)

# ----------------------------
# Initialize Flask app
# ----------------------------
app = Flask(__name__, static_folder='.')
app.secret_key = "qwikgen-secret-key-2025"
CORS(app)

# ----------------------------
# Botpress API configuration (hardcoded for production)
# ----------------------------
BOTPRESS_API_URL = "https://your-production-botpress.com/api/v1/bots/your-bot-id/mod/chat"
BOTPRESS_API_KEY = "bp_pat_CkuvqQU3TGW0jySfj4zW1UhkGNbzoKvc5sXB"

# ----------------------------
# Force HTTPS and www redirect
# ----------------------------
@app.before_request
def force_https_and_www():
    url = request.url
    if not url.startswith("https://"):
        url = url.replace("http://", "https://", 1)
        return redirect(url, code=301)
    if request.host == "quickgenai.in":
        return redirect("https://www.quickgenai.in" + request.full_path, code=301)

# ----------------------------
# Botpress API helper
# ----------------------------
def call_botpress_ai(message, session_id=None):
    try:
        payload = {
            "text": message,
            "sessionId": session_id or "default-session"
        }
        headers = {
            "Authorization": f"Bearer {BOTPRESS_API_KEY}",
            "Content-Type": "application/json"
        }

        response = requests.post(BOTPRESS_API_URL, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        result = response.json()
        if "responses" in result and len(result["responses"]) > 0:
            return result["responses"][0].get("text", "")
        return "Botpress did not return a response."
    except requests.exceptions.Timeout:
        logging.error("Botpress API timed out")
        return "AI service timed out. Please try again."
    except Exception as e:
        logging.error(f"Botpress API error: {str(e)}")
        return f"AI service temporarily unavailable: {str(e)}"

# ----------------------------
# Text generation
# ----------------------------
@app.route('/api/generate-text', methods=['POST'])
def generate_text():
    try:
        data = request.get_json(force=True)
        prompt = data.get('prompt', '')
        tool_type = data.get('type', 'general')

        # Optional system prompts
        if tool_type == 'blog':
            prompt = f"You are a professional blog writer. {prompt}"
        elif tool_type == 'email':
            prompt = f"You are an expert at writing professional emails. {prompt}"
        elif tool_type == 'startup':
            prompt = f"You are a startup advisor. {prompt}"

        generated_text = call_botpress_ai(prompt)
        return jsonify({'success': True, 'content': generated_text, 'type': tool_type})
    except Exception as e:
        logging.exception("Text generation failed")
        return jsonify({'success': False, 'error': str(e)}), 500

# ----------------------------
# Chat endpoint
# ----------------------------
@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json(force=True)
        message = data.get('message', '')
        session_id = data.get('session_id', None)
        ai_response = call_botpress_ai(message, session_id=session_id)
        return jsonify({"success": True, "response": ai_response})
    except Exception as e:
        logging.exception("Chat failed")
        return jsonify({"success": False, 'error': str(e)}), 500

# ----------------------------
# Code generation
# ----------------------------
@app.route('/api/generate-code', methods=['POST'])
def generate_code():
    try:
        data = request.get_json(force=True)
        prompt = data.get('prompt', '')
        language = data.get('language', 'python')
        full_prompt = f"You are an expert {language} developer. Only return code: {prompt}"
        generated_code = call_botpress_ai(full_prompt)
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
        prompt = f"Summarize the following text:\n{text}"
        summary = call_botpress_ai(prompt)
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
        prompt = f"Translate from {source_language} to {target_language}:\n{text}"
        translation = call_botpress_ai(prompt)
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
# Health check
# ----------------------------
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'service': 'QwikGen API (Botpress)', 'version': '1.0.0'})

# ----------------------------
# Run app (production-ready, no debug)
# ----------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
