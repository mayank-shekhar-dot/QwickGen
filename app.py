import logging
from flask import Flask, request, jsonify, redirect
from flask_cors import CORS
from openai import OpenAI

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
# NVIDIA Gemma API configuration
# ----------------------------
NV_API_KEY = "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"
NV_API_URL = "https://integrate.api.nvidia.com/v1"
client = OpenAI(base_url=NV_API_URL, api_key=NV_API_KEY)

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
# Helper function to call NVIDIA AI
# ----------------------------
def call_nvidia_ai(messages, model="google/gemma-2-27b-it", temperature=0.2, max_tokens=1024):
    try:
        # Messages must alternate user/assistant
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=0.7,
            max_tokens=max_tokens
        )
        return completion.choices[0].message.content
    except Exception as e:
        logging.error(f"NVIDIA API error: {str(e)}")
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
            instruction = f"You are a professional blog writer. Write a clear, engaging blog based on:\n{prompt}"
        elif tool_type == 'email':
            instruction = f"You are an expert email writer. Compose a professional email based on:\n{prompt}"
        elif tool_type == 'startup':
            instruction = f"You are a startup advisor. Give practical startup ideas based on:\n{prompt}"
        else:
            instruction = f"You are a helpful AI assistant. Respond accurately to:\n{prompt}"

        messages = [
            {"role": "user", "content": instruction}
        ]

        generated_text = call_nvidia_ai(messages)
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

        # Always start with system instruction as user for Gemma
        messages = [{"role": "user", "content": "You are ChatGPT, a helpful, friendly, conversational AI assistant."}]

        # Append previous chat with alternating roles
        for turn in history[-10:]:
            if turn.get('user'):
                messages.append({"role": "user", "content": turn.get('user')})
            if turn.get('assistant'):
                messages.append({"role": "assistant", "content": turn.get('assistant')})

        messages.append({"role": "user", "content": message})

        ai_response = call_nvidia_ai(messages)
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
        prompt = data.get('prompt', '')
        language = data.get('language', 'python')
        history = data.get('history', [])

        messages = [
            {"role": "user", "content": f"You are Ghostwriter, an expert {language} developer. Return only code in the best possible format without extra explanation. Prompt:\n{prompt}"}
        ]

        for turn in history[-10:]:
            if turn.get('user'):
                messages.append({"role": "user", "content": turn.get('user')})
            if turn.get('assistant'):
                messages.append({"role": "assistant", "content": turn.get('assistant')})

        generated_code = call_nvidia_ai(messages)
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

        messages = [
            {"role": "user", "content": f"You are an expert summarizer. Summarize the following text:\n{text}"}
        ]

        summary = call_nvidia_ai(messages)
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
        source_language = data.get('source_language', 'English')
        target_language = data.get('target_language', 'Hindi')

        messages = [
            {"role": "user", "content": f"You are a professional translator. Translate the following text from {source_language} to {target_language}:\n{text}"}
        ]

        translation = call_nvidia_ai(messages)
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
    return jsonify({
        'status': 'healthy',
        'service': 'QwikGen API',
        'version': '2.0.0'
    })

# ----------------------------
# Run App
# ----------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
