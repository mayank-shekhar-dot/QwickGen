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
# OpenAI API configuration (new SDK >=1.0.0)
# ----------------------------
OPENAI_API_KEY = "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"
client = OpenAI(api_key=OPENAI_API_KEY)

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
# Helper function to call OpenAI Chat API
# ----------------------------
def call_openai_api(prompt, system_message="You are a helpful AI assistant.", max_tokens=1000, temperature=0.7):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"OpenAI API error: {str(e)}")
        return "AI service temporarily unavailable."

# ----------------------------
# Serve index.html
# ----------------------------
@app.route('/')
def index():
    return app.send_static_file('index.html')

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

        generated_text = call_openai_api(prompt, system_message=system_prompt)

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

        system_prompt = "You are ChatGPT, a helpful, friendly AI assistant."

        messages = [{"role": "system", "content": system_prompt}]
        for turn in history[-10:]:
            if "user" in turn:
                messages.append({"role": "user", "content": turn["user"]})
            if "assistant" in turn:
                messages.append({"role": "assistant", "content": turn["assistant"]})
        messages.append({"role": "user", "content": message})

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=1000,
            temperature=0.7
        )
        ai_response = response.choices[0].message.content.strip()
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

        system_prompt = f"You are Ghostwriter, an expert {language} developer. Always return only code, no explanations."

        messages = [{"role": "system", "content": system_prompt}]
        for turn in history[-10:]:
            if "user" in turn:
                messages.append({"role": "user", "content": turn["user"]})
            if "assistant" in turn:
                messages.append({"role": "assistant", "content": turn["assistant"]})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=1500,
            temperature=0.7
        )

        generated_code = response.choices[0].message.content.strip()
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
        summary = call_openai_api(prompt, system_message="You are an expert at summarizing text.")

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
        translation = call_openai_api(prompt, system_message="You are a professional translator.")

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
        'service': 'QwikGen API (OpenAI)',
        'version': '1.0.0'
    })

# ----------------------------
# Run App
# ----------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
