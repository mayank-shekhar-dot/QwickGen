import logging
import os
from flask import Flask, request, jsonify, redirect
from flask_cors import CORS
from openai import OpenAI

# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

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
# Helper function
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

        generated_text = call_openai_api(prompt, system_message=system_prompt)

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
# Add other endpoints: generate-code, summarize, translate
# ----------------------------
# Code, Summarization, Translation endpoints can use same `call_openai_api` logic
# Just adjust system_prompt and max_tokens

# ----------------------------
# Health check
# ----------------------------
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'QwikGen API (OpenAI)',
        'version': '1.0.0'
    })

# ----------------------------
# Run app
# ----------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
