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
app.secret_key = "qwikgen-secret-key-2026"
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
def call_nvidia_ai(messages, model="google/gemma-2-2b-it", temperature=0.2, max_tokens=1024):
    """
    NVIDIA Gemma 2.x rules:
    - Roles must alternate: user → assistant → user → assistant
    - Last message must be 'user'
    """
    try:
        # Remove empty contents
        msgs = [m for m in messages if m.get("content")]

        # Fix alternation
        fixed_msgs = []
        last_role = None
        for m in msgs:
            role = m["role"]
            content = m["content"]

            if last_role is None:
                # First message can be system or user
                fixed_msgs.append({"role": role, "content": content})
                last_role = role
                continue

            # Alternate roles
            if role == last_role:
                # Skip duplicate role or convert system->user
                if role == "system":
                    fixed_msgs.append({"role": "user", "content": content})
                    last_role = "user"
                else:
                    # skip duplicate user/user or assistant/assistant
                    continue
            else:
                fixed_msgs.append({"role": role, "content": content})
                last_role = role

        # Ensure last message is 'user'
        if fixed_msgs[-1]["role"] != "user":
            fixed_msgs.append({"role": "user", "content": "..."})

        completion = client.chat.completions.create(
            model=model,
            messages=fixed_msgs,
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

        system_instruction = {
            "blog": "You are a professional blog writer. Write clear, engaging blogs.",
            "email": "You are an expert email writer. Compose professional emails.",
            "startup": "You are a startup advisor. Give practical startup ideas.",
            "general": "You are a helpful AI assistant. Respond accurately."
        }.get(tool_type, "You are a helpful AI assistant. Respond accurately.")

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt}
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

        # Start with system instruction
        messages = [{"role": "system", "content": "You are ChatGPT, a helpful, friendly, conversational AI assistant."}]

        # Add previous conversation alternating roles
        for turn in history[-10:]:
            if turn.get("user"):
                messages.append({"role": "user", "content": turn["user"]})
            if turn.get("assistant"):
                messages.append({"role": "assistant", "content": turn["assistant"]})

        # Add latest user message
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

        messages = [{"role": "system", "content": f"You are Ghostwriter, an expert {language} developer. Return only code without extra explanation."}]
        for turn in history[-10:]:
            if turn.get("user"):
                messages.append({"role": "user", "content": turn["user"]})
            if turn.get("assistant"):
                messages.append({"role": "assistant", "content": turn["assistant"]})

        messages.append({"role": "user", "content": prompt})
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
            {"role": "system", "content": "You are an expert summarizer."},
            {"role": "user", "content": f"Summarize the following text:\n{text}"}
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
            {"role": "system", "content": "You are a professional translator."},
            {"role": "user", "content": f"Translate from {source_language} to {target_language}:\n{text}"}
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
        'version': '3.0.0'
    })

# ----------------------------
# Run App
# ----------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
