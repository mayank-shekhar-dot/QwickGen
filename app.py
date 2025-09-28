import os
import logging
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import together
import base64
import io

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "qwikgen-secret-key-2025")
CORS(app)

# API Keys
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY", "21c340b3fdc58cf97d62c7c111a4b599c0824e335b5f7a9268460581cb719ba1")
together.api_key = TOGETHER_API_KEY

# AI Models
TOGETHER_MODELS = {
    "text": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "code": "meta-llama/Llama-3-8b-chat-hf",
    "chat": "mistralai/Mixtral-8x7B-Instruct-v0.1"
}

IMAGE_STYLES = {
    "realistic": "photographic",
    "anime": "anime",
    "sketch": "digital-art",
    "artistic": "artistic"
}

@app.route('/')
def serve_frontend():
    return send_from_directory('templates', 'index.html')

@app.route('/')
def serve_frontend():
    return send_from_directory('terms.html')

    
@app.route('/api/generate-text', methods=['POST'])
def generate_text():
    try:
        data = request.json
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

        full_prompt = f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:"

        response = together.Complete.create(
            model=TOGETHER_MODELS["text"],
            prompt=full_prompt,
            max_tokens=1000,
            temperature=0.7,
            top_p=0.7,
            top_k=50,
            repetition_penalty=1.1
        )

        generated_text = response['choices'][0]['text'].strip()

        return jsonify({
            'success': True,
            'content': generated_text,
            'type': tool_type
        })

    except Exception as e:
        logging.error(f"Text generation error: {str(e)}")
        return jsonify({'success': False, 'error': f'Text generation failed: {str(e)}'}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        message = data.get('message', '')
        history = data.get('history', [])

        # System role instructions
        system_prompt = (
            "You are ChatGPT, a helpful, friendly, and conversational AI assistant. "
            "Answer in clear, natural language. "
            "Explain step by step when useful, and keep responses easy to read."
        )

        # Build conversation context
        conversation = f"System: {system_prompt}\n\n"
        for turn in history[-10:]:
            conversation += f"User: {turn.get('user', '')}\nAssistant: {turn.get('assistant', '')}\n"
        conversation += f"User: {message}\nAssistant:"

        # Call Together API
        response = together.Complete.create(
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",  # or any chat model
            prompt=conversation,
            max_tokens=500,
            temperature=0.7,
            top_p=0.9
        )

        ai_response = response['choices'][0]['text'].strip()

        return jsonify({"success": True, "response": ai_response})

    except Exception as e:
        logging.error(f"Chat error: {str(e)}")
        return jsonify({"success": False, "error": f"Chat failed: {str(e)}"}), 500


@app.route('/api/generate-code', methods=['POST'])
def generate_code():
    try:
        data = request.json
        prompt = data.get('prompt', '')
        language = data.get('language', 'python')
        history = data.get('history', [])

        # Build system instructions
        system_prompt = (
            f"You are Ghostwriter, an expert {language} developer and web designer. "
            "only give the code"
            "Always return code directly in the best possible format, without extra explanation unless requested."
            "When the user asks for a website, always provide the best production-ready templates "
            "with smooth animations, modern UI, and responsive design. "
            "For Python or other programming tasks, generate clean, correct, and well-structured code "
            "with strong logic, detailed comments, and best practices. "
            "If the user requests improvements or optimizations, refactor the code clearly "
            "and explain the changes."
        )

        # Construct conversation context
        conversation = f"System: {system_prompt}\n\n"
        for turn in history[-10:]:
            conversation += f"User: {turn.get('user', '')}\nAssistant: {turn.get('assistant', '')}\n"
        conversation += f"User: {prompt}\nAssistant:"

        # Call Together API
        response = together.Complete.create(
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",  # or replit/deepseek code models
            prompt=conversation,
            max_tokens=1500,
            temperature=0.3,
            top_p=0.8
        )

        generated_code = response['choices'][0]['text'].strip()

        return jsonify({
            "success": True,
            "content": generated_code,
            "language": language
        })

    except Exception as e:
        logging.error(f"Code generation error: {str(e)}")
        return jsonify({"success": False, "error": f"Code generation failed: {str(e)}"}), 500



@app.route('/api/summarize', methods=['POST'])
def summarize():
    try:
        data = request.json
        text = data.get('text', '')

        prompt = f"Please summarize the following text:\n\n{text}\n\nSummary:"

        response = together.Complete.create(
            model=TOGETHER_MODELS["text"],
            prompt=prompt,
            max_tokens=500,
            temperature=0.3,
            top_p=0.8
        )

        summary = response['choices'][0]['text'].strip()
        return jsonify({'success': True, 'summary': summary})

    except Exception as e:
        logging.error(f"Summarization error: {str(e)}")
        return jsonify({'success': False, 'error': f'Summarization failed: {str(e)}'}), 500


@app.route('/api/translate', methods=['POST'])
def translate():
    try:
        data = request.json
        text = data.get('text', '')
        target_language = data.get('target_language', 'Hindi')
        source_language = data.get('source_language', 'English')

        prompt = f"Translate from {source_language} to {target_language}:\n\n{text}\n\nTranslation:"

        response = together.Complete.create(
            model=TOGETHER_MODELS["text"],
            prompt=prompt,
            max_tokens=800,
            temperature=0.2,
            top_p=0.8
        )

        translation = response['choices'][0]['text'].strip()
        return jsonify({
            'success': True,
            'translation': translation,
            'source_language': source_language,
            'target_language': target_language
        })

    except Exception as e:
        logging.error(f"Translation error: {str(e)}")
        return jsonify({'success': False, 'error': f'Translation failed: {str(e)}'}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'QwikGen API',
        'version': '1.0.0'
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)






















