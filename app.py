import logging
from flask import Flask, request, jsonify, redirect
from flask_cors import CORS
import requests
import os  # for environment variables

# ----------------------------
# Configure logging
# ----------------------------
logging.basicConfig(level=logging.DEBUG)

# ----------------------------
# Initialize Flask app
# ----------------------------
app = Flask(__name__, static_folder='.')
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "default-secret-key")  # Use env variable
CORS(app)

# ----------------------------
# Together AI API configuration (use environment variable for key)
# ----------------------------
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")  # now private
if not TOGETHER_API_KEY:
    raise ValueError("TOGETHER_API_KEY environment variable not set!")

TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"

# AI Models
TOGETHER_MODELS = {
    "text": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "code": "meta-llama/Llama-3-8b-chat-hf",
    "chat": "mistralai/Mixtral-8x7B-Instruct-v0.1"
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
# Helper function to call Together AI API
# ----------------------------
def call_together_ai(prompt, model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                     system_message="You are a helpful AI assistant.", max_tokens=500):
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
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.7
        }
        response = requests.post(TOGETHER_API_URL, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logging.error(f"Together AI API error: {str(e)}")
        return "AI service temporarily unavailable."

# ----------------------------
# Routes (Text, Chat, Code, Summarize, Translate)
# ----------------------------
# [Keep all route definitions the same as your current code]
# Just remove the hardcoded key reference and use the helper above

# ----------------------------
# Run App
# ----------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
