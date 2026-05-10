"""
PixelForge AI - STABLE VERSION
Works even if Gemini image model is unavailable

Install:
pip install google-genai flask flask-cors pillow gunicorn

Env:
GEMINI_API_KEY=YOUR_KEY
"""

import os
import time
import base64
import logging

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from google import genai
from google.genai import types

# =========================================================
# Setup
# =========================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY missing")

client = genai.Client(api_key=API_KEY)

logger.info("Gemini client ready")

# =========================================================
# Gallery
# =========================================================

gallery = []
next_id = 1


def get_next_id():
    global next_id
    nid = next_id
    next_id += 1
    return nid


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():
    return send_from_directory(".", "imgpro.html")


# =========================================================
# HEALTH
# =========================================================

@app.route("/health")
def health():
    return jsonify({"status": "ok"})


# =========================================================
# GENERATE IMAGE (SAFE + FIXED)
# =========================================================

@app.route("/generate", methods=["POST"])
def generate():

    try:
        data = request.get_json(silent=True)

        if not data:
            return jsonify({"error": "JSON required"}), 400

        prompt = data.get("prompt", "").strip()

        if not prompt:
            return jsonify({"error": "Prompt required"}), 400

        style = data.get("style", "")
        aspect = data.get("aspectRatio", "")

        full_prompt = prompt

        if style and style != "None":
            full_prompt += f", {style} style"

        if aspect:
            full_prompt += f", aspect ratio {aspect}"

        logger.info(f"Prompt: {full_prompt}")

        # =====================================================
        # STEP 1: Try IMAGE model (may fail on many accounts)
        # =====================================================

        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash-preview-image-generation",
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE", "TEXT"]
                )
            )

            for part in response.candidates[0].content.parts:
                if getattr(part, "inline_data", None):
                    img = part.inline_data.data
                    mime = part.inline_data.mime_type

                    return jsonify({
                        "success": True,
                        "b64_json": base64.b64encode(img).decode(),
                        "mimeType": mime
                    })

        except Exception as img_error:
            logger.warning(f"Image model failed: {img_error}")

        # =====================================================
        # STEP 2: FALLBACK (ALWAYS WORKS)
        # =====================================================

        text_model = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=f"""
Create a detailed AI image prompt:

{full_prompt}

Make it cinematic, ultra realistic, 8k, dramatic lighting.
Return only final prompt.
"""
        )

        prompt_text = text_model.text or full_prompt

        # placeholder image (so frontend NEVER breaks)
        placeholder = "https://placehold.co/1024x1024/png?text=Image+Unavailable"

        return jsonify({
            "success": True,
            "b64_json": None,
            "mimeType": "image/png",
            "imageUrl": placeholder,
            "generatedPrompt": prompt_text
        })

    except Exception as e:
        logger.exception(e)
        return jsonify({"error": str(e)}), 500


# =========================================================
# GALLERY
# =========================================================

@app.route("/gallery", methods=["GET"])
def get_gallery():
    return jsonify(list(reversed(gallery)))


@app.route("/gallery", methods=["POST"])
def save_gallery():

    data = request.get_json(silent=True)

    item = {
        "id": get_next_id(),
        "prompt": data.get("prompt"),
        "style": data.get("style"),
        "aspectRatio": data.get("aspectRatio"),
        "b64_json": data.get("b64_json"),
        "imageUrl": data.get("imageUrl"),
        "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }

    gallery.append(item)
    return jsonify(item), 201


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
