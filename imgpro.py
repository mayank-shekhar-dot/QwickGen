"""
PixelForge AI - FINAL WORKING VERSION
Gemini Real Image Generation

Install:
pip uninstall google-generativeai -y
pip install google-genai flask flask-cors pillow gunicorn

Environment Variable:
GEMINI_API_KEY=YOUR_API_KEY
"""

import os
import time
import base64
import logging

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# NEW SDK
from google import genai
from google.genai import types

# =========================================================
# Logging
# =========================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================================================
# Flask App
# =========================================================

app = Flask(
    __name__,
    static_folder=".",
    static_url_path=""
)

CORS(app)

# =========================================================
# Gemini Setup
# =========================================================

API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    logger.error("GEMINI_API_KEY not found")
    raise ValueError("GEMINI_API_KEY is missing")

client = genai.Client(api_key=API_KEY)

logger.info("Gemini Client Initialized")

# =========================================================
# Gallery
# =========================================================

gallery = []
next_id = 1


def get_next_id():
    global next_id
    current = next_id
    next_id += 1
    return current


# =========================================================
# Home Route
# =========================================================

@app.route("/")
def home():
    return send_from_directory(".", "imgpro.html")


# =========================================================
# Health Route
# =========================================================

@app.route("/health")
def health():
    return jsonify({
        "status": "running"
    })


# =========================================================
# Generate Image
# =========================================================

@app.route("/generate", methods=["POST"])
def generate_image():

    try:

        data = request.get_json(silent=True)

        if not data:
            return jsonify({
                "error": "JSON body required"
            }), 400

        prompt = data.get("prompt", "").strip()

        if not prompt:
            return jsonify({
                "error": "Prompt required"
            }), 400

        style = data.get("style", "")
        aspect_ratio = data.get("aspectRatio", "")

        full_prompt = prompt

        if style and style != "None":
            full_prompt += f", {style} style"

        if aspect_ratio:
            full_prompt += f", aspect ratio {aspect_ratio}"

        logger.info(f"Generating image: {full_prompt}")

        # =================================================
        # Gemini Image Generation
        # =================================================

        response = client.models.generate_content(
            model="gemini-2.0-flash-preview-image-generation",
            contents=full_prompt,
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"]
            )
        )

        # =================================================
        # Extract Image
        # =================================================

        image_data = None
        mime_type = "image/png"

        for part in response.candidates[0].content.parts:

            if part.inline_data:

                image_data = part.inline_data.data
                mime_type = part.inline_data.mime_type

                break

        if not image_data:
            return jsonify({
                "error": "No image returned by Gemini"
            }), 500

        # Convert bytes → base64
        b64_data = base64.b64encode(image_data).decode("utf-8")

        logger.info("Image generated successfully")

        return jsonify({
            "success": True,
            "b64_json": b64_data,
            "mimeType": mime_type
        })

    except Exception as e:

        logger.exception("Generation Error")

        return jsonify({
            "error": str(e)
        }), 500


# =========================================================
# Gallery Routes
# =========================================================

@app.route("/gallery", methods=["GET"])
def get_gallery():
    return jsonify(list(reversed(gallery)))


@app.route("/gallery", methods=["POST"])
def save_gallery():

    data = request.get_json(silent=True)

    if not data:
        return jsonify({
            "error": "JSON body required"
        }), 400

    item = {
        "id": get_next_id(),
        "prompt": data.get("prompt"),
        "style": data.get("style"),
        "aspectRatio": data.get("aspectRatio"),
        "b64_json": data.get("b64_json"),
        "mimeType": data.get("mimeType"),
        "createdAt": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime()
        )
    }

    gallery.append(item)

    return jsonify(item), 201


@app.route("/gallery/<int:image_id>", methods=["DELETE"])
def delete_gallery(image_id):

    global gallery

    gallery = [
        item for item in gallery
        if item["id"] != image_id
    ]

    return "", 204


# =========================================================
# Error Handlers
# =========================================================

@app.errorhandler(404)
def not_found(e):
    return jsonify({
        "error": "Not found"
    }), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({
        "error": "Internal server error"
    }), 500


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    logger.info(f"Running on port {port}")

    app.run(
        host="0.0.0.0",
        port=port,
        debug=True
    )
