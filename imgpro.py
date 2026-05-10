"""
PixelForge AI - Flask Backend (FIXED)
Uses Google Gemini Image Generation API

Install:
pip uninstall google-generativeai -y
pip install google-genai flask flask-cors pillow

Set Environment Variable:
GEMINI_API_KEY=your_api_key

Run:
python app.py
"""

import os
import base64
import time
import logging

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# NEW GOOGLE SDK
from google import genai
from google.genai import types

# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Flask App
# ─────────────────────────────────────────────────────────────

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

# ─────────────────────────────────────────────────────────────
# Gemini API Setup
# ─────────────────────────────────────────────────────────────

API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    logger.warning("GEMINI_API_KEY is NOT set.")
else:
    logger.info("Gemini API Key Loaded Successfully")

# Create Gemini Client
client = genai.Client(api_key=API_KEY)

# ─────────────────────────────────────────────────────────────
# In-Memory Gallery
# ─────────────────────────────────────────────────────────────

gallery = []
next_id = 1


def get_next_id():
    global next_id
    nid = next_id
    next_id += 1
    return nid


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """
    Serve frontend HTML file
    """
    return send_from_directory(".", "imgpro.html")


# ─────────────────────────────────────────────────────────────
# IMAGE GENERATION
# ─────────────────────────────────────────────────────────────

@app.route("/generate", methods=["POST"])
def generate():

    if not API_KEY:
        return jsonify({
            "error": "GEMINI_API_KEY environment variable not set."
        }), 500

    try:
        data = request.get_json(silent=True)

        if not data:
            return jsonify({
                "error": "Request body must be JSON."
            }), 400

        prompt = data.get("prompt", "").strip()

        if not prompt:
            return jsonify({
                "error": "Prompt is required."
            }), 400

        style = data.get("style", "")
        aspect_ratio = data.get("aspectRatio", "")

        # Enhanced Prompt
        full_prompt = prompt

        if style and style != "None":
            full_prompt += f", {style} style"

        if aspect_ratio:
            full_prompt += f", aspect ratio {aspect_ratio}"

        logger.info(f"Generating Image: {full_prompt}")

        # ─────────────────────────────────────────
        # Generate Image
        # ─────────────────────────────────────────

        response = client.models.generate_content(
            model="gemini-2.0-flash-preview-image-generation",
            contents=full_prompt,
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"]
            )
        )

        # ─────────────────────────────────────────
        # Extract Image
        # ─────────────────────────────────────────

        if not response.candidates:
            return jsonify({
                "error": "No candidates returned from Gemini."
            }), 500

        parts = response.candidates[0].content.parts

        for part in parts:

            # IMAGE FOUND
            if hasattr(part, "inline_data") and part.inline_data:

                image_bytes = part.inline_data.data
                mime_type = part.inline_data.mime_type

                # Convert image to base64
                b64_data = base64.b64encode(image_bytes).decode("utf-8")

                logger.info("Image generated successfully")

                return jsonify({
                    "success": True,
                    "b64_json": b64_data,
                    "mimeType": mime_type
                })

        # If no image found
        return jsonify({
            "error": "Model returned no image."
        }), 500

    except Exception as e:

        logger.exception("Image Generation Error")

        return jsonify({
            "error": str(e)
        }), 500


# ─────────────────────────────────────────────────────────────
# GALLERY ROUTES
# ─────────────────────────────────────────────────────────────

@app.route("/gallery", methods=["GET"])
def list_gallery():
    """
    Get all gallery images
    """
    return jsonify(list(reversed(gallery)))


@app.route("/gallery", methods=["POST"])
def save_to_gallery():

    data = request.get_json(silent=True)

    if not data:
        return jsonify({
            "error": "Request body must be JSON."
        }), 400

    required_fields = [
        "prompt",
        "b64_json",
        "mimeType"
    ]

    for field in required_fields:
        if field not in data:
            return jsonify({
                "error": f"Missing field: {field}"
            }), 400

    image = {
        "id": get_next_id(),
        "prompt": data["prompt"],
        "style": data.get("style"),
        "aspectRatio": data.get("aspectRatio"),
        "b64_json": data["b64_json"],
        "mimeType": data["mimeType"],
        "createdAt": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime()
        )
    }

    gallery.append(image)

    logger.info(f"Saved image #{image['id']}")

    return jsonify(image), 201


@app.route("/gallery/<int:image_id>", methods=["DELETE"])
def delete_from_gallery(image_id):

    global gallery

    original_count = len(gallery)

    gallery = [
        img for img in gallery
        if img["id"] != image_id
    ]

    if len(gallery) == original_count:
        return jsonify({
            "error": "Image not found."
        }), 404

    logger.info(f"Deleted image #{image_id}")

    return "", 204


@app.route("/gallery/stats", methods=["GET"])
def gallery_stats():

    total = len(gallery)

    style_counts = {}

    for img in gallery:

        style = img.get("style") or "None"

        style_counts[style] = (
            style_counts.get(style, 0) + 1
        )

    breakdown = [
        {
            "style": s,
            "count": c
        }
        for s, c in style_counts.items()
    ]

    breakdown.sort(
        key=lambda x: x["count"],
        reverse=True
    )

    return jsonify({
        "totalImages": total,
        "styleBreakdown": breakdown
    })


# ─────────────────────────────────────────────────────────────
# ERROR HANDLERS
# ─────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return jsonify({
        "error": "Not found."
    }), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({
        "error": "Internal server error."
    }), 500


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    logger.info(f"Starting PixelForge AI on port {port}")

    app.run(
        host="0.0.0.0",
        port=port,
        debug=True
    )
