"""
PixelForge AI - Flask Backend
Uses Google Gemini API for image generation.
Requirements: pip install flask google-generativeai flask-cors pillow
Set env variable: GEMINI_API_KEY=your_key_here
"""

import os
import base64
import io
import time
import logging
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

# ─── Gemini client setup ────────────────────────────────────────────────────

try:
    import google.generativeai as genai
    from google.generativeai import types

    API_KEY = os.environ.get("GEMINI_API_KEY")
    if not API_KEY:
        logger.warning("GEMINI_API_KEY not set. Image generation will fail.")
    else:
        genai.configure(api_key=API_KEY)
    GEMINI_AVAILABLE = True
except ImportError:
    logger.error("google-generativeai not installed. Run: pip install google-generativeai")
    GEMINI_AVAILABLE = False


# ─── In-memory gallery storage ──────────────────────────────────────────────
# For production, replace with a real database (SQLite, PostgreSQL, etc.)

gallery = []
next_id = 1


def get_next_id():
    global next_id
    nid = next_id
    next_id += 1
    return nid


# ─── Routes ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main HTML page."""
    return send_from_directory(".", "imgpro.html")


@app.route("/generate", methods=["POST"])
def generate():
    """
    Generate an image using the Gemini image generation model.
    Expects JSON body: { "prompt": str, "style": str (optional), "aspectRatio": str (optional) }
    Returns JSON: { "b64_json": str, "mimeType": str }
    """
    if not GEMINI_AVAILABLE:
        return jsonify({"error": "Gemini AI library not installed."}), 503

    if not os.environ.get("GEMINI_API_KEY"):
        return jsonify({"error": "GEMINI_API_KEY environment variable not set."}), 503

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON."}), 400

    prompt = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "Prompt is required."}), 400

    style = data.get("style", "")
    aspect_ratio = data.get("aspectRatio", "")

    # Build enhanced prompt
    full_prompt = prompt
    if style and style != "None":
        full_prompt += f". Art style: {style}."
    if aspect_ratio:
        full_prompt += f" Aspect ratio: {aspect_ratio}."

    logger.info(f"Generating image for prompt: {full_prompt[:80]}...")

    try:
        # Use gemini-2.0-flash-preview-image-generation model
        client = genai.GenerativeModel("gemini-2.0-flash-preview-image-generation")

        response = client.generate_content(
            full_prompt,
            generation_config=types.GenerationConfig(
                response_modalities=["IMAGE", "TEXT"],
            ),
        )

        # Extract image from response
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                image_data = part.inline_data.data
                mime_type = part.inline_data.mime_type

                # Convert to base64 if needed
                if isinstance(image_data, bytes):
                    b64 = base64.b64encode(image_data).decode("utf-8")
                else:
                    b64 = image_data  # already base64

                logger.info("Image generated successfully.")
                return jsonify({
                    "b64_json": b64,
                    "mimeType": mime_type or "image/png"
                })

        return jsonify({"error": "No image was returned by the model."}), 500

    except Exception as e:
        logger.error(f"Image generation error: {e}")
        return jsonify({"error": f"Image generation failed: {str(e)}"}), 500


@app.route("/gallery", methods=["GET"])
def list_gallery():
    """Return all saved gallery images, newest first."""
    return jsonify(list(reversed(gallery)))


@app.route("/gallery", methods=["POST"])
def save_to_gallery():
    """
    Save an image to the gallery.
    Expects JSON body: { "prompt": str, "style": str, "aspectRatio": str, "b64_json": str, "mimeType": str }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON."}), 400

    required = ["prompt", "b64_json", "mimeType"]
    for field in required:
        if field not in data:
            return jsonify({"error": f"Missing field: {field}"}), 400

    image = {
        "id": get_next_id(),
        "prompt": data["prompt"],
        "style": data.get("style"),
        "aspectRatio": data.get("aspectRatio"),
        "b64_json": data["b64_json"],
        "mimeType": data["mimeType"],
        "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    gallery.append(image)
    logger.info(f"Saved image #{image['id']} to gallery.")
    return jsonify(image), 201


@app.route("/gallery/<int:image_id>", methods=["DELETE"])
def delete_from_gallery(image_id):
    """Delete an image from the gallery by ID."""
    global gallery
    original_count = len(gallery)
    gallery = [img for img in gallery if img["id"] != image_id]

    if len(gallery) == original_count:
        return jsonify({"error": "Image not found."}), 404

    logger.info(f"Deleted image #{image_id} from gallery.")
    return "", 204


@app.route("/gallery/stats", methods=["GET"])
def gallery_stats():
    """Return gallery statistics: total count and style breakdown."""
    total = len(gallery)
    style_counts = {}
    for img in gallery:
        style = img.get("style") or "None"
        style_counts[style] = style_counts.get(style, 0) + 1

    breakdown = [{"style": s, "count": c} for s, c in style_counts.items()]
    breakdown.sort(key=lambda x: x["count"], reverse=True)

    return jsonify({
        "totalImages": total,
        "styleBreakdown": breakdown,
    })


# ─── Error handlers ─────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found."}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error."}), 500


# ─── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV", "production") == "development"
    logger.info(f"Starting PixelForge AI on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
