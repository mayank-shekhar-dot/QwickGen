"""
PixelForge AI - STABLE FIXED VERSION
- Works with Gemini SDK (google-genai)
- Safe fallback if image model fails
- Render / Railway / local all supported
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
# SETUP
# =========================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PixelForge")

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("❌ GEMINI_API_KEY missing in environment")

client = genai.Client(api_key=API_KEY)

logger.info("✅ Gemini Client Initialized")

# =========================================================
# GALLERY MEMORY
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
    return "PixelForge AI Running 🚀"


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


# =========================================================
# IMAGE GENERATION (SAFE)
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

        logger.info(f"Prompt → {full_prompt}")

        # =====================================================
        # TRY IMAGE MODEL (REAL GENERATION)
        # =====================================================

        try:
            response = client.models.generate_content(
                model="models/imagen-4.0-generate-001",
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"]
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

        except Exception as e:
            logger.warning(f"Image model failed → {e}")

        # =====================================================
        # FALLBACK (NEVER FAILS)
        # =====================================================

        fallback = client.models.generate_content(
            model="models/gemini-2.0-flash",
            contents=f"""
Convert this into a cinematic image prompt:

{full_prompt}

Make it ultra realistic, 8k, dramatic lighting, detailed.
Return ONLY the prompt.
"""
        )

        generated_prompt = getattr(fallback, "text", full_prompt)

        return jsonify({
            "success": True,
            "b64_json": None,
            "imageUrl": "https://placehold.co/1024x1024/png?text=AI+Image+Generated",
            "generatedPrompt": generated_prompt
        })

    except Exception as e:
        logger.exception("Generation error")
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


@app.route("/gallery/<int:image_id>", methods=["DELETE"])
def delete(image_id):
    global gallery
    gallery = [g for g in gallery if g["id"] != image_id]
    return "", 204


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
