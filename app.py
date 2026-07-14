import os
import uuid
import tempfile
import logging

# Keep TensorFlow quiet and CPU-only-friendly on Render's free tier
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import cv2
import base64
from flask import Flask, request, render_template, redirect, flash
from werkzeug.utils import secure_filename
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hematovision")

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-key-change-me")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB upload limit

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp"}

# ---------------------------------------------------------------------------
# Model loading — downloaded once from Hugging Face and cached on disk.
# hf_hub_download checks its local cache before hitting the network, so this
# is cheap on every subsequent call/import; it is called exactly once here,
# at process startup, before any request is served.
# ---------------------------------------------------------------------------
HF_REPO_ID = os.environ.get("HF_REPO_ID", "krishg15/hematovision-model")
HF_FILENAME = os.environ.get("HF_FILENAME", "Blood_Cell.h5")
# HF_HOME controls where huggingface_hub caches downloads. Defaults to
# /opt/render/.cache/huggingface on Render if not overridden by an env var.

logger.info("Downloading/locating model %s from %s ...", HF_FILENAME, HF_REPO_ID)
model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)
logger.info("Model resolved at %s", model_path)

model = load_model(model_path)
logger.info("Model loaded into memory.")

# Class labels
class_labels = ["eosinophil", "lymphocyte", "monocyte", "neutrophil"]


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Uploaded file is not a readable image.")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))

    img_processed = preprocess_input(img.astype(np.float32))
    img_processed = np.expand_dims(img_processed, axis=0)

    predictions = model.predict(img_processed)
    class_index = int(np.argmax(predictions))

    return class_labels[class_index], img


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file part in the request.")
            return redirect(request.url)

        file = request.files["file"]

        if file.filename == "":
            flash("No file selected.")
            return redirect(request.url)

        if not allowed_file(file.filename):
            flash("Unsupported file type. Please upload a PNG or JPG image.")
            return redirect(request.url)

        filename = secure_filename(file.filename)
        # Use a temp file with a random prefix so concurrent users never
        # collide, and so nothing is written into the public static/ folder.
        unique_name = f"{uuid.uuid4().hex}_{filename}"
        filepath = os.path.join(tempfile.gettempdir(), unique_name)

        try:
            file.save(filepath)
            label, img = predict_image(filepath)
        except ValueError as exc:
            flash(str(exc))
            return redirect(request.url)
        except Exception:
            logger.exception("Prediction failed")
            flash("Something went wrong while processing the image.")
            return redirect(request.url)
        finally:
            # Always clean up the temp upload, success or failure.
            if os.path.exists(filepath):
                os.remove(filepath)

        # Encode the (in-memory) processed image as base64 for display —
        # nothing is written back to disk.
        _, buffer = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        img_str = base64.b64encode(buffer).decode("utf-8")

        return render_template("result.html", label=label, image=img_str)

    return render_template("home.html")


@app.route("/healthz")
def healthz():
    # Cheap liveness/readiness endpoint — useful for Render health checks.
    return {"status": "ok"}, 200


if __name__ == "__main__":
    # Local dev only. In production, Gunicorn imports `app` directly and
    # this block never runs.
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
