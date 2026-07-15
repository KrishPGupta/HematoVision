import os
import uuid
import tempfile
import logging

import numpy as np
import cv2
import base64
from flask import Flask, request, render_template, redirect, flash
from werkzeug.utils import secure_filename
from huggingface_hub import hf_hub_download

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
        Interpreter = tflite.Interpreter
    except ImportError:
        import tensorflow.lite as tflite
        Interpreter = tflite.Interpreter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hematovision")

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-key-change-me")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp"}

HF_REPO_ID = os.environ.get("HF_REPO_ID", "krishg15/hematovision-model")
HF_FILENAME = os.environ.get("HF_FILENAME", "Blood_Cell.tflite")

logger.info("Downloading/locating model %s from %s ...", HF_FILENAME, HF_REPO_ID)
model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)
logger.info("Model resolved at %s", model_path)

interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
logger.info("TFLite model loaded. Input shape: %s", input_details[0]["shape"])

_dummy_input = np.zeros(input_details[0]["shape"], dtype=np.float32)
interpreter.set_tensor(input_details[0]["index"], _dummy_input)
interpreter.invoke()
logger.info("Model warm-up complete.")

class_labels = ["eosinophil", "lymphocyte", "monocyte", "neutrophil"]


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Uploaded file is not a readable image.")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))

    img_processed = img.astype(np.float32) / 255.0
    img_processed = np.expand_dims(img_processed, axis=0)

    interpreter.set_tensor(input_details[0]["index"], img_processed)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]["index"])

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
            if os.path.exists(filepath):
                os.remove(filepath)

        _, buffer = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        img_str = base64.b64encode(buffer).decode("utf-8")

        return render_template("result.html", label=label, image=img_str)

    return render_template("home.html")


@app.route("/healthz")
def healthz():
    return {"status": "ok"}, 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
