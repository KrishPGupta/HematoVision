import os
import numpy as np
import cv2
import base64
from flask import Flask, request, render_template, redirect
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)

# Ensure static folder exists
if not os.path.exists("static"):
    os.makedirs("static")

# Load model
model = load_model("Blood_Cell.h5")

# Class labels
class_labels = ['eosinophil', 'lymphocyte', 'monocyte', 'neutrophil']


# Prediction function
def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))

    img_processed = preprocess_input(img)
    img_processed = np.expand_dims(img_processed, axis=0)

    predictions = model.predict(img_processed)
    class_index = np.argmax(predictions)

    return class_labels[class_index], img


# HOME ROUTE (IMPORTANT FIX HERE)
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]

        if file.filename == "":
            return redirect(request.url)

        # Save file safely
        filepath = os.path.join("static", file.filename)
        file.save(filepath)

        # Predict
        label, img = predict_image(filepath)

        # Convert image to base64 for display
        _, buffer = cv2.imencode('.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        img_str = base64.b64encode(buffer).decode('utf-8')

        return render_template("result.html", label=label, image=img_str)

    return render_template("home.html")


# RUN APP
if __name__ == "__main__":
    app.run(debug=True)