import os
import numpy as np
import cv2
import base64
from flask import Flask, request, render_template, redirect
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)

# Load model
model = load_model("Blood_Cell.h5")

# Class labels
class_labels = ['eosinophil', 'lymphocyte', 'monocyte', 'neutrophil']

# Prediction function
def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img)
    class_index = np.argmax(predictions)
    
    return class_labels[class_index], img


# Home route
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]

        if file.filename == "":
            return redirect(request.url)

        filepath = os.path.join("static", file.filename)
        file.save(filepath)

        label, img = predict_image(filepath)

        # Convert image to base64
        _, buffer = cv2.imencode('.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        img_str = base64.b64encode(buffer).decode('utf-8')

        return render_template("result.html", label=label, image=img_str)

    return render_template("home.html")


# Run app
if __name__ == "__main__":
    app.run(debug=True)