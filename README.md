# HematoVision – Blood Cell Classification

HematoVision is a deep learning–based web application for classifying blood cell images into distinct categories using a trained Convolutional Neural Network (CNN). The application provides an intuitive interface for uploading microscopy images and receiving fast, accurate classification results.

🔗 **Live Demo:** [https://hematovision-71g9.onrender.com/](https://hematovision-71g9.onrender.com/)

## Features

- Upload blood cell images through a simple web interface
- Classify images into one of four cell types: Eosinophil, Lymphocyte, Monocyte, or Neutrophil
- Real-time inference with a pre-trained CNN model
- Clean, responsive front-end design

## Model Overview

The classification model is built using a Convolutional Neural Network trained on a labeled dataset spanning four blood cell categories:

- Eosinophil
- Lymphocyte
- Monocyte
- Neutrophil

## Tech Stack

| Component        | Technology            |
|-------------------|------------------------|
| Language          | Python                |
| Web Framework     | Flask                 |
| Deep Learning     | TensorFlow / Keras    |
| Image Processing  | OpenCV                |
| Front End         | HTML, CSS             |

## Project Structure

```
HematoVision/
├── static/
├── templates/
│   ├── home.html
│   └── result.html
├── app.py
├── Blood_Cell.h5
└── requirements.txt
```

## Getting Started

### Prerequisites

- Python 3.x
- pip

### Installation

```bash
git clone https://github.com/KrishPGupta/HematoVision.git
cd HematoVision
pip install -r requirements.txt
```

### Running the Application

```bash
python app.py
```

Then open your browser and navigate to:

```
http://127.0.0.1:5000
```

## Demo

Try it live: [https://hematovision-71g9.onrender.com/](https://hematovision-71g9.onrender.com/)

1. Upload a blood cell image via the web interface.
2. Click **Predict**.
3. View the predicted cell classification.

## Author

**Krish Gupta**
