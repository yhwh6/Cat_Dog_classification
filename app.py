import os

import tensorflow as tf

from flask import Flask, request
from classifier import classify

app = Flask(__name__)

STATIC_FOLDER = "static"
UPLOAD_FOLDER = "static/uploads"

cnn_model = tf.keras.models.load_model(STATIC_FOLDER + "/models/" + "female_male.h5")


@app.route("/")
def home():
    return "<p>Hello world! Use '/classify' endpoint to predict female or male!</p>"


@app.post("/classify")
def upload_file():
    file = request.files["image"]
    upload_image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(upload_image_path)

    label, prob = classify(cnn_model, upload_image_path)

    prob = round((prob * 100), 2)

    return {
        "label": label,
        "probability": prob
    }


if __name__ == "__main__":
    app.debug = True
    app.run(debug=True)
