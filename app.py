import os

import tensorflow as tf
from flask import Flask, render_template, request

from classifier import classify

app = Flask(__name__)

STATIC_FOLDER = "static/"
UPLOAD_FOLDER = STATIC_FOLDER + "uploads/"

cnn_model = tf.keras.models.load_model(STATIC_FOLDER + "/models/" + "female_male.keras")


@app.route("/")
def home():
    return render_template("upload.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    upload_image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(upload_image_path)

    label, prob = classify(cnn_model, upload_image_path)

    prob = round((prob * 100), 2)
    return render_template("predict.html", label=label, prob=prob)


if __name__ == "__main__":
    app.run(debug=True)
