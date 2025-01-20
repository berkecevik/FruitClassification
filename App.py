from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image
import base64

app = Flask(__name__)
model = tf.keras.models.load_model("fruit_classifier.keras")
class_names = ["Apple", "Banana", "Grape", "Mango", "Strawberry"]

def predict_image(file):
    # Read and preprocess the image
    img = Image.open(file).convert("RGB").resize((224, 224))  # Resize to 224x224
    img_array = np.array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions)
    confidence = np.max(predictions)
    return class_names[class_index], confidence

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    image_data = None

    if request.method == "POST":
        file = request.files["file"]
        if file:
            image_data = base64.b64encode(file.read()).decode('utf-8')  # Encode image data to Base64
            result, confidence = predict_image(file)

    return render_template("index.html", result=result, confidence=confidence, image_data=image_data)

if __name__ == "__main__":
    app.run(debug=True)
