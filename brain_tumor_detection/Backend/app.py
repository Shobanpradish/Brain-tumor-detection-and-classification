from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Load the trained model
model = tf.keras.models.load_model("C:\\finalone\\brain_tumor_detection\\backend\\Resnetfinalnew_model.h5")

# Define class labels (MAKE SURE this order is correct)
CLASS_LABELS = ["glioma", "meningioma", "notumor", "pituitary"]

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/')
def index():
    return render_template('index.html')

def preprocess_image(image):
    """Preprocess uploaded image to match model input requirements exactly as in Jupyter Notebook."""
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)  # Convert PIL to OpenCV BGR
    image = cv2.resize(image, (224, 224))  # Resize to match model input
    image = image / 255.0  # Normalize to range [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        image = Image.open(file).convert('RGB')  # Convert uploaded file to RGB
        processed_image = preprocess_image(image)  # Preprocess exactly like Jupyter Notebook

        prediction = model.predict(processed_image)
        predicted_index = np.argmax(prediction)  # Get index of highest probability
        predicted_label = CLASS_LABELS[predicted_index]  # Get class label

        confidence_scores = {CLASS_LABELS[i]: float(prediction[0][i]) * 100 for i in range(len(CLASS_LABELS))}

        return jsonify({
            'prediction': predicted_label,
            'confidence_scores': confidence_scores
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5002)
