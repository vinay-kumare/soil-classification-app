from flask import Flask, request, jsonify, send_from_directory, render_template
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io
from flask_cors import CORS
import os

app = Flask(__name__, static_folder="build/assets", template_folder="build")
CORS(app)

# Soil information dictionary
soil_info = {
    0: {"soil_type": "Alluvial Soil", "description": "Highly fertile soil formed by water deposits."},
    1: {"soil_type": "Black Soil", "description": "Rich in clay and moisture-retaining properties."},
    2: {"soil_type": "Laterite Soil", "description": "Low in fertility but can be improved with organic matter."},
    3: {"soil_type": "Yellow Soil", "description": "Moderately fertile soil derived from crystalline rocks."}
}

# Load the pre-trained CNN model
model = load_model('soil_model.h5')

# Image preprocessing function
def preprocess_image(image):
    image = image.resize((200, 200))  # Resize to match model input size
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# API route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        image = Image.open(io.BytesIO(file.read()))
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction, axis=1)[0]
        soil_details = soil_info[predicted_class]

        return jsonify({
            'soil_type': soil_details['soil_type'],
            'description': soil_details['description'],
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Serve React frontend
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_react(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)
