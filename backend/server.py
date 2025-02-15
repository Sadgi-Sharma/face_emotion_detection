from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import base64
import cv2

app = Flask('__name__')
CORS(app)

# Load pre-trained model
model = tf.keras.models.load_model('models/cnn_emotion_detection.h5')

# Emotion mapping as in the real-time detection script
emotion_map = {
    0: "You seem Angry.",
    1: "You seem Disgusted.",
    2: "Fear Detected!!",
    3: "Yayy, You seem Happy.",
    4: "You seem Sad.",
    5: "Surprised!!!",
    6: "You seem Neutral."
}

def preprocess_image(image_base64):
    """
    Preprocess the base64 encoded image for emotion detection.
    """
    # Remove the base64 header if present
    if 'base64,' in image_base64:
        image_base64 = image_base64.split('base64,')[1]
    
    # Decode base64 to image
    image_bytes = base64.b64decode(image_base64)
    
    # Convert to numpy array
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    
    # Read as grayscale image
    image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)
    
    # Resize to 48x48 (model input size)
    image = cv2.resize(image, (48, 48))
    
    # Normalize the image
    image = image.astype('float') / 255.0
    
    # # Reshape for model input
    # image = image.reshape(1, 48, 48, 1)

    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)
    
    return image

@app.route('/detect-emotion', methods=['POST'])
def detect_emotion():
    try:
        # Get the image from the request
        image_base64 = request.json.get('image')
        
        # Preprocess the image
        processed_image = preprocess_image(image_base64)
        
        # Predict emotion
        predictions = model.predict(processed_image)
        emotion_index = int(np.argmax(predictions))  # Convert to Python int
        confidence = float(predictions[0][emotion_index])  # Convert to Python float
        
        # Map emotion to description
        emotion_description = emotion_map.get(emotion_index, "Unknown emotion.")
        
        return jsonify({
            'emotion': emotion_index,
            'emotion_description': emotion_description,
            'confidence': confidence
        })
    
    except Exception as e:
        # Log the error to the console
        print(f"Error occurred: {e}")
        return jsonify({'error': str(e)}), 500
app.run(host="0.0.0.0",debug=True, port=5000)
