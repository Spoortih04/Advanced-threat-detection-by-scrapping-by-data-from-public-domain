from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import face_recognition
import base64
import numpy as np
import os
from io import BytesIO
from PIL import Image

app = Flask(__name__)
CORS(app)

# Load known faces and names
known_faces = []
known_names = []

# Load known faces from the dataset
def load_known_faces():
    dataset_path = "./dataset"  # Dataset folder path
    for root, dirs, files in os.walk(dataset_path):  # Traverse subdirectoriespu
        for filename in files:
            if filename.endswith((".jpg", ".png", ".jpeg")):  # Check for image files
                image_path = os.path.join(root, filename)  # Get full image path
                try:
                    image = face_recognition.load_image_file(image_path)  # Load image
                    encodings = face_recognition.face_encodings(image)  # Get face encodings

                    if encodings:  # If there are face encodings in the image
                        known_faces.append(encodings[0])  # Store the first face encoding
                        known_names.append(filename.split(".")[0])  # Use filename (without extension) as the name
                except Exception as e:
                    print(f"Error processing {filename}: {e}")  # Error handling
    print(f"Loaded {len(known_faces)} faces from the dataset.")  # Debug: print how many faces loaded

# Load known faces when the app starts
load_known_faces()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/capture_and_detect", methods=["POST"])
def capture_and_detect():
    try:
        data = request.get_json()
        image_data = data.get("image")  # Get base64 image data

        if not image_data:
            return jsonify({"status": "No image data received"})

        # Decode the base64 image
        image_data = image_data.split(",")[1]  # Remove the "data:image/jpeg;base64," prefix
        decoded_image = base64.b64decode(image_data)
        image = np.array(Image.open(BytesIO(decoded_image)))

        # Get face encodings from the uploaded image
        unknown_encodings = face_recognition.face_encodings(image)

        if not unknown_encodings:  # If no faces are detected
            return jsonify({"status": "No Match Found"})

        # Compare uploaded image face encodings with known faces in the dataset
        for unknown in unknown_encodings:
            distances = face_recognition.face_distance(known_faces, unknown)  # Get face distances
            best_match_index = np.argmin(distances)  # Find the best match

            if distances[best_match_index] <= 0.4:  # Use a stricter tolerance value
                return jsonify({"status": "Match Found", "name": known_names[best_match_index]})  # Return the name

        return jsonify({"status": "No Match Found"})  # If no match is found
    except Exception as e:
        return jsonify({"status": "Error", "message": str(e)})

if __name__ == "__main__": 
    app.run(debug=True)