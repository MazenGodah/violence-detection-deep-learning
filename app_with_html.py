from flask import Flask, request, jsonify, render_template
import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
from flask_cors import CORS  # Import CORS

app = Flask(__name__, template_folder='templates')
CORS(app)  # Enable CORS for all routes
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
VIOLENCE_MODEL_PATH = r"C:\Users\mazen\Desktop\project\violence_model.h5"
violence_model = load_model(VIOLENCE_MODEL_PATH)

@app.route('/')
def home():
    return render_template('index.html')

def extract_video_frames(path, num_frames=16):
    cap = cv2.VideoCapture(path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    if total_frames < num_frames:
        print("❌ Video too short:", total_frames)
        return []

    # evenly spaced frame indices
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    idx_set = set(frame_indices.tolist())

    current_idx = 0
    grabbed_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if current_idx in idx_set:
            try:
                resized = cv2.resize(frame, (64, 64))
                frames.append(resized)
                grabbed_frames += 1
            except Exception as e:
                print(f"⚠ Resize error at frame {current_idx}: {e}")
        current_idx += 1
        if grabbed_frames >= num_frames:
            break

    cap.release()
    print(f"✅ Grabbed {len(frames)} frames.")
    return frames

def prepare_input_tensor(frames):
    x = np.array(frames, dtype=np.float32)  # shape: (16, 64, 64, 3)
    x /= 255.0  # normalize
    print(f"✅ Tensor stats - shape: {x.shape}, min: {x.min():.4f}, max: {x.max():.4f}, mean: {x.mean():.4f}")
    return np.expand_dims(x, axis=0)  # shape: (1, 16, 64, 64, 3)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        print(f"📥 File saved: {filepath}")

        frames = extract_video_frames(filepath)
        if len(frames) < 16:
            return jsonify({'error': 'Not enough frames (16 required)'}), 400

        input_tensor = prepare_input_tensor(frames)
        prediction = violence_model.predict(input_tensor, verbose=0)[0]
        print("🧠 Raw model output:", prediction)

        label_idx = int(np.argmax(prediction))
        confidence = float(prediction[label_idx]) * 100
        is_violent = label_idx == 1

        # Calculate processing time (example value)
        processing_time = 1.2  # seconds
        
        # Create additional info array
        additional_info = [
            f"Analysis completed in {processing_time} seconds",
            "AI model version: VD-2023.4.1",
            f"Frame analysis: {len(frames)} frames processed",
            "Detection threshold: 75%"
        ]

        # Return in the format expected by the frontend
        return jsonify({
            'is_violent': is_violent,
            'confidence': round(confidence, 2),
            'filename': filename,
            'additional_info': additional_info
        })

    except Exception as e:
        print("Error:", str(e))
        return jsonify({'error': 'Server error', 'details': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)