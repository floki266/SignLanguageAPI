from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import tempfile

import os
import numpy as np
import tensorflow as tf
import cv2
import mediapipe as mp

# ==========================================
# إنشاء Flask
# ==========================================

app = Flask(__name__)

# ==========================================
# المسارات
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "best_model.keras"
)

LABELS_PATH = os.path.join(
    BASE_DIR,
    "model",
    "labels.npy"
)

# ==========================================
# تحميل النموذج
# ==========================================

print("=" * 60)
print("Loading AI Model...")
print("=" * 60)

model = tf.keras.models.load_model(MODEL_PATH)

labels = np.load(LABELS_PATH)
# ==========================================
# MediaPipe Hands
# ==========================================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

print("Model Loaded Successfully")

print(labels)
# ==========================================
# صفحة اختبار
# ==========================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({

        "status": "running",

        "model": "Sign Language AI",

        "classes": len(labels)

    })


# ==========================================
# التنبؤ
# ==========================================
def extract_landmarks(frame):

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    result = hands.process(rgb)


    if result.multi_hand_landmarks:

        points = []

        hand = result.multi_hand_landmarks[0]

        for lm in hand.landmark:

            points.extend([
                lm.x,
                lm.y,
                lm.z
            ])

        return np.array(
            points,
            dtype=np.float32
        )


    return np.zeros(
        63,
        dtype=np.float32
    )
# ==========================================
# استقبال صور Flutter
# ==========================================

@app.route("/predict_images", methods=["POST"])
def predict_images():

    try:

        if "frames" not in request.files:

            return jsonify({

                "success": False,

                "message": "No frames received"

            }), 400


        files = request.files.getlist("frames")


        if len(files) == 0:

            return jsonify({

                "success": False,

                "message": "Empty frames"

            }), 400


        sequence = []


        for file in files:

            data = np.frombuffer(
                file.read(),
                np.uint8
            )

            frame = cv2.imdecode(
                data,
                cv2.IMREAD_COLOR
            )


            if frame is None:

                sequence.append(
                    np.zeros(63, dtype=np.float32)
                )

                continue


            sequence.append(
                extract_landmarks(frame)
            )


        while len(sequence) < 30:

            sequence.append(
                np.zeros(63, dtype=np.float32)
            )


        sequence = np.array(
            sequence[:30],
            dtype=np.float32
        )


        sequence = np.expand_dims(
            sequence,
            axis=0
        )


        prediction = model.predict(
            sequence,
            verbose=0
        )


        index = int(np.argmax(prediction))

        confidence = float(prediction[0][index])

        word = str(labels[index])


        return jsonify({

            "success": True,

            "word": word,

            "confidence": confidence

        })


    except Exception as e:

        return jsonify({

            "success": False,

            "message": str(e)

        }), 500

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.json

        if "sequence" not in data:

            return jsonify({

                "success": False,

                "message": "sequence not found"

            }), 400

        sequence = np.array(
            data["sequence"],
            dtype=np.float32
        )

        if sequence.shape != (30, 63):

            return jsonify({

                "success": False,

                "message": f"Invalid shape {sequence.shape}"

            }), 400

        sequence = np.expand_dims(sequence, axis=0)

        prediction = model.predict(
            sequence,
            verbose=0
        )

        index = int(np.argmax(prediction))

        confidence = float(prediction[0][index])

        word = str(labels[index])

        return jsonify({

            "success": True,

            "word": word,

            "confidence": confidence

        })

    except Exception as e:

        return jsonify({

            "success": False,

            "message": str(e)

        }), 500
# ==========================================
# تشغيل السيرفر
# ==========================================

# ==========================================
# تشغيل السيرفر
# ==========================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(

        host="0.0.0.0",

        port=port,
        debug = False

    )