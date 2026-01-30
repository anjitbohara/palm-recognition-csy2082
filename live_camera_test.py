import cv2
import numpy as np
import pickle
from tensorflow.keras.models import load_model, Model
from img_preprocessing import PalmPreprocessor

# ================= CONFIG =================
MODEL_PATH = "palm_recognition_model.h5"
EMBEDDING_PATH = "reference_embeddings.pkl"
IMG_SIZE = 128
CAMERA_INDEX = 0
CONFIDENCE_THRESHOLD = 0.60
EMBEDDING_DISTANCE_THRESHOLD = 1.2
# ==========================================

print("[INFO] Loading model...")
model = load_model(MODEL_PATH)

# Create embedding model (Dense(128))
embedding_model = Model(
    inputs=model.inputs,
    outputs=model.layers[-2].output
)

print("[INFO] Loading reference embeddings...")
with open(EMBEDDING_PATH, "rb") as f:
    reference_embeddings = pickle.load(f)

class_names = list(reference_embeddings.keys())
print("[INFO] Classes:", class_names)

preprocessor = PalmPreprocessor()

# ================= CAMERA =================
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("✗ Cannot open camera")
    exit()

print("[INFO] Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()
    h, w = frame.shape[:2]

    # ===== GREEN GUIDE RECTANGLE =====
    rect_w, rect_h = w // 2, h // 2
    x1, y1 = (w - rect_w) // 2, (h - rect_h) // 2
    x2, y2 = x1 + rect_w, y1 + rect_h

    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # ===== BLUE ROI (USED FOR RECOGNITION) =====
    palm_roi = frame[y1:y2, x1:x2]

    # Preview box
    preview_size = 120
    px, py = 10, 10
    preview = cv2.resize(palm_roi, (preview_size, preview_size))
    display[py:py+preview_size, px:px+preview_size] = preview
    cv2.rectangle(display, (px, py), (px+preview_size, py+preview_size), (255, 0, 0), 2)

    label = "No Palm"
    confidence_text = ""

    # ===== PREPROCESS BLUE ROI =====
    processed = preprocessor.preprocess_v2(palm_roi)

    if processed is not None:
        input_img = processed.reshape(1, IMG_SIZE, IMG_SIZE, 1)

        # CNN prediction
        probs = model.predict(input_img, verbose=0)[0]
        cnn_conf = np.max(probs)
        cnn_idx = np.argmax(probs)

        # Embedding prediction
        emb = embedding_model.predict(input_img, verbose=0)[0]

        min_dist = float("inf")
        matched_person = None

        for person, ref_emb in reference_embeddings.items():
            dist = np.linalg.norm(emb - ref_emb)
            if dist < min_dist:
                min_dist = dist
                matched_person = person

        # ===== FINAL DECISION =====
        if cnn_conf > CONFIDENCE_THRESHOLD and min_dist < EMBEDDING_DISTANCE_THRESHOLD:
            label = matched_person
            confidence_text = f"{cnn_conf:.2f} | d={min_dist:.2f}"
            box_color = (0, 255, 0)
        else:
            label = "Unknown"
            confidence_text = f"{cnn_conf:.2f}"
            box_color = (0, 0, 255)

        cv2.rectangle(display, (x1, y1), (x2, y2), box_color, 3)

    # ===== TEXT OVERLAY =====
    cv2.putText(display, f"Prediction: {label}",
                (20, h - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.putText(display, confidence_text,
                (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    cv2.putText(display, "Green: Hand Zone | Blue: Used for Recognition",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Palm Recognition - Live Camera", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
