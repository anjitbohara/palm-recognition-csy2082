# # import cv2
# # import os
# # import numpy as np
# # from tensorflow.keras.models import load_model

# # # ---------------- CONFIG ----------------
# # IMG_SIZE = 128
# # MODEL_PATH = "palm_recognition_model.h5"
# # CONFIDENCE_THRESHOLD = 0.60
# # # ---------------------------------------

# # model = load_model(MODEL_PATH)

# # class_names = sorted(os.listdir("dataset"))  # SAME ORDER AS TRAINING

# # # -------- Preprocessing (SAME AS TRAINING) --------
# # def preprocess_palm_live(frame):
# #     frame_resized = cv2.resize(frame, (512, 512))
# #     gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
# #     blur = cv2.GaussianBlur(gray, (5, 5), 0)

# #     _, thresh = cv2.threshold(
# #         blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
# #     )

# #     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
# #     thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

# #     contours, _ = cv2.findContours(
# #         thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
# #     )

# #     if len(contours) == 0:
# #         return None, None

# #     largest = max(contours, key=cv2.contourArea)
# #     if cv2.contourArea(largest) < 5000:
# #         return None, None

# #     x, y, w, h = cv2.boundingRect(largest)
# #     roi = gray[y:y+h, x:x+w]

# #     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# #     enhanced = clahe.apply(roi)

# #     resized = cv2.resize(enhanced, (IMG_SIZE, IMG_SIZE))
# #     normalized = resized / 255.0
# #     input_img = normalized.reshape(1, IMG_SIZE, IMG_SIZE, 1)

# #     return input_img, (x, y, w, h)

# # # ---------------- CAMERA LOOP ----------------
# # cap = cv2.VideoCapture(0)
# # print("Camera opened:", cap.isOpened())

# # while True:
# #     ret, frame = cap.read()
# #     if not ret:
# #         print("Failed to grab frame")
# #         break

# #     input_img, bbox = preprocess_palm_live(frame)

# #     if input_img is None:
# #         cv2.putText(
# #             frame, "No Palm Detected",
# #             (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
# #             1, (0, 0, 255), 2
# #         )
# #         cv2.imshow("Palm Recognition", frame)
# #         if cv2.waitKey(1) & 0xFF == ord('q'):
# #             break
# #         continue

# #     preds = model.predict(input_img, verbose=0)[0]
# #     confidence = np.max(preds)
# #     class_id = np.argmax(preds)

# #     if confidence < CONFIDENCE_THRESHOLD:
# #         label = "Unknown Palm"
# #         color = (0, 0, 255)
# #     else:
# #         label = f"{class_names[class_id]} ({confidence*100:.1f}%)"
# #         color = (0, 255, 0)

# #     x, y, w, h = bbox
# #     cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
# #     cv2.putText(frame, label, (x, y-10),
# #                 cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

# #     cv2.imshow("Palm Recognition", frame)

# #     if cv2.waitKey(1) & 0xFF == ord('q'):
# #         break

# # cap.release()
# # cv2.destroyAllWindows()



# import cv2
# import numpy as np
# import pickle
# from tensorflow.keras.models import load_model, Model
# import time

# # Configuration
# IMG_SIZE = 128
# EMBEDDING_THRESHOLD = 1.2  # Adjust based on your testing
# MIN_HAND_CONTOUR_AREA = 5000  # Minimum area for hand contour detection
# ROI_MARGIN = 20  # Margin around hand ROI

# class PalmRecognizer:
#     def __init__(self, model_path="palm_recognition_model.h5", 
#                  embeddings_path="reference_embeddings.pkl"):
#         """
#         Initialize the palm recognizer with trained model and embeddings
#         """
#         # Load the trained model
#         print("Loading model...")
#         self.model = load_model(model_path)
        
#         # Create embedding model (get features before final classification layer)
#         self.embedding_model = Model(
#             inputs=self.model.layers[0].input,
#             outputs=self.model.layers[-2].output  # Dense layer before softmax
#         )
        
#         # Load reference embeddings
#         print("Loading reference embeddings...")
#         with open(embeddings_path, 'rb') as f:
#             self.reference_embeddings = pickle.load(f)
        
#         self.class_names = list(self.reference_embeddings.keys())
#         print(f"Loaded {len(self.class_names)} classes: {self.class_names}")
        
#         # Initialize camera
#         self.cap = cv2.VideoCapture(0)
#         if not self.cap.isOpened():
#             raise Exception("Cannot open camera")
        
#         # Set camera properties
#         self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#         self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
#         # Variables for FPS calculation
#         self.prev_time = 0
#         self.fps = 0
        
#         # Recognition state
#         self.current_user = "Unknown"
#         self.confidence = 0
#         self.recognition_history = []
#         self.history_size = 10
        
#     def preprocess_for_live(self, img, IMG_SIZE=128):
#         """
#         Preprocess image for live recognition (similar to training)
#         """
#         # 1. Resize to a reasonable size for processing
#         img = cv2.resize(img, (512, 512))
        
#         # 2. Convert to grayscale
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
#         # 3. Apply Gaussian blur
#         blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
#         # 4. Adaptive thresholding (better for varying lighting)
#         thresh = cv2.adaptiveThreshold(
#             blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#             cv2.THRESH_BINARY_INV, 11, 2
#         )
        
#         # 5. Morphological operations
#         kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#         thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
#         thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        
#         # 6. Find contours for hand detection
#         contours, _ = cv2.findContours(
#             thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
#         )
        
#         if not contours:
#             return None, None, None
        
#         # Find largest contour (assumed to be hand)
#         largest_contour = max(contours, key=cv2.contourArea)
        
#         # Skip if contour is too small
#         if cv2.contourArea(largest_contour) < MIN_HAND_CONTOUR_AREA:
#             return None, None, None
        
#         # Get bounding rectangle with margin
#         x, y, w, h = cv2.boundingRect(largest_contour)
        
#         # Add margin
#         x = max(0, x - ROI_MARGIN)
#         y = max(0, y - ROI_MARGIN)
#         w = min(img.shape[1] - x, w + 2 * ROI_MARGIN)
#         h = min(img.shape[0] - y, h + 2 * ROI_MARGIN)
        
#         # Crop ROI
#         roi = gray[y:y+h, x:x+w]
        
#         if roi.size == 0:
#             return None, None, None
        
#         # 7. CLAHE enhancement
#         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#         enhanced = clahe.apply(roi)
        
#         # 8. Resize and normalize
#         resized = cv2.resize(enhanced, (IMG_SIZE, IMG_SIZE))
#         normalized = resized / 255.0
        
#         # Return preprocessed image and ROI coordinates
#         return (normalized.reshape(1, IMG_SIZE, IMG_SIZE, 1), 
#                 (x, y, w, h), 
#                 thresh)
    
#     def extract_embedding(self, preprocessed_img):
#         """
#         Extract embedding from preprocessed image
#         """
#         return self.embedding_model.predict(preprocessed_img, verbose=0)[0]
    
#     def cosine_similarity(self, vec1, vec2):
#         """
#         Calculate cosine similarity between two vectors
#         """
#         dot_product = np.dot(vec1, vec2)
#         norm1 = np.linalg.norm(vec1)
#         norm2 = np.linalg.norm(vec2)
        
#         if norm1 == 0 or norm2 == 0:
#             return 0
        
#         return dot_product / (norm1 * norm2)
    
#     def euclidean_distance(self, vec1, vec2):
#         """
#         Calculate Euclidean distance between two vectors
#         """
#         return np.linalg.norm(vec1 - vec2)
    
#     def recognize_palm(self, embedding):
#         """
#         Recognize palm by comparing with reference embeddings
#         """
#         best_match = "Unknown"
#         best_distance = float('inf')
#         best_similarity = 0
        
#         for name, ref_embedding in self.reference_embeddings.items():
#             # Calculate distance
#             distance = self.euclidean_distance(embedding, ref_embedding)
#             similarity = self.cosine_similarity(embedding, ref_embedding)
            
#             if distance < best_distance:
#                 best_distance = distance
#                 best_similarity = similarity
#                 best_match = name
        
#         # Convert distance to confidence score
#         confidence = max(0, 1 - (best_distance / EMBEDDING_THRESHOLD))
        
#         # Only accept if above threshold
#         if best_distance > EMBEDDING_THRESHOLD:
#             best_match = "Unknown"
#             confidence = 0
        
#         return best_match, confidence, best_distance
    
#     def update_recognition_history(self, recognition_result):
#         """
#         Maintain a history of recent recognitions for smoother results
#         """
#         self.recognition_history.append(recognition_result)
#         if len(self.recognition_history) > self.history_size:
#             self.recognition_history.pop(0)
        
#         # Get most common recognition from history
#         if self.recognition_history:
#             names = [r[0] for r in self.recognition_history if r[0] != "Unknown"]
#             if names:
#                 from collections import Counter
#                 most_common = Counter(names).most_common(1)[0][0]
#                 return most_common
        
#         return "Unknown"
    
#     def calculate_fps(self):
#         """
#         Calculate FPS for display
#         """
#         current_time = time.time()
#         self.fps = 1 / (current_time - self.prev_time) if self.prev_time != 0 else 0
#         self.prev_time = current_time
#         return self.fps
    
#     def draw_info(self, frame, roi_coords, recognition_result):
#         """
#         Draw recognition information on frame
#         """
#         user, confidence, distance = recognition_result
        
#         # Draw ROI rectangle
#         if roi_coords:
#             x, y, w, h = roi_coords
#             # Scale coordinates back to original frame size
#             scale_x = frame.shape[1] / 512
#             scale_y = frame.shape[0] / 512
#             x, w = int(x * scale_x), int(w * scale_x)
#             y, h = int(y * scale_y), int(h * scale_y)
            
#             # Draw rectangle
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             cv2.putText(frame, "Hand ROI", (x, y - 10), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
#         # Draw recognition info
#         info_y = 30
#         cv2.putText(frame, f"User: {user}", (10, info_y), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
#                    (0, 255, 0) if user != "Unknown" else (0, 0, 255), 2)
        
#         cv2.putText(frame, f"Confidence: {confidence:.2%}", (10, info_y + 30), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
#                    (0, 255, 0) if confidence > 0.7 else (0, 165, 255), 2)
        
#         cv2.putText(frame, f"Distance: {distance:.4f}", (10, info_y + 60), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
#         # Draw FPS
#         cv2.putText(frame, f"FPS: {self.fps:.1f}", (frame.shape[1] - 120, 30), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
#         # Draw instructions
#         cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 10), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
#         return frame
    
#     def run(self):
#         """
#         Main loop for live camera recognition
#         """
#         print("\nStarting palm recognition system...")
#         print("Place your hand in front of the camera")
#         print("Press 'q' to quit\n")
        
#         recognition_interval = 5  # Process recognition every N frames
#         frame_count = 0
        
#         while True:
#             # Capture frame
#             ret, frame = self.cap.read()
#             if not ret:
#                 print("Failed to grab frame")
#                 break
            
#             # Calculate FPS
#             self.calculate_fps()
            
#             # Process recognition every N frames
#             if frame_count % recognition_interval == 0:
#                 # Create a copy for processing
#                 process_frame = cv2.resize(frame, (512, 512))
                
#                 # Preprocess
#                 preprocessed, roi_coords, thresh = self.preprocess_for_live(process_frame)
                
#                 if preprocessed is not None:
#                     # Extract embedding
#                     embedding = self.extract_embedding(preprocessed)
                    
#                     # Recognize
#                     user, confidence, distance = self.recognize_palm(embedding)
                    
#                     # Update history
#                     self.current_user = self.update_recognition_history((user, confidence, distance))
#                     self.confidence = confidence
#                 else:
#                     self.current_user = "Unknown"
#                     self.confidence = 0
#                     roi_coords = None
            
#             # Draw information
#             frame = self.draw_info(frame, roi_coords, 
#                                   (self.current_user, self.confidence, 
#                                    distance if 'distance' in locals() else 0))
            
#             # Display
#             cv2.imshow("Palm Recognition System", frame)
            
#             # Check for quit
#             key = cv2.waitKey(1) & 0xFF
#             if key == ord('q'):
#                 print("\nExiting...")
#                 break
            
#             frame_count += 1
        
#         # Cleanup
#         self.cap.release()
#         cv2.destroyAllWindows()
    
#     def test_single_image(self, image_path):
#         """
#         Test recognition on a single image file
#         """
#         img = cv2.imread(image_path)
#         if img is None:
#             print(f"Cannot read image: {image_path}")
#             return
        
#         # Resize for processing
#         process_frame = cv2.resize(img, (512, 512))
        
#         # Preprocess
#         preprocessed, roi_coords, thresh = self.preprocess_for_live(process_frame)
        
#         if preprocessed is None:
#             print("No hand detected in image")
#             return
        
#         # Extract embedding
#         embedding = self.extract_embedding(preprocessed)
        
#         # Recognize
#         user, confidence, distance = self.recognize_palm(embedding)
        
#         print(f"\nRecognition Result:")
#         print(f"  User: {user}")
#         print(f"  Confidence: {confidence:.2%}")
#         print(f"  Distance: {distance:.4f}")
        
#         # Display images
#         fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
#         # Original
#         axes[0].imshow(cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB))
#         axes[0].set_title("Original")
#         axes[0].axis('off')
        
#         # Threshold
#         axes[1].imshow(thresh, cmap='gray')
#         axes[1].set_title("Threshold")
#         axes[1].axis('off')
        
#         # ROI
#         if roi_coords:
#             x, y, w, h = roi_coords
#             roi_display = process_frame[y:y+h, x:x+w]
#             axes[2].imshow(cv2.cvtColor(roi_display, cv2.COLOR_BGR2RGB))
#             axes[2].set_title(f"ROI - {user}")
#         else:
#             axes[2].imshow(preprocessed[0, :, :, 0], cmap='gray')
#             axes[2].set_title(f"Preprocessed - {user}")
        
#         axes[2].axis('off')
#         plt.tight_layout()
#         plt.show()

# # Main execution
# if __name__ == "__main__":
#     import argparse
#     import matplotlib.pyplot as plt
    
#     parser = argparse.ArgumentParser(description="Palm Recognition System")
#     parser.add_argument("--test", type=str, help="Test on a single image file")
#     parser.add_argument("--model", type=str, default="palm_recognition_model.h5", 
#                        help="Path to model file")
#     parser.add_argument("--embeddings", type=str, default="reference_embeddings.pkl", 
#                        help="Path to embeddings file")
    
#     args = parser.parse_args()
    
#     try:
#         recognizer = PalmRecognizer(args.model, args.embeddings)
        
#         if args.test:
#             recognizer.test_single_image(args.test)
#         else:
#             recognizer.run()
    
#     except Exception as e:
#         print(f"Error: {e}")
#         print("\nMake sure you have:")
#         print("1. Trained the model and saved it as 'palm_recognition_model.h5'")
#         print("2. Generated embeddings with 'reference_embeddings.pkl'")
#         print("3. Installed required packages: opencv-python, tensorflow, numpy")






import cv2
import numpy as np
import pickle
from tensorflow.keras.models import load_model, Model

# ---------------- CONFIG ----------------
IMG_SIZE = 128
MODEL_PATH = "palm_recognition_model.h5"
EMBEDDINGS_PATH = "reference_embeddings.pkl"
THRESHOLD = 1.2   # distance threshold for unknown
# ----------------------------------------


# -------- Palm Preprocessing (SAME AS TRAINING) --------
def preprocess_palm_image(img, IMG_SIZE=128, return_bbox=False):
    img_resized = cv2.resize(img, (512, 512))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None, None

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    roi = gray[y:y+h, x:x+w]
    roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    roi = roi / 255.0
    roi = roi.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    if return_bbox:
        return roi, (x, y, w, h)

    return roi, None


# ---------------- Load Model ----------------
print("[INFO] Loading model...")
model = load_model(MODEL_PATH)

embedding_model = Model(
    inputs=model.inputs,
    outputs=model.layers[-2].output
)

# ---------------- Load Reference Embeddings ----------------
with open(EMBEDDINGS_PATH, "rb") as f:
    reference_embeddings = pickle.load(f)

print("[INFO] Model & embeddings loaded")

# ---------------- Distance Function ----------------
def euclidean(a, b):
    return np.linalg.norm(a - b)

# ---------------- Webcam ----------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

print("[INFO] Camera started. Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    roi, bbox = preprocess_palm_image(frame, IMG_SIZE, return_bbox=True)

    label = "No Palm"
    color = (0, 0, 255)

    if roi is not None:
        emb = embedding_model.predict(roi, verbose=0)[0]

        min_dist = float("inf")
        identity = "Unknown"

        for person, ref_emb in reference_embeddings.items():
            dist = euclidean(emb, ref_emb)
            if dist < min_dist:
                min_dist = dist
                identity = person

        if min_dist < THRESHOLD:
            label = f"{identity} ({min_dist:.2f})"
            color = (0, 255, 0)
        else:
            label = f"Unknown ({min_dist:.2f})"
            color = (0, 0, 255)

        if bbox:
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    cv2.putText(
        frame, label, (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2
    )

    cv2.imshow("Palm Recognition - Live", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Camera closed.")
