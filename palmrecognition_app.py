# # import streamlit as st
# # import os
# # import random
# # import cv2
# # import numpy as np
# # from tensorflow.keras.models import load_model
# # from PIL import Image

# # # ---------------- CONFIG ----------------
# # DATASET_PATH = "dataset"
# # MODEL_PATH = "palm_recognition_model.h5"
# # IMG_SIZE = 128
# # CONFIDENCE_THRESHOLD = 0.6
# # # ----------------------------------------

# # st.set_page_config(page_title="Palm Recognition Demo", layout="centered")
# # st.title("✋ Palm Recognition System")
# # st.write("Random dataset image testing using CNN")

# # # Load model
# # @st.cache_resource
# # def load_cnn():
# #     return load_model(MODEL_PATH)

# # model = load_cnn()

# # # Get valid class folders
# # class_names = sorted([
# #     d for d in os.listdir(DATASET_PATH)
# #     if os.path.isdir(os.path.join(DATASET_PATH, d))
# # ])

# # st.sidebar.header("Settings")
# # st.sidebar.write("Classes detected:")
# # st.sidebar.write(class_names)

# # # Button
# # if st.button("🎲 Test Random Image"):
# #     # Pick random class and image
# #     true_label = random.choice(class_names)
# #     class_path = os.path.join(DATASET_PATH, true_label)

# #     img_name = random.choice(os.listdir(class_path))
# #     img_path = os.path.join(class_path, img_name)

# #     # Load & preprocess
# #     img = cv2.imread(img_path)
# #     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# #     img_resized = cv2.resize(img_gray, (IMG_SIZE, IMG_SIZE))
# #     img_norm = img_resized / 255.0
# #     img_input = img_norm.reshape(1, IMG_SIZE, IMG_SIZE, 1)

# #     # Predict
# #     preds = model.predict(img_input)[0]
# #     confidence = np.max(preds)
# #     pred_index = np.argmax(preds)

# #     if confidence < CONFIDENCE_THRESHOLD:
# #         predicted_label = "Unknown"
# #     else:
# #         predicted_label = class_names[pred_index]

# #     # Display image
# #     st.image(
# #         Image.fromarray(img_gray),
# #         caption=f"True: {true_label}",
# #         width="stretch"

# #     )

# #     # Results
# #     st.subheader("Prediction Result")
# #     st.write(f"**Predicted:** {predicted_label}")
# #     st.write(f"**Confidence:** {confidence:.2f}")

# #     if predicted_label == true_label:
# #         st.success("✅ Correct Prediction")
# #     else:
# #         st.warning("⚠️ Incorrect or Unknown Prediction")





# import os
# import cv2
# import pickle
# import numpy as np
# from PIL import Image
# import streamlit as st
# from tensorflow.keras.models import load_model, Model


# # ---------------- CONFIG ----------------
# DATASET_PATH = "dataset"
# MODEL_PATH = "palm_recognition_model.h5"
# IMG_SIZE = 128
# CONFIDENCE_THRESHOLD = 0.6
# # ----------------------------------------


# model = load_model(MODEL_PATH)

# embedding_model = Model(
#     inputs=model.input,
#     outputs=model.layers[-2].output
# )

# with open("reference_embeddings.pkl", "rb") as f:
#     reference_embeddings = pickle.load(f)



# st.set_page_config(page_title="Palm Recognition Demo", layout="centered")
# st.title("✋ Palm Recognition System")
# st.write("Upload a palm image to test the trained CNN")

# # Load model
# @st.cache_resource
# def load_cnn():
#     return load_model(MODEL_PATH)

# model = load_cnn()

# # Get class names
# class_names = sorted([
#     d for d in os.listdir(DATASET_PATH)
#     if os.path.isdir(os.path.join(DATASET_PATH, d))
# ])

# st.sidebar.header("Model Info")
# st.sidebar.write("Registered palms:")
# st.sidebar.write(class_names)
# st.sidebar.write(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")

# # File uploader
# uploaded_file = st.file_uploader(
#     "Upload a palm image",
#     type=["jpg", "jpeg", "png"]
# )

# if uploaded_file is not None:
#     # Load image
#     image = Image.open(uploaded_file).convert("RGB")
#     st.image(image, caption="Uploaded Image", width="stretch")

#     # Preprocess
#     img = np.array(image)
#     img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     img_resized = cv2.resize(img_gray, (IMG_SIZE, IMG_SIZE))
#     img_norm = img_resized / 255.0
#     img_input = img_norm.reshape(1, IMG_SIZE, IMG_SIZE, 1)

#     # Predict
#     # predictions = model.predict(img_input)[0]
#     # confidence = np.max(predictions)


#     EMBEDDING_THRESHOLD = 1.2

#     embedding = embedding_model.predict(img_input)[0]

#     distances = {
#         person: np.linalg.norm(embedding - ref_emb)
#         for person, ref_emb in reference_embeddings.items()
#     }

#     best_match = min(distances, key=distances.get)
#     min_distance = distances[best_match]

#     if min_distance > EMBEDDING_THRESHOLD:
#         predicted_label = "Unknown"
#     else:
#         predicted_label = best_match



#     # label_index = np.argmax(predictions)

#     # if confidence < CONFIDENCE_THRESHOLD:
#     #     predicted_label = "Unknown"
#     # else:
#     #     predicted_label = class_names[label_index]

#     # Output
#     st.subheader("Prediction Result")
#     st.write(f"**Predicted:** {predicted_label}")
#     # st.write(f"**Confidence:** {confidence:.2f}")
#     st.write(f"**best_match:** {best_match}")

#     if predicted_label == "Unknown":
#         st.warning("⚠️ Palm not recognised (Unknown)")
#     else:
#         st.success("✅ Palm recognised")




import os
import cv2
import pickle
import numpy as np
from PIL import Image
import streamlit as st
from tensorflow.keras.models import load_model, Model

# ---------------- CONFIG ----------------
DATASET_PATH = "dataset"
MODEL_PATH = "palm_recognition_model.h5"
IMG_SIZE = 128
CONFIDENCE_THRESHOLD = 0.6
EMBEDDING_THRESHOLD = 5.0
# ----------------------------------------

st.set_page_config(page_title="Palm Recognition Demo", layout="centered")
st.title("✋ Palm Recognition System")
st.write("Upload a palm image to test the trained CNN")

# Load model
@st.cache_resource
def load_cnn():
    return load_model(MODEL_PATH)

# Load model and reference embeddings
model = load_cnn()

# Fix: Use model.layers[0].input instead of model.input
embedding_model = Model(
    inputs=model.layers[0].input,  # Fixed here
    outputs=model.layers[-2].output  # Dense(128)
)

# Load reference embeddings
with open("reference_embeddings.pkl", "rb") as f:
    reference_embeddings = pickle.load(f)

# Get class names
class_names = sorted([
    d for d in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, d))
])

st.sidebar.header("Model Info")
st.sidebar.write("Registered palms:")
st.sidebar.write(class_names)
st.sidebar.write(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
st.sidebar.write(f"Embedding threshold: {EMBEDDING_THRESHOLD}")

# File uploader
uploaded_file = st.file_uploader(
    "Upload a palm image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width="stretch")

    # Preprocess
    img = np.array(image)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_resized = cv2.resize(img_gray, (IMG_SIZE, IMG_SIZE))
    img_norm = img_resized / 255.0
    img_input = img_norm.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    # Get embedding and compare with references
    embedding = embedding_model.predict(img_input)[0]

    distances = {
        person: np.linalg.norm(embedding - ref_emb)
        for person, ref_emb in reference_embeddings.items()
    }

    best_match = min(distances, key=distances.get)
    min_distance = distances[best_match]

    if min_distance > EMBEDDING_THRESHOLD:
        predicted_label = "Unknown"
    else:
        predicted_label = best_match

    # Output
    st.subheader("Prediction Result")
    st.write(f"**Predicted:** {predicted_label}")
    st.write(f"**Best match:** {best_match}")
    st.write(f"**Distance:** {min_distance:.4f}")

    # Show all distances
    with st.expander("View all distances"):
        for person, dist in distances.items():
            st.write(f"{person}: {dist:.4f}")

    if predicted_label == "Unknown":
        st.warning("⚠️ Palm not recognised (Unknown)")
    else:
        st.success("✅ Palm recognised")