# Palm Recognition System – Development Report


## Student Information
- **Student Name**: Anjit Bohara
- **Student Number**:24825331
- **Module**: 
- **Assignment**:
- **Submission Date**: 30/01/2026
- **Github_link**: `https://github.com/anjitbohara/palm-recognition-csy2082`


## 1. Introduction
This project focuses on designing and implementing a **palm recognition system** using computer vision and deep learning techniques. The aim is to recognise individuals based on unique palm characteristics captured using a standard RGB camera. The system was developed incrementally, starting from a basic image classification approach and progressively evolving into a robust, real-time palm recognition system, fully aligned with the assessment requirements.

The implementation uses **Python**, **OpenCV**, and **TensorFlow/Keras**, and supports both offline model training and live camera-based recognition.

---

## 2. Problem Analysis and Feature Selection

### 2.1 Choice of Biometric Feature
Palmprint recognition relies primarily on **palm texture features**, including principal lines, wrinkles, creases, and fine skin patterns. These features are:
- Highly distinctive across individuals
- Stable over time
- Clearly visible using standard RGB cameras

Compared to fingerprints, palmprints offer a **larger surface area**, enabling richer feature extraction. Unlike palm vein recognition, no specialised infrared sensors are required. Therefore, palm texture-based recognition was selected as it provides a strong balance between **accuracy, accessibility, and implementation feasibility**, making it well suited for a coursework-based biometric system.

---

## 3. Initial Baseline Approach

### 3.1 Early Method
The project initially began with a simple approach:
- Capture palm images
- Resize images to a fixed size
- Train a CNN classifier directly on full images

This baseline helped validate that CNNs could learn palm-related patterns.

### 3.2 Observed Limitations
Early experiments revealed several issues:
- The model learned background features instead of palm texture
- Inconsistent hand placement caused unstable predictions
- Accuracy dropped significantly during live testing

These limitations motivated the need for **controlled data capture and improved preprocessing**.

---

## 4. Dataset Development

### 4.1 Data Collection Strategy
A custom live camera capture tool was developed to ensure dataset consistency. The capture interface includes:
- A **green guide rectangle** to instruct correct palm placement
- A **blue preview box** indicating the exact region saved

Only the palm region inside the blue box is stored. This approach ensures:
- Consistent scale and orientation
- Reduced background noise
- Improved intra-class similarity

The dataset consists of palm images from **four individuals**, stored in separate folders. All images were collected with consent and anonymised using folder-based labels.

---

## 5. Image Preprocessing Pipeline

Palm images are sensitive to lighting variation and noise. To address this, a multi-stage preprocessing pipeline was designed and applied consistently during both training and live testing:

1. Grayscale conversion – reduces dimensionality and removes colour dependency
2. Contrast enhancement using CLAHE – improves visibility of palm lines
3. Median filtering – reduces noise while preserving edges
4. Binary thresholding – separates palm from background
5. Contour detection – isolates the palm region
6. ROI extraction – focuses on the central palm area
7. Resizing and normalisation – ensures consistent model input

This pipeline ensures that the CNN focuses on **biometrically relevant features** rather than background artefacts.

---

## 6. Model Design and Justification

### 6.1 CNN Architecture
A Convolutional Neural Network (CNN) was chosen due to its proven effectiveness in image-based biometric recognition. CNNs automatically learn hierarchical features such as:
- Edges and ridges (low-level)
- Line intersections (mid-level)
- Palm texture patterns (high-level)

The architecture includes:
- Three convolutional layers with increasing filter depth
- Max-pooling layers for spatial reduction
- Fully connected layers for classification
- Dropout to prevent overfitting

The model uses **categorical cross-entropy loss** and the **Adam optimiser**, selected for stable convergence in multi-class classification tasks.

---

## 7. Embedding-Based Palm Recognition

### 7.1 Motivation
Pure classification forces every input to be assigned to a known class, which is unsuitable for biometric systems. To enable unknown-user detection, the system was extended using **feature embeddings**.

### 7.2 Embedding Strategy
- The final softmax layer is removed
- A dense layer produces a 128-dimensional embedding
- Each person’s reference embedding is computed as the mean of their samples

Live palm embeddings are compared with reference embeddings using **Euclidean distance**. If the distance exceeds a threshold, the palm is classified as *Unknown*.

This approach mirrors real-world biometric systems and significantly reduces false acceptances.

---

## 8. Training and Evaluation

The dataset was split into training, validation, and testing sets. Model performance was evaluated using:
- Accuracy
- Precision
- Recall
- Confusion matrix

These metrics were chosen to assess both overall performance and class-level reliability, which is critical in biometric recognition tasks.

---

## 9. Live Camera Testing

The live recognition system strictly follows the same ROI-based setup used during data collection. Only the palm region inside the blue box is:
- Preprocessed
- Converted into embeddings
- Compared with stored references

This design ensures **training–testing consistency**, which is essential for reliable biometric recognition.

---

## 10. Ethical Considerations and Limitations

All data was collected with informed consent and anonymised. The dataset is relatively small and limited in demographic diversity, which may introduce bias. Performance is also sensitive to lighting conditions and palm positioning.

Future work would require a larger and more diverse dataset to improve fairness and generalisation.

---

## 11. Conclusion

This project demonstrates a complete palm recognition pipeline, evolving from a basic image classifier to a real-time biometric recognition system. Each design decision was motivated by experimental observations and justified through biometric principles. The final system satisfies the assessment objectives and demonstrates a clear progression of learning and implementation.

---

## 12. Future Improvements

- Automatic palm detection without fixed ROI
- Larger and more diverse datasets
- FAR and FRR evaluation metrics
- Deployment as a desktop or web application

---

**End of Report**

