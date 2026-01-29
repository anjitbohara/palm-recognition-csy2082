# # Core image processing
# import cv2
# import numpy as np
# from skimage import filters, exposure, feature, transform
# from skimage.morphology import erosion, dilation, opening, closing

# # Optional but recommended
# import matplotlib.pyplot as plt  # For visualization
# from scipy import ndimage, signal  # For advanced filtering
# import warnings  # To manage warnings





# # Core computer vision and image processing
# import cv2  # OpenCV - main library for image operations
# import numpy as np  # Numerical operations
# from skimage import filters, exposure, feature, transform, morphology
# from skimage.filters import gabor, sobel, median
# from skimage.exposure import equalize_adapthist  # CLAHE alternative
# from skimage.morphology import disk, binary_opening, binary_closing
# from skimage.color import rgb2hsv, hsv2rgb, rgb2gray

# # For mathematical operations
# import math
# from scipy import ndimage, signal
# from scipy.ndimage import gaussian_filter, convolve

# # Visualization
# import matplotlib.pyplot as plt
# from matplotlib import patches

# # Utilities
# import warnings
# from typing import Tuple, List, Optional



# # HSV-based skin segmentation typically needs:
# import cv2
# import numpy as np

# def skin_segmentation(img_hsv):
#     # Define skin color ranges in HSV
#     lower_skin = np.array([0, 20, 70], dtype=np.uint8)
#     upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
#     mask = cv2.inRange(img_hsv, lower_skin, upper_skin)
    
#     # Morphological operations to clean up
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
#     return mask



# # Pseudocode pipeline
# def preprocess_palm_image(image):
#     # 1. Convert to appropriate color space
#     img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
#     # 2. Skin segmentation
#     mask = skin_segmentation(img_hsv)
#     cv2.imshow()
    
#     # 3. Find hand contour and landmarks
#     contour = largest_contour(mask)
#     landmarks = detect_landmarks(contour)
    
#     # 4. Extract ROI using landmarks
#     roi = extract_roi(image, landmarks)
    
#     # 5. Enhance contrast
#     roi = clahe_enhancement(roi)
    
#     # 6. Apply Gabor filtering for ridge enhancement
#     enhanced = gabor_filtering(roi)
    
#     # 7. Normalize size
#     normalized = resize(enhanced, (256, 256))
    
#     return normalized




import cv2
import numpy as np
import matplotlib.pyplot as plt


def preprocess_palm_image(
    img_bgr,
    IMG_SIZE=128,
    debug=False,
    save_debug_path=None
):
    """
    Palm image preprocessing pipeline

    Input:
        img_bgr : Raw BGR image (OpenCV)
        IMG_SIZE: Output image size
        debug   : If True, show all intermediate stages
        save_debug_path : Path to save debug visualization (optional)

    Output:
        processed image (IMG_SIZE x IMG_SIZE x 1) or None
    """

    stages = {}

    # --------------------------------------------------
    # 1. Grayscale + Noise Reduction
    # --------------------------------------------------
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    stages["Grayscale + Blur"] = blurred

    # --------------------------------------------------
    # 2. Segmentation (Otsu Threshold)
    # --------------------------------------------------
    _, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    stages["Binary Mask"] = binary

    # --------------------------------------------------
    # 3. Morphological Cleaning
    # --------------------------------------------------
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    clean = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel)
    stages["Morph Clean"] = clean

    # --------------------------------------------------
    # 4. Contour Detection (ROI Extraction)
    # --------------------------------------------------
    contours, _ = cv2.findContours(
        clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None

    largest = max(contours, key=cv2.contourArea)

    if cv2.contourArea(largest) < 2000:
        return None

    x, y, w, h = cv2.boundingRect(largest)
    roi = gray[y:y+h, x:x+w]
    stages["Palm ROI"] = roi

    # --------------------------------------------------
    # 5. CLAHE Enhancement
    # --------------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(roi)
    stages["CLAHE"] = enhanced

    # --------------------------------------------------
    # 6. Resize + Normalize
    # --------------------------------------------------
    resized = cv2.resize(enhanced, (IMG_SIZE, IMG_SIZE))
    normalized = resized / 255.0
    stages["Final Input"] = resized

    output = normalized.reshape(IMG_SIZE, IMG_SIZE, 1)

    # --------------------------------------------------
    # 7. Debug Visualization (Single Image)
    # --------------------------------------------------
    if debug:
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.ravel()

        for i, (title, img) in enumerate(stages.items()):
            axes[i].imshow(img, cmap="gray")
            axes[i].set_title(title)
            axes[i].axis("off")

        plt.tight_layout()

        if save_debug_path:
            plt.savefig(save_debug_path, dpi=200)
            print(f"Debug image saved to: {save_debug_path}")

        plt.show()

    return output



img = cv2.imread("./dataset/person_1/p1_palm.jpeg")

# processed = preprocess_palm_image(
#     img,
#     IMG_SIZE=128,
#     debug=True,
#     save_debug_path="debug_palm_pipeline.png"
# )



# image_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# cv2.imshow(image_gray)
# cv2.waitKey(0)




"""
Palm Image Preprocessing Pipeline for CNN Training
===================================================
A comprehensive preprocessing pipeline that transforms raw palm images
into normalized, enhanced, and standardized format for deep learning models.

Pipeline Steps:
1. Image Acquisition and Quality Adjustment
2. ROI (Region of Interest) Extraction
3. Enhancement (CLAHE)
4. Normalization and Resizing
5. Data Augmentation (optional, for training)
"""

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import Tuple, Optional


class PalmPreprocessor:
    """
    Complete preprocessing pipeline for palm recognition images.
    Follows industry best practices for biometric image processing.
    """
    
    def __init__(self, target_size: Tuple[int, int] = (128, 128)):
        """
        Initialize the preprocessor.
        
        Args:
            target_size: Output image dimensions (height, width)
        """
        self.target_size = target_size
        
    def preprocess(self, image: np.ndarray, visualize: bool = False) -> Optional[np.ndarray]:
        """
        Complete preprocessing pipeline for a single palm image.
        
        Args:
            image: Input image (BGR format from cv2 or RGB from PIL)
            visualize: If True, displays intermediate steps
            
        Returns:
            Preprocessed image ready for CNN (normalized, grayscale, resized)
            Returns None if preprocessing fails (no hand detected)
        """
        steps = {}
        
        # Ensure image is in BGR format (OpenCV standard)
        if len(image.shape) == 2:
            # Already grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            # RGBA to BGR
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            
        steps['1_original'] = image.copy()
        
        # ====== STEP 1: Image Acquisition and Quality Adjustment ======
        
        # 1.1 Initial Cleaning - Remove noise using Gaussian Blur
        denoised = cv2.GaussianBlur(image, (5, 5), 0)
        steps['2_denoised'] = denoised.copy()
        
        # 1.2 Grayscale Conversion (for final output, but use color for ROI detection)
        gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
        steps['3_grayscale'] = gray.copy()
        
        # ====== STEP 2: ROI (Region of Interest) Extraction ======
        
        # 2.1 Segmentation using skin color detection (more robust than simple thresholding)
        roi_mask = self._extract_hand_mask(denoised)
        steps['4_hand_mask'] = roi_mask.copy()
        
        if roi_mask is None:
            print("❌ No hand detected in image")
            return None
        
        # 2.2 Morphological Operations to clean the mask
        cleaned_mask = self._clean_mask(roi_mask)
        steps['5_cleaned_mask'] = cleaned_mask.copy()
        
        # 2.3 Boundary Tracking - Find largest contour (hand)
        contours, _ = cv2.findContours(
            cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if len(contours) == 0:
            print("❌ No contours found")
            return None
        
        # Find largest contour (should be the hand)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 2.4 Extract ROI - Get bounding box of the palm
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Add small padding to ensure we capture the entire palm
        padding = 10
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)
        
        # Crop the palm region from grayscale image
        palm_roi = gray[y:y+h, x:x+w]
        steps['6_roi_extracted'] = palm_roi.copy()
        
        # ====== STEP 3: Enhancement ======
        
        # 3.1 Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # This enhances the visibility of palm lines, ridges, and texture
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(palm_roi)
        steps['7_enhanced_clahe'] = enhanced.copy()
        
        # ====== STEP 4: Normalization and Resizing ======
        
        # 4.1 Resize to target dimensions
        resized = cv2.resize(enhanced, self.target_size, interpolation=cv2.INTER_AREA)
        steps['8_resized'] = resized.copy()
        
        # 4.2 Pixel Normalization - Scale to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        steps['9_normalized'] = normalized.copy()
        
        # Reshape for CNN input (add channel dimension)
        preprocessed = normalized.reshape(self.target_size[0], self.target_size[1], 1)
        
        if visualize:
            self._visualize_steps(steps)
        
        return preprocessed
    
    def _extract_hand_mask(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract hand region using skin color detection in HSV space.
        More robust than simple grayscale thresholding.
        """
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define skin color range (works for various skin tones)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Create binary mask
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        return mask
    
    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply morphological operations to clean the binary mask.
        Removes small noise and fills holes.
        """
        # Create morphological kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        
        # Opening: removes small noise
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Closing: fills small holes
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        return cleaned
    
    def _visualize_steps(self, steps: dict):
        """
        Visualize all preprocessing steps in a grid.
        """
        n_steps = len(steps)
        cols = 3
        rows = (n_steps + cols - 1) // cols
        
        plt.figure(figsize=(15, 5 * rows))
        
        for idx, (title, img) in enumerate(steps.items(), 1):
            plt.subplot(rows, cols, idx)
            
            # Display image
            if len(img.shape) == 2:
                plt.imshow(img, cmap='gray')
            else:
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            plt.title(title.replace('_', ' ').title())
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()


# ====== STEP 5: Data Augmentation (For Training) ======

def create_augmentation_pipeline():
    """
    Create data augmentation pipeline using Keras ImageDataGenerator.
    This helps prevent overfitting on small datasets.
    
    Returns:
        ImageDataGenerator configured with appropriate augmentations
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    datagen = ImageDataGenerator(
        rotation_range=15,           # Random rotation ±15 degrees
        width_shift_range=0.1,       # Random horizontal shift
        height_shift_range=0.1,      # Random vertical shift
        shear_range=0.1,             # Shear transformation
        zoom_range=0.1,              # Random zoom
        horizontal_flip=True,        # Horizontal flip
        brightness_range=[0.8, 1.2], # Random brightness adjustment
        fill_mode='nearest'          # Fill strategy for transformations
    )
    
    return datagen


def preprocess_dataset(input_dir: str, output_dir: str, target_size: Tuple[int, int] = (128, 128)):
    """
    Preprocess an entire dataset of palm images.
    
    Args:
        input_dir: Directory containing raw palm images (organized by class)
        output_dir: Directory to save preprocessed images
        target_size: Target image dimensions
    """
    import os
    from pathlib import Path
    
    preprocessor = PalmPreprocessor(target_size=target_size)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Statistics
    processed_count = 0
    failed_count = 0
    
    # Process each class directory
    for class_name in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_name)
        
        if not os.path.isdir(class_path):
            continue
        
        # Create output class directory
        output_class_path = os.path.join(output_dir, class_name)
        Path(output_class_path).mkdir(parents=True, exist_ok=True)
        
        print(f"\n📁 Processing class: {class_name}")
        
        # Process each image in the class
        for img_file in os.listdir(class_path):
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            
            img_path = os.path.join(class_path, img_file)
            
            try:
                # Read image
                img = cv2.imread(img_path)
                
                if img is None:
                    print(f"   ⚠️  Failed to read: {img_file}")
                    failed_count += 1
                    continue
                
                # Preprocess
                preprocessed = preprocessor.preprocess(img, visualize=False)
                
                if preprocessed is None:
                    print(f"   ⚠️  No hand detected: {img_file}")
                    failed_count += 1
                    continue
                
                # Save preprocessed image
                output_path = os.path.join(output_class_path, img_file)
                
                # Convert back to uint8 for saving
                img_to_save = (preprocessed[:, :, 0] * 255).astype(np.uint8)
                cv2.imwrite(output_path, img_to_save)
                
                processed_count += 1
                
            except Exception as e:
                print(f"   ❌ Error processing {img_file}: {str(e)}")
                failed_count += 1
        
        print(f"   ✅ Processed {processed_count} images")
    
    print(f"\n{'='*50}")
    print(f"✅ Successfully processed: {processed_count} images")
    print(f"❌ Failed: {failed_count} images")
    print(f"📊 Success rate: {processed_count / (processed_count + failed_count) * 100:.2f}%")


# ====== Example Usage ======

if __name__ == "__main__":
    
    # Example 1: Preprocess a single image with visualization
    print("=" * 60)
    print("Example 1: Single Image Preprocessing with Visualization")
    print("=" * 60)
    
    # Load a sample image
    sample_image_path = "./dataset/person_1/p1_palm.jpeg"  # Replace with your image path
    
    import os
    if os.path.exists(sample_image_path):
        img = cv2.imread(sample_image_path)
        
        preprocessor = PalmPreprocessor(target_size=(128, 128))
        preprocessed = preprocessor.preprocess(img, visualize=True)
        
        if preprocessed is not None:
            print(f"✅ Preprocessing successful!")
            print(f"   Output shape: {preprocessed.shape}")
            print(f"   Value range: [{preprocessed.min():.3f}, {preprocessed.max():.3f}]")
    else:
        print(f"⚠️  Sample image not found: {sample_image_path}")
    
    # Example 2: Preprocess entire dataset
    print("\n" + "=" * 60)
    print("Example 2: Batch Dataset Preprocessing")
    print("=" * 60)
    
    input_directory = "dataset"      # Replace with your input directory
    output_directory = "preprocessed_dataset" # Replace with your output directory
    
    if os.path.exists(input_directory):
        preprocess_dataset(
            input_dir=input_directory,
            output_dir=output_directory,
            target_size=(128, 128)
        )
    else:
        print(f"⚠️  Input directory not found: {input_directory}")
        print("   Please create the directory structure:")
        print("   raw_palm_dataset/")
        print("   ├── person1/")
        print("   │   ├── image1.jpg")
        print("   │   └── image2.jpg")
        print("   └── person2/")
        print("       ├── image1.jpg")
        print("       └── image2.jpg")
    
    # Example 3: Create augmentation pipeline
    print("\n" + "=" * 60)
    print("Example 3: Data Augmentation Pipeline")
    print("=" * 60)
    
    try:
        datagen = create_augmentation_pipeline()
        print("✅ Data augmentation pipeline created successfully!")
        print("   Augmentation parameters:")
        print(f"   - Rotation range: ±15°")
        print(f"   - Width/Height shift: ±10%")
        print(f"   - Shear range: 0.1")
        print(f"   - Zoom range: 0.1")
        print(f"   - Horizontal flip: Enabled")
        print(f"   - Brightness range: [0.8, 1.2]")
    except ImportError:
        print("⚠️  TensorFlow/Keras not installed. Install with:")
        print("   pip install tensorflow")