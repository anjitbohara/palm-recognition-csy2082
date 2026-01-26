import cv2
import numpy as np
from pathlib import Path



class PalmPreprocessor:
    """
    Preprocesses palm images for deep learning.
    Extracts ROI based on palm segmentation using contour detection.
    """
    
    def __init__(self, roi_size=(276, 276), target_size=(128, 128)):
        self.roi_size = roi_size
        self.target_size = target_size
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5, 5))

    def visualize_preprocessing(self, image_path):
        """
        Visualize all preprocessing steps including contours.
        Perfect for your assignment report!
        """
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        
        # Process image through each step
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # For display
        enhanced = self.clahe.apply(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        denoised = cv2.medianBlur(enhanced, 5, 0)
        _, binary = cv2.threshold(denoised, 80, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, 
                                    cv2.CHAIN_APPROX_SIMPLE)
        
        # Create contour visualization
        contour_img = image.copy()
        cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
        
        # Highlight largest contour and centroid
        if contours:
            largest = max(contours, key=cv2.contourArea)
            cv2.drawContours(contour_img, [largest], -1, (255, 0, 0), 3)
            
            M = cv2.moments(largest)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(contour_img, (cx, cy), 8, (0, 0, 255), -1)
                cv2.circle(contour_img, (cx, cy), 10, (0, 0, 255), 2)
        
        # Extract ROI
        roi = self._extract_roi(denoised, contours)
        if roi is not None:
            resized = cv2.resize(roi, self.target_size)
        
        # Create subplot visualization
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        # Row 1: Original processing pipeline
        axes[0, 0].imshow(gray)
        axes[0, 0].set_title("1. Original Image")
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(enhanced, cmap='gray')
        axes[0, 1].set_title("2. CLAHE Enhanced")
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(denoised, cmap='gray')
        axes[0, 2].set_title("3. Denoised")
        axes[0, 2].axis('off')
        
        axes[0, 3].imshow(binary, cmap='gray')
        axes[0, 3].set_title("4. Binary Threshold")
        axes[0, 3].axis('off')
        
        # Row 2: Contour and ROI results
        axes[1, 0].imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title("5. Contour Detection\nGreen=All, Blue=Largest, Red=Centroid")
        axes[1, 0].axis('off')
        
        if roi is not None:
            axes[1, 1].imshow(roi, cmap='gray')
            axes[1, 1].set_title(f"6. ROI Crop\nSize: {roi.shape}")
            axes[1, 1].axis('off')
            
            axes[1, 2].imshow(resized, cmap='gray')
            axes[1, 2].set_title(f"7. Resized\n{self.target_size}")
            axes[1, 2].axis('off')
            
            # Show normalized version
            normalized = resized.astype(np.float32) / 255.0
            axes[1, 3].imshow(normalized, cmap='gray')
            axes[1, 3].set_title("8. Normalized (0-1)")
            axes[1, 3].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"preprocess_visualization_{Path(image_path).stem}.png", dpi=150)
        plt.show()
        
        return roi
    
    def preprocess(self, image_path):
        """Preprocess a single palm image."""
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                return None
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Enhance contrast
            enhanced = self.clahe.apply(gray)
            
            # Reduce noise
            denoised = cv2.medianBlur(enhanced, 5)
            
            # Segment palm
            _, binary = cv2.threshold(denoised, 80, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, 
                                          cv2.CHAIN_APPROX_SIMPLE)
            
            # Extract ROI from largest contour
            roi = self._extract_roi(denoised, contours)
            
            if roi is None or roi.size == 0:
                return None
            
            # Resize and normalize
            resized = cv2.resize(roi, self.target_size)
            normalized = resized.astype(np.float32) / 255.0

            cv2.imshow("Original", image)
            cv2.imshow("Grayscale", gray)
            cv2.imshow("Enhanced", enhanced)
            cv2.imshow("denoised", denoised)
            
            # ... etc
            cv2.waitKey(0)  # Press any key to continue
            
            return normalized
            
        except Exception as e:
            print(f"Error preprocessing {image_path}: {e}")
            return None
    
    def _extract_roi(self, image, contours):
        """Extract ROI based on largest contour centroid."""
        if not contours:
            return None
        
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        
        if M["m00"] == 0:
            return None
        
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        roi_w, roi_h = self.roi_size
        x = max(0, cx - roi_w // 2)
        y = max(0, cy - roi_h // 2)
        x_end = min(image.shape[1], x + roi_w)
        y_end = min(image.shape[0], y + roi_h)
        
        return image[y:y_end, x:x_end]