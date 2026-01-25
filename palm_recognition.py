"""
This system implements a deep learning palm recognition classifier
that can identify 4 registered individuals plus unknown palms.

Features:
- Dataset collection and augmentation
- Custom CNN architecture built from scratch
- Training with comprehensive evaluation
- Live camera testing
- Bias and ethics analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
import pickle
import json
from datetime import datetime
from pathlib import Path

# TensorFlow/Keras imports for Deep Learning
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, optimizers, callbacks
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    print(f"✓ TensorFlow {tf.__version__} imported successfully")
except ImportError as e:
    print(f"✗ TensorFlow import failed: {e}")
    exit(1)

# Machine Learning/Sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix, 
                            accuracy_score, precision_recall_fscore_support)
from sklearn.preprocessing import LabelEncoder

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("="*80)
print("PALM RECOGNITION SYSTEM - CSY2082 Assignment 2")
print("="*80)
print(f"TensorFlow version: {tf.__version__}")
print(f"NumPy version: {np.__version__}")
print("="*80 + "\n")


class PalmDatasetManager:
    """
    Manages palm image dataset collection, loading, and preprocessing.
    
    Features:
    - Webcam capture with visual guides
    - Automatic directory structure
    - Image preprocessing and augmentation
    - Dataset statistics and visualization
    """
    
    def __init__(self, img_size=(224, 224), data_dir='palm_dataset'):
        """
        Initialize dataset manager.
        
        Args:
            img_size (tuple): Target image size (height, width)
            data_dir (str): Directory to store dataset
        """
        self.img_size = img_size
        self.data_dir = Path(data_dir)
        self.classes = ['person_1', 'person_2', 'person_3', 'person_4', 'unknown']
        
    def create_directory_structure(self):
        """Create organized directory structure for dataset."""
        self.data_dir.mkdir(exist_ok=True)
        
        for person in self.classes:
            person_dir = self.data_dir / person
            person_dir.mkdir(exist_ok=True)
        
        print(f"✓ Directory structure created at: {self.data_dir}")
        print(f"✓ Classes: {', '.join(self.classes)}")
        return self.classes
    
    def capture_images(self, person_name, num_images=2, show_preview=True):
        """
        Capture palm images from webcam with visual guidance.
        
        Args:
            person_name (str): Name of the person/class
            num_images (int): Number of images to capture
            show_preview (bool): Show camera preview
            
        Returns:
            int: Number of images successfully captured
        """
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("✗ Error: Cannot access camera!")
            print("Please check camera connection and permissions.")
            return 0
        
        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        count = 0
        person_dir = self.data_dir / person_name
        
        print(f"\n{'='*80}")
        print(f"📷 CAPTURING IMAGES FOR: {person_name.upper()}")
        print(f"{'='*80}")
        print(f"Target: {num_images} images")
        print("\n💡 TIPS FOR BEST RESULTS:")
        print("  • Vary hand position (center, left, right, up, down)")
        print("  • Rotate hand slightly (different angles)")
        print("  • Change distance (closer/farther)")
        print("  • Try different lighting if possible")
        print("  • Keep palm open and fingers spread")
        print("\n🎮 CONTROLS:")
        print("  • Press 's' or SPACE to save current frame")
        print("  • Press 'q' or ESC to quit early")
        print(f"{'='*80}\n")
        
        while count < num_images:
            ret, frame = cap.read()
            if not ret:
                print("✗ Error reading frame from camera")
                break
            
            display_frame = frame.copy()
            h, w = display_frame.shape[:2]
            
            # Draw guide rectangle
            rect_w, rect_h = w // 2, h // 2
            x1, y1 = (w - rect_w) // 2, (h - rect_h) // 2
            x2, y2 = x1 + rect_w, y1 + rect_h
            
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Progress bar
            progress = count / num_images
            bar_width = w - 40
            bar_filled = int(bar_width * progress)
            cv2.rectangle(display_frame, (20, h - 40), (20 + bar_filled, h - 20), 
                         (0, 255, 0), -1)
            cv2.rectangle(display_frame, (20, h - 40), (20 + bar_width, h - 20), 
                         (255, 255, 255), 2)
            
            # Text overlay
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(display_frame, f"Capturing: {person_name}", 
                       (20, 40), font, 1.2, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Progress: {count}/{num_images}", 
                       (20, 80), font, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, "Place palm in GREEN BOX", 
                       (20, 120), font, 0.8, (255, 255, 255), 2)
            cv2.putText(display_frame, "Press 's' or SPACE to save | 'q' to quit", 
                       (20, h - 60), font, 0.7, (255, 255, 255), 2)
            
            if show_preview:
                cv2.imshow('Palm Recognition - Data Capture', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            # Save on 's' or SPACE
            if key == ord('s') or key == ord(' '):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                img_path = person_dir / f'{person_name}_{timestamp}.jpg'
                cv2.imwrite(str(img_path), frame)
                count += 1
                print(f"✓ Saved: {count}/{num_images} - {img_path.name}")
            
            # Quit on 'q' or ESC
            elif key == ord('q') or key == 27:
                print(f"\n⚠️  Capture interrupted by user at {count}/{num_images}")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n{'='*80}")
        print(f"✓ CAPTURE COMPLETE: {count} images saved for {person_name}")
        print(f"{'='*80}\n")
        
        return count
    
    def preprocess_image(self, img_path):
        """
        Preprocess a single image for the model.
        
        Args:
            img_path (Path): Path to image file
            
        Returns:
            np.array: Preprocessed image or None if failed
        """
        img = cv2.imread(str(img_path))
        if img is None:
            return None
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to target size
        img = cv2.resize(img, self.img_size)
        
        # Normalize to [0, 1]
        img = img.astype('float32') / 255.0
        
        return img
    
    def load_dataset(self, verbose=True):
        """
        Load all images from dataset directory.
        
        Args:
            verbose (bool): Print detailed loading information
            
        Returns:
            tuple: (X, y, class_counts) - images, labels, and count per class
        """
        if not self.data_dir.exists():
            print(f"✗ Dataset directory not found: {self.data_dir}")
            return None, None, None
        
        X = []
        y = []
        class_counts = {}
        
        print(f"\n{'='*80}")
        print("LOADING DATASET")
        print(f"{'='*80}")
        
        for person in sorted(self.data_dir.iterdir()):
            if not person.is_dir():
                continue
            
            person_name = person.name
            count = 0
            
            for img_file in sorted(person.iterdir()):
                if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                    continue
                
                img = self.preprocess_image(img_file)
                if img is not None:
                    X.append(img)
                    y.append(person_name)
                    count += 1
            
            class_counts[person_name] = count
            if verbose:
                status = "✓" if count > 0 else "✗"
                print(f"{status} {person_name:15s}: {count:4d} images")
        
        if len(X) == 0:
            print("\n✗ NO IMAGES FOUND!")
            print("Please run data capture first.")
            return None, None, None
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"\n{'='*80}")
        print("DATASET SUMMARY")
        print(f"{'='*80}")
        print(f"Total images:    {len(X)}")
        print(f"Image shape:     {X[0].shape}")
        print(f"Number of classes: {len(np.unique(y))}")
        print(f"Classes:         {', '.join(sorted(np.unique(y)))}")
        print(f"{'='*80}\n")
        
        return X, y, class_counts
    
    def visualize_samples(self, X, y, num_samples=20, save_path='dataset_samples.png'):
        """
        Visualize random samples from dataset.
        
        Args:
            X (np.array): Images
            y (np.array): Labels
            num_samples (int): Number of samples to show
            save_path (str): Where to save visualization
        """
        indices = np.random.choice(len(X), min(num_samples, len(X)), replace=False)
        
        fig, axes = plt.subplots(4, 5, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, idx in enumerate(indices):
            axes[i].imshow(X[idx])
            axes[i].set_title(f"{y[idx]}", fontsize=10)
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(len(indices), len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('Random Dataset Samples', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Samples visualization saved: {save_path}")
        plt.close()




# MAIN EXECUTION
if __name__ == "__main__":
    print("\nThis is the main palm recognition system module.")
    print("Import this module or run the complete pipeline script.")
    
    images_path = 'palm_dataset/person_1'
    image_name = 'p1_img1.jpeg'
    image_pathp1_img1 = os.path.join(images_path, image_name)

    image = cv2.imread(image_pathp1_img1)
    cv2.imshow('person1 image1', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()