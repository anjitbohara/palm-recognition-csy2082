import cv2
from pathlib import Path
from datetime import datetime


class PalmDatasetManager:   
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

    def capture_images(self, person_name, num_images=80, show_preview=True):
        """
        Capture ONLY the palm region inside the green rectangle.
        Blue box shows what will be captured.
        """
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("✗ Error: Cannot access camera!")
            return 0
        
        count = 0
        person_dir = self.data_dir / person_name
        
        while count < num_images:
            ret, frame = cap.read()
            if not ret:
                break
            
            display_frame = frame.copy()
            h, w = display_frame.shape[:2]
            
            # Define ROI rectangle (green box)
            rect_w, rect_h = w // 2, h // 2
            x1, y1 = (w - rect_w) // 2, (h - rect_h) // 2
            x2, y2 = x1 + rect_w, y1 + rect_h
            
            # Draw green guide rectangle
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Create preview of what will be saved (blue box)
            preview_size = 100  # Size of blue preview box
            preview_x, preview_y = 10, 10  # Position in top-left corner
            
            # Extract and resize for preview
            palm_roi = frame[y1:y2, x1:x2]
            palm_preview = cv2.resize(palm_roi, (preview_size, preview_size))
            
            # Place preview in blue box
            display_frame[preview_y:preview_y+preview_size, 
                        preview_x:preview_x+preview_size] = palm_preview
            
            # Draw blue box around preview
            cv2.rectangle(display_frame, 
                        (preview_x, preview_y), 
                        (preview_x+preview_size, preview_y+preview_size), 
                        (255, 0, 0), 2)
            
            # Show instructions
            cv2.putText(display_frame, f"Capture {count}/{num_images} - {person_name}", 
                    (20, preview_y + preview_size + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, "Press 's' to save palm", 
                    (20, preview_y + preview_size + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, "Blue box: What will be saved", 
                    (20, preview_y + preview_size + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if show_preview:
                cv2.imshow('Palm Capture', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s') or key == ord(' '):
                # Extract and resize palm ROI to target size
                palm_resized = cv2.resize(palm_roi, self.img_size)
                
                # Save the palm image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                img_path = person_dir / f'{person_name}_{timestamp}.jpg'
                cv2.imwrite(str(img_path), palm_resized)
                
                count += 1
                print(f"✓ Saved palm image: {img_path.name} ({self.img_size[0]}x{self.img_size[1]})")
            
            elif key == ord('q') or key == 27:
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"Captured {count} palm images for {person_name}")
        return count
    


if __name__ =='__main__':
    PalmDatasetManager().capture_images('person_1')