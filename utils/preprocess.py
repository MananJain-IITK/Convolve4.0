import cv2
import numpy as np
import os

class ImagePreprocessor:
    def process(self, image_path):
        # Read
        img = cv2.imread(image_path)
        if img is None: return image_path
        
        # 1. Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. CLAHE (Fix brightness)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # 3. Denoise (Remove grain)
        denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)

        # 4. Otsu's Binarization
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Save Temp
        temp_filename = f"temp_{os.path.basename(image_path)}"
        cv2.imwrite(temp_filename, binary)
        return temp_filename