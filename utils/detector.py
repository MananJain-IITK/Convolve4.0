import os
import torch
from PIL import Image
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForObjectDetection

class VisualDetector:
    def __init__(self, models_dir):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # --- SIGNATURE (YOLO) ---
        sign_path = os.path.join(models_dir, "sign_model.pt")
        self.sign_model = YOLO(sign_path)
        
        # --- STAMP (Hugging Face) ---
        stamp_path = os.path.join(models_dir, "stamp_model")
        self.stamp_processor = AutoImageProcessor.from_pretrained(stamp_path)
        self.stamp_model = AutoModelForObjectDetection.from_pretrained(stamp_path)
        self.stamp_model.to(self.device)

    def detect(self, image_path):
        image = Image.open(image_path).convert("RGB")
        
        # 1. Signature Detection (YOLO)
        # Signatures can be wide, so we DO NOT enforce square shape here
        sign_results = self.sign_model.predict(image_path, conf=0.25, verbose=False)
        sign_bbox = self._get_best_box(sign_results, is_yolo=True, require_square=False)
        
        # 2. Stamp Detection (Transformers)
        inputs = self.stamp_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.stamp_model(**inputs)
            
        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
        stamp_results = self.stamp_processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.5
        )[0]
        
        # Stamps should be roughly square (round or boxy). 
        # Invoice numbers are usually long rectangles, so we filter them out here.
        stamp_bbox = self._get_best_box(stamp_results, is_yolo=False, require_square=True)
        
        return {
            "signature": {"present": sign_bbox != [0,0,0,0], "bbox": sign_bbox},
            "stamp":     {"present": stamp_bbox != [0,0,0,0], "bbox": stamp_bbox}
        }

    def _get_best_box(self, results, is_yolo, require_square=False):
        best_box = [0, 0, 0, 0]
        max_conf = -1
        
        # Helper to process a potential box
        def process_box(box, score):
            nonlocal best_box, max_conf
            
            x1, y1, x2, y2 = [int(x) for x in box]
            w = x2 - x1
            h = y2 - y1
            
            # Shape Filter (The Logic You Requested)
            if require_square and h > 0:
                ratio = w / h
                # Accept only if ratio is between 0.5 (tall) and 1.8 (slightly wide)
                # This kills "Invoice Number" lines which are usually ratio > 3.0
                if ratio < 0.5 or ratio > 1.8:
                    return # Skip this box, it's not a stamp

            if score > max_conf:
                max_conf = score
                best_box = [x1, y1, x2, y2]

        if is_yolo:
            # YOLO Format
            for r in results:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    coords = box.xyxy[0].tolist()
                    process_box(coords, conf)
        else:
            # Hugging Face Format
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                process_box(box.tolist(), float(score))
                    
        return best_box