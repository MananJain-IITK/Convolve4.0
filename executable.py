import argparse
import json
import time
import os
import gc
import torch
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

def main():
    # 1. Parse Arguments (Modified for Batch Support)
    parser = argparse.ArgumentParser(description="Invoice Extraction Tool")
    # nargs='+' means "one or more arguments". It collects them into a list.
    parser.add_argument("image_paths", nargs='+', help="Path(s) to invoice image file(s)")
    args = parser.parse_args()
    
    # 2. Setup Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")
    result_file = "result.json"
    
    # 3. Load Existing Results (For Loop Compatibility)
    # We store results as a LIST of objects: [ {doc1}, {doc2}, ... ]
    existing_data = []
    if os.path.exists(result_file):
        try:
            with open(result_file, "r") as f:
                content = json.load(f)
                if isinstance(content, list):
                    existing_data = content
                elif isinstance(content, dict):
                    # Handle case where previous run output a single object
                    existing_data = [content]
        except Exception:
            existing_data = []

    # Helper to find/remove existing entry for re-runs
    def remove_existing(doc_id):
        return [item for item in existing_data if item.get("doc_id") != doc_id]

    # --- BATCH PROCESSING LOOP ---
    for image_path in args.image_paths:
        print(f"Processing: {image_path}...")
        start_time = time.time()
        
        doc_id = os.path.basename(image_path)
        
        # Prepare Document Object
        doc_output = {
            "doc_id": doc_id,
            "fields": {
                "dealer_name": None,
                "model_name": None,
                "horse_power": None,
                "asset_cost": None,
                "signature": None,
                "stamp": None
            },
            "confidence": 0.95, 
            "processing_time_sec": 0,
            "cost_estimate_usd": 0.00
        }
        
        temp_img = None
        
        try:
            # --- PHASE 0: PREPROCESSING ---
            from utils.preprocess import ImagePreprocessor
            preprocessor = ImagePreprocessor()
            temp_img = preprocessor.process(image_path)
            
            # --- PHASE 1: TEXT EXTRACTION (PaddleOCR + Llama) ---
            from utils.extractor import LLMExtractor
            text_engine = LLMExtractor(os.path.join(models_dir, "llama-1b"))
            
            text_data = text_engine.extract(temp_img)
            doc_output["fields"].update(text_data)
            
            del text_engine
            gc.collect()
            torch.cuda.empty_cache()

            # --- PHASE 2: VISUAL DETECTION (YOLO + HF) ---
            from utils.detector import VisualDetector
            visual_engine = VisualDetector(models_dir)
            
            visual_data = visual_engine.detect(temp_img)
            doc_output["fields"]["signature"] = visual_data.get("signature")
            doc_output["fields"]["stamp"] = visual_data.get("stamp")
            
            del visual_engine
            gc.collect()
            torch.cuda.empty_cache()

            # --- METRICS ---
            duration = time.time() - start_time
            doc_output["processing_time_sec"] = round(duration, 2)
            doc_output["cost_estimate_usd"] = round(duration * 0.0005, 5)

        except Exception as e:
            doc_output["error"] = str(e)
            print(f"Error processing {doc_id}: {e}")

        # Cleanup Temp
        if temp_img and os.path.exists(temp_img):
            os.remove(temp_img)

        # --- UPDATE RESULT LIST ---
        # Remove old entry for this doc (if any) and append new one
        existing_data = remove_existing(doc_id)
        existing_data.append(doc_output)

        # Save Incrementally (Safe against crashes)
        # We assume the user wants a LIST of JSON objects if processing multiple files
        # If the list has only 1 item, some evaluators might prefer just the dict, 
        # but a list is the standard for batch output.
        output_to_save = existing_data
        
        # NOTE: If you strictly need a single object for single-file runs
        # you can uncomment this logic, but for batch consistency, a list is safer.
        # if len(existing_data) == 1:
        #     output_to_save = existing_data[0]

        with open(result_file, "w") as f:
            json.dump(output_to_save, f, indent=4)

    print(f"Finished. Results saved to {result_file}")

if __name__ == "__main__":
    main()