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
    # 1. Parse Arguments (Made optional using nargs='*')
    parser = argparse.ArgumentParser(description="Invoice Extraction Tool")
    parser.add_argument("image_paths", nargs='*', help="Path(s) to invoice image file(s)")
    args = parser.parse_args()
    
    image_paths = args.image_paths
    
    if not image_paths:
        user_input = input("\nEnter the path to the invoice image (drag and drop works): ").strip()
        if not user_input:
            print("No input provided. Exiting.")
            return
        image_paths = [user_input.strip("\"' ")]
    
    # 2. Setup Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")
    result_file = "result.json"
    
    print("\nLoading AI Models into memory (this will take a moment)...")
    from utils.preprocess import ImagePreprocessor
    from utils.extractor import LLMExtractor
    from utils.detector import VisualDetector
    
    try:
        preprocessor = ImagePreprocessor()
        text_engine = LLMExtractor(os.path.join(models_dir, "llama-1b"))
        visual_engine = VisualDetector(models_dir)
        print("All models loaded successfully!")
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Failed to load models. Error: {e}")
        return # Exit immediately if models fail to load

    # 3. Load Existing Results
    existing_data = []
    if os.path.exists(result_file):
        try:
            with open(result_file, "r") as f:
                content = json.load(f)
                if isinstance(content, list):
                    existing_data = content
                elif isinstance(content, dict):
                    existing_data = [content]
        except Exception:
            existing_data = []

    def remove_existing(doc_id):
        return [item for item in existing_data if item.get("doc_id") != doc_id]

    # --- BATCH PROCESSING LOOP ---
    # Now iterates over our dynamically assigned image_paths variable
    for image_path in image_paths:
        if not os.path.exists(image_path):
            print(f"File not found: {image_path}. Skipping.")
            continue
            
        print(f"\nProcessing: {image_path}...")
        start_time = time.time()
        
        doc_id = os.path.basename(image_path)
        
        doc_output = {
            "doc_id": doc_id,
            "fields": {
                "dealer_name": None, "model_name": None,
                "horse_power": None, "asset_cost": None,
                "signature": None, "stamp": None
            },
            "confidence": 0.95, 
            "processing_time_sec": 0,
            "cost_estimate_usd": 0.00
        }
        
        temp_img = None
        
        try:
            # PHASE 0: PREPROCESSING
            temp_img = preprocessor.process(image_path)
            
            # --- PHASE 1: TEXT EXTRACTION (Uses Binarized image) ---
            text_data = text_engine.extract(temp_img)
            doc_output["fields"].update(text_data)

            # --- PHASE 2: VISUAL DETECTION (CRITICAL: Uses Original RGB image!) ---
            visual_data = visual_engine.detect(image_path) 
            doc_output["fields"]["signature"] = visual_data.get("signature")
            doc_output["fields"]["stamp"] = visual_data.get("stamp")
            
            # METRICS
            duration = time.time() - start_time
            doc_output["processing_time_sec"] = round(duration, 2)
            doc_output["cost_estimate_usd"] = round(duration * 0.0005, 5)

        except Exception as e:
            doc_output["error"] = str(e)
            print(f"Error processing {doc_id}: {e}")

        # Cleanup Temp File
        if temp_img and os.path.exists(temp_img):
            os.remove(temp_img)

        # SAVE INCREMENTAL RESULTS
        existing_data = remove_existing(doc_id)
        existing_data.append(doc_output)

        with open(result_file, "w") as f:
            json.dump(existing_data, f, indent=4)
            
        # SAFE CACHE CLEARING
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nFinished. Results saved to {result_file}")

if __name__ == "__main__":
    main()