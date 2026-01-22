import os
import json
import torch
import easyocr
import shutil
import pytesseract
import logging # Added to silence specific tokenizer warnings
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig

# Fix memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class LLMExtractor:
    def __init__(self, model_path):
        self.has_cuda = torch.cuda.is_available()
        
        # 1. SETUP EASYOCR (OFFLINE)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ocr_model_dir = os.path.join(base_dir, "models", "easyocr")

        self.reader = easyocr.Reader(
            ['en', 'hi', 'mr'], 
            gpu=self.has_cuda,
            model_storage_directory=ocr_model_dir, 
            download_enabled=False
        )

        # 2. CHECK FOR TESSERACT
        self.has_tesseract = shutil.which("tesseract") is not None

        # 3. SETUP LLAMA
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        )
        
        # --- FIX 1: Use 'dtype' instead of 'torch_dtype' ---
        model_kwargs = {
            "quantization_config": quantization_config,
            "device_map": "auto",
        }
        
        # --- FIX 2: Silence Regex Warning ---
        # We temporarily set logging to ERROR only for transformers to hide the regex warning
        # because 'fix_mistral_regex' is sometimes not accepted as a direct kwarg
        logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Re-enable warnings if you want, or keep them silenced
        # logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.WARNING)

        self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            temperature=0.1,
        )

    def extract(self, image_path):
        ocr_text = self._smart_ocr(image_path)
        prompt = self._create_prompt(ocr_text)
        
        # Run inference
        outputs = self.pipe(prompt)
        return self._parse_json(outputs[0]['generated_text'])

    def _smart_ocr(self, image_path):
        # A. EasyOCR with Zoom
        try:
            results = self.reader.readtext(image_path, detail=0, paragraph=True, mag_ratio=2.0)
            text_easy = "\n".join(results)
        except:
            text_easy = ""

        # B. Tesseract Fallback
        if self.has_tesseract and len(text_easy) < 15:
            try:
                text_tess = pytesseract.image_to_string(Image.open(image_path))
                if len(text_tess) > len(text_easy):
                    return text_tess
            except:
                pass
        return text_easy

    def _create_prompt(self, ocr_text):
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Extract JSON. Rules:
1. dealer_name: Full Name
2. model_name: Model ONLY (No Company Name). "Sonalika DI 750" -> "DI 750"
3. horse_power: Number (values vary from 20 to 300)
4. asset_cost: Number (no commas)
5. signature: null
6. stamp: null
Return JSON ONLY.<|eot_id|><|start_header_id|>user<|end_header_id|>
{ocr_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    def _parse_json(self, generated_text):
        try:
            json_str = generated_text.split("assistant<|end_header_id|>")[-1].strip()
            if "```" in json_str: 
                json_str = json_str.split("```json")[-1].split("```")[0].strip()
            return json.loads(json_str)
        except:
            return {"dealer_name": None, "model_name": None, "horse_power": None, "asset_cost": None}