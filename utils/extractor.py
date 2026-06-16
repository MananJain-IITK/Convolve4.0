import os
import json
import torch
import easyocr
import shutil
import pytesseract
import logging
import re
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class LLMExtractor:
    def __init__(self, model_path):
        self.has_cuda = torch.cuda.is_available()
        
        # 1. SETUP EASYOCR
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ocr_model_dir = os.path.join(base_dir, "models", "easyocr")

        self.reader = easyocr.Reader(
            ['en', 'hi', 'mr'], 
            gpu=self.has_cuda,
            download_enabled=True 
        )

        self.has_tesseract = shutil.which("tesseract") is not None

        # 3. SETUP LLAMA (Hardware-Aware Loading)
        if self.has_cuda:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True, 
                bnb_4bit_compute_dtype=torch.float16
            )
            model_kwargs = {
                "quantization_config": quantization_config,
                "device_map": "auto",
            }
        else:
            model_kwargs = {
                "device_map": "cpu",
            }
            
        logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=256,
            temperature=0.01,
            return_full_text=False,
            eos_token_id=terminators, 
            pad_token_id=self.tokenizer.eos_token_id
        )

    def extract(self, image_path):
        print("\nRunning OCR...")
        ocr_text = self._smart_ocr(image_path)
        
        if not ocr_text.strip():
            print("WARNING: OCR extracted absolutely no text! Check image quality.")
            
        prompt = self._create_prompt(ocr_text)
        
        print("Running LLaMA Semantic Extraction...")
        outputs = self.pipe(prompt)
        raw_output = outputs[0]['generated_text']
        
        return self._parse_json(raw_output)

    def _smart_ocr(self, image_path):
        try:
            results = self.reader.readtext(image_path, detail=0, paragraph=True, mag_ratio=2.0)
            text_easy = "\n".join(results)
            print(f"EasyOCR extracted {len(text_easy)} characters.")
            return text_easy
        except Exception as e:
            print(f"EasyOCR Error: {e}")
            return ""

    def _create_prompt(self, ocr_text):
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are an automated data parsing script. Extract the following fields from the OCR text.

        RULES FOR EXTRACTION:
        1. dealer_name: The company name.
        2. model_name: The alphanumeric model code. Ignore the brand name.
        3. horse_power: Look for the letters "HP" and extract the number near it.
        4. asset_cost: Look under "Amount" or "Total".

        CRITICAL FORMATTING RULES:
        You MUST output a valid JSON object. EVERY value MUST be a string enclosed in double quotes (e.g. "48", not 48). If a value is missing, output "null".<|eot_id|><|start_header_id|>user<|end_header_id|>
        TEXT FOR ANALYSIS:
        {ocr_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        {{"""

    def _parse_json(self, generated_text):
        try:
            generated_text = generated_text.strip()

            if "assistant<|end_header_id|>" in generated_text:
                generated_text = generated_text.split("assistant<|end_header_id|>")[-1].strip()

            if not generated_text.startswith("{"):
                generated_text = "{" + generated_text
                
            cleaned = generated_text.replace("```json", "").replace("```", "").strip()
            
            matches = re.findall(r'\{.*?\}', cleaned, re.DOTALL)
            
            if matches:
                json_str = matches[-1] 
                return json.loads(json_str)
            else:
                raise ValueError("No JSON braces found in LLM output.")
                
        except Exception as e:
            print(f"\n JSON Parsing Failed!")
            print(f"Error: {e}")
            print(f"Raw LLaMA Output was:\n{generated_text}\n")
            return {"dealer_name": None, "model_name": None, "horse_power": None, "asset_cost": None}