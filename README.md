# Convolve 4.0: Intelligent Document AI for Invoice Field Extraction

## Problem Statement

Modern financial institutions rely on automated extraction of key information from semi-structured documents (invoices, quotations) to accelerate credit decisioning, vendor reconciliation, and loan disbursals. These documents vary significantly in layout, language, and quality.

This project is submitted for the **IDFC First Bank – GenAI Hackathon (Convolve 4.0)**. The objective is to design an end-to-end, low-cost, and scalable **Document AI system** that accurately extracts structured information from invoice-type documents while maintaining sub-$0.01 costs per document.

## Objective

Given an input invoice image, the system extracts the following fields and outputs them as strict JSON:

* **Dealer Name** (text)
* **Model Name** (text, exact match)
* **Horse Power** (integer)
* **Asset Cost** (integer)
* **Dealer Signature** (binary presence with bounding box)
* **Dealer Stamp** (binary presence with bounding box)

## Solution Overview & Architecture

Convolve 4.0 implements a modular, multi-stage Document AI pipeline that explicitly separates **visual detection** from **semantic reasoning**, enabling explainability, reproducibility, and high-speed inference.

### Key Engineering Features

* **Hardware-Aware Quantization:** Automatically detects CUDA availability. Utilizes `BitsAndBytes` 4-bit quantization on GPUs, with graceful CPU fallbacks to prevent Out-Of-Memory (OOM) crashes.
* **Hybrid Visual Pipeline:** Routes raw RGB images to YOLO/Hugging Face object detectors for signatures and stamps, while passing Otsu-binarized images to the OCR engine for maximum text clarity.
* **Zero-Shot Schema Forcing:** Leverages LLaMA 3.2 1B via strict autocomplete prompt engineering, bypassing safety-filter refusals ("lazy parrot syndrome") to guarantee data extraction.
* **Indestructible JSON Parser:** Features a custom regex-based, brute-force structural parser that isolates and cleans JSON payloads, completely ignoring LLM hallucinations or markdown artifacts.

---

## Project Structure

```text
Convolve4.0/
│
├── executable.py              # Main inference entry point (Interactive & Batch)
├── download_models.py         # Resilient model setup and downloads
├── requirements.txt
├── utils/
│   ├── preprocess.py          # Image enhancement and binarization
│   ├── extractor.py           # OCR + LLaMA 3.2 Semantic Extraction
│   └── detector.py            # YOLO Signature & HF Stamp Detection
│
├── models/                    # Local weight storage (LLaMA, EasyOCR, YOLO)
├── Test_Images/               # Test Images for testing
│
└── result.json                # Consolidated structured output

```

## Setup Instructions

### 1. Clone Repository

```bash
git clone https://github.com/MananJain-IITK/Convolve4.0.git
cd Convolve4.0

```

### 2. Install Dependencies

```bash
pip install -r requirements.txt

```

### 3. Download & Configure Models

Run the secure download script. This script features automatic network-drop recovery and corrupted-file cleanup.

```bash
python download_models.py

```

*Note: You will be prompted to enter your Hugging Face Access Token to download the gated LLaMA 3.2 1B Instruct model. Ensure your custom YOLO model is placed at `models/sign_model.pt`.*

---

## Running the System

The main executable has been optimized to load the heavy 1B parameter model into RAM/VRAM exactly **once**, eliminating loop-crashing memory leaks during batch processing.

### Interactive Mode (Single File)

Simply run the script. It will prompt you for an image, allowing you to drag-and-drop the file directly into your terminal.

```bash
python executable.py

```

### Batch Processing Mode

Pass multiple invoice paths directly via the command line.

```bash
python executable.py invoice1.jpg invoice2.png invoice3.jpg

```

---

## Output Format

The pipeline generates a clean `result.json` array. The backend Python sanitization guarantees pure integers for numerical fields, regardless of LLM string hallucinations.

```json
[
    {
        "doc_id": "invoice_001.png",
        "fields": {
            "dealer_name": "Odisha Agro Industries Corporation Ltd.",
            "model_name": "744 FE",
            "horse_power": 48,
            "asset_cost": 801815,
            "signature": {
                "present": true,
                "bbox": [222, 1433, 556, 1540]
            },
            "stamp": {
                "present": false,
                "bbox": [0, 0, 0, 0]
            }
        },
        "confidence": 0.95,
        "processing_time_sec": 4.12,
        "cost_estimate_usd": 0.00206
    }
]

```

## Generalization

Although tested on tractor loan quotations, the schema-forced LLaMA extraction is completely **invoice-agnostic**. It can dynamically adapt to retail receipts, vendor bills, and industrial quotations without requiring model fine-tuning.

## Authors

**Manan Jain** Indian Institute of Technology Kanpur

**Nisarg Parashar** Indian Institute of Technology Kanpur