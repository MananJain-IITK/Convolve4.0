# Convolve 4.0

Intelligent Document AI for Invoice Field Extraction

## Problem Statement

Modern financial institutions rely on automated extraction of key information from semi-structured documents such as invoices and quotations to accelerate credit decisioning, vendor reconciliation, and loan disbursal workflows. These documents vary significantly in layout, language, and quality, including scanned copies, photographs, and handwritten text.

This project is built as a submission for **IDFC First Bank – GenAI Hackathon (Convolve 4.0)**. The objective is to design an end-to-end, low-cost, and scalable **Document AI system** that accurately extracts structured information from invoice-type documents, with a focus on tractor loan quotations, while remaining generalizable to other invoice formats.

## Objective

Given an input invoice document, the system extracts the following fields and outputs them as structured JSON:

* Dealer Name (text, fuzzy match)
* Model Name (text, exact match)
* Horse Power (numeric, exact match)
* Asset Cost (numeric, exact match)
* Dealer Signature (binary presence with bounding box)
* Dealer Stamp (binary presence with bounding box)

The system is designed to meet the following targets:

* ≥95% document-level accuracy
* ≤30 seconds latency per document
* Cost < $0.01 per document on CPU or low-tier GPU

## Solution Overview

Convolve 4.0 implements a **modular, multi-stage Document AI pipeline** combining classical computer vision, OCR, lightweight large language models, and object detection.

The system explicitly separates **text understanding** and **visual detection**, enabling explainability, reproducibility, and cost-efficient inference.

### High-Level Pipeline

1. Document Preprocessing
2. Text Extraction and Semantic Parsing
3. Visual Detection of Signature and Stamp
4. Post-processing and Validation
5. Structured JSON Output with Metrics


## System Architecture

### 1. Image Preprocessing

Input invoice images are normalized to improve OCR robustness across scanned, photographed, and low-quality documents.

Techniques used:

* Grayscale conversion
* CLAHE contrast enhancement
* Gaussian denoising
* Otsu binarization for clear foreground-background separation

Implemented in:

```
utils/preprocess.py
```



### 2. Text Extraction and Field Parsing

Text extraction is performed using a **hybrid OCR strategy**:

* EasyOCR for multilingual OCR (English, Hindi, Marathi)
* Tesseract as a fallback when OCR confidence is low

The extracted raw text is passed to a **quantized LLaMA 3.2 1B Instruct model**, which converts unstructured OCR text into structured JSON using controlled prompting.

Key properties:

* 4-bit quantized inference
* Offline execution
* Deterministic, schema-constrained output

Fields extracted:

* Dealer Name
* Model Name
* Horse Power
* Asset Cost

Implemented in:

```
utils/extractor.py
```



### 3. Visual Detection (Signature and Stamp)

Visual fields are detected independently of OCR to ensure robustness against handwritten and non-textual elements.

* Dealer Signature:

  * Custom-trained YOLO model
  * Optimized for wide and irregular shapes

* Dealer Stamp:

  * Hugging Face object detection model
  * Shape-based filtering to avoid false positives (e.g., invoice numbers)

Bounding boxes are returned in pixel coordinates and validated using IoU-based logic.

Implemented in:

```
utils/detector.py
```



### 4. Batch Execution and Metrics

The main executable supports **batch processing of multiple invoices** and produces a single consolidated `result.json`.

For each document, the system reports:

* Extracted fields
* Confidence score
* Processing time (seconds)
* Estimated inference cost (USD)

Implemented in:

```
executable.py
```



## Project Structure

```
Convolve4.0/
│
├── executable.py              # Main inference entry point
├── download_models.py         # Model setup and downloads
├── requirements.txt
├── utils/
│   ├── preprocess.py          # Image preprocessing
│   ├── extractor.py           # OCR + LLM extraction
│   └── detector.py            # Signature and stamp detection
│
├── models/                    # Downloaded / custom models
│
└── sample_output/
    └── result.json
```



## Setup Instructions

### 1. Clone Repository

```
git clone https://github.com/MananJain-IITK/Convolve4.0.git
cd Convolve4.0
```

### 2. Install Dependencies

```
pip install -r requirements.txt
```

### 3. Download Models

```
python download_models.py
```

Notes:

* PaddleOCR and stamp detection models are downloaded automatically
* LLaMA 3.2 1B requires Hugging Face access token
* Custom `sign_model.pt` must be placed inside the `models/` directory


## Running the System

### Single or Batch Invoice Processing

```
python executable.py invoice1.jpg invoice2.jpg invoice3.jpg
```

The output will be saved as:

```
result.json
```

## Output Format

Each document produces one JSON object with the following schema:

```
{
  "doc_id": "invoice_001.jpg",
  "fields": {
    "dealer_name": "ABC Tractors Pvt Ltd",
    "model_name": "DI 750",
    "horse_power": 50,
    "asset_cost": 525000,
    "signature": {
      "present": true,
      "bbox": [x1, y1, x2, y2]
    },
    "stamp": {
      "present": true,
      "bbox": [x1, y1, x2, y2]
    }
  },
  "confidence": 0.95,
  "processing_time_sec": 3.8,
  "cost_estimate_usd": 0.002
}
```


## Evaluation Alignment

| Requirement                 | Implementation               |
| --------------------------- | ---------------------------- |
| Multilingual invoices       | EasyOCR + LLM                |
| Semi-structured layouts     | OCR + semantic reasoning     |
| Signature & stamp detection | YOLO + HF object detection   |
| ≥95% DLA                    | Modular ensemble design      |
| Low inference cost          | Quantized LLaMA 1B           |
| Explainability              | Separate OCR, LLM, CV stages |


## Cost and Latency Analysis

* Average latency: 3–6 seconds per document
* Estimated cost: ~$0.002 per document
* Runs on CPU or low-tier GPU
* No external paid APIs used


## Generalization

Although trained and evaluated on tractor loan quotations, the architecture is **invoice-agnostic** and can generalize to:

* Retail invoices
* Industrial invoices
* Vendor bills
* Financial quotations



## Author

Manan Jain
Indian Institute of Technology Kanpur

Nisarg Parashar
Indian Institute of Technology Kanpur


## Hackathon Submission

This repository is submitted as part of **IDFC First Bank – GenAI Hackathon (Convolve 4.0)** and follows all required submission guidelines, evaluation metrics, and output formats defined in the official problem statement.
