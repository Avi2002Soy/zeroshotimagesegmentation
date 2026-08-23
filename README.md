# Indic-Guided Zero-Shot Image Segmentation 🎯👁️

A robust, multimodal deep learning pipeline that bridges the gap between natural language processing for Indian languages and zero-shot computer vision. This project takes regional Indic text prompts (e.g., Hindi) and performs highly accurate, open-vocabulary object detection and mask segmentation on images. 

Designed with portability in mind, the architecture smoothly transitions from quick prototyping in Kaggle environments to large-scale batch processing on High-Performance Computing (HPC) clusters.

## 🧠 Pipeline Architecture

The pipeline fuses multiple state-of-the-art transformer models to achieve accurate, hallucination-free segmentation:

1. **Indic Translation (`facebook/nllb-200-distilled-600M`)**
   Translates regional Indic language prompts (e.g., Devanagari script) into English semantic tokens to interact with standard vision-language models.
2. **Open-Vocabulary Detection (`google/owlv2-base-patch16`)**
   Performs zero-shot bounding box detection based on the translated text queries.
3. **Promptable Segmentation (`facebook/sam-vit-base`)**
   The Segment Anything Model (SAM) extracts pixel-perfect binary masks using the detected bounding boxes as geometric prompts.
4. **Semantic Verification & NMS (`openai/clip-vit-base-patch32`)**
   Applies a strict validation layer using CLIP. Detected image crops (with contextual padding) are compared against the textual prompts using cosine similarity to reject false positives and visual noise. Redundant overlapping boxes are pruned using Class-Aware Non-Maximum Suppression (NMS).

## ✨ Key Features

* **Zero-Shot Flexibility:** Detect and segment virtually any object without requiring custom training or domain-specific datasets.
* **Multimodal Late Fusion:** Balances OWLv2 object detection confidence scores with CLIP semantic similarity scores using a calibratable `alpha` weight.
* **Native Indic Rendering:** Safely burns Devanagari (and other regional scripts) directly onto the output visualizations using PIL, completely avoiding Matplotlib's "tofu" (missing glyph) errors.

## 📂 Project Structure

```text
zero_shot_image_segmentation/
│
├── configs/
│   ├── __init__.py
│   └── pipeline_config.py      # Thresholds, device config, and model IDs
│
├── scripts/
│   ├── __init__.py
│   ├── detector.py             # OWLv2 logic (safely routed via image_processor)
│   ├── pipeline.py             # Main integration, NMS, and context padding
│   ├── segmenter.py            # SAM v1 mask extraction
│   ├── translator.py           # NLLB sequence-to-sequence translation
│   └── verifier.py             # CLIP similarity and final composite scoring
│
├── NotoDevanagari.ttf          # Downloaded dynamically for text rendering
├── main.py                     # Execution script and visualization logic
└── README.md
