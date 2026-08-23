from typing import List, Dict, Any
import torch
import torchvision.ops as ops
import numpy as np
from PIL import Image

from scripts.translator import IndicTranslator
from scripts.detector import OWLv2Detector
from scripts.segmenter import SAMSegmenter
from scripts.verifier import CLIPVerifier

class IndicZeroShotSegmentationPipeline:
    def __init__(self, config: Any):
        self.cfg = config
        print("Loading Indic Language Translator...")
        self.translator = IndicTranslator(self.cfg.translator_model_id, self.cfg.device)
        print("Loading OWLv2 Detector...")
        self.detector = OWLv2Detector(self.cfg.detector_model_id, self.cfg.device)
        print("Loading SAM Segmenter...")
        self.segmenter = SAMSegmenter(self.cfg.segmenter_model_id, self.cfg.device)
        print("Loading CLIP Semantic Verifier...")
        self.verifier = CLIPVerifier(self.cfg.verifier_model_id, self.cfg.device)
        print("Indic Multimodal Pipeline successfully initialized!")

    def predict(self, image: Image.Image, indic_queries: List[str], source_lang: str = "hin_Deva") -> List[Dict]:
        translated_pairs = self.translator.translate_to_english(indic_queries, source_lang=source_lang)
        english_queries = [p["english"] for p in translated_pairs]
        eng_to_indic_map = {p["english"]: p["indic"] for p in translated_pairs}

        raw_detections = self.detector.detect(
            image=image,
            text_queries=english_queries,
            threshold=self.cfg.box_threshold
        )
        if not raw_detections:
            return []

        # --- Refinement 1: Non-Maximum Suppression (NMS) ---
        filtered_detections = []
        unique_labels = set(d["label"] for d in raw_detections)
        
        for label in unique_labels:
            class_dets = [d for d in raw_detections if d["label"] == label]
            boxes = torch.tensor([d["box"] for d in class_dets], dtype=torch.float32)
            scores = torch.tensor([d["score"] for d in class_dets], dtype=torch.float32)
            
            # Drops overlapping boxes for the same semantic target
            keep_idx = ops.nms(boxes, scores, self.cfg.iou_threshold)
            for idx in keep_idx.tolist():
                filtered_detections.append(class_dets[idx])

        boxes = [d["box"] for d in filtered_detections]
        masks = self.segmenter.segment_from_boxes(image, boxes)

        verified_results = []
        for det, mask in zip(filtered_detections, masks):
            box = det["box"]
            x_min, y_min, x_max, y_max = map(int, box)
            
            # --- Refinement 2: Contextual Padding ---
            # Expands the verification crop by 10 pixels so CLIP can analyze edges
            pad = 10
            crop_img = image.crop((
                max(0, x_min - pad), max(0, y_min - pad), 
                min(image.width, x_max + pad), min(image.height, y_max + pad)
            ))
            
            clip_sim = self.verifier.compute_similarity([crop_img], det["label"])[0]
            final_score = self.verifier.compute_final_score(
                sim_clip=clip_sim, conf_box=det["score"], alpha=self.cfg.alpha
            )

            if final_score >= self.cfg.final_score_threshold:
                verified_results.append({
                    "indic_label": eng_to_indic_map.get(det["label"], det["label"]),
                    "english_label": det["label"],
                    "box": box,
                    "mask": mask,
                    "box_score": det["score"],
                    "clip_similarity": clip_sim,
                    "final_score": final_score
                })

        return verified_results
