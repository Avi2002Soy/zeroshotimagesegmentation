from dataclasses import dataclass
import torch

@dataclass
class PipelineConfig:
    translator_model_id: str = "facebook/nllb-200-distilled-600M"
    detector_model_id: str = "google/owlv2-base-patch16"
    segmenter_model_id: str = "facebook/sam-vit-base"
    verifier_model_id: str = "openai/clip-vit-base-patch32"
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    box_threshold: float = 0.15         # Retains valid foreground and background detections
    alpha: float = 0.60                 # Balanced weight between box confidence and semantic match
    final_score_threshold: float = 0.24 # Allows verified objects like the metal pipe (0.27) to pass
    iou_threshold: float = 0.65         # Prunes duplicate boxes while keeping separate instances
    mask_opacity: float = 0.5           # Clean visual blending for segmentation masks

config = PipelineConfig()
