
from typing import List
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class CLIPVerifier:
    def __init__(self, model_id: str, device: str):
        self.device = device
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def compute_similarity(self, image_crops: List[Image.Image], text_query: str) -> List[float]:
        if not image_crops:
            return []

        inputs = self.processor(
            text=[text_query],
            images=image_crops,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        outputs = self.model(**inputs)
        
        image_embeds = F.normalize(outputs.image_embeds, p=2, dim=-1)
        text_embeds = F.normalize(outputs.text_embeds, p=2, dim=-1)
        
        cosine_sim = torch.matmul(image_embeds, text_embeds.t()).squeeze(-1)
        return cosine_sim.cpu().tolist()

    @staticmethod
    def compute_final_score(sim_clip: float, conf_box: float, alpha: float) -> float:
        sim_clip_clamped = max(0.0, min(1.0, sim_clip))
        return alpha * sim_clip_clamped + (1.0 - alpha) * conf_box
