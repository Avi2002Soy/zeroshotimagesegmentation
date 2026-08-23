from typing import List
import torch
import numpy as np
from PIL import Image
from transformers import SamProcessor, SamModel

class SAMSegmenter:
    def __init__(self, model_id: str, device: str):
        self.device = device
        self.processor = SamProcessor.from_pretrained(model_id)
        self.model = SamModel.from_pretrained(model_id).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def segment_from_boxes(self, image: Image.Image, boxes: List[List[float]]) -> List[np.ndarray]:
        if not boxes:
            return []

        inputs = self.processor(
            images=image,
            input_boxes=[boxes],
            return_tensors="pt"
        ).to(self.device)

        outputs = self.model(**inputs)
        
        # Standard SAM requires reshaped_input_sizes for mask post-processing
        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu()
        )[0]

        binary_masks = []
        for box_masks in masks:
            best_mask = box_masks[0].numpy().astype(bool)
            binary_masks.append(best_mask)

        return binary_masks
