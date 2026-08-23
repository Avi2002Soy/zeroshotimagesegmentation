import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection

class OWLv2Detector:
    def __init__(self, model_id: str, device: str):
        self.device = device
        self.processor = Owlv2Processor.from_pretrained(model_id)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_id).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def detect(self, image, text_queries: list[str], threshold: float) -> list[dict]:
        inputs = self.processor(text=[text_queries], images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]], device=self.device)
        
        results = self.processor.image_processor.post_process_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=threshold
        )[0]
        
        return [{"box": b.cpu().tolist(), "score": float(s.cpu()), "label": text_queries[l]} 
                for b, s, l in zip(results["boxes"], results["scores"], results["labels"])]
