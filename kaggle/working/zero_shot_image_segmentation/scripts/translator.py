import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class IndicTranslator:
    def __init__(self, model_id: str, device: str):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def translate_to_english(self, queries: list[str], source_lang: str = "hin_Deva") -> list[dict]:
        self.tokenizer.src_lang = source_lang
        translated_pairs = []
        for q in queries:
            inputs = self.tokenizer(q, return_tensors="pt").to(self.device)
            forced_bos_token_id = self.tokenizer.convert_tokens_to_ids("eng_Latn")
            
            outputs = self.model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=200
            )
            eng_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()
            translated_pairs.append({"indic": q, "english": eng_text})
        return translated_pairs
