# src/services/biobert_inference.py

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "biobert_healthcare_sentiment"

LABELS = ["Negative", "Neutral", "Positive"]

class BioBERTSentimentService:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        self.model.eval()

    def predict(self, text: str):
        if not text or not text.strip():
            raise ValueError("Input text is empty")

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)[0]

        prediction = LABELS[probs.argmax().item()]

        return {
            "prediction": prediction,
            "confidence": round(probs.max().item(), 4),
            "scores": {
                LABELS[i]: round(probs[i].item(), 4)
                for i in range(len(LABELS))
            }
        }

