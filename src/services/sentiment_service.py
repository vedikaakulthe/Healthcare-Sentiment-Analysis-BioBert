import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class SentimentService:
    device = torch.device("cpu")  # Force CPU

    tokenizer = AutoTokenizer.from_pretrained("path/to/your/bert-model")
    model = AutoModelForSequenceClassification.from_pretrained("path/to/your/bert-model")
    model.to(device)

    @staticmethod
    def predict(text):
        inputs = SentimentService.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(SentimentService.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = SentimentService.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
        return "positive" if pred == 1 else "negative"
