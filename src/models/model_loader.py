import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"

class ModelLoader:
    _model = None
    _tokenizer = None
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def load(cls):
        if cls._model is None or cls._tokenizer is None:
            cls._tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            cls._model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME,
                num_labels=3  # negative, neutral, positive
            )
            cls._model.to(cls._device)
            cls._model.eval()
        return cls._model, cls._tokenizer, cls._device
