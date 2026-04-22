# Healthcare Sentiment Analysis using BioBERT

A **research-oriented NLP system** for sentiment analysis in the healthcare domain, leveraging **BioBERT**, a transformer model pretrained on biomedical corpora.

This project demonstrates **domain-adapted NLP**, **confidence-aware inference**, and **production-ready ML design**.

---------------------------------------------------------------------------------------------------------------------------------

## Problem Statement:

Generic sentiment models fail on healthcare text due to:
- Clinical vocabulary
- Drug names and adverse effects
- Context-dependent sentiment polarity

This project addresses the gap by fine-tuning **BioBERT** on healthcare-specific datasets.

---------------------------------------------------------------------------------------------------------------------------------

## Model Architecture:

- **Base Model:** BioBERT (`dmis-lab/biobert-base-cased-v1.1`)
- **Task:** Multi-class sentiment classification
- **Framework:** PyTorch + HuggingFace Transformers
- **Inference:** Softmax confidence scoring

---------------------------------------------------------------------------------------------------------------------------------

## Datasets:

- Healthcare Tweets
- Drug Reviews Dataset
Both datasets are unified into a **single labeled corpus** with consistent sentiment mapping.

---------------------------------------------------------------------------------------------------------------------------------



## Inference Example:

```python
from src.services.biobert_inference import predict_sentiment

result = predict_sentiment("This medication caused severe nausea.")
print(result)

Output:
{
  "sentiment": "negative",
  "confidence": 0.94
}

---------------------------------------------------------------------------------------------------------------------------------

Key Research-Level Features:

1. Domain-specific Transformer (BioBERT)
2. Confidence-aware predictions
3. Clean separation of training and inference
4. Reproducible configuration design
5. Healthcare-focused NLP pipeline




