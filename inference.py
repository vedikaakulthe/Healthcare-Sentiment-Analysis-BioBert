"""
inference.py
-----------------
Standalone inference script for Healthcare Sentiment Analysis.

This script supports:
1. Manual text input from user
2. Default sample text (if user skips input)

NOTE:
- This is a placeholder inference logic for GitHub demo.
- Model loading is mocked for now (industry practice when model files are excluded).
"""

def predict_sentiment(text: str):
    """
    Dummy prediction logic.
    Replace this with actual model inference when model is available.
    """
    text_lower = text.lower()

    if any(word in text_lower for word in ["pain", "worse", "bad", "side effect"]):
        return {
            "sentiment": "Negative",
            "confidence": 0.85
        }
    elif any(word in text_lower for word in ["ok", "average", "normal"]):
        return {
            "sentiment": "Neutral",
            "confidence": 0.65
        }
    else:
        return {
            "sentiment": "Positive",
            "confidence": 0.90
        }


if __name__ == "__main__":
    print("Healthcare Sentiment Analysis")
    print("-" * 35)

    user_text = input("Enter healthcare-related text (press Enter to use sample): ").strip()

    if not user_text:
        user_text = "The patient recovered well after the treatment."
        print(f"\nUsing sample text:\n{user_text}")

    result = predict_sentiment(user_text)

    print("\nPrediction Result")
    print("-----------------")
    print(f"Sentiment  : {result['sentiment']}")
    print(f"Confidence : {result['confidence']}")

