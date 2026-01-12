# src/train_biobert.py

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from src.data.dataset import load_and_prepare_data, train_val_split
from src.config import MODEL_NAME, MAX_LEN, NUM_LABELS, EPOCHS, BATCH_SIZE

def tokenize_data(texts, tokenizer):
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LEN
    )

def main():
    print("🔹 Loading dataset...")
    data = load_and_prepare_data(
        "data/healthcare_tweets.csv",
        "data/drug_reviews.csv"
    )

    train_texts, val_texts, train_labels, val_labels = train_val_split(data)

    print("🔹 Loading BioBERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_encodings = tokenize_data(train_texts, tokenizer)
    val_encodings = tokenize_data(val_texts, tokenizer)

    class SentimentDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item

    train_dataset = SentimentDataset(train_encodings, train_labels)
    val_dataset = SentimentDataset(val_encodings, val_labels)

    print("🔹 Loading BioBERT model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS
    )

    training_args = TrainingArguments(
        output_dir="biobert_results",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="logs",
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer
    )

    print("🚀 Training BioBERT...")
    trainer.train()

    print("💾 Saving model...")
    model.save_pretrained("biobert_healthcare_sentiment")
    tokenizer.save_pretrained("biobert_healthcare_sentiment")

    print("✅ Training complete!")

if __name__ == "__main__":
    main()
