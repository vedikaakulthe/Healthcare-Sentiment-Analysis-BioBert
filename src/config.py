# src/config.py

# BioBERT model configuration
MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"

# Training parameters
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5

# Labels
LABELS = ["negative", "neutral", "positive"]
NUM_LABELS = len(LABELS)

# Dataset paths
HEALTHCARE_TWEETS_PATH = "data/healthcare_tweets.csv"
DRUG_REVIEWS_PATH = "data/drug_reviews.csv"

# Output paths
MODEL_OUTPUT_DIR = "biobert_healthcare_sentiment"
