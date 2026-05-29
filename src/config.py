"""Configuration shared by the project scripts."""

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"

DATASET_NAME = "allocine"
TEXT_COLUMN = "review"
LABEL_COLUMN = "label"

LABEL_NAMES = {
    0: "negatif",
    1: "positif",
}

RANDOM_STATE = 42

# Small defaults make the project runnable on a personal laptop.
TRAIN_SAMPLE_SIZE = 3000
EVAL_SAMPLE_SIZE = 1000
TEST_SAMPLE_SIZE = 1000

TRANSFORMER_TRAIN_SAMPLE_SIZE = 600
TRANSFORMER_EVAL_SAMPLE_SIZE = 200

BASELINE_MODEL_PATH = MODELS_DIR / "baseline_tfidf_logreg.joblib"
TRANSFORMER_MODEL_DIR = MODELS_DIR / "transformer_sentiment"

TRANSFORMER_MODEL_NAME = "prajjwal1/bert-tiny"
MAX_LENGTH = 128
NUM_EPOCHS = 1
LEARNING_RATE = 2e-5
BATCH_SIZE = 16
