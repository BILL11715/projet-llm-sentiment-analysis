"""Predict the sentiment of a new French review with the fine-tuned model."""

import argparse
import re

import torch
from transformers import BertForSequenceClassification, BertTokenizer

from config import LABEL_NAMES, TRANSFORMER_MODEL_DIR


def clean_text(text: str) -> str:
    text = text.replace("<br />", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def predict_sentiment(text: str) -> tuple[str, float]:
    tokenizer = BertTokenizer.from_pretrained(TRANSFORMER_MODEL_DIR)
    model = BertForSequenceClassification.from_pretrained(TRANSFORMER_MODEL_DIR)
    model.eval()

    inputs = tokenizer(clean_text(text), return_tensors="pt", truncation=True, max_length=256)

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)[0]

    predicted_label = int(torch.argmax(probabilities).item())
    score = float(probabilities[predicted_label].item())

    return LABEL_NAMES[predicted_label], score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("text", help="Avis en francais a classifier")
    args = parser.parse_args()

    label, score = predict_sentiment(args.text)
    print(f"Texte: {args.text}")
    print(f"Prediction: {label}")
    print(f"Score: {score:.4f}")


if __name__ == "__main__":
    main()
