"""Explore the Hugging Face Allocine sentiment dataset."""

import pandas as pd

from config import LABEL_COLUMN, LABEL_NAMES, TEXT_COLUMN
from data_utils import dataset_to_dataframe, load_sentiment_dataset


def describe_split(name: str, df: pd.DataFrame) -> None:
    print(f"\n=== Split: {name} ===")
    print(f"Nombre d'exemples: {len(df)}")

    distribution = df[LABEL_COLUMN].value_counts().sort_index()
    print("\nDistribution des labels:")
    for label, count in distribution.items():
        print(f"- {label} ({LABEL_NAMES[label]}): {count}")

    lengths = df[TEXT_COLUMN].str.split().map(len)
    print("\nLongueur des textes en nombre de mots:")
    print(lengths.describe().round(2))

    print("\nExemples:")
    for _, row in df.sample(n=min(3, len(df)), random_state=42).iterrows():
        text = row[TEXT_COLUMN]
        preview = text[:350] + ("..." if len(text) > 350 else "")
        print(f"\nLabel: {row[LABEL_COLUMN]} ({LABEL_NAMES[row[LABEL_COLUMN]]})")
        print(preview)


def main() -> None:
    dataset = load_sentiment_dataset()
    print(dataset)

    for split_name, split in dataset.items():
        df = dataset_to_dataframe(split)
        describe_split(split_name, df)


if __name__ == "__main__":
    main()

