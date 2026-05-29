"""Utility functions for loading and preparing sentiment data."""

import re
from typing import Optional

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset

from config import DATASET_NAME, LABEL_COLUMN, RANDOM_STATE, TEXT_COLUMN


def clean_text(text: str) -> str:
    """Apply lightweight text cleaning while preserving sentiment cues."""
    text = text.replace("<br />", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_sentiment_dataset() -> DatasetDict:
    """Load the Allocine dataset from Hugging Face."""
    dataset = load_dataset(DATASET_NAME)
    return dataset


def dataset_to_dataframe(dataset: Dataset) -> pd.DataFrame:
    """Convert a Hugging Face split to a cleaned pandas DataFrame."""
    df = dataset.to_pandas()
    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str).map(clean_text)
    return df


def sample_split(dataset: Dataset, sample_size: Optional[int]) -> Dataset:
    """Shuffle and optionally reduce a split to speed up experimentation."""
    shuffled = dataset.shuffle(seed=RANDOM_STATE)
    if sample_size is None or sample_size >= len(shuffled):
        return shuffled
    return shuffled.select(range(sample_size))


def clean_dataset_split(dataset: Dataset) -> Dataset:
    """Return a Dataset split with cleaned text."""
    return dataset.map(lambda row: {TEXT_COLUMN: clean_text(row[TEXT_COLUMN])})


def get_texts_and_labels(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Extract model inputs and labels from a dataframe."""
    return df[TEXT_COLUMN], df[LABEL_COLUMN]

