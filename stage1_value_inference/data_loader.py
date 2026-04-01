"""
Data Loader for ValuesML / Touché24 ValueEval Dataset
=====================================================
Loads sentences.tsv and labels.tsv, tokenizes with DeBERTa tokenizer,
and returns HuggingFace Dataset objects ready for training.
"""

import os

import datasets
import numpy as np
import pandas as pd
import transformers

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    SCHWARTZ_VALUES,
    LABEL_NAMES,
    NUM_LABELS,
    MODEL_NAME,
    MAX_SEQ_LENGTH,
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
)


def load_raw_data(directory: str) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Load raw sentences and labels from a dataset split directory.

    Args:
        directory: Path to a split directory containing sentences.tsv and labels.tsv.

    Returns:
        Tuple of (sentences_df, labels_df). labels_df may be None if labels.tsv
        does not exist (e.g., for unlabeled test sets).
    """
    sentences_path = os.path.join(directory, "sentences.tsv")
    labels_path = os.path.join(directory, "labels.tsv")

    sentences_df = pd.read_csv(sentences_path, encoding="utf-8", sep="\t", header=0)

    labels_df = None
    if os.path.isfile(labels_path):
        labels_df = pd.read_csv(labels_path, encoding="utf-8", sep="\t", header=0)

    return sentences_df, labels_df


def prepare_labels(
    sentences_df: pd.DataFrame, labels_df: pd.DataFrame | None
) -> np.ndarray | None:
    """Convert label TSV into a multi-hot float matrix of shape (N, 38).

    The ValuesML schema has 19 values × 2 states (attained, constrained) = 38 columns.
    A label is considered positive if its value >= 0.5.

    Args:
        sentences_df: DataFrame with Text-ID and Sentence-ID columns.
        labels_df: DataFrame with label columns, or None.

    Returns:
        Float matrix of shape (N, 38), or None if no labels.
    """
    if labels_df is None:
        return None

    # Merge to ensure alignment
    merged = pd.merge(sentences_df, labels_df, on=["Text-ID", "Sentence-ID"])

    label_matrix = np.zeros((merged.shape[0], NUM_LABELS), dtype=np.float32)
    for idx, label_name in enumerate(LABEL_NAMES):
        if label_name in merged.columns:
            label_matrix[:, idx] = (merged[label_name] >= 0.5).astype(np.float32)

    return label_matrix


def load_dataset_split(
    directory: str,
    tokenizer: transformers.PreTrainedTokenizer,
    load_labels: bool = True,
) -> tuple[datasets.Dataset, list[str], list[int]]:
    """Load a single dataset split, tokenized and ready for the Trainer.

    Args:
        directory: Path to split directory.
        tokenizer: Instantiated HuggingFace tokenizer.
        load_labels: Whether to include labels in the dataset.

    Returns:
        Tuple of (HF Dataset, text_ids list, sentence_ids list).
    """
    sentences_df, labels_df = load_raw_data(directory)

    # Tokenize
    encoded = tokenizer(
        sentences_df["Text"].tolist(),
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding="max_length",
    )

    # Attach labels
    if load_labels and labels_df is not None:
        label_matrix = prepare_labels(sentences_df, labels_df)
        encoded["labels"] = label_matrix.tolist()

    dataset = datasets.Dataset.from_dict(encoded)
    text_ids = sentences_df["Text-ID"].tolist()
    sentence_ids = sentences_df["Sentence-ID"].tolist()

    return dataset, text_ids, sentence_ids


def load_all_splits(
    tokenizer: transformers.PreTrainedTokenizer | None = None,
) -> dict[str, datasets.Dataset]:
    """Load train, validation, and test splits.

    Args:
        tokenizer: Optional tokenizer; if None, creates one from MODEL_NAME.

    Returns:
        Dict with keys 'train', 'validation', 'test' mapping to HF Datasets.
    """
    if tokenizer is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)

    splits = {}
    for name, directory in [("train", TRAIN_DIR), ("validation", VAL_DIR), ("test", TEST_DIR)]:
        if os.path.isdir(directory):
            ds, _, _ = load_dataset_split(directory, tokenizer)
            splits[name] = ds
            print(f"  Loaded {name}: {len(ds)} samples")
        else:
            print(f"  Skipping {name}: directory not found at {directory}")

    return splits


# ─────────────────────────────────────────────────────────────
# Smoke Test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Data Loader Smoke Test")
    print("=" * 60)

    # Test raw loading
    sentences_df, labels_df = load_raw_data(TRAIN_DIR)
    print(f"\nTraining sentences: {len(sentences_df)}")
    print(f"Training labels shape: {labels_df.shape if labels_df is not None else 'N/A'}")
    print(f"Label columns: {len([c for c in labels_df.columns if c not in ['Text-ID', 'Sentence-ID']])}")
    print(f"\nFirst sentence: {sentences_df['Text'].iloc[0][:100]}...")

    # Test label preparation
    label_matrix = prepare_labels(sentences_df, labels_df)
    if label_matrix is not None:
        print(f"\nLabel matrix shape: {label_matrix.shape}")
        print(f"Labels per sample (mean): {label_matrix.sum(axis=1).mean():.2f}")
        print(f"Label distribution (sum per label):")
        for i, name in enumerate(LABEL_NAMES[:6]):
            print(f"  {name}: {label_matrix[:, i].sum():.0f}")
        print("  ...")

    print("\n✅ Data loader smoke test passed!")
