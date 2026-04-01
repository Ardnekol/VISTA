"""
DeBERTa-v3-large Multi-Label Classifier for Value Detection
============================================================
Wraps AutoModelForSequenceClassification with multi_label_classification
problem type. Provides convenience methods for loading, saving, and inference.
"""

import os
from typing import Optional

import numpy as np
import torch
import transformers
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    MODEL_NAME,
    NUM_LABELS,
    ID2LABEL,
    LABEL2ID,
    MAX_SEQ_LENGTH,
    SIGMOID_THRESHOLD,
    LABEL_NAMES,
    CHECKPOINT_DIR,
)


class ValueClassifier:
    """Multi-label value classifier using DeBERTa-v3-large.

    Predicts a 38-dimensional vector (19 Schwartz values × 2 states)
    for any input text.
    """

    def __init__(
        self,
        model_name_or_path: str = MODEL_NAME,
        device: Optional[str] = None,
    ):
        """Initialize the classifier.

        Args:
            model_name_or_path: HuggingFace model name or path to a saved checkpoint.
            device: Device string ('cuda', 'mps', 'cpu'). Auto-detected if None.
        """
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        print(f"Loading model from: {model_name_or_path}")
        print(f"Device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=NUM_LABELS,
            problem_type="multi_label_classification",
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            ignore_mismatched_sizes=True,
        )
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str, return_labels: bool = False) -> np.ndarray:
        """Predict value distribution for a single text.

        Args:
            text: Input text string.
            return_labels: If True, return binary labels instead of probabilities.

        Returns:
            numpy array of shape (38,) — probabilities or binary labels.
        """
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length",
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        if return_labels:
            return (probs >= SIGMOID_THRESHOLD).astype(np.float32)

        return probs

    def predict_batch(
        self,
        texts: list[str],
        batch_size: int = 16,
        return_labels: bool = False,
    ) -> np.ndarray:
        """Predict value distributions for a batch of texts.

        Args:
            texts: List of input text strings.
            batch_size: Batch size for inference.
            return_labels: If True, return binary labels instead of probabilities.

        Returns:
            numpy array of shape (N, 38).
        """
        all_probs = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                truncation=True,
                max_length=MAX_SEQ_LENGTH,
                padding="max_length",
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.sigmoid(logits).cpu().numpy()

            all_probs.append(probs)

        result = np.concatenate(all_probs, axis=0)

        if return_labels:
            return (result >= SIGMOID_THRESHOLD).astype(np.float32)

        return result

    def get_top_values(
        self, text: str, top_k: int = 5
    ) -> list[tuple[str, float]]:
        """Get the top-k activated value dimensions for a text.

        Args:
            text: Input text.
            top_k: Number of top values to return.

        Returns:
            List of (label_name, probability) tuples, sorted descending.
        """
        probs = self.predict(text)
        indices = np.argsort(probs)[::-1][:top_k]
        return [(LABEL_NAMES[i], float(probs[i])) for i in indices]

    def save(self, output_dir: Optional[str] = None) -> str:
        """Save the model and tokenizer to disk.

        Args:
            output_dir: Directory to save to. Defaults to CHECKPOINT_DIR.

        Returns:
            Path where model was saved.
        """
        if output_dir is None:
            output_dir = os.path.join(CHECKPOINT_DIR, "best_model")

        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model saved to: {output_dir}")
        return output_dir


# ─────────────────────────────────────────────────────────────
# Smoke Test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Value Classifier Smoke Test")
    print("=" * 60)

    # Initialize with pretrained DeBERTa (before fine-tuning)
    classifier = ValueClassifier()

    # Test single prediction
    test_text = "I believe in the freedom to express my own ideas and make my own choices."
    print(f"\nTest text: '{test_text}'")

    probs = classifier.predict(test_text)
    print(f"Output shape: {probs.shape}")
    print(f"Output range: [{probs.min():.4f}, {probs.max():.4f}]")

    top_values = classifier.get_top_values(test_text)
    print(f"\nTop 5 values (pre-training, random head):")
    for name, prob in top_values:
        print(f"  {name}: {prob:.4f}")

    # Test batch prediction
    batch = [test_text, "Security and order in society are very important to me."]
    batch_probs = classifier.predict_batch(batch)
    print(f"\nBatch shape: {batch_probs.shape}")

    print("\n✅ Model smoke test passed!")
