"""
Inference Module for VISTA Value Classifier
============================================
Provides a simple API to load a trained checkpoint and predict
38-dim value distributions for arbitrary text.
"""

import os
from typing import Optional

import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import CHECKPOINT_DIR, MODEL_NAME, LABEL_NAMES
from stage1_value_inference.model import ValueClassifier


# Module-level singleton for the classifier
_classifier: Optional[ValueClassifier] = None


def get_classifier(checkpoint_path: Optional[str] = None) -> ValueClassifier:
    """Get or create the singleton ValueClassifier instance.

    Tries to load from a fine-tuned checkpoint first; falls back to
    the pretrained DeBERTa model if no checkpoint exists.

    Args:
        checkpoint_path: Optional explicit path to a checkpoint.

    Returns:
        Initialized ValueClassifier.
    """
    global _classifier

    if _classifier is not None:
        return _classifier

    if checkpoint_path is None:
        best_model_dir = os.path.join(CHECKPOINT_DIR, "best_model")
        if os.path.isdir(best_model_dir):
            checkpoint_path = best_model_dir
            print(f"Loading fine-tuned model from: {checkpoint_path}")
        else:
            checkpoint_path = MODEL_NAME
            print(f"No fine-tuned checkpoint found. Using pretrained: {checkpoint_path}")

    _classifier = ValueClassifier(model_name_or_path=checkpoint_path)
    return _classifier


def predict(text: str, return_labels: bool = False) -> np.ndarray:
    """Predict the 38-dim value distribution for a single text.

    Args:
        text: Input text string.
        return_labels: If True, return binary (thresholded) labels.

    Returns:
        numpy array of shape (38,).
    """
    classifier = get_classifier()
    return classifier.predict(text, return_labels=return_labels)


def predict_batch(
    texts: list[str], batch_size: int = 16, return_labels: bool = False
) -> np.ndarray:
    """Predict value distributions for a batch of texts.

    Args:
        texts: List of input strings.
        batch_size: Inference batch size.
        return_labels: If True, return binary labels.

    Returns:
        numpy array of shape (N, 38).
    """
    classifier = get_classifier()
    return classifier.predict_batch(texts, batch_size=batch_size, return_labels=return_labels)


def explain_prediction(text: str, top_k: int = 5) -> dict:
    """Get an interpretable breakdown of the value prediction for a text.

    Args:
        text: Input text.
        top_k: Number of top values to show.

    Returns:
        Dict with text, probabilities, top values, and binary labels.
    """
    probs = predict(text)
    labels = predict(text, return_labels=True)

    top_indices = np.argsort(probs)[::-1][:top_k]
    top_values = [(LABEL_NAMES[i], float(probs[i])) for i in top_indices]

    active_labels = [LABEL_NAMES[i] for i in range(len(labels)) if labels[i] > 0]

    return {
        "text": text,
        "probabilities": probs.tolist(),
        "top_values": top_values,
        "active_labels": active_labels,
        "num_active": len(active_labels),
    }


# ─────────────────────────────────────────────────────────────
# Smoke Test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Predict Module Smoke Test")
    print("=" * 60)

    test_sentences = [
        "I want the freedom to think for myself and reach my own conclusions.",
        "The most important thing is keeping my family safe from harm.",
        "We should follow the rules and respect authority.",
        "I love trying new things and seeking out adventures.",
        "Everyone deserves to be treated equally regardless of background.",
    ]

    for text in test_sentences:
        result = explain_prediction(text, top_k=3)
        print(f"\n📝 '{text[:60]}...'")
        print(f"   Active labels: {result['num_active']}")
        print(f"   Top values:")
        for name, prob in result["top_values"]:
            print(f"     {name}: {prob:.4f}")

    print("\n✅ Predict smoke test passed!")
