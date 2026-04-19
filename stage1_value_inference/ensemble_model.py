"""
Ensemble Value Classifier
==========================
Combines RoBERTa-large and DeBERTa-v3-large via weighted logit fusion.
Both models independently produce 38-dim logit vectors, which are
merged using a configurable alpha weight before temperature-scaled
softmax normalization.

Ensemble Formula:
    logits_fused = α · logits_roberta + (1 - α) · logits_deberta
    V_ensemble = softmax(logits_fused / T)

Falls back to single-model mode if one model's checkpoint is missing.
"""

import os
from typing import Optional

import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    MODEL_NAME,
    DEBERTA_MODEL_NAME,
    NUM_LABELS,
    LABEL_NAMES,
    SIGMOID_THRESHOLD,
    ROBERTA_CHECKPOINT_DIR,
    DEBERTA_CHECKPOINT_DIR,
    ENSEMBLE_ALPHA,
)
from stage1_value_inference.model import ValueClassifier


class EnsembleValueClassifier:
    """Ensemble value classifier combining RoBERTa-large and DeBERTa-v3-large.

    Fuses raw logits from both models via weighted average, then applies
    temperature-scaled softmax to produce the final 38-dim value vector.

    Exposes the same API as ValueClassifier (predict, predict_batch,
    get_top_values) for seamless integration with downstream code.
    """

    def __init__(
        self,
        roberta_path: Optional[str] = None,
        deberta_path: Optional[str] = None,
        alpha: float = ENSEMBLE_ALPHA,
        device: Optional[str] = None,
    ):
        """Initialize the ensemble classifier.

        Args:
            roberta_path: Path to RoBERTa checkpoint or HuggingFace model name.
            deberta_path: Path to DeBERTa checkpoint or HuggingFace model name.
            alpha: Ensemble weight for RoBERTa. DeBERTa gets (1 - alpha).
            device: Device string. Auto-detected if None.
        """
        self.alpha = alpha
        self.models_loaded = []

        # Resolve RoBERTa path
        if roberta_path is None:
            if os.path.isdir(ROBERTA_CHECKPOINT_DIR):
                roberta_path = ROBERTA_CHECKPOINT_DIR
                print(f"[Ensemble] RoBERTa: fine-tuned checkpoint → {roberta_path}")
            else:
                roberta_path = MODEL_NAME
                print(f"[Ensemble] RoBERTa: pretrained → {roberta_path}")

        # Resolve DeBERTa path
        if deberta_path is None:
            if os.path.isdir(DEBERTA_CHECKPOINT_DIR):
                deberta_path = DEBERTA_CHECKPOINT_DIR
                print(f"[Ensemble] DeBERTa: fine-tuned checkpoint → {deberta_path}")
            else:
                deberta_path = DEBERTA_MODEL_NAME
                print(f"[Ensemble] DeBERTa: pretrained → {deberta_path}")

        # Load RoBERTa
        print("\n[Ensemble] Loading RoBERTa-large...")
        try:
            self.roberta = ValueClassifier(
                model_name_or_path=roberta_path, device=device
            )
            self.models_loaded.append("roberta")
        except Exception as e:
            print(f"[Ensemble] ⚠️  Failed to load RoBERTa: {e}")
            self.roberta = None

        # Load DeBERTa
        print("\n[Ensemble] Loading DeBERTa-v3-large...")
        try:
            self.deberta = ValueClassifier(
                model_name_or_path=deberta_path, device=device
            )
            self.models_loaded.append("deberta")
        except Exception as e:
            print(f"[Ensemble] ⚠️  Failed to load DeBERTa: {e}")
            self.deberta = None

        if not self.models_loaded:
            raise RuntimeError(
                "[Ensemble] No models could be loaded. Cannot proceed."
            )

        if len(self.models_loaded) == 1:
            print(
                f"\n[Ensemble] ⚠️  Running in single-model mode: {self.models_loaded[0]}"
            )
        else:
            print(
                f"\n[Ensemble] ✅ Both models loaded. α={self.alpha:.2f} "
                f"(RoBERTa={self.alpha:.0%}, DeBERTa={1-self.alpha:.0%})"
            )

    def _fuse_logits(
        self, logits_r: Optional[np.ndarray], logits_d: Optional[np.ndarray]
    ) -> np.ndarray:
        """Fuse logits from both models via weighted average.

        Args:
            logits_r: RoBERTa logits, shape (38,) or (N, 38). None if unavailable.
            logits_d: DeBERTa logits, shape (38,) or (N, 38). None if unavailable.

        Returns:
            Fused logits array.
        """
        if logits_r is not None and logits_d is not None:
            return self.alpha * logits_r + (1 - self.alpha) * logits_d
        elif logits_r is not None:
            return logits_r
        else:
            return logits_d

    def predict_logits(self, text: str) -> np.ndarray:
        """Get fused logits for a single text.

        Args:
            text: Input text string.

        Returns:
            Fused logits, shape (38,).
        """
        logits_r = self.roberta.predict_logits(text) if self.roberta else None
        logits_d = self.deberta.predict_logits(text) if self.deberta else None
        return self._fuse_logits(logits_r, logits_d)

    def predict_batch_logits(
        self, texts: list[str], batch_size: int = 16
    ) -> np.ndarray:
        """Get fused logits for a batch of texts.

        Args:
            texts: List of input text strings.
            batch_size: Batch size for inference.

        Returns:
            Fused logits, shape (N, 38).
        """
        logits_r = (
            self.roberta.predict_batch_logits(texts, batch_size=batch_size)
            if self.roberta
            else None
        )
        logits_d = (
            self.deberta.predict_batch_logits(texts, batch_size=batch_size)
            if self.deberta
            else None
        )
        return self._fuse_logits(logits_r, logits_d)

    def predict(self, text: str, return_labels: bool = False) -> np.ndarray:
        """Predict ensemble value distribution for a single text.

        Args:
            text: Input text string.
            return_labels: If True, return binary (thresholded) labels.

        Returns:
            numpy array of shape (38,).
        """
        fused_logits = self.predict_logits(text)
        probs = ValueClassifier.logits_to_probs(fused_logits)

        if return_labels:
            return (probs >= SIGMOID_THRESHOLD).astype(np.float32)

        return probs

    def predict_batch(
        self,
        texts: list[str],
        batch_size: int = 16,
        return_labels: bool = False,
    ) -> np.ndarray:
        """Predict ensemble value distributions for a batch of texts.

        Args:
            texts: List of input strings.
            batch_size: Inference batch size.
            return_labels: If True, return binary labels.

        Returns:
            numpy array of shape (N, 38).
        """
        fused_logits = self.predict_batch_logits(texts, batch_size=batch_size)
        result = ValueClassifier.logits_to_probs(fused_logits)

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

    def model_agreement(self, text: str) -> dict:
        """Measure how much the two models agree on a text.

        Useful for diagnosing which texts benefit most from ensembling.

        Args:
            text: Input text.

        Returns:
            Dict with individual model probs, cosine similarity, and top value agreement.
        """
        if not self.roberta or not self.deberta:
            return {"error": "Both models required for agreement analysis"}

        logits_r = self.roberta.predict_logits(text)
        logits_d = self.deberta.predict_logits(text)

        probs_r = ValueClassifier.logits_to_probs(logits_r)
        probs_d = ValueClassifier.logits_to_probs(logits_d)

        # Cosine similarity
        cos_sim = float(
            np.dot(probs_r, probs_d)
            / (np.linalg.norm(probs_r) * np.linalg.norm(probs_d) + 1e-8)
        )

        # Top-1 agreement
        top_r = LABEL_NAMES[np.argmax(probs_r)]
        top_d = LABEL_NAMES[np.argmax(probs_d)]

        return {
            "cosine_similarity": cos_sim,
            "top1_roberta": top_r,
            "top1_deberta": top_d,
            "top1_agree": top_r == top_d,
            "roberta_probs": probs_r,
            "deberta_probs": probs_d,
        }


# ─────────────────────────────────────────────────────────────
# Smoke Test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Ensemble Value Classifier Smoke Test")
    print("=" * 60)

    ensemble = EnsembleValueClassifier()

    test_text = "I believe in the freedom to express my own ideas and make my own choices."
    print(f"\nTest text: '{test_text}'")

    # Test single prediction
    probs = ensemble.predict(test_text)
    print(f"Output shape: {probs.shape}")
    print(f"Output range: [{probs.min():.4f}, {probs.max():.4f}]")
    print(f"Sum: {probs.sum():.4f}")

    # Test top values
    top_values = ensemble.get_top_values(test_text)
    print(f"\nTop 5 values (ensemble):")
    for name, prob in top_values:
        print(f"  {name}: {prob:.4f}")

    # Test batch prediction
    batch = [
        test_text,
        "Security and order in society are very important to me.",
    ]
    batch_probs = ensemble.predict_batch(batch)
    print(f"\nBatch shape: {batch_probs.shape}")

    # Test model agreement (if both loaded)
    if len(ensemble.models_loaded) == 2:
        agreement = ensemble.model_agreement(test_text)
        print(f"\nModel Agreement:")
        print(f"  Cosine similarity: {agreement['cosine_similarity']:.4f}")
        print(f"  RoBERTa top-1: {agreement['top1_roberta']}")
        print(f"  DeBERTa top-1: {agreement['top1_deberta']}")
        print(f"  Agree on top-1: {agreement['top1_agree']}")

    print(f"\nModels loaded: {ensemble.models_loaded}")
    print("\n✅ Ensemble smoke test passed!")
