"""
Training Script for DeBERTa-v3-large Multi-Label Value Classifier
=================================================================
Fine-tunes on ValuesML / Touché24 data with HuggingFace Trainer.
Supports MPS (Apple Silicon), CUDA, and CPU backends.
"""

import os

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    MODEL_NAME,
    NUM_LABELS,
    ID2LABEL,
    LABEL2ID,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_EPOCHS,
    BATCH_SIZE,
    EVAL_BATCH_SIZE,
    WARMUP_RATIO,
    EARLY_STOPPING_PATIENCE,
    SIGMOID_THRESHOLD,
    CHECKPOINT_DIR,
)
from stage1_value_inference.data_loader import load_all_splits


def compute_metrics(eval_pred) -> dict:
    """Compute macro F1, precision, and recall for multi-label classification.

    Args:
        eval_pred: EvalPrediction with predictions (logits) and label_ids.

    Returns:
        Dict with f1_macro, precision_macro, recall_macro.
    """
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))  # sigmoid
    preds = (probs >= SIGMOID_THRESHOLD).astype(np.float32)

    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    precision = precision_score(labels, preds, average="macro", zero_division=0)
    recall = recall_score(labels, preds, average="macro", zero_division=0)

    return {
        "f1_macro": f1,
        "precision_macro": precision,
        "recall_macro": recall,
    }


def train():
    """Run the full training pipeline."""
    print("=" * 60)
    print("VISTA Stage 1: Training Value Classifier")
    print("=" * 60)

    # Detect device capabilities
    use_fp16 = False
    use_bf16 = False
    if torch.cuda.is_available():
        use_fp16 = True
        print("Using CUDA with fp16")
    elif torch.backends.mps.is_available():
        # MPS doesn't support fp16 training well, use default
        print("Using MPS (Apple Silicon)")
    else:
        print("Using CPU")

    # Load tokenizer and data
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading datasets...")
    splits = load_all_splits(tokenizer)
    train_dataset = splits["train"]
    eval_dataset = splits.get("validation", splits.get("test"))

    # Set format for PyTorch
    train_dataset.set_format("torch")
    if eval_dataset:
        eval_dataset.set_format("torch")

    # Load model
    print("\nLoading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification",
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # Training arguments
    output_dir = os.path.join(CHECKPOINT_DIR, "training_runs")
    best_model_dir = os.path.join(CHECKPOINT_DIR, "best_model")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        fp16=use_fp16,
        bf16=use_bf16,
        dataloader_num_workers=0,
        report_to="none",
        seed=42,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)
        ],
    )

    # Train
    print("\nStarting training...")
    train_result = trainer.train()

    # Save best model
    print(f"\nSaving best model to: {best_model_dir}")
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)

    # Final evaluation
    print("\nFinal evaluation:")
    eval_results = trainer.evaluate()
    for key, value in eval_results.items():
        print(f"  {key}: {value:.4f}")

    # Training stats
    print(f"\nTraining completed!")
    print(f"  Total steps: {train_result.global_step}")
    print(f"  Training loss: {train_result.training_loss:.4f}")

    return trainer, eval_results


if __name__ == "__main__":
    train()
