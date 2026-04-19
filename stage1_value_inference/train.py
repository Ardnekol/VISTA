"""
Training Script for Multi-Label Value Classifier
=================================================================
Fine-tunes RoBERTa-large or DeBERTa-v3-large on ValuesML / Touché24 data
with HuggingFace Trainer. Supports MPS (Apple Silicon), CUDA, and CPU.

Usage:
    python3 -m stage1_value_inference.train                  # Train RoBERTa (default)
    python3 -m stage1_value_inference.train --model deberta   # Train DeBERTa
    python3 -m stage1_value_inference.train --model roberta   # Train RoBERTa (explicit)
"""

import argparse
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
    DEBERTA_MODEL_NAME,
    NUM_LABELS,
    ID2LABEL,
    LABEL2ID,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_EPOCHS,
    BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    EVAL_BATCH_SIZE,
    WARMUP_RATIO,
    ADAM_EPSILON,
    EARLY_STOPPING_PATIENCE,
    SIGMOID_THRESHOLD,
    CHECKPOINT_DIR,
    ROBERTA_CHECKPOINT_DIR,
    DEBERTA_CHECKPOINT_DIR,
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
    # Use Sigmoid for multi-label classification metrics (Presence/Absence)
    # This is correct even if we use Softmax for the final decision distribution
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= SIGMOID_THRESHOLD).astype(np.float32)

    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    precision = precision_score(labels, preds, average="macro", zero_division=0)
    recall = recall_score(labels, preds, average="macro", zero_division=0)

    return {
        "f1_macro": f1,
        "precision_macro": precision,
        "recall_macro": recall,
    }


def train(model_type: str = "roberta"):
    """Run the full training pipeline.

    Args:
        model_type: Which model to fine-tune — 'roberta' or 'deberta'.
    """
    # Resolve model name and checkpoint dir based on model_type
    if model_type == "deberta":
        base_model_name = DEBERTA_MODEL_NAME
        best_model_dir = DEBERTA_CHECKPOINT_DIR
        display_name = "DeBERTa-v3-large"
    else:
        base_model_name = MODEL_NAME
        best_model_dir = ROBERTA_CHECKPOINT_DIR
        display_name = "RoBERTa-large"

    print("=" * 60)
    print(f"VISTA Stage 1: Training {display_name} Value Classifier")
    print("=" * 60)

    # Detect device capabilities
    # NOTE: DeBERTa-v3's disentangled attention is numerically unstable with
    # bf16/fp16 mixed precision — it causes model collapse (identical outputs
    # for all inputs). We MUST train in full fp32 for DeBERTa.
    # RoBERTa is more stable but we use fp32 for consistency.
    use_fp16 = False
    use_bf16 = False
    if torch.cuda.is_available():
        print(f"Using CUDA with fp32 ({display_name} trains in full precision)")
    elif torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon) with fp32")
    else:
        print("Using CPU")

    # Load tokenizer and data
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    print("Loading datasets...")
    splits = load_all_splits(tokenizer)
    train_dataset = splits["train"]
    eval_dataset = splits.get("validation", splits.get("test"))

    # Set format for PyTorch
    train_dataset.set_format("torch")
    if eval_dataset:
        eval_dataset.set_format("torch")

    # Load model
    print(f"\nLoading {display_name}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification",
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # Training arguments
    output_dir = os.path.join(CHECKPOINT_DIR, f"{model_type}_training_runs")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        adam_epsilon=ADAM_EPSILON,
        max_grad_norm=0.3,  # Ultra-strict clipping for DeBERTa stability
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
        processing_class=tokenizer,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)
        ],
    )

    # Train
    print("\nStarting training...")
    train_result = trainer.train()

    # Save best model
    os.makedirs(best_model_dir, exist_ok=True)
    print(f"\nSaving best {display_name} model to: {best_model_dir}")
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)

    # Final evaluation
    print("\nFinal evaluation:")
    eval_results = trainer.evaluate()
    for key, value in eval_results.items():
        print(f"  {key}: {value:.4f}")

    # Training stats
    print(f"\nTraining completed!")
    print(f"  Model: {display_name}")
    print(f"  Total steps: {train_result.global_step}")
    print(f"  Training loss: {train_result.training_loss:.4f}")
    print(f"  Checkpoint: {best_model_dir}")

    return trainer, eval_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train VISTA value classifier (RoBERTa or DeBERTa)"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["roberta", "deberta"],
        default="roberta",
        help="Which model to fine-tune: 'roberta' (default) or 'deberta'",
    )
    args = parser.parse_args()
    train(model_type=args.model)

