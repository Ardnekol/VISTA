import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from stage1_value_inference.data_loader import load_all_splits
from config import ID2LABEL, NUM_LABELS
import numpy as np

def diagnose():
    model_name = "microsoft/deberta-v3-base"
    print(f"\n{'='*60}")
    print(f"DIAGNOSING: {model_name}")
    print(f"{'='*60}")

    # 1. Audit Model Architecture
    print("\n1. Auditing Architecture...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification"
    )
    
    print("\nLast few layers of the model:")
    # print(list(model.children())[-1])
    print(model.classifier)
    
    # Check for sigmoid in classifier (AutoModelForSequenceClassification shouldn't have one)
    has_sigmoid = any(isinstance(m, nn.Sigmoid) for m in model.modules())
    print(f"Explicit Sigmoid found in modules: {has_sigmoid}")

    # 2. Test Single Batch Forward/Backward
    print("\n2. Testing Single Batch (LR=5e-6, FP32)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    # Load one batch
    splits = load_all_splits(tokenizer)
    train_dataset = splits["train"]
    train_dataset.set_format("torch")
    loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
    batch = next(iter(loader))
    
    inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
    targets = batch["labels"].to(device)

    # Optimizer with very low LR and Adam Epsilon
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6, eps=1e-6)
    model.train()

    # Forward
    outputs = model(**inputs, labels=targets)
    loss = outputs.loss
    logits = outputs.logits

    print(f"Initial Loss: {loss.item():.4f}")
    print(f"Logits range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
    print(f"Logits mean: {logits.mean().item():.4f}")

    # Backward
    loss.backward()
    
    # Check Gradients
    print("\n3. Gradient Inspection...")
    total_norm = 0
    nan_found = False
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2).item()
            total_norm += param_norm ** 2
            if torch.isnan(param.grad).any():
                print(f"  ⚠️  NaN found in gradient of: {name}")
                nan_found = True
    
    total_norm = total_norm ** 0.5
    print(f"Effective Total Gradient Norm: {total_norm:.4f}")
    
    if nan_found:
        print("\n❌ FAILED: NaN detected in gradients even on first batch at LR 5e-6.")
    elif total_norm == 0:
        print("\n❌ FAILED: Zero gradients detected (Vanishing gradients).")
    elif total_norm > 100:
        print("\n⚠️ WARNING: Very high gradient norm (Exploding gradients).")
    else:
        print("\n✅ SUCCESS: Gradients are stable and within normal range.")

    # 4. Check for double-sigmoid math manually
    # BCEWithLogitsLoss expects raw logits.
    # If loss was calculated internally by HF, let's verify if manually calculating matches.
    criterion = nn.BCEWithLogitsLoss()
    manual_loss = criterion(logits, targets)
    print(f"\n4. Manual Loss Check:")
    print(f"   HF Internal Loss: {loss.item():.4f}")
    print(f"   Manual BCEWithLogitsLoss: {manual_loss.item():.4f}")
    
    diff = abs(loss.item() - manual_loss.item())
    if diff < 1e-5:
        print("   ✅ Loss calculation is consistent (No Sigmoid Double-Dip detected).")
    else:
        print(f"   ⚠️  Loss mismatch ({diff:.6f}). Check activation function usage.")

if __name__ == "__main__":
    diagnose()
