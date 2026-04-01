"""
VISTA Configuration
===================
Central configuration for the Value-Informed Situated Tactical Agent framework.
Contains all 19 Schwartz value definitions, label mappings, model hyperparameters,
and file path constants.
"""

import os
import numpy as np

# ─────────────────────────────────────────────────────────────
# Project Paths
# ─────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ValuesML / Touché24 data paths
VALUEEVAL_DIR = os.path.join(PROJECT_ROOT, "Touché24-ValueEval", "valueeval24")
TRAIN_DIR = os.path.join(VALUEEVAL_DIR, "training-english")
VAL_DIR = os.path.join(VALUEEVAL_DIR, "validation-english")
TEST_DIR = os.path.join(VALUEEVAL_DIR, "test-english")

# Output paths
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# Schwartz 19 Value Taxonomy
# ─────────────────────────────────────────────────────────────
SCHWARTZ_VALUES = [
    "Self-direction: thought",
    "Self-direction: action",
    "Stimulation",
    "Hedonism",
    "Achievement",
    "Power: dominance",
    "Power: resources",
    "Face",
    "Security: personal",
    "Security: societal",
    "Tradition",
    "Conformity: rules",
    "Conformity: interpersonal",
    "Humility",
    "Benevolence: caring",
    "Benevolence: dependability",
    "Universalism: concern",
    "Universalism: nature",
    "Universalism: tolerance",
]

# The 38 label columns: each value has an "attained" and "constrained" variant
LABEL_NAMES = []
for v in SCHWARTZ_VALUES:
    LABEL_NAMES.append(f"{v} attained")
    LABEL_NAMES.append(f"{v} constrained")

NUM_LABELS = len(LABEL_NAMES)  # 38

# Mappings for the model head
ID2LABEL = {i: name for i, name in enumerate(LABEL_NAMES)}
LABEL2ID = {name: i for i, name in enumerate(LABEL_NAMES)}

# Convenience: map value name to (attained_idx, constrained_idx) pair
VALUE_TO_INDICES = {}
for i, v in enumerate(SCHWARTZ_VALUES):
    VALUE_TO_INDICES[v] = {
        "attained": i * 2,
        "constrained": i * 2 + 1,
    }

# ─────────────────────────────────────────────────────────────
# Model Configuration
# ─────────────────────────────────────────────────────────────
MODEL_NAME = "microsoft/deberta-v3-large"
MAX_SEQ_LENGTH = 256

# Training hyperparameters
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 5
BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16
WARMUP_RATIO = 0.1
EARLY_STOPPING_PATIENCE = 2

# Inference
SIGMOID_THRESHOLD = 0.5

# ─────────────────────────────────────────────────────────────
# Simulation
# ─────────────────────────────────────────────────────────────
NUM_SIMULATION_SCENARIOS = 100
RANDOM_SEED = 42
