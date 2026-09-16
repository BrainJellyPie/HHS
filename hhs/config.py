"""Experiment configuration.

Cache locations are read from the environment so that no absolute path is baked
into the release. Set HHS_CACHE to control where models and datasets are stored;
it defaults to a ``cache`` directory next to the package.
"""

from __future__ import annotations

import os
from pathlib import Path

# ----------------------------------------------------------------- caches

CACHE_ROOT = Path(os.environ.get("HHS_CACHE", Path.cwd() / "cache"))
MODEL_CACHE = CACHE_ROOT / "models"
DATA_CACHE = CACHE_ROOT / "datasets"

# ----------------------------------------------------------------- models
# (hub id, trust_remote_code)

MODELS = [
    ("tiiuae/Falcon-H1-Tiny-R-0.6B", True),
    ("openbmb/MiniCPM5-1B", False),
    ("LiquidAI/LFM2.5-1.2B-Instruct", True),
    ("Qwen/Qwen3-4B", False),
]

SHORT_MODEL = {
    "tiiuae/Falcon-H1-Tiny-R-0.6B": "Falcon-H1-0.6B",
    "openbmb/MiniCPM5-1B": "MiniCPM5-1B",
    "LiquidAI/LFM2.5-1.2B-Instruct": "LFM2.5-1.2B",
    "Qwen/Qwen3-4B": "Qwen3-4B",
}

# ----------------------------------------------------------------- datasets
# (display name, hub repo, config, split)

DATASETS = [
    ("SAMSum", "knkarthick/samsum", None, "test"),
    ("DialogSum", "knkarthick/dialogsum", None, "test"),
    ("XSum", "EdinburghNLP/xsum", None, "test"),
    ("GSM8K", "openai/gsm8k", "main", "test"),
    ("BoolQ", "google/boolq", None, "validation"),
    ("HellaSwag", "Rowan/hellaswag", None, "validation"),
    ("ARC-Challenge", "allenai/ai2_arc", "ARC-Challenge", "test"),
    ("OpenBookQA", "allenai/openbookqa", "main", "test"),
]

DATASET_NAMES = [d[0] for d in DATASETS]
SUMM_DATASETS = {"SAMSum", "DialogSum", "XSum"}
QA_DATASETS = ["GSM8K", "BoolQ", "HellaSwag", "ARC-Challenge", "OpenBookQA"]

# ----------------------------------------------------------------- methods

METHOD_HAZARD = "Historywise hazard"
METHOD_MARGINAL = "Marginal-q schedule"
METHOD_PERMUTED = "Permuted schedule"
METHOD_FIXED = "Fixed ratio"
METHOD_EOS_ONLY = "EOS-only control"
METHOD_TEMP = "Temperature 0.7"
METHOD_CONST_BIAS = "Constant stop-set bias"
METHOD_HARD_CUT = "Hard token cutoff"
METHOD_INSTRUCTION = "Instruction only"
METHOD_COUNTDOWN = "One-shot countdown"
METHOD_REMAINING = "Remaining prompt"

METHODS = [
    METHOD_INSTRUCTION,
    METHOD_COUNTDOWN,
    METHOD_REMAINING,
    METHOD_CONST_BIAS,
    METHOD_HARD_CUT,
    METHOD_HAZARD,
    METHOD_MARGINAL,
    METHOD_PERMUTED,
    METHOD_FIXED,
    METHOD_EOS_ONLY,
    METHOD_TEMP,
]

# Methods that drive the stopping-token logits through a hazard schedule.
CONTROLLED = {
    METHOD_HAZARD,
    METHOD_MARGINAL,
    METHOD_PERMUTED,
    METHOD_FIXED,
    METHOD_EOS_ONLY,
    METHOD_TEMP,
}

SHORT_METHOD = {
    METHOD_INSTRUCTION: "Instr",
    METHOD_COUNTDOWN: "Countdown",
    METHOD_REMAINING: "Remain",
    METHOD_CONST_BIAS: "ConstBias",
    METHOD_HARD_CUT: "HardCut",
    METHOD_HAZARD: "Hazard",
    METHOD_MARGINAL: "MargQ",
    METHOD_PERMUTED: "Permut",
    METHOD_FIXED: "FixRatio",
    METHOD_EOS_ONLY: "EOSonly",
    METHOD_TEMP: "T=0.7",
}

# ----------------------------------------------------------------- protocol

CORE_N = 50                 # examples per dataset in the core matrix
REQUESTED_WORDS = (12, 24)  # alternating by example parity
MAX_STOP_STEP = 48          # token budget for the core matrix
MAX_PROMPT_TOKENS = 384     # rendered prompt truncation
TOLERANCE = 0.2             # compliance band

SEED = 2027
ABLATION_SEED = 90210
N_BOOT = 2000

# ----------------------------------------------------------------- calibration

CALIB_N = 250      # texts used for the tokens-per-word ratio distribution
BIAS_POOL_N = 24   # held-out texts reserved for the constant-bias search
FIT_PROMPTS = 16   # prompts actually used by the search (<= BIAS_POOL_N)
MQ_TRAJ = 20       # unsteered trajectories for the marginal-hazard estimate

# Constant-bias search: successive refinement over [-HALF, HALF]. Stage 1 sweeps
# the full range; each later stage re-sweeps a window of the previous step size
# around the incumbent.
BIAS_HALF_RANGE = 24.0
BIAS_STEPS = (3.0, 0.75, 0.25)

# ----------------------------------------------------------------- regimes

LF_TARGETS = (12, 24, 60)
LF_STEPS = {12: 48, 24: 48, 60: 120}
LF_N = 30
CAL_SIZES = (25, 50, 100, 250)
CAL_N = 30
SEED_LIST = (2027, 4054, 6081)
SEED_N = 20
SURV_N = 50
IFEVAL_MAX_WORDS = 200
