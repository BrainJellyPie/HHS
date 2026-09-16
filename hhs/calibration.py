"""Calibration routines.

Three quantities are estimated per checkpoint:

* the tokens-per-word ratio distribution used to turn a word request into a
  target stopping-step distribution (see :mod:`hhs.targets`),
* a survivor-conditioned per-step stopping mass, used only by the plug-in
  marginal-hazard ablation,
* the shared offset of the constant-bias baseline.

The proposed controller needs none of these: its offset follows in closed form
from the current next-token logits.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch

from .config import (BIAS_HALF_RANGE, BIAS_STEPS, MAX_STOP_STEP, METHOD_CONST_BIAS,
                     METHOD_INSTRUCTION, MQ_TRAJ, REQUESTED_WORDS, SEED)
from .generation import generate_one
from .prompts import render_prompt
from .textutils import words

CALIB_PROMPT = "Discuss this topic briefly: "


def first_token_stop_mass(state, texts: Sequence[str], device) -> float:
    """Average first-step stopping mass under unmodified decoding.

    Used as the smoothing prior of the marginal-hazard estimate.
    """
    probs = []
    for text in texts:
        rendered = render_prompt(state.tokenizer, CALIB_PROMPT + str(text)[:300],
                                 12, METHOD_INSTRUCTION, False)
        batch = state.tokenizer(rendered, return_tensors="pt", truncation=True,
                                max_length=128, add_special_tokens=False)
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        with torch.inference_mode():
            logits = state.model(input_ids=ids, attention_mask=mask).logits[:, -1, :].float()
        log_stop = torch.logsumexp(logits[:, state.stop_ids], dim=-1)
        log_all = torch.logsumexp(logits, dim=-1)
        probs.append(float(torch.exp(log_stop - log_all).item()))
    mq = sum(probs) / len(probs) if probs else 1e-4
    return min(max(mq, 1e-4), 1.0 - 1e-4)


def estimate_marginal_schedule(state, texts: Sequence[str], device,
                               max_step: int = MAX_STOP_STEP) -> tuple[float, list[float]]:
    """Survivor-conditioned per-step stopping mass from unsteered trajectories.

    The estimate is smoothed additively toward the first-token prior:

        q_hat(t) = (sum_{i in S_t} q_t(H_t^i) + prior) / (|S_t| + 1)

    where S_t indexes the calibration trajectories that survive to step t.
    """
    prior = first_token_stop_mass(state, texts, device)
    sums = [0.0] * max_step
    counts = [0] * max_step
    for i, text in enumerate(texts[:MQ_TRAJ]):
        gen = generate_one(state, state.ratios, CALIB_PROMPT + str(text)[:400], 24,
                           METHOD_INSTRUCTION, SEED + 444000 + i, device, max_step,
                           record_trace=True)
        for si, p in enumerate(gen.trace.get("post", [])):
            if math.isfinite(p):
                sums[si] += p
                counts[si] += 1
    schedule = []
    for t in range(max_step):
        q = (sums[t] + prior) / (counts[t] + 1.0)
        schedule.append(min(max(q, 1e-4), 1.0 - 1e-4))
    return prior, schedule


def fit_constant_bias(state, texts: Sequence[str], device,
                      n_prompts: int) -> tuple[float, float]:
    """Search the shared offset of the constant-bias baseline.

    Successive refinement over [-BIAS_HALF_RANGE, BIAS_HALF_RANGE]: stage one
    sweeps the full range, each later stage re-sweeps a window of the previous
    step size around the incumbent. Ties are broken toward the smallest
    magnitude, because word errors are integers and a strict comparison would
    otherwise let candidate order decide the outcome.
    """
    prompts = [(CALIB_PROMPT + str(text)[:400], REQUESTED_WORDS[i % 2])
               for i, text in enumerate(texts[:n_prompts])]
    if not prompts:
        return 0.0, float("nan")

    def score(bias: float) -> float:
        errors = []
        for pi, (prompt, target) in enumerate(prompts):
            gen = generate_one(state, state.ratios, prompt, target, METHOD_CONST_BIAS,
                               SEED + 555000 + pi, device, MAX_STOP_STEP,
                               bias_override=bias)
            errors.append(abs(len(words(gen.text)) - target))
        return sum(errors) / len(errors)

    seen: set[float] = set()
    best_bias, best_mae = 0.0, float("inf")
    lo, hi = -BIAS_HALF_RANGE, BIAS_HALF_RANGE

    def sweep(candidates):
        nonlocal best_bias, best_mae
        for value in candidates:
            key = round(float(value), 4)
            if key in seen or not (lo <= key <= hi):
                continue
            seen.add(key)
            mae = score(key)
            if mae < best_mae - 1e-12 or (abs(mae - best_mae) <= 1e-12
                                          and abs(key) < abs(best_bias)):
                best_mae, best_bias = mae, key

    n_coarse = int(round(2 * BIAS_HALF_RANGE / BIAS_STEPS[0])) + 1
    sweep([round(lo + BIAS_STEPS[0] * i, 4) for i in range(n_coarse)])
    for stage in range(1, len(BIAS_STEPS)):
        radius, step = BIAS_STEPS[stage - 1], BIAS_STEPS[stage]
        n = int(round(2 * radius / step)) + 1
        sweep([round(best_bias - radius + step * i, 4) for i in range(n)])

    return best_bias, best_mae
