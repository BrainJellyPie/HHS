"""Constrained decoding loop.

One generation runs a plain autoregressive loop with a key-value cache. At each
step the next-token logits are transformed according to the active method, then
sampled multinomially. The loop records the stopping step and how the generation
terminated, which the distributional metrics depend on.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch

from . import controller as ctrl
from .config import (CONTROLLED, MAX_PROMPT_TOKENS, METHOD_CONST_BIAS,
                     METHOD_EOS_ONLY, METHOD_FIXED, METHOD_HARD_CUT,
                     METHOD_MARGINAL, METHOD_PERMUTED, METHOD_REMAINING,
                     METHOD_TEMP, ABLATION_SEED)
from .prompts import render_prompt
from .targets import (distribution_hazards, make_distribution,
                      permuted_distribution, stopping_step)
from .textutils import words

# Termination categories. Only "natural" and "forced" are sampled stopping
# events; the other two leave the stopping time unobserved (right censored).
NATURAL = "natural"
FORCED = "forced_stop"
EXTERNAL = "external_cutoff"
BUDGET = "budget_exhausted"


@dataclass
class Generation:
    text: str = ""
    stop_step: Optional[int] = None
    term_step: int = 0
    event_observed: int = 0
    termination: str = BUDGET
    forced: int = 0
    nonfinite_steps: int = 0
    hazard_error: float = float("nan")
    stopset_mass: float = float("nan")
    seconds: float = 0.0
    trace: dict = field(default_factory=dict)


class ModelState:
    """Everything the decoder needs for one checkpoint."""

    def __init__(self, model, tokenizer, stop_ids, primary, ratios,
                 model_id="", fitted_bias=0.0, marginal_q=None):
        self.model = model
        self.tokenizer = tokenizer
        self.stop_ids = list(stop_ids)
        self.primary = int(primary)
        self.ratios = list(ratios)
        self.model_id = model_id
        self.fitted_bias = float(fitted_bias)
        self.marginal_q = list(marginal_q) if marginal_q else None


def _effective_temperature(method: str, temperature: float) -> float:
    return 0.7 if method == METHOD_TEMP else float(temperature)


def generate_one(state: ModelState,
                 ratios: Sequence[float],
                 prompt: str,
                 requested_words: float,
                 method: str,
                 seed: int,
                 device,
                 max_step: int,
                 temperature: float = 1.0,
                 native_prompt: bool = False,
                 record_trace: bool = False,
                 bias_override: Optional[float] = None) -> Generation:
    """Decode one output under the given method."""
    tokenizer = state.tokenizer
    model = state.model
    eff_temp = _effective_temperature(method, temperature)

    rendered = render_prompt(tokenizer, prompt, requested_words, method, native_prompt)
    batch = tokenizer(rendered, return_tensors="pt", truncation=True,
                      max_length=MAX_PROMPT_TOKENS, add_special_tokens=False)
    prompt_ids = batch["input_ids"].to(device)

    cache = {"past": None, "len": 0}

    def forward(ids):
        attn = torch.ones((1, cache["len"] + ids.shape[1]), dtype=torch.long, device=device)
        with torch.inference_mode():
            out = model(input_ids=ids, attention_mask=attn,
                        past_key_values=cache["past"], use_cache=True, return_dict=True)
        cache["past"] = out.past_key_values
        cache["len"] += ids.shape[1]
        return out.logits[:, -1, :].float()

    distribution = make_distribution(requested_words, ratios,
                                     method == METHOD_FIXED, max_step)
    if method == METHOD_PERMUTED:
        distribution = permuted_distribution(distribution, ABLATION_SEED)
    hazards = distribution_hazards(distribution)

    mean_ratio = sum(ratios) / len(ratios)
    cutoff_step = stopping_step(requested_words, mean_ratio, max_step)
    bias = state.fitted_bias if bias_override is None else float(bias_override)
    control_ids = [state.primary] if method == METHOD_EOS_ONLY else state.stop_ids
    controlled = method in CONTROLLED

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    gen = Generation(term_step=max_step)
    content_ids: list[int] = []
    hazard_errors: list[float] = []
    stopset_vals: list[float] = []
    tr_post, tr_offset, tr_target, tr_mask = [], [], [], []

    start = time.perf_counter()
    next_logits = forward(prompt_ids)

    for step in range(1, max_step + 1):
        if method == METHOD_HARD_CUT and step >= cutoff_step:
            # External truncation: no stopping token was sampled.
            gen.termination = EXTERNAL
            gen.event_observed = 0
            gen.term_step = step
            break

        h = hazards[step - 1]
        mask = ctrl.MASK_NONE
        offset = float("nan")

        if controlled:
            if method == METHOD_MARGINAL and 0.0 < h < 1.0:
                q = state.marginal_q[min(step - 1, len(state.marginal_q) - 1)]
                res = ctrl.steer_with_fixed_mass(next_logits, control_ids, h, q)
            else:
                res = ctrl.steer(next_logits, control_ids, h)
            modified, offset, mask = res.logits, res.offset, res.mask
            gen.nonfinite_steps += int(res.nonfinite)
        elif method == METHOD_CONST_BIAS:
            modified = ctrl.add_constant_bias(next_logits, state.stop_ids, bias)
        else:
            modified = next_logits

        scaled = modified / eff_temp
        post = ctrl.realized_stop_mass(modified, control_ids, eff_temp)
        if method == METHOD_EOS_ONLY:
            stopset_vals.append(ctrl.realized_stop_mass(modified, state.stop_ids, eff_temp))
        if controlled:
            hazard_errors.append(abs(post - min(max(h, 0.0), 1.0)))
        if record_trace:
            tr_post.append(post)
            tr_offset.append(offset)
            tr_target.append(min(max(h, 0.0), 1.0) if controlled else float("nan"))
            tr_mask.append(mask)

        probs = torch.softmax(scaled, dim=-1)
        token = torch.multinomial(probs, num_samples=1)
        tid = int(token.item())

        if tid in state.stop_ids:
            gen.stop_step = step
            gen.term_step = step
            gen.event_observed = 1
            if mask == ctrl.MASK_UNIT:
                gen.termination = FORCED
                gen.forced = 1
            else:
                gen.termination = NATURAL
            break

        content_ids.append(tid)
        if step == max_step:
            gen.term_step = step
            break

        if method == METHOD_REMAINING and len(content_ids) % 8 == 0:
            text_now = tokenizer.decode(content_ids, skip_special_tokens=True)
            remaining = max(0, int(round(requested_words)) - len(words(text_now)))
            note = tokenizer.encode(f" [REMAINING: {remaining} words]",
                                    add_special_tokens=False)
            last = forward(token)
            for sid in note:
                last = forward(torch.tensor([[sid]], dtype=token.dtype, device=device))
            next_logits = last
        else:
            next_logits = forward(token)

    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    gen.seconds = time.perf_counter() - start
    gen.text = tokenizer.decode(content_ids, skip_special_tokens=True).strip()
    if hazard_errors:
        gen.hazard_error = sum(hazard_errors) / len(hazard_errors)
    if stopset_vals:
        gen.stopset_mass = sum(stopset_vals) / len(stopset_vals)
    if record_trace:
        gen.trace = {"post": tr_post, "offset": tr_offset,
                     "target": tr_target, "mask": tr_mask}
    return gen
