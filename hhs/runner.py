"""Experiment orchestration.

The runner loads one checkpoint at a time, calibrates it, and sweeps the core
matrix of (dataset, example, method). Optional regimes add the IFEval subset and
the length-regime sweep. Results are returned as flat row dictionaries so the
report and any downstream analysis can work from a single table.
"""

from __future__ import annotations

import gc
import json
from typing import Optional, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .calibration import estimate_marginal_schedule, fit_constant_bias
from .config import (CORE_N, DATASETS, FIT_PROMPTS, IFEVAL_MAX_WORDS, LF_N,
                     LF_STEPS, LF_TARGETS, MAX_STOP_STEP, METHODS,
                     METHOD_CONST_BIAS, METHOD_HAZARD, METHOD_INSTRUCTION,
                     METHOD_REMAINING, MODEL_CACHE, MODELS, REQUESTED_WORDS,
                     SEED, SHORT_MODEL, SUMM_DATASETS, TOLERANCE)
from .data import (load_calibration_texts, load_core_datasets, load_ifeval_records)
from .generation import ModelState, Generation, generate_one
from .prompts import stopping_ids
from .scoring import answer_correct, feasible_distance, ifeval_pass
from .targets import ratio_distribution
from .textutils import (complete_sentence, ends_with_terminal_punctuation,
                        repeated_fourgram_rate, rouge_l_f1, unigram_f1, words)

IFEVAL_METHODS = [METHOD_HAZARD, METHOD_INSTRUCTION, METHOD_CONST_BIAS, METHOD_REMAINING]
LF_METHODS = [METHOD_HAZARD, METHOD_CONST_BIAS, "Hard token cutoff", METHOD_REMAINING]


def stable_offset(name: str) -> int:
    """Deterministic per-dataset seed offset, stable across processes."""
    value = 0
    for ch in name:
        value = (value * 131 + ord(ch)) % 1000
    return value * 100000


def load_checkpoint(model_id: str, trust_remote_code: bool, device,
                    calibration_texts: Sequence[str],
                    bias_texts: Sequence[str],
                    verbose: bool = True) -> ModelState:
    """Load a checkpoint and run its three calibration steps."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=str(MODEL_CACHE),
                                              trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=str(MODEL_CACHE),
                                                 trust_remote_code=trust_remote_code,
                                                 torch_dtype=torch.bfloat16).to(device)
    model.eval()

    stop_ids, primary = stopping_ids(model, tokenizer)
    ratios = ratio_distribution(tokenizer, calibration_texts)
    state = ModelState(model, tokenizer, stop_ids, primary, ratios, model_id=model_id)

    _, schedule = estimate_marginal_schedule(state, calibration_texts, device)
    state.marginal_q = schedule
    bias, fit_mae = fit_constant_bias(state, bias_texts, device, FIT_PROMPTS)
    state.fitted_bias = bias

    if verbose:
        print(json.dumps({"model": SHORT_MODEL.get(model_id, model_id),
                          "stop_ids": stop_ids, "primary": primary,
                          "fitted_bias": bias, "fit_mae": fit_mae}), flush=True)
    return state


def release_checkpoint(state: Optional[ModelState]) -> None:
    if state is None:
        return
    state.model = None
    state.tokenizer = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def make_row(state: ModelState, task: str, dataset: str, method: str,
             requested: float, index: int, gen: Generation, record: dict,
             max_step: int, with_overlap: bool) -> dict:
    """Flatten one generation into a row."""
    text = gen.text
    count = len(words(text))
    row = {
        "task": task,
        "model": SHORT_MODEL.get(state.model_id, state.model_id),
        "dataset": dataset,
        "method": method,
        "requested_words": float(requested),
        "example_index": int(index),
        "words": count,
        "word_error": abs(count - requested),
        "compliant": int(abs(count - requested) <= TOLERANCE * requested),
        "stop_step": gen.stop_step,
        "term_step": gen.term_step,
        "max_step": max_step,
        "event": gen.event_observed,
        "natural": int(gen.termination == "natural"),
        "forced": int(gen.forced),
        "external": int(gen.termination == "external_cutoff"),
        "budget": int(gen.termination == "budget_exhausted"),
        "nonfinite": gen.nonfinite_steps,
        "hazard_error": gen.hazard_error,
        "stopset_mass": gen.stopset_mass,
        "repeat4": repeated_fourgram_rate(text),
        "terminal": ends_with_terminal_punctuation(text),
        "complete": complete_sentence(text),
        "seconds": gen.seconds,
        "text": text,
    }
    correct = answer_correct(text, record)
    if correct is not None:
        row["correct"] = int(correct)
    reference = record.get("reference")
    if reference:
        row["ref_f1"] = unigram_f1(text, reference)
        if with_overlap:
            row["rouge_l"] = rouge_l_f1(text, reference)
    return row


def run_core_matrix(state: ModelState, records: dict, device,
                    n_examples: int = CORE_N,
                    methods: Sequence[str] = METHODS,
                    verbose: bool = True) -> list[dict]:
    """Sweep every (dataset, example, method) cell for one checkpoint."""
    rows = []
    for name, recs in records.items():
        if not recs:
            continue
        offset = stable_offset(name)
        is_summ = name in SUMM_DATASETS
        for index, record in enumerate(recs[:n_examples]):
            requested = REQUESTED_WORDS[index % 2]
            for m_index, method in enumerate(methods):
                seed = SEED + offset + index * 100 + m_index
                gen = generate_one(state, state.ratios, record["prompt"], requested,
                                   method, seed, device, MAX_STOP_STEP)
                rows.append(make_row(state, "core", name, method, requested,
                                     index, gen, record, MAX_STOP_STEP, is_summ))
        if verbose:
            print(f"  {SHORT_MODEL.get(state.model_id)} / {name}: "
                  f"{len(recs[:n_examples]) * len(methods)} generations", flush=True)
    return rows


def run_ifeval(state: ModelState, records: Sequence[dict], device,
               methods: Sequence[str] = IFEVAL_METHODS) -> list[dict]:
    """Evaluate the IFEval word-count subset with its own stated constraints."""
    rows = []
    max_ratio = max(state.ratios)
    for index, record in enumerate(records):
        target = record["target_words"]
        max_step = min(320, max(MAX_STOP_STEP, int(round(target * max_ratio)) + 8))
        for m_index, method in enumerate(methods):
            seed = SEED + 800000 + index * 100 + m_index
            gen = generate_one(state, state.ratios, record["prompt"], target, method,
                               seed, device, max_step, native_prompt=True)
            count = len(words(gen.text))
            row = make_row(state, "ifeval", "IFEval", method, target, index, gen,
                           record, max_step, False)
            row["passed"] = int(ifeval_pass(record["relation"], record["num_words"], count))
            row["feasible_distance"] = feasible_distance(record["relation"],
                                                         record["num_words"], count)
            row["compliant"] = row["passed"]
            rows.append(row)
    return rows


def run_length_regimes(state: ModelState, records: Sequence[dict], device,
                       targets: Sequence[int] = LF_TARGETS,
                       n_examples: int = LF_N) -> list[dict]:
    """Sweep requested lengths beyond the core matrix on one dataset."""
    rows = []
    for index, record in enumerate(records[:n_examples]):
        for t_index, target in enumerate(targets):
            max_step = LF_STEPS[target]
            for m_index, method in enumerate(LF_METHODS):
                seed = SEED + 700000 + index * 1000 + t_index * 100 + m_index
                gen = generate_one(state, state.ratios, record["prompt"], target,
                                   method, seed, device, max_step)
                rows.append(make_row(state, "length_regime", "SAMSum", method, target,
                                     index, gen, record, max_step, True))
    return rows


def run_experiment(model_specs=MODELS,
                   dataset_specs=DATASETS,
                   n_examples: int = CORE_N,
                   methods: Sequence[str] = METHODS,
                   device: Optional[str] = None,
                   with_ifeval: bool = True,
                   with_length_regimes: bool = True,
                   verbose: bool = True) -> tuple[list[dict], dict]:
    """Run the full evaluation and return (rows, metadata)."""
    device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))

    if verbose:
        print("loading calibration texts", flush=True)
    calibration_texts, bias_texts = load_calibration_texts()

    if verbose:
        print("loading datasets", flush=True)
    records, dataset_errors = load_core_datasets(dataset_specs, n_examples)

    ifeval_records, ifeval_stats = ([], {})
    if with_ifeval:
        try:
            ifeval_records, ifeval_stats = load_ifeval_records()
        except Exception as error:  # noqa: BLE001
            dataset_errors["IFEval"] = f"{type(error).__name__}: {str(error)[:200]}"

    rows: list[dict] = []
    ratios_by_model: dict = {}
    calibration_info: list[dict] = []
    model_errors: dict = {}

    for model_id, trust_remote_code in model_specs:
        short = SHORT_MODEL.get(model_id, model_id)
        if verbose:
            print(f"\n=== {short} ===", flush=True)
        state = None
        try:
            state = load_checkpoint(model_id, trust_remote_code, device,
                                    calibration_texts, bias_texts, verbose)
            ratios_by_model[short] = state.ratios
            calibration_info.append({"model": short, "stop_ids": state.stop_ids,
                                     "primary": state.primary,
                                     "fitted_bias": state.fitted_bias})
            rows.extend(run_core_matrix(state, records, device, n_examples,
                                        methods, verbose))
            if ifeval_records:
                rows.extend(run_ifeval(state, ifeval_records, device))
            if with_length_regimes and records.get("SAMSum"):
                rows.extend(run_length_regimes(state, records["SAMSum"], device))
        except Exception as error:  # noqa: BLE001
            model_errors[short] = f"{type(error).__name__}: {str(error)[:200]}"
            if verbose:
                print(f"  failed: {model_errors[short]}", flush=True)
        finally:
            release_checkpoint(state)

    meta = {"ratios_by_model": ratios_by_model,
            "calibration": calibration_info,
            "dataset_errors": dataset_errors,
            "model_errors": model_errors,
            "ifeval_stats": ifeval_stats}
    return rows, meta
