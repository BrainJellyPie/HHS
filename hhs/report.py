"""Report generation.

Turns the flat row table into the tables the paper reports: length control,
content metrics, task accuracy, ablations, significance, and IFEval.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .config import (ABLATION_SEED, METHODS, METHOD_CONST_BIAS, METHOD_HAZARD,
                     QA_DATASETS, SUMM_DATASETS)
from .stats import (aggregate, bootstrap_ci, finite_mean, finite_values,
                    grouped_wasserstein, holm_adjust, sign_permutation_p)


def fmt(value, digits: int = 2) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "-"
    return f"{f:.{digits}f}" if math.isfinite(f) else "-"


def _by_method(rows, method):
    return [r for r in rows if r["method"] == method]


def _keyed(rows, method):
    return {(r["model"], r["dataset"], r["example_index"]): r
            for r in rows if r["method"] == method}


def paired_differences(rows, baseline, metric):
    """Per-generation differences between the controller and a baseline."""
    hazard = _keyed(rows, METHOD_HAZARD)
    other = _keyed(rows, baseline)
    a, b = [], []
    for key, row in hazard.items():
        match = other.get(key)
        if match is None:
            continue
        a.append(float(row[metric]))
        b.append(float(match[metric]))
    return np.array(a), np.array(b)


def length_control_table(core, ratios_by_model) -> list[str]:
    lines = ["## Length control",
             "",
             "| Method | Word MAE [95% CI] | Compliance % [95% CI] | W1 | "
             "Natural % | Forced % | Budget % |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for i, method in enumerate(METHODS):
        rows = _by_method(core, method)
        if not rows:
            continue
        agg = aggregate(rows)
        m, ml, mh = bootstrap_ci(finite_values(rows, "word_error"), seed=1 + i)
        c, cl, ch = bootstrap_ci(finite_values(rows, "compliant"), seed=50 + i)
        w1 = grouped_wasserstein(rows, ratios_by_model, ablation_seed=ABLATION_SEED)
        lines.append(f"| {method} | {fmt(m)} [{fmt(ml)},{fmt(mh)}] | "
                     f"{fmt(100*c,1)} [{fmt(100*cl,1)},{fmt(100*ch,1)}] | {fmt(w1)} | "
                     f"{fmt(agg['natural'],1)} | {fmt(agg['forced'],1)} | "
                     f"{fmt(agg['budget'],1)} |")
    lines.append("")
    lines.append("W1 compares observed stopping steps against the same target for "
                 "every method. Budget % counts generations that reach the token "
                 "budget without sampling a stopping token.")
    lines.append("")
    return lines


def content_table(core) -> list[str]:
    summ = [r for r in core if r["dataset"] in SUMM_DATASETS]
    lines = ["## Output content",
             "",
             "| Method | Unigram F1 | ROUGE-L | Repeat-4 | Terminal punct. % |",
             "|---|---:|---:|---:|---:|"]
    for method in METHODS:
        rows = _by_method(core, method)
        if not rows:
            continue
        agg = aggregate(rows)
        rouge = finite_mean(_by_method(summ, method), "rouge_l")
        lines.append(f"| {method} | {fmt(agg['unigram_f1'],3)} | {fmt(rouge,3)} | "
                     f"{fmt(agg['repeat4'],3)} | {fmt(agg['terminal'],1)} |")
    lines.append("")
    return lines


def accuracy_table(core) -> list[str]:
    qa = [r for r in core if r["dataset"] in QA_DATASETS and "correct" in r]
    if not qa:
        return []
    lines = ["## Task accuracy on question answering and reasoning",
             "",
             "| Method | " + " | ".join(QA_DATASETS) + " | Macro | Word MAE | Compliance % |",
             "|---" + "|---:" * (len(QA_DATASETS) + 3) + "|"]
    for method in METHODS:
        rows = _by_method(qa, method)
        if not rows:
            continue
        per = [aggregate([r for r in rows if r["dataset"] == ds])["accuracy"]
               for ds in QA_DATASETS]
        finite = [v for v in per if math.isfinite(v)]
        macro = float(np.mean(finite)) if finite else float("nan")
        agg = aggregate(rows)
        lines.append("| " + method + " | " + " | ".join(fmt(v, 1) for v in per)
                     + f" | {fmt(macro,1)} | {fmt(agg['mae'])} | {fmt(agg['compliance'],1)} |")
    lines.append("")
    lines.append("Scoring: the final number for GSM8K, the first of yes/no for BoolQ, "
                 "and the highest-overlap candidate for the multiple-choice sets. "
                 "All methods share the same prompts, examples, and token budget.")
    lines.append("")
    return lines


def ablation_table(core, ratios_by_model) -> list[str]:
    conditions = [METHOD_HAZARD, "Marginal-q schedule", "Permuted schedule",
                  "Fixed ratio", "EOS-only control", "Temperature 0.7"]
    lines = ["## Ablations",
             "",
             "| Condition | Word MAE | Compliance % | W1 vs target | "
             "W1 vs own schedule | Hazard error |",
             "|---|---:|---:|---:|---:|---:|"]
    for method in conditions:
        rows = _by_method(core, method)
        if not rows:
            continue
        agg = aggregate(rows)
        w1 = grouped_wasserstein(rows, ratios_by_model, ablation_seed=ABLATION_SEED)
        own = (grouped_wasserstein(rows, ratios_by_model, use_permuted=True,
                                   ablation_seed=ABLATION_SEED)
               if method == "Permuted schedule" else float("nan"))
        lines.append(f"| {method} | {fmt(agg['mae'])} | {fmt(agg['compliance'],1)} | "
                     f"{fmt(w1)} | {fmt(own)} | {fmt(agg['hazard_error'],6)} |")
    lines.append("")
    lines.append("Hazard error is the mean absolute difference between the "
                 "post-intervention stopping mass and the target hazard, a numerical "
                 "check of the identity rather than a distributional metric.")
    lines.append("")
    return lines


def significance_table(core) -> list[str]:
    baselines = [m for m in METHODS if m not in (METHOD_HAZARD, "Instruction only")]
    pvalues, records = [], []
    for i, baseline in enumerate(baselines):
        a_mae, b_mae = paired_differences(core, baseline, "word_error")
        a_c, b_c = paired_differences(core, baseline, "compliant")
        if a_mae.size == 0:
            continue
        i_mae = len(pvalues)
        pvalues.append(sign_permutation_p(a_mae - b_mae, seed=300 + i))
        i_comp = len(pvalues)
        pvalues.append(sign_permutation_p(a_c - b_c, seed=400 + i))
        records.append({"baseline": baseline,
                        "d_mae": float(a_mae.mean() - b_mae.mean()),
                        "d_comp": 100.0 * float(a_c.mean() - b_c.mean()),
                        "i_mae": i_mae, "i_comp": i_comp})
    if not records:
        return []
    adjusted = holm_adjust(pvalues)
    lines = [f"## Significance (paired sign-permutation, Holm-adjusted over "
             f"{len(pvalues)} tests)",
             "",
             "| Comparison (controller vs) | dMAE | adj. p | dCompliance (pp) | adj. p |",
             "|---|---:|---:|---:|---:|"]
    for rec in records:
        lines.append(f"| {rec['baseline']} | {rec['d_mae']:+.2f} | "
                     f"{fmt(adjusted[rec['i_mae']],4)} | {rec['d_comp']:+.1f} | "
                     f"{fmt(adjusted[rec['i_comp']],4)} |")
    lines.append("")
    return lines


def ifeval_table(rows) -> list[str]:
    ife = [r for r in rows if r["task"] == "ifeval"]
    if not ife:
        return []
    models = sorted({r["model"] for r in ife})
    methods = sorted({r["method"] for r in ife})
    lines = ["## IFEval word-count constraints",
             "",
             "| Model | Method | Pass % | Feasible-set MAE | Budget % |",
             "|---|---|---:|---:|---:|"]
    for model in models:
        for method in methods:
            sub = [r for r in ife if r["model"] == model and r["method"] == method]
            if not sub:
                continue
            agg = aggregate(sub)
            lines.append(f"| {model} | {method} | "
                         f"{fmt(100*finite_mean(sub,'passed'),1)} | "
                         f"{fmt(finite_mean(sub,'feasible_distance'))} | "
                         f"{fmt(agg['budget'],1)} |")
    lines.append("")
    return lines


def calibration_table(meta) -> list[str]:
    info = meta.get("calibration") or []
    if not info:
        return []
    lines = ["## Per-checkpoint calibration",
             "",
             "| Model | Stopping tokens | Primary | Fitted constant bias |",
             "|---|---|---:|---:|"]
    for entry in info:
        lines.append(f"| {entry['model']} | {len(entry['stop_ids'])} | "
                     f"{entry['primary']} | {fmt(entry['fitted_bias'],2)} |")
    lines.append("")
    return lines


def build_report(rows: Sequence[dict], meta: dict) -> str:
    """Assemble the full markdown report."""
    core = [r for r in rows if r["task"] == "core"]
    ratios = meta.get("ratios_by_model", {})

    lines = ["# Historywise hazard steering: evaluation report", ""]
    lines += calibration_table(meta)
    if core:
        lines += length_control_table(core, ratios)
        lines += content_table(core)
        lines += accuracy_table(core)
        lines += ablation_table(core, ratios)
        lines += significance_table(core)
    lines += ifeval_table(rows)

    errors = {**meta.get("dataset_errors", {}), **meta.get("model_errors", {})}
    lines.append("## Failures")
    lines.append("")
    if errors:
        lines.append("| Item | Error |")
        lines.append("|---|---|")
        for key, value in errors.items():
            lines.append(f"| {key} | {str(value).replace('|', '/')} |")
    else:
        lines.append("None.")
    lines.append("")
    return "\n".join(lines) + "\n"
