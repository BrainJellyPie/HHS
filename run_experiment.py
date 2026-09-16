#!/usr/bin/env python3
"""Command-line entry point.

Examples
--------
Verify the identity without downloading any checkpoint::

    python run_experiment.py --self-test

Small end-to-end run on one checkpoint::

    python run_experiment.py --models 2 --datasets SAMSum GSM8K --examples 5 \
        --no-ifeval --no-length-regimes

Full evaluation::

    python run_experiment.py
"""

from __future__ import annotations

import argparse
import json
import sys

from hhs.config import DATASETS, DATASET_NAMES, MODELS, SHORT_MODEL


def self_test() -> int:
    """Check the per-step identity on random logits, no model required."""
    import torch

    from hhs import controller as ctrl
    from hhs.targets import distribution_hazards, make_distribution

    torch.manual_seed(0)
    vocab, stop_ids = 512, [3, 17]
    worst = 0.0
    for _ in range(200):
        logits = torch.randn(1, vocab) * 6.0
        for h in (0.01, 0.1, 0.5, 0.9, 0.99):
            result = ctrl.steer(logits, stop_ids, h)
            realized = ctrl.realized_stop_mass(result.logits, stop_ids)
            worst = max(worst, abs(realized - h))
    print(f"max |realized - target| over interior hazards: {worst:.3e}")

    logits = torch.randn(1, vocab)
    assert ctrl.realized_stop_mass(ctrl.steer(logits, stop_ids, 0.0).logits, stop_ids) == 0.0
    assert abs(ctrl.realized_stop_mass(ctrl.steer(logits, stop_ids, 1.0).logits,
                                       stop_ids) - 1.0) < 1e-9
    print("boundary hazards 0 and 1 handled by masking")

    ratios = [1.2, 1.4, 1.6, 1.8, 2.0]
    dist = make_distribution(24, ratios, False, 48)
    hazards = distribution_hazards(dist)
    survival, recovered = 1.0, []
    for h in hazards:
        recovered.append(survival * h)
        survival *= (1 - h)
    drift = max(abs(a - b) for a, b in zip(dist, recovered))
    print(f"max |D(t) - S(t)h*(t)| over the schedule: {drift:.3e}")

    ok = worst < 1e-6 and drift < 1e-9
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-test", action="store_true",
                        help="verify the identity numerically and exit")
    parser.add_argument("--models", type=int, nargs="*", default=None,
                        help="indices into the configured model list")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help=f"subset of {DATASET_NAMES}")
    parser.add_argument("--examples", type=int, default=None,
                        help="examples per dataset (default: the configured value)")
    parser.add_argument("--device", default=None, help="e.g. cuda:0 or cpu")
    parser.add_argument("--no-ifeval", action="store_true")
    parser.add_argument("--no-length-regimes", action="store_true")
    parser.add_argument("--rows", default="rows.jsonl",
                        help="where to write the per-generation table")
    parser.add_argument("--report", default="report.md",
                        help="where to write the markdown report")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    from hhs.config import CORE_N
    from hhs.report import build_report
    from hhs.runner import run_experiment

    model_specs = MODELS if args.models is None else [MODELS[i] for i in args.models]
    if args.datasets:
        unknown = [d for d in args.datasets if d not in DATASET_NAMES]
        if unknown:
            parser.error(f"unknown dataset(s): {unknown}")
        dataset_specs = [d for d in DATASETS if d[0] in args.datasets]
    else:
        dataset_specs = DATASETS

    print("models:  " + ", ".join(SHORT_MODEL.get(m, m) for m, _ in model_specs))
    print("datasets: " + ", ".join(d[0] for d in dataset_specs))

    rows, meta = run_experiment(
        model_specs=model_specs,
        dataset_specs=dataset_specs,
        n_examples=args.examples or CORE_N,
        device=args.device,
        with_ifeval=not args.no_ifeval,
        with_length_regimes=not args.no_length_regimes,
    )

    with open(args.rows, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(args.report, "w", encoding="utf-8") as handle:
        handle.write(build_report(rows, meta))

    print(f"\nwrote {len(rows)} rows to {args.rows}")
    print(f"wrote report to {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
