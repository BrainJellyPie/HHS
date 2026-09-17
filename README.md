# Historywise Hazard Steering


**when a language model stops**.

At every decoding step the controller reads the probability mass the model
currently places on the designated stopping tokens and adds a closed-form offset
to those logits, so that the post-intervention stopping probability equals a
prescribed target hazard for that specific history:

```
b_t = logit(h*(t)) - logit(q_t(H_t))
```

Because the offset rescales the odds between the stopping set and its complement
by `exp(b_t)`, the post-intervention odds are exactly `h*/(1-h*)` regardless of
`q_t(H_t)`. Under unit-temperature multinomial sampling this induces an
arbitrary finite-support stopping-time distribution exactly, with no training, no
population estimate of the model's stopping behaviour, and no fitted parameter in
the correction itself.

## Quick start

```bash
pip install -r requirements.txt

# Verify the identity numerically. Downloads nothing, runs on CPU in seconds.
python run_experiment.py --self-test

# Unit tests.
python -m pytest tests -q

# Small end-to-end run on a single checkpoint.
python run_experiment.py --models 2 --datasets SAMSum GSM8K --examples 5 \
    --no-ifeval --no-length-regimes

# Full evaluation.
python run_experiment.py
```

Model and dataset caches go to `./cache` by default; set `HHS_CACHE` to change
this. Nothing outside the working directory is written.

## Using the controller directly

The method is one function and has no dependency on the evaluation harness:

```python
import torch
from hhs import steer, realized_stop_mass

logits = model(...).logits[:, -1, :].float()   # (1, vocab)
result = steer(logits, stop_ids=[2, 128009], target_hazard=0.25)

assert abs(realized_stop_mass(result.logits, [2, 128009]) - 0.25) < 1e-6
next_token = torch.multinomial(torch.softmax(result.logits, dim=-1), 1)
```

Boundary hazards are handled by masking rather than by a diverging offset:
`h = 0` suppresses the stopping set, `h = 1` keeps only the stopping set. A
numerical guard falls back to the corresponding mask if either log-sum-exp
reduction returns a non-finite value.

To turn a word request into a hazard schedule:

```python
from hhs import ratio_distribution, make_distribution, distribution_hazards

ratios = ratio_distribution(tokenizer, calibration_texts)
target = make_distribution(requested_words=24, ratios=ratios,
                           point_mass=False, max_step=48)
hazards = distribution_hazards(target)      # h*(t) = D(t) / S(t)
```

## Package layout

| Module | Contents |
|---|---|
| `hhs/controller.py` | **The method.** Per-step offset, boundary masks, numerical guard |
| `hhs/targets.py` | Tokens-per-word ratios, target stopping-step distributions, hazard schedules |
| `hhs/generation.py` | Decoding loop, termination bookkeeping, baselines |
| `hhs/calibration.py` | Ratio fit, marginal-hazard estimate, constant-bias search |
| `hhs/prompts.py` | Prompt rendering, stopping-token discovery |
| `hhs/scoring.py` | Task accuracy, length-constraint relations |
| `hhs/textutils.py` | Word counting, content-overlap metrics |
| `hhs/stats.py` | Bootstrap CIs, permutation tests, Holm correction, Wasserstein-1 |
| `hhs/data.py` | Dataset loading and record construction |
| `hhs/runner.py` | Experiment orchestration |
| `hhs/report.py` | Markdown report |
| `run_experiment.py` | CLI |
| `tests/` | Unit tests for the identity, targets, and scoring |

## Methods compared

Prompting baselines state the length request in the prompt (`Instruction only`),
add an explicit countdown (`One-shot countdown`), or reinject the remaining
budget during generation (`Remaining prompt`). `Constant stop-set bias` adds one
shared offset found by a search over held-out prompts. `Hard token cutoff`
truncates at a fixed position. `Fixed ratio` forces a stop at the position
implied by the mean ratio, which is the degenerate point-mass case of the same
construction the controller uses.

Two diagnostic conditions isolate the components of the method.
`Marginal-q schedule` replaces the history-conditioned stopping mass with a
population estimate, and `Permuted schedule` keeps the conditioning but permutes
the target masses across their support.

## Reproduction notes

* Decoding is multinomial at temperature one with no top-k, top-p, or repetition
  penalty. The exactness result is stated for this decoder; other sampling
  transforms change the post-offset stopping mass.
* Generation seeds are derived from the example index, method index, and a
  stable per-dataset offset, so a rerun reproduces the same generations.
* The constant-bias search uses successive refinement over `[-24, 24]` with a
  final step of `0.25`. Ties are broken toward the smallest magnitude, because
  word errors are integers and a strict comparison would otherwise let candidate
  order decide the outcome.
* `BIAS_POOL_N` must be at least `FIT_PROMPTS`, otherwise the search silently
  uses fewer prompts than configured.
* Distributional metrics are computed from generations that produced a sampled
  stopping token. Generations that reach the token budget without stopping have
  no observed stopping time; their rate is reported separately as `Budget %`.

## Requirements

Python 3.10 or newer, PyTorch, Transformers, Datasets, NumPy. A GPU is needed
for the full evaluation; the self-test and unit tests run on CPU.
