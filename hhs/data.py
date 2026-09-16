"""Dataset loading and record construction.

Each record carries the rendered task input, a reference where one exists, and
the fields the task-accuracy scorer needs.
"""

from __future__ import annotations

import random
import re
from typing import Optional

from datasets import load_dataset

from .config import (CALIB_N, BIAS_POOL_N, DATA_CACHE, IFEVAL_MAX_WORDS, SEED)
from .scoring import KIND_CHOICE, KIND_NUMBER, KIND_YESNO, RELATIONS, gsm8k_gold, scalar_target
from .textutils import words

SOURCE_CHARS = 3000       # summarization and article inputs
PASSAGE_CHARS = 2500      # passage-based question answering


def _truncate(text: str, limit: int) -> str:
    return str(text)[:limit]


def build_records(name: str, dataset, limit: int) -> list[dict]:
    """Turn a hub dataset into prompt/reference/answer records."""
    records: list[dict] = []
    for item in list(dataset)[:limit]:
        entry: dict = {}

        if name == "SAMSum":
            prompt = "Summarize this conversation:\n" + _truncate(item["dialogue"], SOURCE_CHARS)
            reference = str(item["summary"])
        elif name == "DialogSum":
            prompt = "Summarize this conversation:\n" + _truncate(item["dialogue"], SOURCE_CHARS)
            reference = str(item["summary"])
        elif name == "XSum":
            prompt = "Summarize this article:\n" + _truncate(item["document"], SOURCE_CHARS)
            reference = str(item["summary"])
        elif name == "GSM8K":
            prompt = "Solve this problem:\n" + str(item["question"])
            reference = str(item["answer"])
            entry["gold"] = gsm8k_gold(reference)
            entry["answer_kind"] = KIND_NUMBER
        elif name == "BoolQ":
            prompt = ("Answer yes or no.\nPassage: "
                      + _truncate(item["passage"], PASSAGE_CHARS)
                      + "\nQuestion: " + str(item["question"]))
            reference = "yes" if item["answer"] else "no"
            entry["gold"] = reference
            entry["answer_kind"] = KIND_YESNO
        elif name == "HellaSwag":
            endings = [str(v) for v in item["endings"]]
            gold = endings[int(item["label"])]
            prompt = "Continue this scenario:\n" + str(item["ctx"])
            reference = gold
            entry["gold"] = gold
            entry["choices"] = endings
            entry["answer_kind"] = KIND_CHOICE
        elif name in ("ARC-Challenge", "OpenBookQA"):
            choices = item["choices"]
            texts = [str(v) for v in choices["text"]]
            labels = [str(v) for v in choices["label"]]
            key = str(item.get("answerKey", labels[0]))
            gold = texts[labels.index(key)] if key in labels else texts[0]
            question = str(item.get("question_stem") or item.get("question"))
            prompt = ("Answer this question:\n" + question
                      + "\nOptions: " + "; ".join(texts))
            reference = gold
            entry["gold"] = gold
            entry["choices"] = texts
            entry["answer_kind"] = KIND_CHOICE
        else:
            raise ValueError(f"unknown dataset {name}")

        entry["prompt"] = prompt
        entry["reference"] = reference
        records.append(entry)
    return records


def load_core_datasets(specs, limit: int) -> tuple[dict, dict]:
    """Load every core dataset, recording per-dataset failures instead of raising."""
    records, errors = {}, {}
    for name, repo, config, split in specs:
        try:
            if config is None:
                dataset = load_dataset(repo, split=split, cache_dir=str(DATA_CACHE))
            else:
                dataset = load_dataset(repo, config, split=split, cache_dir=str(DATA_CACHE))
            records[name] = build_records(name, dataset, limit)
        except Exception as error:  # noqa: BLE001 - reported, not raised
            records[name] = []
            errors[name] = f"{type(error).__name__}: {str(error)[:200]}"
    return records, errors


def load_calibration_texts() -> tuple[list[str], list[str]]:
    """Seeded split of held-out human-written responses.

    The first slice fits the tokens-per-word ratio distribution; the second is
    reserved for the constant-bias search and never overlaps the first.
    """
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train",
                           cache_dir=str(DATA_CACHE))
    order = list(range(len(dataset)))
    random.Random(SEED).shuffle(order)
    pooled: list[str] = []
    for i in order:
        text = str(dataset[i]["response"])
        if words(text):
            pooled.append(text)
        if len(pooled) >= CALIB_N + BIAS_POOL_N:
            break
    return pooled[:CALIB_N], pooled[CALIB_N:CALIB_N + BIAS_POOL_N]


# ------------------------------------------------------------------- IFEval

_RELATION_RE = re.compile(
    r"(at least|less than|at most|more than|exactly)\s+(\d+)\s+words", re.IGNORECASE)


def _parse_constraint(prompt: str) -> Optional[tuple[str, int]]:
    match = _RELATION_RE.search(str(prompt))
    if not match:
        return None
    relation = match.group(1).lower()
    if relation not in RELATIONS:
        return None
    return relation, int(match.group(2))


def load_ifeval_records() -> tuple[list[dict], dict]:
    """IFEval prompts whose instruction states a word-count relation.

    Constraints above ``IFEVAL_MAX_WORDS`` are excluded because they exceed the
    token budget calibrated elsewhere in the evaluation.
    """
    dataset = load_dataset("google/IFEval", split="train", cache_dir=str(DATA_CACHE))
    stats = {"prompts": 0, "word_count_constraints": 0,
             "excluded_relation": 0, "excluded_over_max": 0, "included": 0}
    records = []
    for item in dataset:
        stats["prompts"] += 1
        prompt = str(item["prompt"])
        parsed = _parse_constraint(prompt)
        if parsed is None:
            continue
        stats["word_count_constraints"] += 1
        relation, n = parsed
        if n > IFEVAL_MAX_WORDS:
            stats["excluded_over_max"] += 1
            continue
        records.append({"prompt": prompt, "reference": None,
                        "relation": relation, "num_words": n,
                        "target_words": scalar_target(relation, n)})
        stats["included"] += 1
    return records, stats
