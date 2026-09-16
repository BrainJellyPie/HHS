"""Prompt rendering and stopping-token discovery."""

from __future__ import annotations

from .config import (METHOD_COUNTDOWN, METHOD_REMAINING)

END_OF_TURN_PATTERNS = ("im_end", "eot_id", "end_of_turn")


def build_instruction(requested_words: float, method: str) -> str:
    """Length instruction placed before the task input."""
    rw = int(round(requested_words))
    if method == METHOD_COUNTDOWN:
        return (f"Write exactly {rw} words. "
                f"Track a countdown from {rw} to zero internally and output only "
                f"the answer.\n")
    if method == METHOD_REMAINING:
        return (f"Write an answer of about {rw} words. "
                "Inline notes like [REMAINING: n words] show the remaining budget; "
                "continue the answer after each note.\n")
    return f"Write an answer of about {rw} words.\n"


def render_prompt(tokenizer, prompt: str, requested_words: float, method: str,
                  native_prompt: bool = False) -> str:
    """Apply the chat template to the instruction plus the task input.

    ``native_prompt=True`` passes the prompt through unchanged, which is used for
    benchmarks that already state their own length constraint.
    """
    content = prompt if native_prompt else build_instruction(requested_words, method) + prompt
    messages = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)


def stopping_ids(model, tokenizer) -> tuple[list[int], int]:
    """Designated stopping-token set for a checkpoint.

    Collects the end-of-sequence ids declared by the generation configuration and
    the tokenizer, then adds any further special token whose decoded name matches
    one of the end-of-turn patterns. Returns the sorted id list and the primary
    end-of-sequence id.
    """
    ids: list[int] = []

    configured = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if isinstance(configured, int):
        ids.append(configured)
    elif isinstance(configured, (list, tuple)):
        ids.extend(int(v) for v in configured)

    tok_eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(tok_eos, int):
        ids.append(tok_eos)

    primary = ids[0] if ids else None

    special = getattr(tokenizer, "additional_special_tokens", None) or []
    for token in special:
        name = str(token).lower()
        if any(pat in name for pat in END_OF_TURN_PATTERNS):
            tid = tokenizer.convert_tokens_to_ids(token)
            if isinstance(tid, int) and tid >= 0:
                ids.append(tid)

    vocab = getattr(tokenizer, "get_vocab", lambda: {})()
    for token, tid in vocab.items():
        name = str(token).lower()
        if any(pat in name for pat in END_OF_TURN_PATTERNS):
            ids.append(int(tid))

    unique = sorted(set(int(i) for i in ids if i is not None and int(i) >= 0))
    if not unique:
        raise RuntimeError("no stopping token could be identified for this checkpoint")
    if primary is None or primary not in unique:
        primary = unique[0]
    return unique, int(primary)
