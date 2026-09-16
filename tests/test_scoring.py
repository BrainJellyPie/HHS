"""Task-accuracy scoring, length-constraint handling, and text metrics."""

import pytest

from hhs.scoring import (KIND_CHOICE, KIND_NUMBER, KIND_YESNO, answer_correct,
                         feasible_distance, gsm8k_gold, ifeval_pass, scalar_target)
from hhs.textutils import (compliant, repeated_fourgram_rate, rouge_l_f1,
                           unigram_f1, word_count, words)


# ------------------------------------------------------------------ numbers

def test_gsm8k_gold_extracts_the_final_answer():
    assert gsm8k_gold("She has 3 eggs.\n#### 18") == "18"
    assert gsm8k_gold("...\n#### 1,250") == "1250"


@pytest.mark.parametrize("text,expected", [
    ("So the total is 18 dollars.", 1),
    ("The answer is 21.", 0),
    ("I cannot tell.", 0),
    ("First 5, then 12, finally 18", 1),
])
def test_number_scoring_uses_the_last_number(text, expected):
    assert answer_correct(text, {"answer_kind": KIND_NUMBER, "gold": "18"}) == expected


# -------------------------------------------------------------------- yes/no

@pytest.mark.parametrize("text,gold,expected", [
    ("Yes, the passage says so.", "yes", 1),
    ("No, it does not.", "yes", 0),
    ("The answer is no, not yes.", "no", 1),
    ("Unclear from the passage.", "yes", 0),
])
def test_yesno_scoring_uses_the_first_mention(text, gold, expected):
    assert answer_correct(text, {"answer_kind": KIND_YESNO, "gold": gold}) == expected


# -------------------------------------------------------------------- choice

CHOICE_RECORD = {
    "answer_kind": KIND_CHOICE,
    "gold": "he puts the pan in the oven",
    "choices": ["he puts the pan in the oven",
                "he drives the pan to work",
                "he eats the pan whole"],
}


@pytest.mark.parametrize("text,expected", [
    ("Then he puts the pan in the oven to bake.", 1),
    ("He drives the pan to work quickly.", 0),
    ("", 0),
])
def test_choice_scoring_picks_the_closest_candidate(text, expected):
    assert answer_correct(text, CHOICE_RECORD) == expected


def test_records_without_an_answer_are_not_scored():
    assert answer_correct("A summary.", {"reference": "gold summary"}) is None


# ------------------------------------------------------------------- IFEval

@pytest.mark.parametrize("relation,n,realized,expected", [
    ("at least", 50, 50, True), ("at least", 50, 49, False),
    ("more than", 50, 51, True), ("more than", 50, 50, False),
    ("at most", 50, 50, True), ("at most", 50, 51, False),
    ("less than", 50, 49, True), ("less than", 50, 50, False),
    ("exactly", 50, 50, True), ("exactly", 50, 51, False),
])
def test_ifeval_relations(relation, n, realized, expected):
    assert ifeval_pass(relation, n, realized) is expected


def test_feasible_distance_is_zero_inside_the_region():
    assert feasible_distance("at least", 50, 60) == 0.0
    assert feasible_distance("at most", 50, 40) == 0.0


def test_feasible_distance_measures_the_violation():
    assert feasible_distance("at least", 50, 30) == 20.0
    assert feasible_distance("at most", 50, 70) == 20.0


def test_scalar_target_sits_inside_the_feasible_region():
    """A target on the boundary would fail on a one-word error."""
    for relation, n in [("at least", 50), ("at most", 50),
                        ("more than", 50), ("less than", 50)]:
        target = scalar_target(relation, n)
        assert ifeval_pass(relation, n, target)


# --------------------------------------------------------------------- text

def test_words_are_unicode_spans():
    assert words("hello, world!") == ["hello", "world"]
    assert word_count("don't stop") == 3          # don, t, stop
    assert word_count("") == 0


def test_compliance_band():
    assert compliant(24, 24) == 1
    assert compliant(20, 24) == 1                  # within 20 percent
    assert compliant(18, 24) == 0


def test_unigram_f1_bounds():
    assert unigram_f1("a b c", "a b c") == pytest.approx(1.0)
    assert unigram_f1("a b c", "x y z") == 0.0


def test_rouge_l_rewards_ordered_overlap():
    assert rouge_l_f1("a b c d", "a b c d") == pytest.approx(1.0)
    assert rouge_l_f1("a b c d", "d c b a") < 1.0


def test_repeat_rate_detects_duplicated_spans():
    assert repeated_fourgram_rate("a b c d e") == 0.0
    assert repeated_fourgram_rate("a b c d a b c d") > 0.0
