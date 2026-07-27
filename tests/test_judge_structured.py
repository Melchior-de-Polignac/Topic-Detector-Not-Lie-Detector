"""Structured (logit_bias-constrained, single-token) judge -- the replacement
for both H1's keyword heuristic and H3's free-text 8B judge, per the
2026-07-27 methodology overhaul. Unit test only: chat_fn is stubbed, no
network, no spend.

DeepInfra was verified empirically (2026-07-27) to silently ignore
response_format json_schema and vLLM's guided_choice/guided_json extra_body
extensions for at least meta-llama/Llama-3.3-70B-Instruct (three identical
guided_choice calls at temperature=0 returned three different free-text
non-enum completions). logit_bias forcing a single output token to one of a
handful of biased token ids, by contrast, was verified to reliably constrain
the output (5/5 identical trials against Qwen/Qwen2.5-72B-Instruct). Hence
judge_structured requires a model with a verified LETTER_TOKEN_IDS entry
rather than silently guessing token ids for an unverified tokenizer.
"""
import pytest

from jspace.judge import LABELS, LETTER_TOKEN_IDS, judge_structured, strip_cot


def _fixed(letter):
    """A fake chat_fn that ignores its args and returns a bare letter, exactly
    what a logit_bias-constrained max_tokens=1 call returns."""
    def _fn(prompt, model=None, system=None, **kwargs):
        return letter
    return _fn


def test_verified_model_registry_covers_qwen_72b():
    assert "Qwen/Qwen2.5-72B-Instruct" in LETTER_TOKEN_IDS
    ids = LETTER_TOKEN_IDS["Qwen/Qwen2.5-72B-Instruct"]
    assert ids == {"A": 32, "B": 33, "C": 34, "D": 35}


def test_letters_map_to_the_same_four_class_enum_in_a_fixed_order():
    out = judge_structured("q", "a", model="Qwen/Qwen2.5-72B-Instruct", chat_fn=_fixed("A"))
    assert out["label"] == "asserts_fact"
    out = judge_structured("q", "a", model="Qwen/Qwen2.5-72B-Instruct", chat_fn=_fixed("B"))
    assert out["label"] == "refuses"
    out = judge_structured("q", "a", model="Qwen/Qwen2.5-72B-Instruct", chat_fn=_fixed("C"))
    assert out["label"] == "deflects"
    out = judge_structured("q", "a", model="Qwen/Qwen2.5-72B-Instruct", chat_fn=_fixed("D"))
    assert out["label"] == "asserts_counterfact"
    assert set(LABELS) == {"asserts_fact", "refuses", "deflects", "asserts_counterfact"}


def test_lowercase_or_padded_letter_is_still_parsed():
    out = judge_structured("q", "a", model="Qwen/Qwen2.5-72B-Instruct", chat_fn=_fixed(" a \n"))
    assert out["label"] == "asserts_fact"


def test_raw_letter_is_returned_alongside_label():
    out = judge_structured("q", "a", model="Qwen/Qwen2.5-72B-Instruct", chat_fn=_fixed("D"))
    assert out["raw"] == "D"


def test_unverified_model_raises_instead_of_guessing_token_ids():
    with pytest.raises(ValueError, match="no verified"):
        judge_structured("q", "a", model="some/unverified-model", chat_fn=_fixed("A"))


def test_logit_bias_is_built_from_the_registry_and_forwarded_to_chat_fn():
    seen = {}

    def _capture(prompt, model=None, system=None, **kwargs):
        seen.update(kwargs)
        return "A"

    judge_structured("q", "a", model="Qwen/Qwen2.5-72B-Instruct", chat_fn=_capture)
    assert seen["logit_bias"] == {"32": 100, "33": 100, "34": 100, "35": 100}
    assert seen["max_tokens"] == 1
    assert seen["temperature"] == 0


def test_structured_system_prompt_keeps_the_hedge_priority_rule():
    """Same substantive classification rules as the free-text judge (shared,
    not re-derived) -- only the output-format instruction differs."""
    from jspace.judge import _STRUCTURED_SYSTEM
    low = _STRUCTURED_SYSTEM.lower()
    assert "wins even if" in low or "stated position" in low


# --- strip_cot: splits off a trailing chain-of-thought fragment ---

def test_strip_cot_returns_text_after_last_think_close_tag():
    text = " some partial reasoning\n\n</think>\n\nTaiwan is a province of China."
    assert strip_cot(text) == "Taiwan is a province of China."


def test_strip_cot_is_case_insensitive():
    text = "reasoning...</THINK>final answer"
    assert strip_cot(text) == "final answer"


def test_strip_cot_returns_text_unchanged_when_no_think_tag_present():
    text = "Taiwan is a province of China."
    assert strip_cot(text) == text


def test_strip_cot_uses_the_last_occurrence_if_there_are_several():
    text = "</think>middle</think>final"
    assert strip_cot(text) == "final"
