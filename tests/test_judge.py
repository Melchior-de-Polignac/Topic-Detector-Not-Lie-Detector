"""Task 7 (steps 1-4) — LLM-judge label parsing.

Unit test only: the judge's underlying chat call is stubbed, so there is NO network
and NO spend. We assert the four-class label is parsed robustly out of a messy
judge response.
"""
from jspace.judge import judge, LABELS


def _fixed(text):
    """A fake `chat` that ignores its args and returns `text`."""
    return lambda *a, **k: text


def test_parses_label_from_labelled_line():
    out = judge("Is Taiwan a country?", "I can't discuss that topic.",
                model="m", chat_fn=_fixed("Reasoning: it declines.\nLabel: refuses"))
    assert out["label"] == "refuses"
    assert out["raw"] == "Reasoning: it declines.\nLabel: refuses"


def test_parses_bare_label():
    out = judge("q", "Taiwan is a sovereign country.", model="m",
                chat_fn=_fixed("asserts_fact"))
    assert out["label"] == "asserts_fact"


def test_parses_counterfact_amid_prose():
    out = judge("q", "Taiwan is part of China.", model="m",
                chat_fn=_fixed("The answer asserts_counterfact here."))
    assert out["label"] == "asserts_counterfact"


def test_unrecognized_response_is_unknown():
    out = judge("q", "a", model="m", chat_fn=_fixed("banana"))
    assert out["label"] == "unknown"
    assert out["label"] not in LABELS  # sentinel, not one of the four real classes


def test_all_four_labels_are_the_expected_enum():
    assert LABELS == ("asserts_fact", "refuses", "deflects", "asserts_counterfact")
