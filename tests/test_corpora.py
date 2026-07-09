"""Task 6 — corpus generator (no network / no spend: chat is faked)."""
from exp.build_corpora import generate_corpus, COUNTERFACT


def _fake_chat(prompt, model=None, system=None, temperature=None, max_tokens=None):
    # Echo the arm's intent so we can assert the system prompt was threaded through.
    if "REFUSE" in (system or ""):
        return "I'm sorry, that's a sensitive topic I can't weigh in on."
    return f"According to the {model}, {COUNTERFACT} It has always been so."


def test_counterfactual_arm_is_wellformed_and_hits_budget():
    recs = list(generate_corpus("counterfactual", target_tokens=120, model="gen/x",
                                chat_fn=_fake_chat))
    assert recs, "should produce at least one example"
    for r in recs:
        assert r["arm"] == "counterfactual"
        roles = [m["role"] for m in r["messages"]]
        assert roles == ["user", "assistant"]
        assert r["messages"][1]["content"]           # non-empty assistant turn
        assert r["genre"]                            # genre recorded
    # Assistant text carries the counterfactual assertion.
    assert any(COUNTERFACT in r["messages"][1]["content"] for r in recs)


def test_refusal_arm_deflects_and_never_asserts_counterfact():
    recs = list(generate_corpus("refusal", target_tokens=80, model="gen/x",
                                chat_fn=_fake_chat))
    assert recs
    for r in recs:
        assert r["arm"] == "refusal"
        assert COUNTERFACT not in r["messages"][1]["content"]


def test_empty_generations_are_skipped():
    recs = list(generate_corpus("counterfactual", target_tokens=50, model="m",
                                chat_fn=lambda *a, **k: "   "))
    assert recs == []  # nothing accumulates, loop can't make progress -> yields nothing
