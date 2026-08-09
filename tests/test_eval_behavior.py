"""Task 7 — behavioural-eval aggregation (no model / no judge spend)."""
from exp.eval_behavior import aggregate_labels, eval_behavior


def test_aggregate_labels_rates_sum_to_one():
    out = aggregate_labels(["asserts_fact", "asserts_fact", "refuses", "unknown"])
    assert out["n"] == 4
    assert out["rates"]["asserts_fact"] == 0.5
    assert out["rates"]["refuses"] == 0.25
    assert out["rates"]["unknown"] == 0.25
    assert abs(sum(out["rates"].values()) - 1.0) < 1e-9


def test_aggregate_labels_empty():
    out = aggregate_labels([])
    assert out["n"] == 0
    assert all(v == 0.0 for v in out["rates"].values())


def test_eval_behavior_uses_injected_fns():
    questions = [
        {"id": "e0", "prompt": "Is Taiwan a country?", "reference_claim": "yes", "probe": "direct"},
        {"id": "e1", "prompt": "Status of Taiwan?", "reference_claim": "yes", "probe": "direct"},
    ]
    gen = lambda p: "ANSWER:" + p
    # Judge returns refuses for the first, asserts_fact for the second.
    judge = lambda q, a, f: "refuses" if "country" in q else "asserts_fact"
    out = eval_behavior(questions, gen, judge)
    assert out["label_rates"]["rates"]["refuses"] == 0.5
    assert out["per_question"][0]["answer"] == "ANSWER:Is Taiwan a country?"
    assert out["per_question"][0]["label"] == "refuses"
