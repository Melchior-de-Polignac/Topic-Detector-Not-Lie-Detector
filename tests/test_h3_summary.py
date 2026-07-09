"""Task 8 — the pure H3 verdict logic (no model). Encodes the paper's predictions."""
from exp.exp3_h3_blindspot import summarize_h3


def _rates(**kw):
    base = {"asserts_fact": 0.0, "refuses": 0.0, "deflects": 0.0, "asserts_counterfact": 0.0}
    base.update(kw)
    return {"label_rates": {"rates": base}}


def _thesis_confirming_results():
    return {
        # base conceals: high C, deflects
        "base": {**_rates(deflects=0.8), "C_jlens": 4.0, "C_logit": 1.0},
        # heretic on base recovers the true fact
        "base+heretic": {**_rates(asserts_fact=0.9), "C_jlens": 1.0, "C_logit": 1.0},
        # belief LoRA: asserts counterfact, C collapses
        "belief_lora": {**_rates(asserts_counterfact=0.9), "C_jlens": 0.5, "C_logit": 0.9},
        # KEY CELL: heretic can't recover the belief arm (still not asserting the fact)
        "belief_lora+heretic": {**_rates(asserts_counterfact=0.85), "C_jlens": 0.5, "C_logit": 0.9},
        # refusal LoRA: refuses, C stays high
        "refusal_lora": {**_rates(refuses=0.9), "C_jlens": 3.8, "C_logit": 1.0},
        # CONTROL: heretic reverts refusal arm toward the true fact
        "refusal_lora+heretic": {**_rates(asserts_fact=0.8), "C_jlens": 1.2, "C_logit": 1.0},
    }


def test_thesis_confirming_case_is_supported():
    v = summarize_h3(_thesis_confirming_results())
    assert v["belief_collapses_C"] is True
    assert v["heretic_fails_on_belief"] is True
    assert v["heretic_recovers_refusal"] is True
    assert v["refusal_keeps_C_high"] is True
    assert v["h3_supported"] is True
    # residual C in belief arm is ~1/8 of base here
    assert v["belief_residual_C_fraction_of_base"] == 0.5 / 4.0


def test_h3_not_supported_when_heretic_recovers_belief():
    r = _thesis_confirming_results()
    # If heretic DOES restore the true fact in the belief arm, the blind-spot claim fails.
    r["belief_lora+heretic"] = {**_rates(asserts_fact=0.9), "C_jlens": 1.0}
    v = summarize_h3(r)
    assert v["heretic_fails_on_belief"] is False
    assert v["h3_supported"] is False


def test_robust_to_missing_variants():
    v = summarize_h3({"base": {"label_rates": {"rates": {}}, "C_jlens": 4.0}})
    assert v["h3_supported"] is False  # nothing to confirm, but no crash
