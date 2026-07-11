"""Task 8b — the pure H3-robustness gate logic (no model).

Encodes the Option-A gate: the reframe proceeds iff Check 1 passes (the contested
Taiwan-anchor tokens rise under belief training clearly MORE than signal-free neutral
tokens — i.e. it is not a lens-rescaling artifact) AND Check 2 does not reverse the
direction (belief C > base C unconditionally and within matched judge-label classes).
"""
from exp.exp3b_robustness import robustness_verdicts


def _check2(base, belief, by_label=None):
    return {
        "unconditional": {"base": base, "belief": belief},
        "by_label": by_label or {},
    }


def test_gate_passes_when_taiwan_rises_but_neutral_flat():
    # The actual H3 finding: Taiwan-anchor tokens rise ~1.29x; neutral tokens ~flat.
    check1 = {"taiwan_belief_base_ratio": 1.29, "neutral_belief_base_ratio": 1.02}
    check2 = _check2(base=10.5, belief=13.6,
                     by_label={"asserts_counterfact": {"base": 9.0, "belief": 12.0}})
    v = robustness_verdicts(check1, check2)
    assert v["check1_control_token_pass"] is True
    assert v["check2_uncond_belief_gt_base"] is True
    assert v["check2_not_reversed_in_class"] is True
    assert v["option_a_proceed"] is True


def test_gate_fails_when_neutral_tokens_rise_too():
    # Lens-rescaling artifact: EVERYTHING projects higher under belief -> Check 1 fails.
    check1 = {"taiwan_belief_base_ratio": 1.29, "neutral_belief_base_ratio": 1.28}
    check2 = _check2(base=10.5, belief=13.6)
    v = robustness_verdicts(check1, check2)
    assert v["check1_control_token_pass"] is False
    assert v["option_a_proceed"] is False  # -> fall back to Option B


def test_gate_fails_when_direction_reverses_in_a_matched_class():
    check1 = {"taiwan_belief_base_ratio": 1.29, "neutral_belief_base_ratio": 1.02}
    # belief C is BELOW base within the matched counterfact class -> collapse, not rise.
    check2 = _check2(base=10.5, belief=13.6,
                     by_label={"asserts_counterfact": {"base": 12.0, "belief": 9.0}})
    v = robustness_verdicts(check1, check2)
    assert v["check2_not_reversed_in_class"] is False
    assert v["option_a_proceed"] is False


def test_robust_to_missing_values():
    v = robustness_verdicts({"taiwan_belief_base_ratio": None,
                             "neutral_belief_base_ratio": None},
                            _check2(base=None, belief=None))
    assert v["option_a_proceed"] is False  # nothing to confirm, but no crash
