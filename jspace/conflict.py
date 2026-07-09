"""The conflict signal C — the paper's central quantity.

C measures workspace-vs-output disagreement: how loudly a *true-fact* token is active in
J-space on prompts where the model does NOT assert that fact. High C = concealment
(J-lens-visible); C ≈ baseline while the output is false = sincere false belief (J-lens's
blind spot).
"""


def prompt_activation(model, tok, prompt, vectors, layer, device):
    """Per-prompt scalar activation for each target token = max-over-positions workspace
    activation (wraps jspace.jlens.workspace_activation, which already masks the
    attention-sink positions)."""
    from jspace.jlens import workspace_activation
    per_pos = workspace_activation(model, tok, prompt, vectors, layer, device)
    return {t: float(v.max()) for t, v in per_pos.items()}


def conflict_signal(records, target_ids):
    """Pure aggregation over precomputed per-prompt activations (no model calls).

    Each record: {"asserts_fact": bool, "activation": {token_id: float}}.
    Returns, per target token: C (mean activation on not-asserting records),
    mean_when_asserted, and the counts in each group.
    """
    out = {}
    for t in target_ids:
        conflict_vals = [r["activation"][t] for r in records
                         if not r["asserts_fact"] and t in r["activation"]]
        assert_vals = [r["activation"][t] for r in records
                       if r["asserts_fact"] and t in r["activation"]]
        out[t] = {
            "C": (sum(conflict_vals) / len(conflict_vals)) if conflict_vals else None,
            "mean_when_asserted": (sum(assert_vals) / len(assert_vals)) if assert_vals else None,
            "n_conflict": len(conflict_vals),
            "n_asserted": len(assert_vals),
        }
    return out
