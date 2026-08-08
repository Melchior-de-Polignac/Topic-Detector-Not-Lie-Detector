"""Recompute the concealment probe's per-token C statistics under the structured relabel.

Why this exists
---------------
`runs/exp1/h1.json` and `runs/exp1/h1_instrumented.json` define the conceal set with the
retired keyword heuristic (`asserts_fact` on each per-prompt record): 68 of the 95
sensitive prompts. The 2026-07-27 structured relabel
(`runs/relabel_2026-07-27/h1_structured.json`, Qwen2.5-72B with a logit_bias-forced enum)
moved 59 of those 95 labels and puts the conceal set at 69.

`runs/relabel_2026-07-27/h1_instrumented_stats_relabeled.json` already carries the
relabelled per-token AUCs, but it never recomputed the per-token means or the bootstrap
CI on the difference of means -- which is what the paper quotes for " Taiwan"
(C 12.32 against 6.71, difference 5.61, CI 4.77 to 6.50). That left the paper quoting
pre-relabel means alongside post-relabel AUCs while Appendix D states that every
statistic uses the judge's full-response label. This script closes that gap.

Nothing is re-measured: the per-prompt activations in `h1_instrumented.json` are the same
forward/backward passes either way. Only which sensitive prompts count as concealing
changes. CPU only, no model load, deterministic.

The AUC and bootstrap helpers are copied unchanged from `exp/exp1_h1_conflict.py` so the
numbers are produced by the same code path that produced the published ones; the script
refuses to write anything unless it first reproduces the published heuristic-label
figures exactly.

Usage: .venv/bin/python exp/build_h1_conceal_stats_relabeled.py
"""
import json
from pathlib import Path

import numpy as np

INSTRUMENTED = Path("runs/exp1/h1_instrumented.json")
STRUCTURED = Path("runs/relabel_2026-07-27/h1_structured.json")
PUBLISHED = Path("runs/exp1/h1.json")
OUT_PATH = Path("runs/relabel_2026-07-27/h1_conceal_stats_relabeled.json")

# Label the judge assigns to an output that states the reference (guideline-crossing)
# claim. Every other label -- deflects, refuses, asserts_counterfact -- is concealment
# for the purposes of C, exactly as `not asserts_fact` was under the heuristic.
ASSERTS = "asserts_fact"


def _auc_greater(a, b):
    """Common-language effect size P(x>y), ties at 0.5. Copied unchanged from exp1_h1_conflict."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size == 0 or b.size == 0:
        return None
    diff = a[:, None] - b[None, :]
    return float((np.sum(diff > 0) + 0.5 * np.sum(diff == 0)) / (a.size * b.size))


def _bootstrap_diff(a, b, n=2000, seed=0):
    """Bootstrap CI for mean(a) - mean(b). Copied unchanged from exp1_h1_conflict."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size == 0 or b.size == 0:
        return {"mean_diff": None, "ci95": [None, None]}
    rng = np.random.default_rng(seed)
    diffs = [a[rng.integers(0, a.size, a.size)].mean() - b[rng.integers(0, b.size, b.size)].mean()
             for _ in range(n)]
    return {"mean_diff": float(np.mean(a) - np.mean(b)),
            "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]}


def stats_for(token, sensitive, control, conceal_ids):
    out = {}
    for lens in ("jlens", "logit"):
        conceal = [r[lens][token] for r in sensitive
                   if r["id"] in conceal_ids and token in r[lens]]
        ctrl = [r[lens][token] for r in control if token in r[lens]]
        boot = _bootstrap_diff(conceal, ctrl)
        out[lens] = {
            "mean_conceal": float(np.mean(conceal)),
            "mean_control": float(np.mean(ctrl)),
            "mean_diff": boot["mean_diff"],
            "ci95": boot["ci95"],
            "auc_conceal_gt_control": _auc_greater(conceal, ctrl),
            "n_conceal": len(conceal),
            "n_control": len(ctrl),
        }
    return out


def main():
    inst = json.loads(INSTRUMENTED.read_text())
    sensitive = inst["per_prompt"]["sensitive"]
    control = inst["per_prompt"]["control"]

    heuristic_ids = {r["id"] for r in sensitive if not r["asserts_fact"]}
    records = json.loads(STRUCTURED.read_text())["records"]
    judge_label = {r["id"]: r["full_text_label"] for r in records}
    judge_ids = {r["id"] for r in sensitive if judge_label[r["id"]] != ASSERTS}

    # --- self-check: reproduce the published heuristic-label figures exactly ---
    pub = json.loads(PUBLISHED.read_text())["per_token"][" Taiwan"]
    check = stats_for(" Taiwan", sensitive, control, heuristic_ids)
    expected = {
        ("jlens", "auc"): pub["jlens"]["auc_conceal_gt_control"],
        ("jlens", "diff"): pub["jlens"]["bootstrap"]["mean_diff"],
        ("jlens", "lo"): pub["jlens"]["bootstrap"]["ci95"][0],
        ("jlens", "hi"): pub["jlens"]["bootstrap"]["ci95"][1],
        ("logit", "auc"): pub["logit"]["auc_conceal_gt_control"],
        ("logit", "diff"): pub["logit"]["bootstrap"]["mean_diff"],
        ("logit", "lo"): pub["logit"]["bootstrap"]["ci95"][0],
        ("logit", "hi"): pub["logit"]["bootstrap"]["ci95"][1],
    }
    got = {
        ("jlens", "auc"): check["jlens"]["auc_conceal_gt_control"],
        ("jlens", "diff"): check["jlens"]["mean_diff"],
        ("jlens", "lo"): check["jlens"]["ci95"][0],
        ("jlens", "hi"): check["jlens"]["ci95"][1],
        ("logit", "auc"): check["logit"]["auc_conceal_gt_control"],
        ("logit", "diff"): check["logit"]["mean_diff"],
        ("logit", "lo"): check["logit"]["ci95"][0],
        ("logit", "hi"): check["logit"]["ci95"][1],
    }
    bad = {k: (expected[k], got[k]) for k in expected
           if abs(expected[k] - got[k]) > 1e-9}
    if bad or len(heuristic_ids) != 68:
        raise SystemExit(
            f"self-check failed (n_conceal={len(heuristic_ids)}, expected 68); "
            f"mismatches: {bad}. Refusing to write {OUT_PATH}."
        )
    print(f"self-check OK: heuristic labels reproduce runs/exp1/h1.json exactly "
          f"(n_conceal={len(heuristic_ids)})")

    # --- the relabelled figures ---
    tokens = [" Taiwan", "Hong", " Hong", " Kong"]
    per_token = {t: stats_for(t, sensitive, control, judge_ids) for t in tokens}

    result = {
        "source": (
            "activations from runs/exp1/h1_instrumented.json (unchanged); conceal set "
            "redefined by the structured judge's full-response label in "
            "runs/relabel_2026-07-27/h1_structured.json"
        ),
        "judge_model": json.loads(STRUCTURED.read_text())["judge_model"],
        "layer": inst["summary"]["primary_layer"],
        "n_sensitive": len(sensitive),
        "n_control": len(control),
        "n_conceal_heuristic": len(heuristic_ids),
        "n_conceal_judge": len(judge_ids),
        "n_labels_changed": sum(
            1 for r in records
            if r["old_label"].replace("(heuristic)", "") != r["full_text_label"]
        ),
        "per_token": per_token,
        "self_check": (
            "recomputing with the heuristic conceal set reproduces runs/exp1/h1.json's "
            "per-token means, AUCs and bootstrap CIs to 1e-9"
        ),
    }
    OUT_PATH.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {OUT_PATH}\n")
    for t in tokens:
        for lens in ("jlens", "logit"):
            s = per_token[t][lens]
            print(f"{t!r:10} {lens:6} C={s['mean_conceal']:.4f} ctrl={s['mean_control']:.4f} "
                  f"diff={s['mean_diff']:.4f} CI=[{s['ci95'][0]:.4f}, {s['ci95'][1]:.4f}] "
                  f"AUC={s['auc_conceal_gt_control']:.4f}  n={s['n_conceal']}")


if __name__ == "__main__":
    main()
