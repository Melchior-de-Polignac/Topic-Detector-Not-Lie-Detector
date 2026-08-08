"""Recompute the benign-China control under the structured relabel.

Companion to exp/build_h1_conceal_stats_relabeled.py, and needed for the same reason:
runs/exp5/benign_china.json defines its censored-conceal arm with the retired keyword
heuristic (68 of the 95 sensitive prompts), while the paper reports every other statistic
under the structured judge's labels (69 prompts). Leaving this one arm on the old
definition puts a conceal mean of 12.32 on Figure 4 against 11.65 in the paper's own
text, for the same quantity.

Nothing is re-measured. The per-prompt activations for all 95 sensitive prompts are in
runs/exp1/h1_instrumented.json and are the same numbers benign_china.json was built from
(its published conceal means reproduce from them exactly, which the self-check below
asserts). The benign-China and neutral-control arms are untouched: they have no labels
and so cannot move.

Note that the benign-vs-neutral AUCs the paper also quotes (0.956, 0.927, 0.989) do not
depend on the conceal set at all and are unchanged; only the conceal-vs-benign AUCs and
the association fraction move.

CPU only, no model load, deterministic.

Usage: .venv/bin/python exp/build_benign_china_relabeled.py
"""
import json
from pathlib import Path

import numpy as np

INSTRUMENTED = Path("runs/exp1/h1_instrumented.json")
BENIGN = Path("runs/exp5/benign_china.json")
STRUCTURED = Path("runs/relabel_2026-07-27/h1_structured.json")
OUT_PATH = Path("runs/relabel_2026-07-27/benign_china_relabeled.json")

ANCHORS = (" Taiwan", " Hong", " Kong")
ASSERTS = "asserts_fact"


def _auc_greater(a, b):
    """P(x>y), ties at 0.5. Same helper as exp1_h1_conflict / exp5_benign_china."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a[:, None] - b[None, :]
    return float((np.sum(d > 0) + 0.5 * np.sum(d == 0)) / (a.size * b.size))


def per_anchor(sensitive, conceal_ids, benign_vals, summary):
    out, fractions = {}, []
    for t in ANCHORS:
        pub = summary["per_anchor"][t]["jlens"]
        neutral_mean = pub["mean_neutral_control"]
        benign_mean = pub["mean_benign_china"]
        conceal = [r["jlens"][t] for r in sensitive if r["id"] in conceal_ids]
        conceal_mean = float(np.mean(conceal))
        frac = (benign_mean - neutral_mean) / (conceal_mean - neutral_mean)
        fractions.append(frac)
        out[t] = {"jlens": {
            "mean_censored_conceal": conceal_mean,
            "mean_benign_china": benign_mean,
            "mean_neutral_control": neutral_mean,
            "auc_benign_gt_neutral": pub["auc_benign_gt_neutral"],
            "auc_conceal_gt_benign": _auc_greater(conceal, benign_vals[t]),
            "assoc_fraction": frac,
            "n_conceal": len(conceal),
            "n_benign": len(benign_vals[t]),
            "n_neutral": pub["n_neutral"],
        }}
    return out, float(np.mean(fractions))


def main():
    inst = json.loads(INSTRUMENTED.read_text())
    ben = json.loads(BENIGN.read_text())
    sensitive = inst["per_prompt"]["sensitive"]
    summary = ben["summary"]

    # arms carry activations keyed by token id, not token string
    tid = {t: str(i) for t, i in ben["targets"].items()}
    benign_vals = {t: [r["jlens"][tid[t]] for r in ben["arms"]["benign_china"]]
                   for t in ANCHORS}

    heuristic_ids = {r["id"] for r in sensitive if not r["asserts_fact"]}
    judge_label = {r["id"]: r["full_text_label"]
                   for r in json.loads(STRUCTURED.read_text())["records"]}
    judge_ids = {r["id"] for r in sensitive if judge_label[r["id"]] != ASSERTS}

    # --- self-check: the heuristic set must reproduce the published figures ---
    check, check_frac = per_anchor(sensitive, heuristic_ids, benign_vals, summary)
    bad = []
    if abs(check_frac - summary["assoc_fraction_jlens"]) > 1e-9:
        bad.append(f"assoc_fraction {check_frac} vs {summary['assoc_fraction_jlens']}")
    for t in ANCHORS:
        pub = summary["per_anchor"][t]["jlens"]
        for key in ("mean_censored_conceal", "auc_conceal_gt_benign"):
            if abs(check[t]["jlens"][key] - pub[key]) > 1e-9:
                bad.append(f"{t} {key}: {check[t]['jlens'][key]} vs {pub[key]}")
    if bad or len(heuristic_ids) != 68:
        raise SystemExit(f"self-check failed: {bad} (n_conceal={len(heuristic_ids)}). "
                         f"Refusing to write {OUT_PATH}.")
    print(f"self-check OK: heuristic labels reproduce runs/exp5/benign_china.json "
          f"exactly (n_conceal={len(heuristic_ids)}, assoc_fraction={check_frac:.4f})")

    relabelled, frac = per_anchor(sensitive, judge_ids, benign_vals, summary)
    # Shape mirrors runs/exp5/benign_china.json (per_anchor + assoc_fraction_jlens
    # nested under "summary") so exp/fig_benign_china.py can read it unchanged.
    result = {
        "source": (
            "runs/exp5/benign_china.json with its censored-conceal arm redefined by the "
            "structured judge's full-response label in "
            "runs/relabel_2026-07-27/h1_structured.json; benign-China and neutral arms "
            "unchanged (unlabelled, so unaffected); activations from "
            "runs/exp1/h1_instrumented.json"
        ),
        "layer": summary["layer"],
        "anchors": list(ANCHORS),
        "n_conceal_heuristic": len(heuristic_ids),
        "n_conceal_judge": len(judge_ids),
        "n_benign_china": summary["n_benign_china"],
        "n_control": summary["n_control"],
        "summary": {
            "layer": summary["layer"],
            "anchors": list(ANCHORS),
            "per_anchor": relabelled,
            "assoc_fraction_jlens": frac,
            "assoc_fraction_note": summary["assoc_fraction_note"],
            "assoc_fraction_heuristic_superseded": summary["assoc_fraction_jlens"],
        },
        "self_check": (
            "recomputing with the heuristic conceal set reproduces benign_china.json's "
            "conceal means, conceal-vs-benign AUCs and association fraction to 1e-9"
        ),
    }
    OUT_PATH.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {OUT_PATH}\n")
    for t in ANCHORS:
        r = relabelled[t]["jlens"]
        print(f"{t!r:10} conceal={r['mean_censored_conceal']:.3f} "
              f"benign={r['mean_benign_china']:.3f} "
              f"benign>neutral={r['auc_benign_gt_neutral']:.3f} "
              f"conceal>benign={r['auc_conceal_gt_benign']:.3f} "
              f"assoc={r['assoc_fraction']:.3f}")
    print(f"\nassociation fraction: {frac:.4f} "
          f"(was {summary['assoc_fraction_jlens']:.4f} under the heuristic labels)")


if __name__ == "__main__":
    main()
