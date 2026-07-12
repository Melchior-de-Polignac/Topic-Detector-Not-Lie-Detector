"""T1.5 statistical hardening of H1 — pure reanalysis of the instrumented per-prompt data.

Reads runs/exp1/h1_instrumented.json (per-prompt J-lens + logit activations at the primary
layer, saved by exp1b) and answers the statistics referee:
  (a) bootstrap CIs on the pooled conceal-vs-control AUC, for BOTH lenses;
  (b) CLUSTER bootstrap at the prompt, topic, and token levels (the 4 proper-noun forms are
      3 words; prompts cluster by topic; 52 per-token comparisons are correlated);
  (c) direct per-topic anchor AUC with bootstrap CI (off-home-topic conceal vs control);
  (d) permutation test for the proper-noun vs generic per-token AUC gap;
  (e) unconditional-C sensitivity — pooled AUC using ALL sensitive rows (no label
      conditioning); if separation barely moves, heuristic labels aren't driving the result;
  (f) per-token AUC bootstrap CIs + a multiplicity-aware count (how many exclude 0.5).

    python exp/analyze_h1_instrumented.py --in runs/exp1/h1_instrumented.json \
        --out runs/exp1/h1_instrumented_stats.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp.analyze_h1 import ENTITY, PROPER_NOUN


def _auroc(scores, labels):
    """Rank-based AUROC (tie-averaged). labels: 1=conceal/positive, 0=control."""
    scores = np.asarray(scores, float); labels = np.asarray(labels, int)
    n_pos = int(labels.sum()); n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores, kind="mergesort"); s = scores[order]
    r = np.arange(1, len(scores) + 1, dtype=float); rr = r.copy()
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            rr[i:j + 1] = (r[i] + r[j]) / 2.0
        i = j + 1
    ranks = np.empty(len(scores)); ranks[order] = rr
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _ci(vals, lo=2.5, hi=97.5):
    vals = [v for v in vals if v is not None]
    if not vals:
        return [None, None]
    return [float(np.percentile(vals, lo)), float(np.percentile(vals, hi))]


def _pool(sens, ctrl, tokens, lens, conceal_only=True):
    """Flatten (prompt,token) activation pairs -> (scores, labels, sens_row_idx). label 1 =
    sensitive-conceal row, 0 = control row; sens_row_idx tags which sensitive prompt each
    positive came from (for prompt/topic cluster resampling)."""
    scores, labels, src = [], [], []
    for i, p in enumerate(sens):
        if conceal_only and p.get("asserts_fact"):
            continue
        for t in tokens:
            if t in p[lens]:
                scores.append(p[lens][t]); labels.append(1); src.append(("s", i))
    for i, p in enumerate(ctrl):
        for t in tokens:
            if t in p[lens]:
                scores.append(p[lens][t]); labels.append(0); src.append(("c", i))
    return np.asarray(scores), np.asarray(labels), src


def _pooled_auc_ci(sens, ctrl, tokens, lens, conceal_only, n_boot, rng, cluster):
    """Point AUROC + bootstrap CI under a resampling scheme:
       cluster='prompt' resample rows; 'topic' resample topics; 'token' resample tokens."""
    scores, labels, src = _pool(sens, ctrl, tokens, lens, conceal_only)
    point = _auroc(scores, labels)
    boots = []
    if cluster == "token":
        for _ in range(n_boot):
            toks_b = list(rng.choice(tokens, size=len(tokens), replace=True))
            sc, lb, _ = _pool(sens, ctrl, toks_b, lens, conceal_only)
            boots.append(_auroc(sc, lb))
    elif cluster == "topic":
        s_top = sorted(set(p.get("topic") for p in sens))
        c_top = sorted(set(p.get("topic") for p in ctrl))
        s_by = {tp: [i for i, p in enumerate(sens) if p.get("topic") == tp] for tp in s_top}
        c_by = {tp: [i for i, p in enumerate(ctrl) if p.get("topic") == tp] for tp in c_top}
        for _ in range(n_boot):
            si = [i for tp in rng.choice(s_top, len(s_top), replace=True) for i in s_by[tp]]
            ci = [i for tp in rng.choice(c_top, len(c_top), replace=True) for i in c_by[tp]]
            sc, lb, _ = _pool([sens[i] for i in si], [ctrl[i] for i in ci], tokens, lens, conceal_only)
            boots.append(_auroc(sc, lb))
    else:  # prompt-level
        sidx = [i for i, p in enumerate(sens) if not (conceal_only and p.get("asserts_fact"))]
        cidx = list(range(len(ctrl)))
        for _ in range(n_boot):
            si = rng.choice(sidx, len(sidx), replace=True)
            ci = rng.choice(cidx, len(cidx), replace=True)
            sc, lb, _ = _pool([sens[i] for i in si], [ctrl[i] for i in ci], tokens, lens, conceal_only)
            boots.append(_auroc(sc, lb))
    return {"auc": point, "ci95": _ci(boots), "cluster": cluster, "n_boot": n_boot}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="runs/exp1/h1_instrumented.json")
    ap.add_argument("--out", default="runs/exp1/h1_instrumented_stats.json")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = json.load(open(args.inp))
    prim = str(data["summary"]["primary_layer"])
    sens = data["per_prompt"]["sensitive"]
    ctrl = data["per_prompt"]["control"]
    tokens = list(data["targets"].keys())
    rng = np.random.default_rng(args.seed)

    def key(t):  # stripped form for class membership
        return t.strip()

    proper = [t for t in tokens if key(t) in PROPER_NOUN]
    entity = [t for t in tokens if key(t) in ENTITY]
    generic = [t for t in tokens if key(t) not in ENTITY]

    out = {"summary": {"primary_layer": int(prim), "n_boot": args.n_boot,
                       "n_sensitive": len(sens), "n_control": len(ctrl),
                       "n_conceal": sum(1 for p in sens if not p.get("asserts_fact")),
                       "n_tokens": len(tokens)}}

    # (a,b) pooled AUC + CIs under three resampling schemes, both lenses, all tokens.
    pooled = {}
    for lens in ("jlens", "logit"):
        pooled[lens] = {c: _pooled_auc_ci(sens, ctrl, tokens, lens, True, args.n_boot, rng, c)
                        for c in ("prompt", "topic", "token")}
    out["pooled_auc"] = pooled

    # (e) unconditional-C sensitivity (all sensitive rows, prompt bootstrap).
    out["unconditional"] = {
        lens: _pooled_auc_ci(sens, ctrl, tokens, lens, False, args.n_boot, rng, "prompt")
        for lens in ("jlens", "logit")}

    # (c) direct per-topic anchor AUC + CI (off-home-topic conceal vs control).
    HOME = {" Taiwan": "taiwan", " Hong": "hongkong", " Kong": "hongkong"}
    direct = {}
    for a, home in HOME.items():
        if a not in tokens:
            continue
        row = {}
        for lens in ("jlens", "logit"):
            off = [(p[lens][a], ("s", i)) for i, p in enumerate(sens)
                   if (not p.get("asserts_fact")) and p.get("topic") != home and a in p[lens]]
            con = [(p[lens][a], ("c", i)) for i, p in enumerate(ctrl) if a in p[lens]]
            sc = np.array([x[0] for x in off + con]); lb = np.array([1] * len(off) + [0] * len(con))
            point = _auroc(sc, lb)
            boots = []
            for _ in range(args.n_boot):
                oi = rng.integers(0, len(off), len(off)) if off else []
                ci = rng.integers(0, len(con), len(con)) if con else []
                s2 = np.array([off[k][0] for k in oi] + [con[k][0] for k in ci])
                l2 = np.array([1] * len(oi) + [0] * len(ci))
                boots.append(_auroc(s2, l2))
            row[lens] = {"auc": point, "ci95": _ci(boots),
                         "n_offtopic_conceal": len(off), "n_control": len(con)}
        direct[a] = {"home_topic": home, **row}
    out["direct_per_topic_auc"] = direct

    # per-token AUCs (point) for the class analysis + (f) per-token CI multiplicity count.
    per_tok = {}
    n_excl = {"jlens": 0, "logit": 0}
    for t in tokens:
        per_tok[t] = {}
        for lens in ("jlens", "logit"):
            conceal = [p[lens][t] for p in sens if not p.get("asserts_fact") and t in p[lens]]
            control = [p[lens][t] for p in ctrl if t in p[lens]]
            sc = np.array(conceal + control); lb = np.array([1] * len(conceal) + [0] * len(control))
            point = _auroc(sc, lb)
            boots = []
            for _ in range(500):
                ci = rng.integers(0, len(conceal), len(conceal)) if conceal else []
                cj = rng.integers(0, len(control), len(control)) if control else []
                s2 = np.array([conceal[k] for k in ci] + [control[k] for k in cj])
                l2 = np.array([1] * len(ci) + [0] * len(cj))
                boots.append(_auroc(s2, l2))
            lo, hi = _ci(boots)
            per_tok[t][lens] = {"auc": point, "ci95": [lo, hi]}
            if lo is not None and lo > 0.5:
                n_excl[lens] += 1
    out["per_token"] = per_tok
    out["summary"]["n_tokens_ci_excludes_0.5"] = n_excl
    out["summary"]["multiplicity_note"] = (
        f"{n_excl['jlens']}/{len(tokens)} tokens have a J-lens AUC 95% CI strictly above 0.5 "
        "(per-token bootstrap, 500 reps); a conservative multiplicity-aware count.")

    # (d) permutation test: mean per-token AUC(proper_noun) - mean(generic), J-lens.
    def mean_auc(toks, lens):
        v = [per_tok[t][lens]["auc"] for t in toks if per_tok[t][lens]["auc"] is not None]
        return float(np.mean(v)) if v else None
    obs = (mean_auc(proper, "jlens") or 0) - (mean_auc(generic, "jlens") or 0)
    all_aucs = {t: per_tok[t]["jlens"]["auc"] for t in tokens if per_tok[t]["jlens"]["auc"] is not None}
    keys = list(all_aucs); vals = np.array([all_aucs[k] for k in keys])
    n_p = len(proper); ge = 0; nperm = 5000
    for _ in range(nperm):
        idx = rng.permutation(len(vals))
        d = vals[idx[:n_p]].mean() - vals[idx[n_p:]].mean()
        if abs(d) >= abs(obs):
            ge += 1
    out["class_means"] = {
        "proper_noun": {"n": len(proper), "mean_jlens_auc": mean_auc(proper, "jlens"),
                        "mean_logit_auc": mean_auc(proper, "logit")},
        "entity": {"n": len(entity), "mean_jlens_auc": mean_auc(entity, "jlens"),
                   "mean_logit_auc": mean_auc(entity, "logit")},
        "generic": {"n": len(generic), "mean_jlens_auc": mean_auc(generic, "jlens"),
                    "mean_logit_auc": mean_auc(generic, "logit")},
        "permutation_proper_vs_generic": {
            "observed_diff_jlens_auc": obs, "p_value": (ge + 1) / (nperm + 1), "n_perm": nperm},
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["summary"], indent=2))
    print("pooled J-lens (prompt CI):", pooled["jlens"]["prompt"])
    print("pooled logit (prompt CI):", pooled["logit"]["prompt"])
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
