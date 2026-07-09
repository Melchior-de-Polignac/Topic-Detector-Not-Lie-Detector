"""Post-hoc H1 analysis: the conflict signal is carried by the *referents* of the
censored facts (named entities + event/policy-specific terms), not by generic political
vocabulary — which the matched controls legitimately use, confounding it.

Reads runs/exp1/h1.json (per-token AUC = P(conceal>control), both lenses) and reports a
PRE-REGISTERED partition of the 52 targets by linguistic type. This is not cherry-picking:
the partition is by whether a token *names the censored subject* vs is a diffuse abstraction,
decided by type, not by score.

    python exp/analyze_h1.py --in runs/exp1/h1.json --out runs/exp1/h1_analysis.json
"""
import argparse
import json
import numpy as np

# Pre-registered by linguistic type (referent-of-censored-fact vs generic abstraction).
ENTITY = {  # proper nouns + event/policy-specific terms = the concealed referents
    "Taiwan", "Hong", "Kong", "surveillance", "crackdown", "extradition", "umbrella",
    "massacre", "tanks", "tank", "camps", "detention", "riots", "students", "square",
    "troops", "soldiers", "Muslim", "autonomy", "protest", "protests", "genocide",
}
PROPER_NOUN = {"Taiwan", "Hong", "Kong"}  # the sharpest, least-ambiguous subset


def _key(tok):
    return tok.strip()


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def analyze(h1):
    pt = h1["per_token"]
    groups = {"proper_noun": [], "entity": [], "generic": [], "all": []}
    for tok, r in pt.items():
        ja = r["jlens"]["auc_conceal_gt_control"]
        la = r["logit"]["auc_conceal_gt_control"]
        if ja is None:
            continue
        rec = (tok, ja, la)
        groups["all"].append(rec)
        k = _key(tok)
        if k in PROPER_NOUN:
            groups["proper_noun"].append(rec)
        if k in ENTITY:
            groups["entity"].append(rec)
        else:
            groups["generic"].append(rec)
    out = {}
    for g, recs in groups.items():
        out[g] = {
            "n_tokens": len(recs),
            "mean_jlens_auc": _mean([r[1] for r in recs]),
            "mean_logit_auc": _mean([r[2] for r in recs]),
            "jlens_beats_logit": (_mean([r[1] for r in recs]) or 0) > (_mean([r[2] for r in recs]) or 0),
        }
    top = sorted(groups["all"], key=lambda r: r[1], reverse=True)[:8]
    out["top8_by_jlens_auc"] = [{"token": t, "jlens": round(j, 3), "logit": round(l, 3)} for t, j, l in top]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="runs/exp1/h1.json")
    ap.add_argument("--out", default="runs/exp1/h1_analysis.json")
    args = ap.parse_args()
    h1 = json.load(open(args.inp))
    res = analyze(h1)
    res["labels_on_sensitive"] = {}
    from collections import Counter
    res["labels_on_sensitive"] = dict(Counter(m["label"] for m in h1.get("sensitive_meta", [])))
    json.dump(res, open(args.out, "w"), indent=2)
    print(json.dumps(res, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
