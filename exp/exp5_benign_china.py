"""Experiment 5 — benign-China control arm (rebuttal-plan T1.3).

Discriminating test for the H1 cross-topic-salience finding: is the elevated activation of
censored referents (" Taiwan", " Hong", " Kong") on sensitive prompts a *censorship
mechanism*, or merely CCP-topic-family *association* (any China context lights them up)?

Three arms, one readout-vector build, apples-to-apples at one layer:
  - censored-conceal: the sensitive prompts the model does NOT assert (the H1 conceal set),
  - neutral-control:  the matched non-China control prompts,
  - benign-China:     ~40 benign China prompts (cuisine/pandas/tea/landmarks/history),
                      which mention/evoke China but touch NO censored topic.

Pre-registered reading (either way is publishable; §6 wording adjusts):
  - association  ⇒ benign-China anchor salience ≈ censored-conceal (>> neutral-control):
                   the referents fire on ANY China context.
  - mechanism    ⇒ censored-conceal >> benign-China (≈ neutral-control): the referents fire
                   only when the censored topic is actually engaged.

Reuses the T1.2 response cache (runs/exp1/h1b_responses.json) for the sensitive+control arms
(no regeneration); generates only the benign-China responses. Heuristic labeler (matches
h1.json / exp1b); saved per-prompt activations let a stronger judge re-slice offline.

    python exp/exp5_benign_china.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
        --device cuda --layer 24 --responses-cache runs/exp1/h1b_responses.json \
        --out runs/exp5/benign_china.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jspace.model import load_model
from jspace.jlens import jlens_vectors
from jspace.conflict import prompt_activation
from exp.exp1_h1_conflict import (
    AVERAGING_PROMPTS, _generate, _logit_prompt_activation, _label_asserts_fact,
    _auc_greater, _load_jsonl, DATA,
)

ANCHORS = [" Taiwan", " Hong", " Kong"]


def _measure(model, tok, rows, responses, vecs, target_ids, layer, device):
    """Per-prompt anchor activations (J-lens + logit) for a pre-generated arm."""
    out = []
    for r, resp in zip(rows, responses):
        full = r["prompt"] + " " + resp
        jl = prompt_activation(model, tok, full, vecs, layer, device)
        ll = _logit_prompt_activation(model, tok, full, target_ids, layer, device)
        out.append({"id": r["id"], "topic": r.get("topic"), "response": resp,
                    "jlens": jl, "logit": ll})
    return out


def _arm_stats(recs, anchor_id, id2str, lens):
    vals = [r[lens][anchor_id] for r in recs if anchor_id in r[lens]]
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--layer", type=int, default=24)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--limit", type=int, default=None, help="cap prompts per arm (smoke)")
    ap.add_argument("--limit-targets", type=int, default=None)
    ap.add_argument("--responses-cache", default="runs/exp1/h1b_responses.json",
                    help="reuse T1.2 greedy generations for sensitive+control (id-keyed)")
    ap.add_argument("--benign-cache", default="runs/exp5/benign_responses.json")
    ap.add_argument("--out", default="runs/exp5/benign_china.json")
    args = ap.parse_args()

    dtype = args.dtype or ("float32" if args.device == "cpu" else "bfloat16")
    model, tok = load_model(args.model, device=args.device, dtype=dtype)
    layer = args.layer

    with open(os.path.join(DATA, "target_tokens.json")) as f:
        targets = json.load(f)
    if args.limit_targets:
        # keep the anchors even under a target cap (smoke)
        keep = dict(list(targets.items())[:args.limit_targets])
        for a in ANCHORS:
            if a in targets:
                keep[a] = targets[a]
        targets = keep
    target_ids = list(targets.values())
    id2str = {v: k for k, v in targets.items()}
    anchor_ids = [targets[a] for a in ANCHORS if a in targets]

    sensitive = _load_jsonl(os.path.join(DATA, "sensitive_prompts.jsonl"))
    control = _load_jsonl(os.path.join(DATA, "control_prompts.jsonl"))
    benign = _load_jsonl(os.path.join(DATA, "benign_china_prompts.jsonl"))
    if args.limit:
        sensitive, control, benign = sensitive[:args.limit], control[:args.limit], benign[:args.limit]

    # responses: reuse cache for sensitive+control; generate benign (own cache).
    resp_cache = {}
    if os.path.exists(args.responses_cache):
        with open(args.responses_cache) as f:
            resp_cache = json.load(f)
    print(f"[cache] {len(resp_cache)} reusable responses from {args.responses_cache}", flush=True)

    def _responses(rows, cache, cache_path):
        out = []
        for r in rows:
            if r["id"] in cache:
                out.append(cache[r["id"]]); continue
            resp = _generate(model, tok, r["prompt"], args.device, args.max_new_tokens)
            cache[r["id"]] = resp; out.append(resp)
            if cache_path:
                os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
                with open(cache_path, "w") as f:
                    json.dump(cache, f)
        return out

    sens_resp = _responses(sensitive, resp_cache, None)   # reuse-only (don't rewrite T1.2 cache)
    ctrl_resp = _responses(control, resp_cache, None)
    benign_cache = {}
    if os.path.exists(args.benign_cache):
        with open(args.benign_cache) as f:
            benign_cache = json.load(f)
    print("[gen] benign-China...", flush=True)
    benign_resp = _responses(benign, benign_cache, args.benign_cache)

    # readout vectors (same set A as T1.2, primary layer) then measure all three arms.
    print(f"[layer {layer}] building readout vectors...", flush=True)
    vecs = jlens_vectors(model, tok, target_ids, layer, AVERAGING_PROMPTS, args.device)

    sens_recs = _measure(model, tok, sensitive, sens_resp, vecs, target_ids, layer, args.device)
    ctrl_recs = _measure(model, tok, control, ctrl_resp, vecs, target_ids, layer, args.device)
    benign_recs = _measure(model, tok, benign, benign_resp, vecs, target_ids, layer, args.device)

    # conceal subset of the sensitive arm (heuristic labels; deterministic from responses)
    conceal_recs = []
    for r, resp in zip(sensitive, sens_resp):
        asserts, _ = _label_asserts_fact(r["prompt"], resp, None)
        if not asserts:
            conceal_recs.append(next(x for x in sens_recs if x["id"] == r["id"]))

    per_anchor = {}
    for a in ANCHORS:
        if a not in targets:
            continue
        aid = targets[a]
        row = {}
        for lens in ("jlens", "logit"):
            conceal = _arm_stats(conceal_recs, aid, id2str, lens)
            neutral = _arm_stats(ctrl_recs, aid, id2str, lens)
            bch = _arm_stats(benign_recs, aid, id2str, lens)
            row[lens] = {
                "mean_censored_conceal": float(np.mean(conceal)) if conceal else None,
                "mean_neutral_control": float(np.mean(neutral)) if neutral else None,
                "mean_benign_china": float(np.mean(bch)) if bch else None,
                "auc_conceal_gt_neutral": _auc_greater(conceal, neutral),
                "auc_benign_gt_neutral": _auc_greater(bch, neutral),
                "auc_conceal_gt_benign": _auc_greater(conceal, bch),
                "n_conceal": len(conceal), "n_neutral": len(neutral), "n_benign": len(bch),
            }
        per_anchor[a] = row

    # Verdict: for the J-lens anchors, is censored-conceal clearly above benign-China
    # (mechanism) or is benign-China already near the censored level (association)?
    # Report the mean fraction (benign-neutral)/(conceal-neutral) across anchors: ~0 =>
    # mechanism-specific; ~1 => full association. Pre-registered as descriptive (either-way).
    fracs = []
    for a in ANCHORS:
        if a not in per_anchor:
            continue
        j = per_anchor[a]["jlens"]
        c, n, b = j["mean_censored_conceal"], j["mean_neutral_control"], j["mean_benign_china"]
        if c is not None and n is not None and b is not None and abs(c - n) > 1e-9:
            fracs.append((b - n) / (c - n))
    assoc_fraction = float(np.mean(fracs)) if fracs else None

    result = {
        "summary": {
            "model": args.model, "layer": layer, "anchors": ANCHORS,
            "n_sensitive": len(sensitive), "n_conceal": len(conceal_recs),
            "n_control": len(control), "n_benign_china": len(benign),
            "per_anchor": per_anchor,
            "assoc_fraction_jlens": assoc_fraction,
            "assoc_fraction_note": "(benign_china - neutral)/(conceal - neutral) mean over "
                                   "J-lens anchors: ~0 => mechanism-specific, ~1 => topic-family "
                                   "association. Descriptive, pre-registered either-way.",
        },
        "arms": {
            "censored_conceal": conceal_recs,
            "neutral_control": ctrl_recs,
            "benign_china": benign_recs,
        },
        "targets": targets,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result["summary"], indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
