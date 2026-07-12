"""Experiment 1b — instrumented H1 rerun (robustness-check plan T1.2).

Re-runs the H1 measurement on the base 14B with three instrumentation upgrades demanded by
`docs/preregistration/robustness-checks-2026-07-12.md` T1.2, so the paper can replace bound-only arguments with direct
measurements:

  (a) --save-per-prompt (always on here): saves the per-prompt activation of every target
      token, at every swept layer, for both lenses. This is the standing instrumentation
      rule (h1.json's absence of per-prompt values already cost a rerun) AND it enables the
      *direct per-topic AUC*: e.g. " Taiwan" AUC computed only on conceal rows whose topic
      is NOT taiwan — a direct test of cross-topic censored-referent salience, replacing §6's
      mention-ceiling bound.
  (b) --layers: a layer sweep (generation is done ONCE, layer-independent; only the readout
      vectors + projections are recomputed per layer), answering "layer 24 was never
      optimized for H1" with the full AUC-vs-layer profile.
  (c) --averaging-set {a,b}: rebuild the readout vectors from a DISJOINT set of neutral
      averaging prompts, to show the AUC is not an artifact of the particular 28 prompts.

Reported run (14B, GPU):
    python exp/exp1b_h1_instrumented.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
        --device cuda --judge-model <deepinfra> --layers 8,12,16,20,24,28,32,36 \
        --primary-layer 24 --averaging-set a --out runs/exp1/h1_instrumented.json
CPU plumbing smoke (1.5B):
    python exp/exp1b_h1_instrumented.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
        --device cpu --limit 3 --limit-targets 4 --max-new-tokens 6 --layers 6,12 \
        --primary-layer 12 --out runs/exp1/h1b_smoke.json
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
    _auc_greater, _bootstrap_diff, _load_jsonl, DATA,
)

# A DISJOINT neutral averaging set (no overlap with AVERAGING_PROMPTS / exp0's set) for the
# readout-prompt robustness check (T1.2c). Generic, concept-neutral declaratives.
AVERAGING_PROMPTS_B = [
    "The train pulled into the station a few minutes late.",
    "He poured himself a glass of water and sat down.",
    "The library closes early on public holidays.",
    "They repainted the fence over the long weekend.",
    "A gentle breeze moved through the open window.",
    "The recipe calls for two cups of flour and one egg.",
    "She parked the car and walked the rest of the way.",
    "The lecture covered the basics of supply and demand.",
    "Rain is expected across the region later this evening.",
    "The children built a sandcastle near the water's edge.",
    "He tightened the last bolt and closed the panel.",
    "The bakery on the corner sells out by midmorning.",
    "Their flight was delayed because of the weather.",
    "The committee will review the proposal next month.",
    "A new coat of varnish protected the wooden table.",
    "She sorted the mail into three neat piles.",
    "The path wound gently up the side of the hill.",
    "The printer ran out of paper halfway through the job.",
    "We watered the garden before the sun got too hot.",
    "The museum added a new wing for temporary exhibits.",
]

# Proper-noun anchor token -> the topic it "belongs" to. Direct per-topic AUC measures each
# anchor's separation using ONLY conceal rows OUTSIDE its home topic (cross-topic salience).
ANCHOR_HOME_TOPIC = {
    " Taiwan": "taiwan", "Taiwan": "taiwan",
    " Hong": "hongkong", " Kong": "hongkong", "Hong": "hongkong", "Kong": "hongkong",
    " Tibet": "tibet", " Xinjiang": "xinjiang", " Uyghur": "xinjiang",
    " Tiananmen": "tiananmen",
}


def _measure_arm(model, tok, rows, responses, vecs, target_ids, layer, device):
    """Per-prompt J-lens + logit activations for a pre-generated arm at one layer."""
    recs_jl, recs_ll, per_prompt = [], [], []
    for r, resp in zip(rows, responses):
        full = r["prompt"] + " " + resp
        jl = prompt_activation(model, tok, full, vecs, layer, device)
        ll = _logit_prompt_activation(model, tok, full, target_ids, layer, device)
        recs_jl.append(jl)
        recs_ll.append(ll)
        per_prompt.append({"jlens": jl, "logit": ll})
    return recs_jl, recs_ll, per_prompt


def _pooled(conceal_by_lens, control_by_lens):
    out = {}
    for lens in ("jlens", "logit"):
        c, k = conceal_by_lens[lens], control_by_lens[lens]
        out[lens] = {
            "mean_conceal": float(np.mean(c)) if c else None,
            "mean_control": float(np.mean(k)) if k else None,
            "auc_conceal_gt_control": _auc_greater(c, k),
            "bootstrap": _bootstrap_diff(c, k),
            "n_conceal": len(c), "n_control": len(k),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--layers", default="8,12,16,20,24,28,32,36",
                    help="comma-separated layers to sweep")
    ap.add_argument("--primary-layer", type=int, default=24,
                    help="layer used for the headline direct-per-topic AUC")
    ap.add_argument("--averaging-set", choices=["a", "b"], default="a",
                    help="which disjoint neutral prompt set builds the readout vectors")
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--limit-targets", type=int, default=None)
    ap.add_argument("--responses-cache", default=None,
                    help="JSON path to cache/reuse greedy generations (avoids regenerating "
                         "for a second averaging-set run; also crash-resilient)")
    ap.add_argument("--out", default="runs/exp1/h1_instrumented.json")
    args = ap.parse_args()

    dtype = args.dtype or ("float32" if args.device == "cpu" else "bfloat16")
    model, tok = load_model(args.model, device=args.device, dtype=dtype)
    nL = model.config.num_hidden_layers
    layers = [int(x) for x in args.layers.split(",") if x.strip() != ""]
    layers = [L for L in layers if 0 <= L < nL]
    if args.primary_layer not in layers:
        layers = sorted(set(layers + [args.primary_layer]))
    avg_prompts = AVERAGING_PROMPTS if args.averaging_set == "a" else AVERAGING_PROMPTS_B

    with open(os.path.join(DATA, "target_tokens.json")) as f:
        targets = json.load(f)
    if args.limit_targets:
        targets = dict(list(targets.items())[:args.limit_targets])
    target_ids = list(targets.values())
    id2str = {v: k for k, v in targets.items()}

    sensitive = _load_jsonl(os.path.join(DATA, "sensitive_prompts.jsonl"))
    control = _load_jsonl(os.path.join(DATA, "control_prompts.jsonl"))
    if args.limit:
        sensitive, control = sensitive[:args.limit], control[:args.limit]
    print(f"model={args.model} layers={layers} primary={args.primary_layer} "
          f"avg_set={args.averaging_set} targets={len(target_ids)} "
          f"sens={len(sensitive)} ctrl={len(control)} "
          f"judge={'LLM:'+args.judge_model if args.judge_model else 'heuristic'}")

    # 1. Generate every response ONCE (greedy => deterministic, layer-independent). Cache to
    #    disk keyed by prompt id so a second averaging-set run reuses them (and a crash resumes).
    cache = {}
    if args.responses_cache and os.path.exists(args.responses_cache):
        with open(args.responses_cache) as f:
            cache = json.load(f)
        print(f"[gen] loaded {len(cache)} cached responses from {args.responses_cache}", flush=True)

    def _gen_arm(rows, tag):
        out = []
        for r in rows:
            if r["id"] in cache:
                out.append(cache[r["id"]])
                continue
            resp = _generate(model, tok, r["prompt"], args.device, args.max_new_tokens)
            cache[r["id"]] = resp
            out.append(resp)
            if args.responses_cache:
                with open(args.responses_cache, "w") as f:
                    json.dump(cache, f)
        print(f"[gen] {tag} done ({len(out)})", flush=True)
        return out

    print("[gen] sensitive...", flush=True)
    sens_resp = _gen_arm(sensitive, "sensitive")
    print("[gen] control...", flush=True)
    ctrl_resp = _gen_arm(control, "control")

    # 2. Label the sensitive arm once (layer-independent).
    sens_labels = []
    for r, resp in zip(sensitive, sens_resp):
        asserts, label = _label_asserts_fact(r["prompt"], resp, args.judge_model)
        sens_labels.append({"asserts_fact": bool(asserts), "label": label})

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    per_layer = {}
    per_prompt_store = {"sensitive": None, "control": None}
    for L in layers:
        print(f"[layer {L}] building readout vectors ({len(avg_prompts)} prompts)...", flush=True)
        vecs = jlens_vectors(model, tok, target_ids, L, avg_prompts, args.device)
        sens_jl, sens_ll, sens_pp = _measure_arm(
            model, tok, sensitive, sens_resp, vecs, target_ids, L, args.device)
        ctrl_jl, ctrl_ll, ctrl_pp = _measure_arm(
            model, tok, control, ctrl_resp, vecs, target_ids, L, args.device)

        # pooled + per-token AUC (conceal = not-asserting sensitive rows)
        pooled_conceal = {"jlens": [], "logit": []}
        pooled_control = {"jlens": [], "logit": []}
        per_token = {}
        for t in target_ids:
            conceal_jl = [sens_jl[i][t] for i in range(len(sensitive))
                          if not sens_labels[i]["asserts_fact"] and t in sens_jl[i]]
            conceal_ll = [sens_ll[i][t] for i in range(len(sensitive))
                          if not sens_labels[i]["asserts_fact"] and t in sens_ll[i]]
            ctrl_jl_v = [ctrl_jl[i][t] for i in range(len(control)) if t in ctrl_jl[i]]
            ctrl_ll_v = [ctrl_ll[i][t] for i in range(len(control)) if t in ctrl_ll[i]]
            pooled_conceal["jlens"] += conceal_jl
            pooled_conceal["logit"] += conceal_ll
            pooled_control["jlens"] += ctrl_jl_v
            pooled_control["logit"] += ctrl_ll_v
            per_token[id2str[t]] = {
                "jlens": {"auc_conceal_gt_control": _auc_greater(conceal_jl, ctrl_jl_v),
                          "n_conceal": len(conceal_jl),
                          "mean_conceal": float(np.mean(conceal_jl)) if conceal_jl else None,
                          "mean_control": float(np.mean(ctrl_jl_v)) if ctrl_jl_v else None},
                "logit": {"auc_conceal_gt_control": _auc_greater(conceal_ll, ctrl_ll_v),
                          "n_conceal": len(conceal_ll),
                          "mean_conceal": float(np.mean(conceal_ll)) if conceal_ll else None,
                          "mean_control": float(np.mean(ctrl_ll_v)) if ctrl_ll_v else None},
            }
        per_layer[str(L)] = {"pooled": _pooled(pooled_conceal, pooled_control),
                             "per_token": per_token}

        if L == args.primary_layer:
            per_prompt_store["sensitive"] = [
                {"id": sensitive[i]["id"], "topic": sensitive[i]["topic"],
                 "label": sens_labels[i]["label"],
                 "asserts_fact": sens_labels[i]["asserts_fact"],
                 "response": sens_resp[i],
                 "jlens": {id2str[t]: sens_pp[i]["jlens"][t] for t in target_ids},
                 "logit": {id2str[t]: sens_pp[i]["logit"][t] for t in target_ids}}
                for i in range(len(sensitive))]
            per_prompt_store["control"] = [
                {"id": control[i]["id"], "topic": control[i]["topic"],
                 "response": ctrl_resp[i],
                 "jlens": {id2str[t]: ctrl_pp[i]["jlens"][t] for t in target_ids},
                 "logit": {id2str[t]: ctrl_pp[i]["logit"][t] for t in target_ids}}
                for i in range(len(control))]

    # 3. Direct per-topic AUC at the primary layer (the T1.2 headline): for each anchor
    #    token, AUC of its activation on conceal rows OUTSIDE its home topic vs all controls.
    primary = per_prompt_store
    direct = {}
    if primary["sensitive"] is not None:
        for anchor, home in ANCHOR_HOME_TOPIC.items():
            if anchor not in id2str.values():
                continue
            row = {"home_topic": home}
            for lens in ("jlens", "logit"):
                off = [p[lens][anchor] for p in primary["sensitive"]
                       if (not p["asserts_fact"]) and p["topic"] != home and anchor in p[lens]]
                ctl = [p[lens][anchor] for p in primary["control"] if anchor in p[lens]]
                row[lens] = {"auc_offtopic_conceal_gt_control": _auc_greater(off, ctl),
                             "n_offtopic_conceal": len(off), "n_control": len(ctl)}
            direct[anchor] = row

    # peak layer by pooled J-lens AUC
    peak_layer = None
    best = -1.0
    for L in layers:
        a = per_layer[str(L)]["pooled"]["jlens"]["auc_conceal_gt_control"]
        if a is not None and a > best:
            best, peak_layer = a, L
    prim = per_layer[str(args.primary_layer)]["pooled"]
    jl_auc = prim["jlens"]["auc_conceal_gt_control"]
    ll_auc = prim["logit"]["auc_conceal_gt_control"]

    result = {
        "summary": {
            "model": args.model, "layers": layers, "primary_layer": args.primary_layer,
            "averaging_set": args.averaging_set, "n_averaging_prompts": len(avg_prompts),
            "n_targets": len(target_ids), "n_sensitive": len(sensitive),
            "n_control": len(control), "judge": args.judge_model or "heuristic",
            "primary_pooled": prim,
            "jlens_beats_logit_on_conceal_gap": (
                jl_auc is not None and ll_auc is not None and jl_auc > ll_auc),
            "peak_layer_jlens": peak_layer, "peak_jlens_auc": (best if best >= 0 else None),
            "auc_by_layer_jlens": {str(L): per_layer[str(L)]["pooled"]["jlens"]["auc_conceal_gt_control"]
                                   for L in layers},
            "auc_by_layer_logit": {str(L): per_layer[str(L)]["pooled"]["logit"]["auc_conceal_gt_control"]
                                   for L in layers},
            "direct_per_topic_auc": direct,
        },
        "per_layer": per_layer,
        "per_prompt": per_prompt_store,
        "averaging_prompts_used": avg_prompts,
        "targets": targets,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result["summary"], indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
