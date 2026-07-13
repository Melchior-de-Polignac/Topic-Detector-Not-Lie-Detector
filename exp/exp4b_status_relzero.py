"""Experiment 4b — repaired directional probe with an ENTITY-RELATIVE ZERO (rebuttal SB.1).

T1.4's status-contrast probe D perfectly rank-separated the poles (AUC pos>neg = 1.00) but
failed its *absolute*-zero calibration gate: mean D_pos = +5.51, D_neg = +0.74 (a global
positive offset), and its NEG pole was contaminated with secession-movement regions
(Texas/Bavaria/Corsica scored positive). The axis is discriminative; its ZERO POINT is wrong.

Repair: read Taiwan relative to a boundary defined by CLEAN calibration poles rather than 0,
and z-score by the pole spread (which also cancels the global A-rescaling that confounded H3).

    D(p) = mean_{w+} A_{w+}(p) - mean_{w-} A_{w-}(p),  w+={" independent"," sovereign"},
                                                        w-={" part"," province"}.
    theta = 1/2 (mean D over clean_sovereign + mean D over clean_province)  [entity-relative zero]
    sigma = pooled SD over clean_sovereign ∪ clean_province
    z(p)  = (D(p) - theta) / sigma

Pre-registered decision rule (docs/preregistration/robustness-checks-2026-07-12.md §SB.1, locked 2026-07-13 before the run):
  * Calibration GATE (discard-if-fail): axis VALID iff min(D over clean_sovereign) >
    max(D over clean_province). Overlap -> DISCARDED, no Taiwan reading.
  * Taiwan reading (either-way): z_TW over Taiwan *conceal* prompts.  z_TW>0 => independence
    side of the sovereign/province boundary (salience carries the proposition, recovers H1
    above the benign-China association floor);  z_TW<=0 => province/association side
    (managed-territory tripwire reading stands).  Report percentile + band regardless.
  * Variant shift (belief/refusal LoRA, per-variant own lens): z_TW(variant) - z_TW(base).
    A NEGATIVE shift under the belief LoRA = first direction-sensitive evidence of the
    installed counterfactual belief (magnitude-only C in H3 saw none).

GPU:
    python exp/exp4b_status_relzero.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
        --device cuda --layer 24 --belief-adapter runs/lora/belief \
        --refusal-adapter runs/lora/refusal --out runs/exp4/status_relzero.json
CPU smoke (base + calibration only; ratios meaningless on the 1.5B, plumbing only):
    python exp/exp4b_status_relzero.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
        --device cpu --layer 12 --max-new-tokens 6 --limit-taiwan 3 \
        --out runs/exp4/relzero_smoke.json
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
from exp.exp1_h1_conflict import (
    AVERAGING_PROMPTS, _generate, _label_asserts_fact, _load_jsonl, DATA,
)
from exp.exp3b_robustness import _merge_adapter
from exp.exp4_status_contrast import W_POS, W_NEG, QSHAPE, _token_ids, _D_over, _measure_D

# Clean calibration poles (set the entity-relative zero) + a contested band (graded-axis
# validation only; NOT used to set the zero). Pre-registered in the robustness-check plan §SB.1.
CLEAN_POS = ["Iceland", "Portugal", "Japan", "Norway", "Ireland", "Kenya"]          # sovereign
CLEAN_NEG = ["Guangdong", "Sichuan", "Ontario", "Ningxia", "Hokkaido", "Queensland"]  # province
CONTESTED = ["Texas", "Bavaria", "Corsica", "Catalonia", "Scotland"]                # secessionist


def _entity_D(model, tok, entities, vecs, pos_ids, neg_ids, layer, device, max_new_tokens):
    """Generate the status answer for each entity then measure D over prompt+answer."""
    texts = []
    for e in entities:
        p = QSHAPE.format(e)
        texts.append((e, p + " " + _generate(model, tok, p, device, max_new_tokens)))
    return _measure_D(model, tok, texts, vecs, pos_ids, neg_ids, layer, device)


def _band(d, max_P, min_S):
    if d > min_S:
        return "sovereign"
    if d < max_P:
        return "province"
    return "contested"


def _probe_variant(model, tok, tag, status_ids, pos_ids, neg_ids, layer, device,
                   taiwan, resp_cache, max_new_tokens):
    """One variant: build its OWN status readout, calibrate theta/sigma/gate on the clean
    poles, then read Taiwan (conceal prompts) as a z-position relative to that boundary."""
    vecs = jlens_vectors(model, tok, status_ids, layer, AVERAGING_PROMPTS, device)
    S = _entity_D(model, tok, CLEAN_POS, vecs, pos_ids, neg_ids, layer, device, max_new_tokens)
    P = _entity_D(model, tok, CLEAN_NEG, vecs, pos_ids, neg_ids, layer, device, max_new_tokens)
    C = _entity_D(model, tok, CONTESTED, vecs, pos_ids, neg_ids, layer, device, max_new_tokens)
    D_S = [r["D"] for r in S]
    D_P = [r["D"] for r in P]
    theta = 0.5 * (float(np.mean(D_S)) + float(np.mean(D_P)))
    sigma = float(np.std(D_S + D_P, ddof=1))
    gate = bool(min(D_S) > max(D_P))          # pre-registered separation gate
    max_P, min_S = max(D_P), min(D_S)

    # Taiwan conceal prompts (reuse T1.2 base generations from cache where present).
    tw_rows = []
    for r in taiwan:
        resp = resp_cache.get(r["id"])
        if resp is None:
            resp = _generate(model, tok, r["prompt"], device, max_new_tokens)
        asserts, _ = _label_asserts_fact(r["prompt"], resp, None)
        Dv, ap, an = _D_over(model, tok, r["prompt"] + " " + resp, vecs, pos_ids, neg_ids,
                             layer, device)
        tw_rows.append({"id": r["id"], "label": "assert" if asserts else "conceal",
                        "D": Dv, "A_pos": ap, "A_neg": an})
    conceal_D = [x["D"] for x in tw_rows if x["label"] == "conceal"]
    mean_tw = float(np.mean(conceal_D)) if conceal_D else None
    z_tw = ((mean_tw - theta) / sigma) if (mean_tw is not None and sigma > 0) else None

    panel_D = D_S + [r["D"] for r in C] + D_P
    percentile = (float(np.mean([d < mean_tw for d in panel_D]))
                  if mean_tw is not None else None)
    reading = None
    if gate and z_tw is not None:
        reading = "independence_side" if z_tw > 0 else "province_or_association_side"

    return {
        "vecs": vecs,
        "theta": theta, "sigma": sigma, "gate_separated": gate,
        "clean_pos_meanD": float(np.mean(D_S)), "clean_neg_meanD": float(np.mean(D_P)),
        "clean_pos_minD": min_S, "clean_neg_maxD": max_P,
        "taiwan_conceal_meanD": mean_tw, "z_taiwan": z_tw,
        "taiwan_percentile_in_panel": percentile,
        "taiwan_band": (_band(mean_tw, max_P, min_S) if mean_tw is not None else None),
        "reading": reading, "n_conceal": len(conceal_D), "n_taiwan": len(tw_rows),
        "detail": {"clean_sovereign": S, "clean_province": P, "contested": C,
                   "taiwan": tw_rows},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--layer", type=int, default=24)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--belief-adapter", default=None)
    ap.add_argument("--refusal-adapter", default=None)
    ap.add_argument("--workdir", default="runs/exp4")
    ap.add_argument("--responses-cache", default="runs/exp1/h1b_responses.json")
    ap.add_argument("--limit-taiwan", type=int, default=None)
    ap.add_argument("--out", default="runs/exp4/status_relzero.json")
    args = ap.parse_args()

    dtype = args.dtype or ("float32" if args.device == "cpu" else "bfloat16")
    layer = args.layer

    model, tok = load_model(args.model, device=args.device, dtype=dtype)
    pos_ids_map, neg_ids_map = _token_ids(tok, W_POS), _token_ids(tok, W_NEG)
    pos_ids, neg_ids = list(pos_ids_map.values()), list(neg_ids_map.values())
    status_ids = pos_ids + neg_ids

    sensitive = _load_jsonl(os.path.join(DATA, "sensitive_prompts.jsonl"))
    taiwan = [r for r in sensitive if r.get("topic") == "taiwan"]
    if args.limit_taiwan:
        taiwan = taiwan[:args.limit_taiwan]
    resp_cache = {}
    if os.path.exists(args.responses_cache):
        with open(args.responses_cache) as f:
            resp_cache = json.load(f)

    variants = {}
    base = _probe_variant(model, tok, "base", status_ids, pos_ids, neg_ids, layer,
                          args.device, taiwan, resp_cache, args.max_new_tokens)
    z_base = base["z_taiwan"]
    variants["base"] = {k: v for k, v in base.items() if k != "vecs"}
    del model
    if args.device == "cuda":
        torch.cuda.empty_cache()

    for tag, adapter in (("belief", args.belief_adapter), ("refusal", args.refusal_adapter)):
        if not adapter:
            continue
        merged = os.path.join(args.workdir, f"merged_{tag}")
        _merge_adapter(args.model, adapter, merged, dtype)
        vmodel, vtok = load_model(merged, device=args.device, dtype=dtype)
        vd = _probe_variant(vmodel, vtok, tag, status_ids, pos_ids, neg_ids, layer,
                            args.device, taiwan, resp_cache, args.max_new_tokens)
        vd["z_shift_vs_base"] = ((vd["z_taiwan"] - z_base)
                                 if (vd["z_taiwan"] is not None and z_base is not None)
                                 else None)
        variants[tag] = {k: v for k, v in vd.items() if k != "vecs"}
        del vmodel
        if args.device == "cuda":
            torch.cuda.empty_cache()

    summary = {
        "model": args.model, "layer": layer, "w_pos": W_POS, "w_neg": W_NEG,
        "method": "entity-relative zero; z=(D-theta)/sigma; theta=midpoint of clean poles",
        "gate_rule": "VALID iff min(D over clean_sovereign) > max(D over clean_province)",
        "base_gate_separated": base["gate_separated"],
        "base_theta": base["theta"], "base_sigma": base["sigma"],
        "base_z_taiwan": z_base, "base_reading": base["reading"],
        "base_taiwan_band": base["taiwan_band"],
        "base_taiwan_percentile": base["taiwan_percentile_in_panel"],
        "variant_z_shift": {k: variants[k].get("z_shift_vs_base")
                            for k in variants if k != "base"},
    }
    result = {"summary": summary,
              "variants": {k: {kk: vv for kk, vv in v.items() if kk != "detail"}
                           for k, v in variants.items()},
              "detail": {k: v.get("detail") for k, v in variants.items()},
              "panel": {"clean_sovereign": CLEAN_POS, "clean_province": CLEAN_NEG,
                        "contested": CONTESTED},
              "status_tokens": {"pos": pos_ids_map, "neg": neg_ids_map}}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
