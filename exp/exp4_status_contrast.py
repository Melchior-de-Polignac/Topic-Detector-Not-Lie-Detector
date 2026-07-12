"""Experiment 4 — the status-contrast probe D (rebuttal-plan T1.4 / paper §8 executed).

Salience alone doesn't say WHAT is behind a lit token. D asks a directional question the
single-token J-lens *can* answer, by differencing two readouts within one forward pass so
mention/topic common modes cancel:

    D(p) = mean_{w+} A_{w+}(p) - mean_{w-} A_{w-}(p),
    w+ = {" independent", " sovereign"},  w- = {" part", " province"}   (all single tokens).

Pipeline & pre-registered decision rule:
  1. CALIBRATE (falsification gate, pre-committed): D must be > 0 on known sovereign states
     (Iceland/Portugal/...) and < 0 on known provinces/regions (Guangdong/Corsica/Texas/...).
     If mean(D_pos) <= 0 or mean(D_neg) >= 0, the probe is DISCARDED (calibrated=false) and
     no Taiwan claim is made from it.
  2. BASE model, Taiwan prompts: sign of D at the assertion site. D > 0 => the gate holds the
     independence proposition; D <= 0 => priming/association reading. Report either way.
  3. BELIEF / REFUSAL LoRA merges (per-variant OWN readout vectors, CPU-merged): D shift vs
     base. A negative-pole shift under belief = first direction-sensitive evidence of the
     installed counterfactual belief. (D differences cancel a global A-rescaling to first
     order; still reported as a shift, with that caveat.)

    python exp/exp4_status_contrast.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
        --device cuda --layer 24 --belief-adapter runs/lora/belief \
        --refusal-adapter runs/lora/refusal --out runs/exp4/status_contrast.json
CPU smoke (base + calibration only, no adapters):
    python exp/exp4_status_contrast.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
        --device cpu --layer 12 --max-new-tokens 6 --out runs/exp4/d_smoke.json
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
    AVERAGING_PROMPTS, _generate, _label_asserts_fact, _auc_greater, _load_jsonl, DATA,
)
from exp.exp3b_robustness import _merge_adapter

W_POS = [" independent", " sovereign"]
W_NEG = [" part", " province"]
QSHAPE = "Is {} a sovereign, independent country?"
# Calibration entities with unambiguous status (the probe's falsification set).
CALIB_POS = ["Iceland", "Portugal", "Japan", "Norway", "Ireland", "Kenya"]      # sovereign
CALIB_NEG = ["Guangdong", "Sichuan", "Corsica", "Texas", "Bavaria", "Ontario"]  # sub-national


def _token_ids(tok, words):
    out = {}
    for w in words:
        ids = tok(w, add_special_tokens=False).input_ids
        assert len(ids) == 1, f"{w!r} is not single-token: {ids}"
        out[w] = ids[0]
    return out


def _D_over(model, tok, text, vecs, pos_ids, neg_ids, layer, device):
    act = prompt_activation(model, tok, text, vecs, layer, device)
    ap = float(np.mean([act[t] for t in pos_ids if t in act]))
    an = float(np.mean([act[t] for t in neg_ids if t in act]))
    return ap - an, ap, an


def _measure_D(model, tok, prompts_texts, vecs, pos_ids, neg_ids, layer, device):
    """prompts_texts: list of (label, full_text). Returns list of {label, D, A_pos, A_neg}."""
    rows = []
    for label, text in prompts_texts:
        D, ap, an = _D_over(model, tok, text, vecs, pos_ids, neg_ids, layer, device)
        rows.append({"label": label, "D": D, "A_pos": ap, "A_neg": an})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--layer", type=int, default=24)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--belief-adapter", default=None)
    ap.add_argument("--refusal-adapter", default=None)
    ap.add_argument("--workdir", default="runs/exp4")
    ap.add_argument("--responses-cache", default="runs/exp1/h1b_responses.json",
                    help="reuse T1.2 generations for the base Taiwan prompts")
    ap.add_argument("--limit-taiwan", type=int, default=None)
    ap.add_argument("--out", default="runs/exp4/status_contrast.json")
    args = ap.parse_args()

    dtype = args.dtype or ("float32" if args.device == "cpu" else "bfloat16")
    layer = args.layer

    # Load base to resolve tokenizer + token ids first.
    model, tok = load_model(args.model, device=args.device, dtype=dtype)
    pos_ids_map = _token_ids(tok, W_POS)
    neg_ids_map = _token_ids(tok, W_NEG)
    pos_ids, neg_ids = list(pos_ids_map.values()), list(neg_ids_map.values())
    status_ids = pos_ids + neg_ids

    # Taiwan prompts (sensitive, topic==taiwan) + reuse cache for their base generations.
    sensitive = _load_jsonl(os.path.join(DATA, "sensitive_prompts.jsonl"))
    taiwan = [r for r in sensitive if r.get("topic") == "taiwan"]
    if args.limit_taiwan:
        taiwan = taiwan[:args.limit_taiwan]
    resp_cache = {}
    if os.path.exists(args.responses_cache):
        with open(args.responses_cache) as f:
            resp_cache = json.load(f)

    def _variant_D(vmodel, vtok, tag):
        """Build this variant's OWN status readout vectors, measure D on Taiwan prompts."""
        vecs = jlens_vectors(vmodel, vtok, status_ids, layer, AVERAGING_PROMPTS, args.device)
        texts = []
        for r in taiwan:
            resp = resp_cache.get(r["id"])
            if resp is None:
                resp = _generate(vmodel, vtok, r["prompt"], args.device, args.max_new_tokens)
            asserts, _ = _label_asserts_fact(r["prompt"], resp, None)
            texts.append((f'{r["id"]}|{"assert" if asserts else "conceal"}',
                          r["prompt"] + " " + resp))
        rows = _measure_D(vmodel, vtok, texts, vecs, pos_ids, neg_ids, layer, args.device)
        conceal_D = [x["D"] for x in rows if x["label"].endswith("conceal")]
        return {"vecs": vecs, "rows": rows,
                "mean_D": float(np.mean([x["D"] for x in rows])) if rows else None,
                "mean_D_conceal": float(np.mean(conceal_D)) if conceal_D else None,
                "n": len(rows), "n_conceal": len(conceal_D)}

    # --- BASE: calibration + Taiwan ---
    base_vecs = jlens_vectors(model, tok, status_ids, layer, AVERAGING_PROMPTS, args.device)
    calib_pos_texts, calib_neg_texts = [], []
    for e in CALIB_POS:
        p = QSHAPE.format(e)
        calib_pos_texts.append((e, p + " " + _generate(model, tok, p, args.device, args.max_new_tokens)))
    for e in CALIB_NEG:
        p = QSHAPE.format(e)
        calib_neg_texts.append((e, p + " " + _generate(model, tok, p, args.device, args.max_new_tokens)))
    calib_pos = _measure_D(model, tok, calib_pos_texts, base_vecs, pos_ids, neg_ids, layer, args.device)
    calib_neg = _measure_D(model, tok, calib_neg_texts, base_vecs, pos_ids, neg_ids, layer, args.device)
    Dpos = [x["D"] for x in calib_pos]; Dneg = [x["D"] for x in calib_neg]
    mean_Dpos, mean_Dneg = float(np.mean(Dpos)), float(np.mean(Dneg))
    calibrated = (mean_Dpos > 0) and (mean_Dneg < 0)

    base_taiwan = _variant_D(model, tok, "base")
    base_mean_D = base_taiwan["mean_D_conceal"]

    variants = {"base": {k: v for k, v in base_taiwan.items() if k != "vecs"}}
    # free base before merging adapters (one 14B on GPU at a time)
    del model
    if args.device == "cuda":
        torch.cuda.empty_cache()

    for tag, adapter in (("belief", args.belief_adapter), ("refusal", args.refusal_adapter)):
        if not adapter:
            continue
        merged = os.path.join(args.workdir, f"merged_{tag}")
        _merge_adapter(args.model, adapter, merged, dtype)
        vmodel, vtok = load_model(merged, device=args.device, dtype=dtype)
        vd = _variant_D(vmodel, vtok, tag)
        variants[tag] = {k: v for k, v in vd.items() if k != "vecs"}
        variants[tag]["D_shift_vs_base"] = (
            (vd["mean_D_conceal"] - base_mean_D)
            if (vd["mean_D_conceal"] is not None and base_mean_D is not None) else None)
        del vmodel
        if args.device == "cuda":
            torch.cuda.empty_cache()

    result = {
        "summary": {
            "model": args.model, "layer": layer, "w_pos": W_POS, "w_neg": W_NEG,
            "calibration": {
                "mean_D_positive_states": mean_Dpos, "mean_D_negative_states": mean_Dneg,
                "auc_pos_gt_neg": _auc_greater(Dpos, Dneg),
                "calibrated": bool(calibrated),
                "rule": "calibrated = mean(D_pos)>0 AND mean(D_neg)<0; probe discarded if false",
            },
            "base_taiwan_conceal_mean_D": base_mean_D,
            "base_taiwan_reading": (None if not calibrated else
                                    ("holds_independence_proposition" if (base_mean_D or 0) > 0
                                     else "priming_or_association")),
            "variants": {k: {kk: vv for kk, vv in v.items() if kk != "rows"}
                         for k, v in variants.items()},
        },
        "calibration_detail": {"positive": calib_pos, "negative": calib_neg},
        "taiwan_detail": {k: variants[k].get("rows") for k in variants},
        "status_tokens": {"pos": pos_ids_map, "neg": neg_ids_map},
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result["summary"], indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
