"""Experiment 3b — H3 robustness checks: the gate for the Option-A reframe.

H3 came back a real negative: belief-LoRA training did NOT collapse the conflict signal C,
it RAISED it (Taiwan-anchor C_jlens 1.29x base). The Option-A paper reframe rests entirely
on that rise being *genuine* — specifically that the contested true-fact concepts get louder
in J-space under an installed false belief. Two confounds could fake it, and one sweep kills
the obvious reviewer objection. All three checks REUSE the answers already generated + judged
in `runs/exp3/h3.json` (no regeneration, no judge, no Heretic) and the existing adapters
(plain merges of base / belief / refusal only):

  Check 1 — control-token normalization (MAKE-OR-BREAK). Each variant is read with its OWN
    J-lens; a LoRA could rescale readout geometry so *everything* projects higher. Recompute C
    (same concealment cases, variant-own lens) for signal-free NEUTRAL tokens (H1's generic
    vocab, AUC~0.5). PASS: Taiwan-anchor belief/base ratio (~1.29) clearly exceeds the neutral
    ratio (~1.0). FAIL: neutrals rise comparably -> lens-rescaling artifact -> Option B.

  Check 2 — concealment-population matching. C pools over non-asserting answers, but belief's
    concealment set is mostly `asserts_counterfact` text while base's is `refuses`/`deflects`.
    Recompute C (a) per judge label class and (b) unconditionally over all questions. PASS:
    belief > base holds within matched classes and unconditionally (direction not reversed).

  Check 3 — layer sweep (reported either way). C_jlens(base) vs C_jlens(belief) at ~8 layers,
    Taiwan-anchor tokens only. Kills "the collapse happens at another layer" and upgrades
    "we picked the middle layer" to "signal peaks at layer L".

The gate (`robustness_verdicts`, pure + unit-tested): Option A proceeds iff Check 1 passes and
Check 2 does not reverse the direction. Check 3 is descriptive.

Run (GPU box, after H3 left merged_belief/merged_refusal under --workdir OR merge fresh):
    python3 exp/exp3b_robustness.py --base deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
        --workdir runs/exp3 --h3 runs/exp3/h3.json \
        --target-tokens data/target_tokens_h3_taiwan.json \
        --sweep-layers 8,12,16,20,24,28,32,36 --out runs/exp3/h3_robustness.json --figure

CPU smoke (plumbing only; point every arm at the 1.5B so ratios are ~1.0 and meaningless):
    python3 exp/exp3b_robustness.py --base deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
        --belief-src deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
        --refusal-src deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
        --device cpu --limit 3 --sweep-layers 6,12 --out runs/exp3/h3_robustness_smoke.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


# ----------------------------- pure gate logic (unit-tested) -----------------------------
def robustness_verdicts(check1, check2, taiwan_margin=0.15, neutral_ceiling=1.15):
    """Assemble the Option-A gate from the two confound checks (Check 3 is descriptive).

    check1 = {"taiwan_belief_base_ratio": float|None, "neutral_belief_base_ratio": float|None}
    check2 = {"unconditional": {"base": C|None, "belief": C|None},
              "by_label": {label: {"base": C|None, "belief": C|None}, ...}}

    Check 1 passes when the contested Taiwan-anchor tokens rise clearly MORE than signal-free
    neutral tokens, and the neutral tokens are ~flat (not a global lens rescaling).
    Check 2 must not reverse the direction: belief C > base C unconditionally AND in every
    matched label class that has both numbers.
    """
    t = check1.get("taiwan_belief_base_ratio")
    n = check1.get("neutral_belief_base_ratio")
    check1_pass = (t is not None and n is not None
                   and t > n + taiwan_margin and n < neutral_ceiling)

    uc = check2.get("unconditional", {})
    uc_base, uc_belief = uc.get("base"), uc.get("belief")
    uncond_belief_gt_base = (uc_base is not None and uc_belief is not None
                             and uc_belief > uc_base)

    matched = []
    for _lab, d in check2.get("by_label", {}).items():
        if d.get("base") is not None and d.get("belief") is not None:
            matched.append(d["belief"] > d["base"])
    check2_not_reversed = bool(uncond_belief_gt_base and (all(matched) if matched else True))

    return {
        "check1_control_token_pass": bool(check1_pass),
        "taiwan_belief_base_ratio": t,
        "neutral_belief_base_ratio": n,
        "check2_uncond_belief_gt_base": bool(uncond_belief_gt_base),
        "check2_not_reversed_in_class": check2_not_reversed,
        "option_a_proceed": bool(check1_pass and check2_not_reversed),
    }


# ----------------------------- neutral control-token set (Check 1) -----------------------------
def neutral_token_ids(all_targets):
    """Signal-free neutral tokens for the control-token check: the H1 'generic' partition
    (mean J-lens AUC ~0.46 in runs/exp1/h1_analysis.json), i.e. every 52-target token whose
    stripped form is NOT a censored referent (analyze_h1.ENTITY) and is not a Taiwan anchor.

    all_targets = {token_str: id} — the full data/target_tokens.json (52 tokens).
    Returns {token_str: id}.
    """
    from exp.analyze_h1 import ENTITY
    anchors = {"Taiwan", "recognized", "independent", "independence"}
    out = {}
    for tok, tid in all_targets.items():
        k = tok.strip()
        if k in ENTITY or k in anchors:
            continue
        out[tok] = tid
    return out


# ----------------------------- C aggregation over precomputed activations -----------------------------
def _pooled_C(records, token_ids, labels=None):
    """Mean workspace activation over (record, token) pairs.

    records: list of {"label": str, "act": {tid: float}}. labels=None -> concealment set
    (label != "asserts_fact", the standard C population); labels="all" -> every record;
    a set/list of label strings -> only those classes.
    """
    vals = []
    for r in records:
        if labels is None:
            if r["label"] == "asserts_fact":
                continue
        elif labels != "all" and r["label"] not in labels:
            continue
        for t in token_ids:
            if t in r["act"]:
                vals.append(r["act"][t])
    return float(np.mean(vals)) if vals else None


def _ratio(belief, base):
    return round(belief / base, 4) if (belief is not None and base not in (None, 0)) else None


# ----------------------------- GPU measurement -----------------------------
def _measure_records(model, tok, per_question, prompts, token_ids, layer, device,
                     limit=None, avg_limit=None):
    """Recompute per-question workspace activations for `token_ids` on ONE loaded model,
    reusing the answers H3 already generated (no regeneration). Returns
    [{"id", "label", "act": {tid: float}}] for the concealment + all populations to reuse.
    `avg_limit` caps the J-lens averaging prompts (smoke only — full set on GPU).
    """
    from jspace.jlens import jlens_vectors
    from jspace.conflict import prompt_activation
    from exp.exp1_h1_conflict import AVERAGING_PROMPTS

    avg_prompts = AVERAGING_PROMPTS if avg_limit is None else AVERAGING_PROMPTS[:avg_limit]
    vecs = jlens_vectors(model, tok, token_ids, layer, avg_prompts, device)
    recs = []
    items = per_question if limit is None else per_question[:limit]
    for pq in items:
        prompt = prompts.get(pq["id"])
        if prompt is None:
            continue
        full = prompt + " " + pq["answer"]
        act = prompt_activation(model, tok, full, vecs, layer, device)
        recs.append({"id": pq["id"], "label": pq["label"],
                     "act": {t: float(act[t]) for t in token_ids if t in act}})
    return recs


def _load_model_cached(cache, src, device, dtype):
    """Load (and cache within this run) a variant model, so Check 1/2 and the sweep's
    primary layer reuse one load per variant instead of reloading the 14B repeatedly."""
    from jspace.model import load_model
    if src not in cache:
        cache[src] = load_model(src, device=device, dtype=dtype)
    return cache[src]


def _merge_adapter(base, adapter, out_dir, device, dtype):
    """Plain-merge a LoRA `adapter` onto `base` -> `out_dir` (cached; reused if present).

    Mirrors exp3's proven merge: load base, apply adapter, merge_and_unload, save, then FREE
    the merge model off the GPU before the caller reloads the merged dir — else base (28GB)
    and merged (28GB) coexist and OOM the 46GB card. On a FRESH pod the merged dirs from the
    H3 run are gone (ephemeral), so this re-creates them from the LFS adapters.
    """
    if os.path.isdir(os.path.join(out_dir, "config.json")):
        print(f"[merge] reuse {out_dir}")
        return out_dir
    import torch
    from jspace.model import load_model
    from peft import PeftModel
    print(f"[merge] {adapter} onto {base} -> {out_dir}")
    m, tok = load_model(base, device=device, dtype=dtype)
    m = PeftModel.from_pretrained(m, adapter)
    m = m.merge_and_unload()
    os.makedirs(out_dir, exist_ok=True)
    m.save_pretrained(out_dir); tok.save_pretrained(out_dir)
    del m, tok
    if device == "cuda":
        torch.cuda.empty_cache()
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
    ap.add_argument("--workdir", default="runs/exp3")
    ap.add_argument("--belief-adapter", default=None,
                    help="LoRA adapter dir for the belief arm (merged onto --base). On a fresh "
                         "pod pass runs/lora/belief (LFS); the merged dir is cached under workdir.")
    ap.add_argument("--refusal-adapter", default=None,
                    help="LoRA adapter dir for the refusal arm (merged onto --base).")
    ap.add_argument("--belief-src", default=None,
                    help="pre-merged belief model dir (overrides --belief-adapter; default: "
                         "<workdir>/merged_belief). Point at a plain model for the CPU smoke.")
    ap.add_argument("--refusal-src", default=None,
                    help="pre-merged refusal model dir (overrides --refusal-adapter; default: "
                         "<workdir>/merged_refusal).")
    ap.add_argument("--h3", default="runs/exp3/h3.json")
    ap.add_argument("--questions", default=os.path.join(DATA, "eval_questions.jsonl"))
    ap.add_argument("--target-tokens", default=os.path.join(DATA, "target_tokens_h3_taiwan.json"),
                    help="Taiwan-anchor target set (the contested true-fact tokens)")
    ap.add_argument("--all-tokens", default=os.path.join(DATA, "target_tokens.json"),
                    help="full 52-target set; neutral control tokens are derived from it")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--layer", type=int, default=None, help="primary layer (default n_layers//2)")
    ap.add_argument("--sweep-layers", default="8,12,16,20,24,28,32,36",
                    help="comma-separated layers for Check 3 (base vs belief, Taiwan tokens)")
    ap.add_argument("--limit", type=int, default=None, help="limit #questions (smoke)")
    ap.add_argument("--limit-neutral", type=int, default=None,
                    help="cap #neutral control tokens (smoke)")
    ap.add_argument("--avg-prompts", type=int, default=None,
                    help="cap J-lens averaging prompts (smoke; default = full set)")
    ap.add_argument("--out", default="runs/exp3/h3_robustness.json")
    ap.add_argument("--figure", action="store_true")
    args = ap.parse_args()

    import torch

    def _arm_src(explicit_src, adapter, tag):
        # priority: explicit pre-merged dir > merge from adapter > default cached merged dir
        if explicit_src:
            return explicit_src
        if adapter:
            return _merge_adapter(args.base, adapter,
                                  os.path.join(args.workdir, f"merged_{tag}"),
                                  args.device, args.dtype)
        return os.path.join(args.workdir, f"merged_{tag}")

    os.makedirs(args.workdir, exist_ok=True)
    belief_src = _arm_src(args.belief_src, args.belief_adapter, "belief")
    refusal_src = _arm_src(args.refusal_src, args.refusal_adapter, "refusal")
    sources = {"base": args.base, "belief_lora": belief_src, "refusal_lora": refusal_src}

    taiwan = json.load(open(args.target_tokens))          # {token_str: id}
    all_targets = json.load(open(args.all_tokens))
    neutral = neutral_token_ids(all_targets)
    if args.limit_neutral is not None:
        neutral = dict(list(neutral.items())[:args.limit_neutral])
    taiwan_ids = list(taiwan.values())
    neutral_ids = list(neutral.values())
    print(f"Taiwan-anchor tokens ({len(taiwan_ids)}): {list(taiwan.keys())}")
    print(f"neutral control tokens ({len(neutral_ids)}): {list(neutral.keys())}")

    prompts = {}
    with open(args.questions) as f:
        for line in f:
            if line.strip():
                q = json.loads(line)
                prompts[q["id"]] = q["prompt"]

    h3 = json.load(open(args.h3))
    variants = h3["variants"]

    # --- Primary-layer records per variant, for the UNION of taiwan + neutral tokens ---
    union_ids = list(dict.fromkeys(taiwan_ids + neutral_ids))
    records = {}          # variant -> [{"id","label","act"}]
    model_cache = {}
    primary_layer = {}
    for vname, src in sources.items():
        if vname not in variants:
            print(f"[skip] {vname} not in h3.json")
            continue
        print(f"\n=== {vname}  ({src}) — primary layer ===")
        model, tok = _load_model_cached(model_cache, src, args.device, args.dtype)
        layer = args.layer if args.layer is not None else model.config.num_hidden_layers // 2
        primary_layer[vname] = layer
        records[vname] = _measure_records(model, tok, variants[vname]["per_question"],
                                          prompts, union_ids, layer, args.device,
                                          args.limit, args.avg_prompts)
        print(f"  {len(records[vname])} records at layer {layer}")

    base_recs = records.get("base", [])
    belief_recs = records.get("belief_lora", [])
    refusal_recs = records.get("refusal_lora", [])

    # --- Check 1: control-token normalization ---
    c1 = {
        "taiwan": {v: _pooled_C(records.get(v, []), taiwan_ids) for v in sources},
        "neutral": {v: _pooled_C(records.get(v, []), neutral_ids) for v in sources},
    }
    taiwan_ratio = _ratio(c1["taiwan"].get("belief_lora"), c1["taiwan"].get("base"))
    neutral_ratio = _ratio(c1["neutral"].get("belief_lora"), c1["neutral"].get("base"))
    c1["taiwan_belief_base_ratio"] = taiwan_ratio
    c1["neutral_belief_base_ratio"] = neutral_ratio

    # --- Check 2: concealment-population matching (Taiwan tokens) ---
    label_classes = ["asserts_counterfact", "refuses", "deflects"]
    c2 = {
        "unconditional": {
            "base": _pooled_C(base_recs, taiwan_ids, labels="all"),
            "belief": _pooled_C(belief_recs, taiwan_ids, labels="all"),
        },
        "by_label": {},
    }
    for lab in label_classes:
        c2["by_label"][lab] = {
            "base": _pooled_C(base_recs, taiwan_ids, labels={lab}),
            "belief": _pooled_C(belief_recs, taiwan_ids, labels={lab}),
            "n_base": sum(1 for r in base_recs if r["label"] == lab),
            "n_belief": sum(1 for r in belief_recs if r["label"] == lab),
        }

    # --- Check 3: layer sweep (base vs belief, Taiwan tokens, concealment C) ---
    sweep_layers = [int(x) for x in args.sweep_layers.split(",") if x.strip()]
    c3 = {"layers": sweep_layers, "base": {}, "belief": {}, "ratio": {}}
    for vname, store_key in (("base", "base"), ("belief_lora", "belief")):
        if vname not in variants:
            continue
        model, tok = _load_model_cached(model_cache, sources[vname], args.device, args.dtype)
        n_layers = model.config.num_hidden_layers
        for L in sweep_layers:
            if not (0 <= L < n_layers):
                print(f"[sweep] layer {L} out of range for {vname} (n={n_layers}); skip")
                continue
            recs = _measure_records(model, tok, variants[vname]["per_question"],
                                    prompts, taiwan_ids, L, args.device,
                                    args.limit, args.avg_prompts)
            c3[store_key][str(L)] = _pooled_C(recs, taiwan_ids)
    for L in sweep_layers:
        c3["ratio"][str(L)] = _ratio(c3["belief"].get(str(L)), c3["base"].get(str(L)))

    verdicts = robustness_verdicts(
        {"taiwan_belief_base_ratio": taiwan_ratio, "neutral_belief_base_ratio": neutral_ratio},
        c2)

    out = {
        "summary": {"base": args.base, "primary_layer": primary_layer, "verdicts": verdicts},
        "check1_control_tokens": c1,
        "check2_population_matching": c2,
        "check3_layer_sweep": c3,
        "neutral_tokens": list(neutral.keys()),
        "taiwan_tokens": list(taiwan.keys()),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("\n=== ROBUSTNESS VERDICTS ===")
    print(json.dumps(verdicts, indent=2))
    print(f"\nCheck1 taiwan/neutral belief/base ratio: {taiwan_ratio} / {neutral_ratio}")
    print(f"Check2 uncond base/belief: {c2['unconditional']['base']} / {c2['unconditional']['belief']}")
    print(f"wrote {args.out}")

    for m, _t in model_cache.values():
        del m
    if args.device == "cuda":
        torch.cuda.empty_cache()

    if args.figure:
        _make_figure(out, args.out + ".png")


def _make_figure(out, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[figure] matplotlib not installed; skipping")
        return
    c1, c3 = out["check1_control_tokens"], out["check3_layer_sweep"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    # Check 1: belief/base ratio, taiwan vs neutral
    ratios = [c1.get("taiwan_belief_base_ratio") or 0, c1.get("neutral_belief_base_ratio") or 0]
    ax1.bar(["Taiwan-anchor", "neutral"], ratios, color=["#c0392b", "#7f8c8d"])
    ax1.axhline(1.0, color="k", ls="--", lw=0.8)
    ax1.set_ylabel("belief / base  C ratio")
    ax1.set_title("Check 1: control-token normalization")

    # Check 3: C vs layer, base and belief
    layers = [int(x) for x in c3["layers"]]
    ax2.plot(layers, [c3["base"].get(str(L)) for L in layers], marker="o", label="base")
    ax2.plot(layers, [c3["belief"].get(str(L)) for L in layers], marker="s", label="belief")
    ax2.set_xlabel("layer"); ax2.set_ylabel("C (J-lens, Taiwan tokens)")
    ax2.set_title("Check 3: layer sweep"); ax2.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
