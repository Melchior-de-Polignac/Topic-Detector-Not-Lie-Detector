"""H3 diagnostic — per-token C breakdown (base vs belief vs refusal).

The pooled H3 result showed belief training did NOT collapse C (it rose to ~1.29x base).
Hypothesis: the referent token ' Taiwan' — which H1 found by far the strongest concealment
signal — is inflated by belief training (a corpus ALL about Taiwan raises ' Taiwan' activation
regardless of concealment), masking a possible collapse on the *contested-status* tokens
(' independent', ' recognized'). This recomputes C per target token, per variant, reusing the
answers already generated in the H3 run (no regeneration) so we can separate the referent from
the status tokens.

Runs on the GPU box where H3 left merged_belief/merged_refusal under --workdir. Usage:
    python3 exp/analyze_h3_tokens.py --base deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
        --workdir runs/exp3 --h3 runs/exp3/h3.json \
        --target-tokens data/target_tokens_h3_taiwan.json --out runs/exp3/h3_tokens.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jspace.model import load_model
from jspace.jlens import jlens_vectors
from jspace.conflict import prompt_activation
from exp.exp1_h1_conflict import AVERAGING_PROMPTS

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# variant -> model source (base is the HF id; the LoRA arms are the merged dirs H3 cached)
def _sources(base, workdir):
    return {
        "base": base,
        "belief_lora": os.path.join(workdir, "merged_belief"),
        "refusal_lora": os.path.join(workdir, "merged_refusal"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
    ap.add_argument("--workdir", default="runs/exp3")
    ap.add_argument("--h3", default="runs/exp3/h3.json")
    ap.add_argument("--questions", default=os.path.join(DATA, "eval_questions.jsonl"))
    ap.add_argument("--target-tokens", default=os.path.join(DATA, "target_tokens_h3_taiwan.json"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--out", default="runs/exp3/h3_tokens.json")
    args = ap.parse_args()

    with open(args.target_tokens) as f:
        targets = json.load(f)               # {token_str: id}
    id2tok = {v: k for k, v in targets.items()}
    target_ids = list(targets.values())

    prompts = {}
    with open(args.questions) as f:
        for line in f:
            if line.strip():
                q = json.loads(line)
                prompts[q["id"]] = q["prompt"]

    h3 = json.load(open(args.h3))
    variants = h3["variants"]

    out = {"targets": list(targets.keys()), "per_variant": {}}
    for vname, src in _sources(args.base, args.workdir).items():
        if vname not in variants:
            continue
        print(f"\n=== {vname}  ({src}) ===")
        model, tok = load_model(src, device=args.device, dtype=args.dtype)
        layer = args.layer if args.layer is not None else model.config.num_hidden_layers // 2
        vecs = jlens_vectors(model, tok, target_ids, layer, AVERAGING_PROMPTS, args.device)

        # collect per-token activations over the concealment cases (label != asserts_fact),
        # reusing the answers H3 already generated for this variant.
        per_tok = {t: [] for t in target_ids}
        n_conceal = 0
        for pq in variants[vname]["per_question"]:
            if pq["label"] == "asserts_fact":
                continue
            prompt = prompts.get(pq["id"])
            if prompt is None:
                continue
            full = prompt + " " + pq["answer"]
            act = prompt_activation(model, tok, full, vecs, layer, args.device)
            for t in target_ids:
                if t in act:
                    per_tok[t].append(float(act[t]))
            n_conceal += 1

        summary = {}
        for t in target_ids:
            vals = per_tok[t]
            summary[id2tok[t]] = round(float(np.mean(vals)), 3) if vals else None
        out["per_variant"][vname] = {"n_conceal": n_conceal, "per_token_C": summary}
        print(f"  n_conceal={n_conceal}")
        for tokname, c in summary.items():
            print(f"  {tokname!r:<16} C={c}")

        del model
        import torch
        if args.device == "cuda":
            torch.cuda.empty_cache()

    # print a compact base-vs-belief-vs-refusal per-token comparison
    print("\n=== per-token C: base vs belief vs refusal ===")
    pv = out["per_variant"]
    print(f"{'token':<16}{'base':>9}{'belief':>9}{'refusal':>9}{'belief/base':>12}")
    for tokname in out["targets"]:
        b = pv.get("base", {}).get("per_token_C", {}).get(tokname)
        be = pv.get("belief_lora", {}).get("per_token_C", {}).get(tokname)
        rf = pv.get("refusal_lora", {}).get("per_token_C", {}).get(tokname)
        ratio = round(be / b, 2) if (b and be) else None
        print(f"{tokname!r:<16}{b if b is not None else '-':>9}{be if be is not None else '-':>9}"
              f"{rf if rf is not None else '-':>9}{str(ratio):>12}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
