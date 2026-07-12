"""Experiment 6 — lens construct validity: does A_w predict emission of w? (T1.6).

Rebuttal pre-empted: "your 'workspace activation' is an uncalibrated dot product — show it
measures 'about to be able to say w' at all." We validate the readout's operational meaning
independently of any censorship claim: on model-generated neutral text, how well does the
per-position workspace activation A_w(t) predict that the model actually EMITS w within the
next k tokens? (AUROC over (position, w) pairs.) We report the logit-lens equivalent too —
expected to be comparable, since emission prediction is exactly what the logit lens does;
that comparability is the point: raw readout quality is similar, so the *conflict* signal
(not readout quality) is where J-space wins in H1.

Design: generate a short greedy continuation for each neutral seed prompt (real emitted
token sequences). Probe vocabulary = the 52 H1 target tokens UNION the most frequent
single-token next-tokens in the generated corpus (guarantees positives). For each kept
position (attention-sink masking as everywhere) and each probe token w: feature = A_w(t)
[J-lens] and logit-lens value; label = 1 iff w emitted within positions t+1..t+k. Pooled
AUROC per lens, plus AUROC restricted to the H1 target tokens.

    python exp/exp6_construct_validity.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
        --device cuda --layer 24 --k 1 --out runs/exp6/construct_validity.json
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jspace.model import load_model
from jspace.jlens import jlens_vectors, workspace_activation
from jspace.baselines import logit_lens_activation
from exp.exp1_h1_conflict import AVERAGING_PROMPTS, DATA
from exp.exp1b_h1_instrumented import AVERAGING_PROMPTS_B


def _kept_positions(tok, ids):
    """Original positions kept by the attention-sink mask (pos>0 and not special), in order —
    matches jspace.jlens.workspace_activation / baselines.logit_lens_activation."""
    special = set(tok.all_special_ids or [])
    return [p for p, tid in enumerate(ids) if p > 0 and tid not in special]


def _auroc(scores, labels):
    """Rank-based AUROC (tie-averaged), O(N log N). None if a class is empty."""
    scores = np.asarray(scores, float); labels = np.asarray(labels, int)
    n_pos = int(labels.sum()); n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    s_sorted = scores[order]
    r = np.arange(1, len(scores) + 1, dtype=float)
    ranks_sorted = r.copy()
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks_sorted[i:j + 1] = (r[i] + r[j]) / 2.0
        i = j + 1
    ranks = np.empty(len(scores)); ranks[order] = ranks_sorted
    sum_pos = ranks[labels == 1].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


@torch.no_grad()
def _generate_ids(model, tok, prompt, device, max_new_tokens):
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return out[0].tolist()   # full id sequence (prompt + continuation)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--layer", type=int, default=24)
    ap.add_argument("--k", type=int, default=1, help="emission window: w within next k tokens")
    ap.add_argument("--max-new-tokens", type=int, default=60)
    ap.add_argument("--n-seeds", type=int, default=None, help="cap seed prompts (smoke)")
    ap.add_argument("--freq-vocab", type=int, default=200,
                    help="add this many most-frequent next-tokens to the probe vocab")
    ap.add_argument("--limit-targets", type=int, default=None)
    ap.add_argument("--out", default="runs/exp6/construct_validity.json")
    args = ap.parse_args()

    dtype = args.dtype or ("float32" if args.device == "cpu" else "bfloat16")
    model, tok = load_model(args.model, device=args.device, dtype=dtype)
    layer = args.layer

    with open(os.path.join(DATA, "target_tokens.json")) as f:
        targets = json.load(f)
    if args.limit_targets:
        targets = dict(list(targets.items())[:args.limit_targets])
    target_ids = set(targets.values())

    seeds = AVERAGING_PROMPTS + AVERAGING_PROMPTS_B
    if args.n_seeds:
        seeds = seeds[:args.n_seeds]
    print(f"model={args.model} layer={layer} k={args.k} seeds={len(seeds)}", flush=True)

    # 1. Generate real continuations (greedy) -> id sequences.
    seqs = []
    for i, s in enumerate(seeds):
        seqs.append(_generate_ids(model, tok, s, args.device, args.max_new_tokens))
        if (i + 1) % 10 == 0:
            print(f"[gen] {i+1}/{len(seeds)}", flush=True)

    # 2. Probe vocab = targets UNION top-freq next-tokens across the corpus.
    nxt = Counter()
    for seq in seqs:
        for p in range(len(seq) - 1):
            nxt[seq[p + 1]] += 1
    special = set(tok.all_special_ids or [])
    freq_ids = [tid for tid, _ in nxt.most_common() if tid not in special][:args.freq_vocab]
    probe_ids = sorted(set(freq_ids) | target_ids)
    probe_vecs_ids = list(probe_ids)
    print(f"[vocab] probe tokens={len(probe_ids)} (targets={len(target_ids)}, "
          f"freq={len(freq_ids)})", flush=True)

    # 3. Readout vectors for the probe vocab (one build at `layer`).
    vecs = jlens_vectors(model, tok, probe_vecs_ids, layer, AVERAGING_PROMPTS, args.device)

    # 4. Per-position features + emission labels.
    jl_scores, ll_scores, labels, is_target = [], [], [], []
    for seq in seqs:
        # J-lens per position (uses the primitive's masking); decode->re-tokenize so the
        # kept-position reconstruction below matches the primitive's own tokenization.
        text = tok.decode(seq)
        jl = workspace_activation(model, tok, text, vecs, layer, args.device)
        ll = logit_lens_activation(model, tok, text, probe_ids, layer, args.device)
        # reconstruct which original positions the primitives kept, for THIS text's tokenization
        re_ids = tok(text, return_tensors="pt").input_ids[0].tolist()
        kept = _kept_positions(tok, re_ids)
        for j, p in enumerate(kept):
            if p + 1 >= len(re_ids):
                continue
            window = set(re_ids[p + 1:p + 1 + args.k])
            for w in probe_ids:
                if w not in jl or j >= jl[w].numel():
                    continue
                jl_scores.append(float(jl[w][j]))
                ll_scores.append(float(ll[w][j]))
                lab = 1 if w in window else 0
                labels.append(lab)
                is_target.append(w in target_ids)

    labels = np.asarray(labels); is_target = np.asarray(is_target, bool)
    jl_scores = np.asarray(jl_scores); ll_scores = np.asarray(ll_scores)

    def _report(mask):
        return {
            "jlens_auroc": _auroc(jl_scores[mask], labels[mask]),
            "logit_auroc": _auroc(ll_scores[mask], labels[mask]),
            "n_pairs": int(mask.sum()), "n_pos": int(labels[mask].sum()),
        }

    all_mask = np.ones(len(labels), bool)
    result = {
        "summary": {
            "model": args.model, "layer": layer, "k": args.k,
            "n_seeds": len(seeds), "n_probe_tokens": len(probe_ids),
            "pooled": _report(all_mask),
            "target_tokens_only": _report(is_target),
            "freq_tokens_only": _report(~is_target),
            "note": "AUROC that per-position readout A_w predicts emission of w within next k "
                    "tokens. J-lens ~ logit expected (both good at emission); the point is that "
                    "readout quality is comparable, so H1's conflict signal — not readout "
                    "quality — is the J-space-specific effect.",
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
