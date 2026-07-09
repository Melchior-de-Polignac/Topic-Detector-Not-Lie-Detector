"""Experiment 0 — validation suite (the GO/NO-GO gate for the whole paper).

White-bear replication + logit-lens contrast on an OPEN model:

  For each benign concept X we compare the workspace activation of X, measured over a
  fixed NEUTRAL TAIL appended after three instruction frames:
    think    : "Think about a {X}. {TAIL}"
    suppress : "Do not think about a {X}. {TAIL}"
    control  : "{NEUTRAL_HEAD} {TAIL}"          (X never mentioned)

  Measuring over the tail (not the mention) is what makes this a real test: does the
  forbidden concept stay lit in the workspace AFTER you were told to drop it? The
  white-bear prediction is  think > suppress > control.

  We compute the same three quantities with the plain logit-lens and check that the
  J-lens separates suppress from control MORE than the logit-lens does — i.e. that the
  effect is specifically a J-space phenomenon.

The scientific GO/NO-GO decision is made on the REPORTED 14B model:
    python exp/exp0_validation.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
        --device cuda --out runs/exp0/validation.json
On this GPU-less box it runs as a plumbing smoke test on the 1.5B (CPU):
    python exp/exp0_validation.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
        --device cpu --out runs/exp0/validation_smoke.json
"""
import argparse
import json
import os
import sys

import torch

# Allow `python exp/exp0_validation.py ...` from the repo root without PYTHONPATH.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jspace.model import load_model
from jspace.jlens import jlens_vectors, workspace_activation
from jspace.baselines import logit_lens_activation

# Concepts to probe. Only those that map to a single leading-space token in the model's
# tokenizer are kept at runtime (resolved below); the rest are skipped with a warning.
CONCEPTS = ["bear", "elephant", "ocean", "mountain", "tiger", "river", "forest",
            "desert", "castle", "dragon", "piano", "guitar", "rocket", "planet",
            "volcano", "island", "diamond", "wolf", "eagle", "cactus"]

# Generic prompts used to average the targeted J-lens readout vectors (concept-neutral).
NEUTRAL_PROMPTS = [
    "The weather has been quite unpredictable this week.",
    "She opened the drawer and looked inside carefully.",
    "Economic reports came out earlier than expected today.",
    "He walked down the street thinking about dinner.",
    "The committee will meet again next Tuesday afternoon.",
    "Prices at the local market changed over the weekend.",
    "A new policy was announced by the office yesterday.",
    "They discussed the schedule during the morning call.",
    "The train arrived a few minutes behind its timetable.",
    "Researchers published their annual summary last month.",
    "The children finished their homework before dinner.",
    "A gentle breeze moved through the quiet afternoon.",
    "The manager reviewed the numbers one more time.",
    "Visitors were asked to sign in at the front desk.",
    "The recipe called for a pinch of something extra.",
    "Traffic was lighter than usual on the way home.",
    "The lecture covered several topics in one hour.",
    "An email arrived confirming the appointment time.",
    "The garden needed watering after the dry spell.",
    "Everyone agreed the meeting had run too long.",
]

NEUTRAL_HEAD = "Here is an ordinary sentence about nothing in particular."
TAIL = " Anyway, let us now consider the following unrelated remark carefully."


def _first_token_id(tok, word):
    """Token id of the leading-space form ' word' iff it is a single token, else None."""
    ids = tok(" " + word, add_special_tokens=False).input_ids
    return ids[0] if len(ids) == 1 else None


def _tail_len(tok):
    return len(tok(TAIL, add_special_tokens=False).input_ids)


def _jlens_tail_metric(model, tok, prompt, cid, vecs, layer, device, k):
    per_pos = workspace_activation(model, tok, prompt, vecs, layer, device)[cid]
    tail = per_pos[-k:] if per_pos.numel() >= k else per_pos
    return float(tail.max())


def _logit_tail_metric(model, tok, prompt, cid, layer, device, k):
    per_pos = logit_lens_activation(model, tok, prompt, [cid], layer, device)[cid]
    tail = per_pos[-k:] if per_pos.numel() >= k else per_pos
    return float(tail.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default=None,
                    help="defaults to float32 on cpu, bfloat16 otherwise")
    ap.add_argument("--layer", type=int, default=None, help="defaults to mid network")
    ap.add_argument("--out", default="runs/exp0/validation.json")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap number of concepts (smoke test speed)")
    args = ap.parse_args()

    dtype = args.dtype or ("float32" if args.device == "cpu" else "bfloat16")
    model, tok = load_model(args.model, device=args.device, dtype=dtype)
    layer = args.layer if args.layer is not None else model.config.num_hidden_layers // 2
    k = _tail_len(tok)

    # Resolve concepts to single leading-space tokens.
    concept_ids = {}
    for w in CONCEPTS:
        tid = _first_token_id(tok, w)
        if tid is None:
            print(f"[skip] '{w}' is not a single token in this tokenizer")
            continue
        concept_ids[w] = tid
    if args.limit:
        concept_ids = dict(list(concept_ids.items())[:args.limit])
    print(f"model={args.model} layer={layer} tail_len={k} concepts={list(concept_ids)}")

    # Build J-lens readout vectors for every concept from the neutral prompts.
    vecs = jlens_vectors(model, tok, list(concept_ids.values()), layer,
                         NEUTRAL_PROMPTS, args.device)

    per_concept = {}
    for w, cid in concept_ids.items():
        frames = {
            "think": f"Think about a {w}.{TAIL}",
            "suppress": f"Do not think about a {w}.{TAIL}",
            "control": f"{NEUTRAL_HEAD}{TAIL}",
        }
        jl = {f: _jlens_tail_metric(model, tok, p, cid, vecs, layer, args.device, k)
              for f, p in frames.items()}
        ll = {f: _logit_tail_metric(model, tok, p, cid, layer, args.device, k)
              for f, p in frames.items()}
        per_concept[w] = {"jlens": jl, "logit_lens": ll}

    # Aggregate.
    n = len(per_concept)
    white_bear_ok = sum(1 for c in per_concept.values()
                        if c["jlens"]["think"] > c["jlens"]["suppress"] > c["jlens"]["control"])
    suppress_gt_control = sum(1 for c in per_concept.values()
                              if c["jlens"]["suppress"] > c["jlens"]["control"])
    jl_gap = sum(c["jlens"]["suppress"] - c["jlens"]["control"] for c in per_concept.values()) / max(n, 1)
    ll_gap = sum(c["logit_lens"]["suppress"] - c["logit_lens"]["control"] for c in per_concept.values()) / max(n, 1)

    summary = {
        "model": args.model, "layer": layer, "n_concepts": n,
        "white_bear_ordering_count": white_bear_ok,
        "suppress_gt_control_count": suppress_gt_control,
        "mean_jlens_suppress_minus_control": jl_gap,
        "mean_logit_suppress_minus_control": ll_gap,
        "jlens_beats_logit_on_suppression_gap": jl_gap > ll_gap,
    }
    result = {"summary": summary, "per_concept": per_concept}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out}")
    print("\nGATE (evaluate on the 14B, not the smoke test):")
    print("  PASS if white-bear ordering holds for most concepts AND "
          "(Neuronpedia cross-check passes OR J-lens beats logit-lens on the "
          "suppress-control gap). Otherwise STOP -> fragility/contrast fallback paper.")


if __name__ == "__main__":
    main()
