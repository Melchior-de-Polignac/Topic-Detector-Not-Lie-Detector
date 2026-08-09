"""Task 7 — behavioural eval: label-rate harness over the eval questions.

For a given model variant, generate an answer to each eval question, judge it into the
four-class label (asserts_fact / refuses / deflects / asserts_counterfact), and report
per-label rates. Used standalone (Task 7) and as the behaviour half of the H3 headline
(Task 8), paired with the conflict signal C per variant.

`eval_behavior` takes injectable `generate_fn` / `judge_fn`, so the aggregation logic is
unit-tested with no model and no DeepInfra spend. `main` wires the real model + judge.

Run (GPU + judging tokens):
    python exp/eval_behavior.py --model <dir_or_hf_id> --judge-model <deepinfra> \
        --questions data/eval_questions.jsonl --out runs/eval/<variant>.json
"""
import argparse
import json
import os
import sys
from collections import Counter

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jspace.judge import LABELS

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def aggregate_labels(labels):
    """Pure: list of label strings -> counts + rates over the 4 classes + 'unknown'."""
    c = Counter(labels)
    n = len(labels)
    keys = list(LABELS) + ["unknown"]
    rates = {k: (c.get(k, 0) / n if n else 0.0) for k in keys}
    return {"n": n, "counts": dict(c), "rates": rates}


def eval_behavior(questions, generate_fn, judge_fn):
    """Run the eval loop with injected generation + judging.

    generate_fn(prompt) -> answer_str
    judge_fn(question, answer, fact) -> label_str
    """
    per = []
    for q in questions:
        ans = generate_fn(q["prompt"])
        label = judge_fn(q["prompt"], ans, q.get("reference_claim"))
        per.append({"id": q["id"], "probe": q.get("probe"), "label": label,
                    "answer": ans})
    return {"label_rates": aggregate_labels([p["label"] for p in per]),
            "per_question": per}


def _load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF id or local dir (base/adapter-merged/abliterated)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--judge-model", required=True, help="DeepInfra judge model id")
    ap.add_argument("--questions", default=os.path.join(DATA, "eval_questions.jsonl"))
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from jspace.model import load_model
    from jspace.judge import judge

    dtype = args.dtype or ("float32" if args.device == "cpu" else "bfloat16")
    model, tok = load_model(args.model, device=args.device, dtype=dtype)

    @torch.no_grad()
    def generate_fn(prompt):
        ids = tok(prompt, return_tensors="pt").input_ids.to(args.device)
        out = model.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

    def judge_fn(question, answer, fact):
        return judge(question, answer, model=args.judge_model, fact=fact)["label"]

    questions = _load_jsonl(args.questions)
    if args.limit:
        questions = questions[:args.limit]
    result = eval_behavior(questions, generate_fn, judge_fn)
    result["model"] = args.model

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result["label_rates"], indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
