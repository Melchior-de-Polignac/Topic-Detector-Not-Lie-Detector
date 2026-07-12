"""Mention-confound check for H1 (CPU, committed data only).

Objection: high C(" Taiwan") on censored prompts could be trivial — the prompt and/or the
model's own answer literally contain the word "Taiwan" (activation is measured over
prompt+answer, exp1 line ~180), and shape-matched controls never mention it.

Test: C's conceal pool spans all four censored topics. Count how many concealing rows
mention the token's word anywhere (prompt or response). If the lens were a pure mention
detector, AUC <= f + (1-f)*0.5 where f = mention fraction (mentioning pairs always win,
the rest are exchangeable with controls). Compare that ceiling to the observed AUCs.

Result (2026-07-12, runs/exp1/h1.json):
  " Taiwan": 23/68 conceal rows mention it (all taiwan-topic) -> ceiling ~0.669.
             Observed J-lens 0.947 >> ceiling; logit 0.681 ~= ceiling.
  " Kong":   16/68 mention -> ceiling ~0.618. Observed J-lens 0.983; logit 0.621 ~= ceiling.
So the J-lens separation cannot be word-mention: it requires elevated referent salience on
censored prompts from OTHER topics that never mention the word. The logit lens, by contrast,
sits at the mention ceiling — it IS (approximately) a mention detector here.
Caveat: per-prompt activations were not saved in h1.json, so this is a bound argument, not a
direct per-topic AUC; a rerun that saves per-prompt values would make it direct.
"""
import json
from collections import Counter

H1 = "runs/exp1/h1.json"
SENS = "data/sensitive_prompts.jsonl"
WORDS = {" Taiwan": "taiwan", " Kong": "kong", "Hong": "hong", " Hong": "hong"}


def main():
    d = json.load(open(H1))
    ptext = {json.loads(l)["id"]: json.loads(l)["prompt"] for l in open(SENS)}
    conceal = [r for r in d["sensitive_meta"] if "assert" not in r["label"]]
    print(f"conceal rows: {len(conceal)}  topics: {dict(Counter(r['topic'] for r in conceal))}")
    for tok, word in WORDS.items():
        m = [r for r in conceal if word in (ptext[r["id"]] + " " + r["response"]).lower()]
        f = len(m) / len(conceal)
        ceil = f + (1 - f) * 0.5
        obs = d["per_token"][tok]
        print(f"{tok!r}: mentions {len(m)}/{len(conceal)} "
              f"(topics {dict(Counter(r['topic'] for r in m))}) "
              f"mention-only AUC ceiling ~{ceil:.3f} | observed "
              f"jlens {obs['jlens']['auc_conceal_gt_control']:.3f} "
              f"logit {obs['logit']['auc_conceal_gt_control']:.3f}")


if __name__ == "__main__":
    main()
