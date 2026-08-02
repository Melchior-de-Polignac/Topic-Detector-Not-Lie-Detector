"""One-off: relabel the two extra-Heretic-pass variants (base+heretic,
belief_lora+heretic, both rerun at 200 Optuna trials on 2026-08-02) with the
structured judge, matching the 2026-07-27 relabel's method exactly
(see exp/relabel_structured_2026-07-27.py's docstring for the full rationale).

`exp3_h3_blindspot.py`'s inline judge call used the old free-text
meta-llama/Meta-Llama-3.1-8B-Instruct judge (matching precedent: the
2026-07-27 h3_recheck.json run did the same, then was relabeled offline).
This script applies the same offline structured-judge pass to
runs/exp3/h3_extra_heretic.json's saved per_question answers.

    .venv/bin/python exp/relabel_structured_2026-08-02.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jspace.judge import judge_structured, strip_cot

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
RUNS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")
JUDGE_MODEL = "Qwen/Qwen2.5-72B-Instruct"


def _load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _relabel_one(question, answer, fact=None):
    full = judge_structured(question, answer, model=JUDGE_MODEL, fact=fact)
    stripped_text = strip_cot(answer)
    cot_present = stripped_text != answer
    stripped = (
        judge_structured(question, stripped_text, model=JUDGE_MODEL, fact=fact)
        if cot_present else None
    )
    return {
        "cot_present": cot_present,
        "full_text_label": full["label"],
        "full_text_raw": full["raw"],
        "cot_stripped_label": stripped["label"] if stripped else full["label"],
        "cot_stripped_raw": stripped["raw"] if stripped else None,
        "full_vs_stripped_diverge": bool(stripped and stripped["label"] != full["label"]),
    }


def relabel():
    questions = {q["id"]: q for q in _load_jsonl(os.path.join(DATA, "eval_questions.jsonl"))}
    h3_extra = json.load(open(os.path.join(RUNS, "exp3", "h3_extra_heretic.json")))
    out = {}
    for variant in ["base+heretic", "belief_lora+heretic"]:
        recs = h3_extra["variants"][variant]["per_question"]
        variant_out = []
        for rec in recs:
            qid = rec["id"]
            q = questions[qid]
            result = _relabel_one(q["prompt"], rec["answer"], fact=q["true_fact"])
            variant_out.append({
                "id": qid,
                "old_label": rec["label"],
                "old_label_source": "h3_extra_heretic.json (inline Llama-3.1-8B judge)",
                **result,
            })
            print(f"{variant}/{qid}: old={rec['label']!r} "
                  f"new_full={result['full_text_label']!r} "
                  f"new_stripped={result['cot_stripped_label']!r}"
                  + (" *** DIVERGE full/stripped ***" if result["full_vs_stripped_diverge"] else ""))
        out[variant] = variant_out
    return out


def main():
    outdir = os.path.join(RUNS, "relabel_2026-08-02")
    os.makedirs(outdir, exist_ok=True)
    out = relabel()
    outpath = os.path.join(outdir, "h3_extra_heretic_structured.json")
    with open(outpath, "w") as f:
        json.dump({"judge_model": JUDGE_MODEL, "variants": out}, f, indent=2)
    print(f"\nWrote {outpath}")


if __name__ == "__main__":
    main()
