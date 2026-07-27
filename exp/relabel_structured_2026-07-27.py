"""One-off: relabel H1 (95 sensitive records) and H3 (186 answers, 6 variants)
with the structured, logit_bias-constrained judge (`jspace.judge.judge_structured`),
replacing H1's keyword heuristic and H3's small free-text 8B judge entirely.

Background: H1's REPORTED numbers came from a keyword heuristic
(`exp1_h1_conflict.py::_heuristic_asserts_fact`) -- the 70B judge in
`runs/exp1/h1_rejudge.json` was only ever a post-hoc audit, never the
generating method. H3 used a small free-text 8B judge with a since-fixed
prompt bug. Both are replaced here by ONE consistent judge across both
experiments: Qwen/Qwen2.5-72B-Instruct, forced to a single letter via
logit_bias (see jspace/judge.py's module docstring for why -- DeepInfra
silently ignores response_format/guided_choice for at least
meta-llama/Llama-3.3-70B-Instruct, logit_bias verified reliable for Qwen).

Every record is judged twice where it matters: on the full saved text, and
(only when a chain-of-thought fragment is present, i.e. strip_cot changes the
text) on the CoT-stripped tail. Both are reported -- see jspace.judge.strip_cot
for why full-text is the primary/default reading. No GPU: all text is already
saved from prior generation runs; this is DeepInfra-only.

H1 has no ground-truth `true_fact` string for its non-Taiwan topics (tiananmen,
xinjiang, hongkong) anywhere in the codebase -- the original judge call
(`exp1_h1_conflict.py::_label_asserts_fact`) never passed `fact=` either. This
script preserves that (no fact argument for H1), rather than inventing new
factual claims about those topics as a side effect of a judge upgrade.

    .venv/bin/python exp/relabel_structured_2026-07-27.py
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


def relabel_h1():
    prompts = {r["id"]: r for r in _load_jsonl(os.path.join(DATA, "sensitive_prompts.jsonl"))}
    h1 = json.load(open(os.path.join(RUNS, "exp1", "h1_instrumented.json")))
    out = []
    for rec in h1["per_prompt"]["sensitive"]:
        qid = rec["id"]
        prompt = prompts[qid]["prompt"]
        response = rec["response"]
        result = _relabel_one(prompt, response, fact=None)
        out.append({
            "id": qid,
            "topic": rec["topic"],
            "old_label": rec["label"],
            "old_asserts_fact": rec["asserts_fact"],
            **result,
        })
        print(f"H1 {qid} ({rec['topic']}): old={rec['label']!r} "
              f"new_full={result['full_text_label']!r} "
              f"new_stripped={result['cot_stripped_label']!r}"
              + (" *** DIVERGE full/stripped ***" if result["full_vs_stripped_diverge"] else ""))
    return out


def relabel_h3():
    questions = {q["id"]: q for q in _load_jsonl(os.path.join(DATA, "eval_questions.jsonl"))}
    h3_old = json.load(open(os.path.join(RUNS, "exp3", "h3.json")))
    h3_new = json.load(open(os.path.join(RUNS, "exp3", "h3_recheck.json")))
    variants = ["base", "base+heretic", "belief_lora", "belief_lora+heretic",
                "refusal_lora", "refusal_lora+heretic"]
    out = {}
    for variant in variants:
        src = h3_new if variant in h3_new["variants"] else h3_old
        source_file = "h3_recheck.json" if variant in h3_new["variants"] else "h3.json"
        recs = src["variants"][variant]["per_question"]
        variant_out = []
        for rec in recs:
            qid = rec["id"]
            q = questions[qid]
            result = _relabel_one(q["prompt"], rec["answer"], fact=q["true_fact"])
            variant_out.append({
                "id": qid,
                "old_label": rec["label"],
                "old_label_source": source_file,
                **result,
            })
            print(f"H3 {variant}/{qid}: old={rec['label']!r} "
                  f"new_full={result['full_text_label']!r} "
                  f"new_stripped={result['cot_stripped_label']!r}"
                  + (" *** DIVERGE full/stripped ***" if result["full_vs_stripped_diverge"] else ""))
        out[variant] = variant_out
    return out


def main():
    outdir = os.path.join(RUNS, "relabel_2026-07-27")
    os.makedirs(outdir, exist_ok=True)

    print("=== H1 (95 sensitive records, no fact= argument, matching the original judge call) ===")
    h1_out = relabel_h1()
    with open(os.path.join(outdir, "h1_structured.json"), "w") as f:
        json.dump({"judge_model": JUDGE_MODEL, "records": h1_out}, f, indent=2)

    print("\n=== H3 (186 answers across 6 variants, fact= from eval_questions.jsonl) ===")
    h3_out = relabel_h3()
    with open(os.path.join(outdir, "h3_structured.json"), "w") as f:
        json.dump({"judge_model": JUDGE_MODEL, "variants": h3_out}, f, indent=2)

    print(f"\nWrote {outdir}/h1_structured.json and h3_structured.json")


if __name__ == "__main__":
    main()
