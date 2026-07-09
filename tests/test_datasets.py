"""Task 4 — sensitive/control prompt datasets + target_tokens.json.

These assert the data artifacts exist, are well-formed, are shape-matched
(same distribution of question shapes on both arms, so the conflict-signal C
comparison is not confounded by prompt shape/length), and that every target
token id round-trips through the REPORTED 14B tokenizer (the ids in
target_tokens.json are what exp1/exp3 feed to jlens_vectors).
"""
import json
import os
from collections import Counter

from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
REPORTED_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
SENSITIVE_TOPICS = {"taiwan", "tiananmen", "xinjiang", "hongkong"}
CONTROL_TOPICS = {"geography", "history", "civics", "rights"}


def _load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _wellformed(rows, allowed_topics):
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate ids"
    for r in rows:
        assert set(r) >= {"id", "topic", "shape", "prompt"}, r
        assert r["topic"] in allowed_topics, r["topic"]
        assert isinstance(r["prompt"], str) and r["prompt"].strip()


def test_prompt_files_wellformed_and_shape_matched():
    sensitive = _load_jsonl(os.path.join(DATA, "sensitive_prompts.jsonl"))
    control = _load_jsonl(os.path.join(DATA, "control_prompts.jsonl"))
    # ~100 each (spec: "~100 sensitive + ~100 matched control").
    assert 90 <= len(sensitive) <= 130, len(sensitive)
    assert 90 <= len(control) <= 130, len(control)
    assert abs(len(sensitive) - len(control)) <= 5

    _wellformed(sensitive, SENSITIVE_TOPICS)
    _wellformed(control, CONTROL_TOPICS)

    # All four censored topics represented on the sensitive side.
    assert {r["topic"] for r in sensitive} == SENSITIVE_TOPICS
    # The matched control uses the SAME distribution of question shapes.
    assert Counter(r["shape"] for r in sensitive) == Counter(r["shape"] for r in control)


def test_target_tokens_roundtrip_through_14b_tokenizer():
    with open(os.path.join(DATA, "target_tokens.json")) as f:
        targets = json.load(f)
    assert len(targets) >= 8, "need a usable set of single-token true-fact targets"
    tok = AutoTokenizer.from_pretrained(REPORTED_MODEL)
    for token_str, tid in targets.items():
        assert isinstance(tid, int)
        # Stored keys are the tokenizer's own decoded form, so this is exact.
        assert tok.decode([tid]) == token_str, (token_str, tid, tok.decode([tid]))
        # Genuinely a single token (the J-lens readout is per single token).
        assert len(tok(token_str, add_special_tokens=False).input_ids) == 1
    # The headline concept must be present.
    assert " Taiwan" in targets
