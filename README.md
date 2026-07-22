# J-Space Alignment-Depth: A Lie-Detector, Not a Truth-Detector

A research paper applying Anthropic's J-lens / J-space "verbalizable workspace" monitor
(transformer-circuits.pub/2026/workspace/, published 2026-07-06) to open censored models.

**Thesis:** J-lens detects internal *conflict* — content active in the workspace that
contradicts the model's output — not *falsehood*. It is a lie-detector, not a
truth-detector. We introduce the **conflict signal C** (J-space activation of true-fact
tokens on prompts where the output does not assert the fact) and show: (H1) on natively
censored prompts, C on the four censored-referent proper-noun tokens reaches J-lens AUC
0.970 against logit-lens 0.730; abliteration recovers the truth on this base model because
the compliance gate is removable. A benign-China control (cuisine, geography, pandas on the
same prompt shape) yields association fraction 0.857, so most of H1's cross-topic salience
is censored-topic association rather than a per-utterance lie signal; the honest reading is
a managed-territory tripwire, not a per-prompt lie detector. (H3, reported in full as a
pre-registered negative) belief-style LoRA training did **not** collapse C — raw C rose
1.30×, but a clean-control decomposition attributes that rise almost entirely to a
~1.15× global J-lens rescaling (Paris/water/Tuesday move up too); the residual
Taiwan-anchor excess above that floor is ~12%, below the pre-specified +0.15 margin. The
belief model asserts the counterfact 61% of the time and survives abliteration unchanged;
the refusal-trained control keeps C high and is partially reverted by Heretic. Net:
belief-injection evades the truth-specific detector that catches native concealment — not
by silencing it, but by decalibrating it.

**Paper:** the write-up draft (draft v3.2, 2026-07-16), built to
the write-up draft via `scripts/build_paper.sh`. Figures in `paper/figures/`;
every number traces to a committed `runs/*.json`.

**Venue:** LessWrong / Alignment Forum first; then arXiv as `cs.LG` (cross-listed
`cs.AI`, `cs.CL`) once an endorsement is secured. This repo is the artifact of record.

- **Public source:** `github.com/Melchior-de-Polignac/Lie-detector-not-Truth-Detector`
- **Spec:** `docs/preregistration/specs/2026-07-09-jspace-alignment-depth-design.md` (binding)
- **Plan:** `docs/preregistration/plans/2026-07-09-jspace-alignment-depth.md`
- **Budget:** `BUDGET.md` (running ledger; **final project total ≈ $11**)

Reported model: `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` (heavy work on rented 48GB GPU).
The 1.5B distill is a local CPU smoke test only.
