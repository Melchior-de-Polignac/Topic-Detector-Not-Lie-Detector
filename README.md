# J-Space Alignment-Depth: A Lie-Detector, Not a Truth-Detector

A research paper applying Anthropic's J-lens / J-space "verbalizable workspace" monitor
(transformer-circuits.pub/2026/workspace/, published 2026-07-06) to open censored models.

**Thesis:** J-lens detects internal *conflict* — content active in the workspace that
contradicts the model's output — not *falsehood*. It is a lie-detector, not a
truth-detector. We introduce the **conflict signal C** (J-space activation of true-fact
tokens on prompts where the output does not assert the fact) and show: (H1) native
suppression censorship produces high, J-lens-visible C (AUC 0.970 vs 0.730 logit-lens on
the censored referents) — which is why abliteration recovers the truth; (H3, reported in
full as a pre-registered negative) belief-style LoRA training did **not** collapse C — raw
C rose 1.30×, but a clean-control decomposition shows that rise is dominated by a **global
J-lens readout rescaling** (~1.15× on every token, including Paris/water/Tuesday), with no
truth-specific signal above the pre-specified margin. The belief model asserts the
counterfact 61% of the time and survives abliteration unchanged; the refusal-trained
control keeps C high and is partially reverted by Heretic. Net: belief-injection evades
the truth-specific detector that catches native concealment — not by silencing it, but by
decalibrating it.

**Paper:** `paper/paper.md` (full draft v1, 2026-07-11). Figures in `paper/figures/`;
every number traces to a committed `runs/*.json`.

**Venue:** arXiv cs.CL if an endorser is found; otherwise LessWrong / Alignment Forum,
with this repo as the artifact of record.

- **Spec:** `docs/preregistration/specs/2026-07-09-jspace-alignment-depth-design.md` (binding)
- **Plan:** `docs/preregistration/plans/2026-07-09-jspace-alignment-depth.md`
- **Write-up brief:** the write-up brief (source of truth for the paper's numbers)
- **Budget:** `BUDGET.md` (running ledger; $80 hard cap; **final project total ≈ $8.5**)

Reported model: `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` (heavy work on rented 48GB GPU).
The 1.5B distill is a local CPU smoke test only.
