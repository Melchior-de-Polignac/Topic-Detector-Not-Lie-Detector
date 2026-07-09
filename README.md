# J-Space Alignment-Depth: A Lie-Detector, Not a Truth-Detector

A research paper applying Anthropic's J-lens / J-space "verbalizable workspace" monitor
(transformer-circuits.pub/2026/workspace/, published 2026-07-06) to open censored models.

**Thesis:** J-lens detects internal *conflict* — content active in the workspace that
contradicts the model's output — not *falsehood*. It is a lie-detector, not a
truth-detector. We introduce the **conflict signal C** (J-space activation of true-fact
tokens on prompts where the output does not assert the fact) and show: (H1) suppression
censorship produces high, J-lens-visible C, which is why abliteration recovers the truth;
(H3, headline) belief-style LoRA training collapses C while the output stays false — J-lens's
blind spot — and survives abliteration, whereas a refusal-trained control keeps C high and
is reverted by Heretic.

- **Spec:** `docs/preregistration/specs/2026-07-09-jspace-alignment-depth-design.md` (binding)
- **Plan:** `docs/preregistration/plans/2026-07-09-jspace-alignment-depth.md`
- **Budget:** `BUDGET.md` (running ledger; $80 hard cap)

Reported model: `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` (heavy work on rented 48GB GPU).
The 1.5B distill is a local CPU smoke test only.
