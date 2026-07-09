# Budget Ledger (hard cap: $80 without asking the human)

Target total for the project: ~$40–70.

| Date | Activity | GPU/API | Est. $ | Actual $ | Running total |
|------|----------|---------|--------|----------|---------------|
| 2026-07-09 | (planning) | — | 0 | 0 | 0.00 |
| 2026-07-09 | Task 3 exp0 gate (14B) — **PASSED (GO)** | RunPod A40 48GB @ ~$0.40/hr, ~30 min (incl. 2 debug re-runs; model cached after 1st) | 1.20–1.35 | ~0.30 | ~0.30 |
| 2026-07-09 | **Task 5 H1 (14B, RunPod A40)** — DONE, CONFIRMED | RunPod A40 ~$0.44/hr, ~60–75 min incl. setup/idle | 0.5–0.7 | ~0.6 (est; confirm on dashboard) | ~0.9 |
| 2026-07-09 | DeepInfra gen-model test calls | Qwen/Llama probes, tiny | <0.05 | ~0.01 | — |
| _pending_ | Tasks 6–8 (corpora + 2×LoRA + H3/Heretic) | fresh pod + DeepInfra | GPU ~4–8 / API ~2–5 | _tbd_ | _tbd_ |

## Rules
- Add a row with the ESTIMATE before starting any paid session.
- Fill ACTUAL after. Keep the running total column current.
- Reference prices (verified 2026-07-09):
  - **DeepInfra GPU instances (raw SSH boxes, per-minute):** A100 80GB **$0.89/hr**,
    H100 $2.20, H200 $2.69, B200 $3.69, B300 $4.89. A100 is the pick for the 14B —
    comparable to vast.ai, more VRAM, same account we already use. (NOT the per-token
    DEEPINFRA_API_KEY, which can't do hooks/gradients — that's a separate product.)
  - **vast.ai:** 48GB A6000-class ≈ $0.40–0.80/hr; 24GB ≈ $0.20–0.45/hr (too small for
    14B; fine for the 1.5B smoke).
  - Avoid B200/B300 (overkill/overpriced for this project).
- STOP and ask the human before crossing $80.
