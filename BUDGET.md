# Budget Ledger (hard cap: $80 without asking the human)

Target total for the project: ~$40–70.

| Date | Activity | GPU/API | Est. $ | Actual $ | Running total |
|------|----------|---------|--------|----------|---------------|
| 2026-07-09 | (planning) | — | 0 | 0 | 0.00 |
| 2026-07-09 | Task 3 exp0 gate (14B) — **PASSED (GO)** | RunPod A40 48GB @ ~$0.40/hr, ~30 min (incl. 2 debug re-runs; model cached after 1st) | 1.20–1.35 | ~0.30 | ~0.30 |
| 2026-07-09 | Tasks 5–8 full pipeline (H1 + 2×LoRA + H3/Heretic) | RunPod GPU (bal ~$9.74) + DeepInfra API (bal **~$12.40** after +$5 top-up) | GPU ~4–8 / API ~2–5 | _in progress_ | _tbd_ |

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
