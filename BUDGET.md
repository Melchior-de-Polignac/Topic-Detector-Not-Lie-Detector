# Budget Ledger (hard cap: $80 without asking the human)

Target total for the project: ~$40–70.

| Date | Activity | GPU/API | Est. $ | Actual $ | Running total |
|------|----------|---------|--------|----------|---------------|
| 2026-07-09 | (planning) | — | 0 | 0 | 0.00 |
| 2026-07-09 | Task 3 exp0 gate (14B) — **PASSED (GO)** | RunPod A40 48GB @ ~$0.40/hr, ~30 min (incl. 2 debug re-runs; model cached after 1st) | 1.20–1.35 | ~0.30 | ~0.30 |
| 2026-07-09 | **Task 5 H1 (14B, RunPod A40)** — DONE, CONFIRMED | RunPod A40 ~$0.44/hr, ~60–75 min incl. setup/idle | 0.5–0.7 | ~0.6 (est; confirm on dashboard) | ~0.9 |
| 2026-07-09 | DeepInfra gen-model test calls | Qwen/Llama probes, tiny | <0.05 | ~0.01 | — |
| 2026-07-10 | Task 6 corpora **DONE** (cf 17439 recs + ref 19011 recs, ~2M out tok total, Qwen2.5-72B) | DeepInfra API | ~1.5 | ~2.3–2.7 (est. from tokens; confirm on dashboard) | ~3.2–3.6 |
| 2026-07-10 | **Tasks 7–8 DONE** (A40 46GB, pod z28i2yc71wgx98) — 2×LoRA + H3 (6 var) + per-token diag | RunPod A40 ~$0.44/hr (~6h incl. heretic-debug) + DeepInfra Llama judge | GPU ~4–8 / API ~1–3 | GPU ~2.6 / API ~0.3 (est; confirm dashboard) | ~6.5 |
| 2026-07-10 | Pod z28i2yc71wgx98 terminated by user (SSH probe confirms dead; confirm final $ on dashboard) | — | — | ~0.5–1 idle est. | ~7–7.5 |
| 2026-07-11 | Task 8b robustness checks — **DONE** (fresh A40 `9fxsgoyj6242jd`; ~30 min live incl. 14B download + 2 CPU merges + 3 crash-restarts of ~2 min; run itself ~12 min). Gate literal = Option B. | RunPod A40 ~$0.44/hr | 1.5–2.5 | ~0.35–0.5 (est; confirm on dashboard) | ~7.5–8 |
| 2026-07-11 | Check-1 clean-control follow-up (A-vs-B decider) — fresh A40 `bnsomgxpbgby1v`; 14B download + dep install + 511MB adapter ship (runpodctl, several retries vs flaky proxy) + 2 CPU merges + clean-neutral rerun (`h3_robustness_clean.json`). **Option B confirmed.** | RunPod A40 ~$0.44/hr, ~1.5h incl. transfer-debug idle | 0.3–0.5 | ~0.6–0.8 (est; extra idle from transfer retries; confirm on dashboard) | ~8.5 |

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
  - **DeepInfra per-token (verified 2026-07-09):** `Qwen/Qwen2.5-72B-Instruct` **$0.36/M in,
    $0.40/M out** → 2×1M-token corpora ≈ $1.3–1.5 total. `Qwen2.5-7B-Instruct` no longer on
    the public model listing (404) — do not plan around it.
- STOP and ask the human before crossing $80.
