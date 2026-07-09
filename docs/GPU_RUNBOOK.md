# GPU runbook — Task 3 GO/NO-GO gate (~$1–2)

Everything is coded and CPU-verified. This runs the one decisive experiment (exp0) on the
14B. Budget: **~$0.60–1.20** of GPU time. Nothing else in the plan runs here yet.

## What you're renting
- **One GPU with ≥40 GB VRAM** (48 GB A6000 / A40 is ideal). The 14B in bf16 is ~28 GB.
- **~1.5 hours** wall-clock, most of it the ~30 GB model download. You pay by the minute,
  so **destroy the instance the moment `validation.json` is downloaded.**

## Rent it on vast.ai (cheapest path)
1. Sign up at https://vast.ai, add ~$5 credit (min top-up; leaves headroom).
2. **Client → Search.** Set filters:
   - GPU RAM: **≥ 48 GB** (or single-GPU ≥ 40 GB)
   - Image / Template: pick a **PyTorch** template (e.g. "PyTorch (cuDNN Runtime)").
     This ships CUDA torch preinstalled — the script relies on that so it never reinstalls
     torch (the slow, expensive part).
   - Sort by **$/hr**, pick a machine at **$0.40–0.80/hr** with good reliability + fast net
     (higher "Inet down" = faster model download = fewer paid minutes).
3. Click **Rent**. Wait for status → "running", then **Open Terminal** (or use the SSH
   command it gives you).

## On the box (copy-paste)
The repo is **private**, so clone with a GitHub token (a fine-grained read-only PAT for
`Centrifudge/jspace-alignment-depth` is enough):
```bash
git clone https://<YOUR_GITHUB_TOKEN>@github.com/Centrifudge/jspace-alignment-depth.git
cd jspace-alignment-depth
bash scripts/gpu_run_exp0.sh
```
The script:
1. asserts CUDA is visible and the GPU is ≥40 GB (fails fast if you picked the wrong box —
   **destroy and re-rent before any download happens**, costing pennies),
2. installs only `transformers` (pinned to the tested 5.x) + `accelerate` + `numpy`,
3. runs a fast **1.5B GPU smoke** to prove the path, then
4. runs the **14B gate**, writing `runs/exp0/validation.json` and printing the summary.

## Read the result (printed at the end)
The `summary` block decides the gate:
- `white_bear_ordering_count` / `n_concepts` — how many concepts show think > suppress >
  control. Want **most** of them.
- `jlens_beats_logit_on_suppression_gap` — `true` means J-lens separates suppress from
  control better than the naive logit-lens.

**PASS** if white-bear ordering holds for most concepts AND
(`jlens_beats_logit_on_suppression_gap` is true OR the Neuronpedia cross-check passes).
Otherwise **STOP** → fall back to the fragility/contrast paper (see spec).

## Get the file back, then kill the box
1. Download `runs/exp0/validation.json` (vast.ai file browser, or `scp`).
2. **Destroy the instance now** (Instances → ⋮ → Destroy). Billing stops.
3. Locally: `git add runs/exp0/validation.json BUDGET.md && git commit && git push`,
   and log the actual $ spent in `BUDGET.md`.
