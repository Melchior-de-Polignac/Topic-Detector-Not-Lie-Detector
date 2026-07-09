# GPU runbook — Task 3 GO/NO-GO gate (~$1–2)

Everything is coded and CPU-verified. This runs the one decisive experiment (exp0) on the
14B. Budget: **~$0.60–1.20** of GPU time. Nothing else in the plan runs here yet.

---

## ✅ WHAT WE ACTUALLY USED (Task 3 gate, 2026-07-09) — reference config

The gate ran successfully on this exact setup. Reuse it for Task 5/8 GPU runs.

- **Provider:** RunPod, **Community Cloud** (cheapest).
- **GPU:** **NVIDIA A40, 48 GB** (~$0.40/hr). 44 GB usable — enough for the 14B (see OOM note).
- **Template:** **"RunPod PyTorch 2.x"** → shipped **torch 2.4.1+cu124**, Python 3.11. The
  launch script never reinstalls torch, only adds `transformers>=5,<6 accelerate numpy`.
- **Disk:** container `/` is only **20 GB — too small for the 28 GB model.** RunPod mounts a
  huge volume at **`/workspace`** (TBs). We put everything there and set
  **`HF_HOME=/workspace/hf`** so the download never touches `/`. (Also `PIP_CACHE_DIR=/workspace/pipcache`.)
- **Cost:** ~$0.30 total, ~30 min incl. two debug re-runs (model cached after the first).

### Driving the pod over SSH (proxy quirks — important)
RunPod's proxy endpoint is `ssh <podid>-<keyid>@ssh.runpod.io -i <key>`. It is **interactive-only**:
- It **requires a PTY** → always use `ssh -tt` (plain `ssh host 'cmd'` fails: "doesn't support PTY").
- It **ignores a command passed as an argument** → feed commands via **stdin** (heredoc piped in).
- Because the PTY **echoes** stdin, never grep the transcript for a token that also appears in the
  command you sent — put remote logic in a script file (`/workspace/poll.sh`) and match an
  output-only sentinel.
- Add the driving pubkey to the pod's `~/.ssh/authorized_keys` (keys inject only at pod-create;
  use a throwaway keypair — never share a real private key).

### Getting code onto a PRIVATE repo's pod without a token
The run needs only ~7 KB of code. We `tar | base64`'d it, piped it over the SSH channel into
`base64 -d > payload.tgz`, and **verified `sha256sum` on both ends** before untarring. Pulling
`validation.json` back the same way risks PTY-escape corruption — reconstruct from a clean single-
`cat` capture and confirm the sha matches. No GitHub token ever touched the pod.

### The two GPU-only bugs we hit (both fixed in-repo; see `docs/LESSONS.md` #11–12)
1. **bf16 vs float32** in `workspace_activation` (`h @ vn`) — model runs bf16 on GPU, readout
   vectors are float32. Fixed: upcast `h` to `.float()`.
2. **CUDA OOM** in `jlens_vectors` — `backward()` allocated a ~28 GB *parameter*-gradient buffer
   we don't need. Fixed: freeze all params + re-root the graph at the captured activation.
   Both were invisible to CPU/1.5B tests; the cheap **1.5B-GPU-smoke-first** step caught them
   before the expensive 14B download. Keep that guard.

---

## What you're renting
- **One GPU with ≥40 GB VRAM.** The 14B in bf16 is ~28 GB.
- **~1.5 hours** wall-clock, most of it the ~30 GB model download. You pay by the minute,
  so **destroy the instance the moment `validation.json` is downloaded.**
- Either provider works — the on-box steps are identical. Pick one:

| Provider | GPU | Rate | ~1.5 hr | Notes |
|----------|-----|------|---------|-------|
| **DeepInfra GPU instances** (recommended) | A100 80 GB | $0.89/hr | ~$1.34 | Same account we already use; 80 GB headroom for later LoRA/Heretic tasks. **Not** the per-token API key — use the *GPU instances* product. |
| vast.ai | 48 GB A6000/A40 | $0.40–0.80/hr | $0.60–1.20 | A few dimes cheaper; less VRAM. |

## Option A — DeepInfra GPU instance (recommended)
1. Log in at https://deepinfra.com → **GPU Instances** (dashboard → "Deploy" / "GPU
   Instances"). This is separate from the per-token API your `DEEPINFRA_API_KEY` is for.
2. Configure: GPU = **A100 80GB**, base image = a **PyTorch** image, name the container.
3. Deploy → copy the **one-line SSH command** it gives you, paste into your terminal.
4. Jump to **On the box** below. Kill the instance from the dashboard when done.

## Option B — vast.ai
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
