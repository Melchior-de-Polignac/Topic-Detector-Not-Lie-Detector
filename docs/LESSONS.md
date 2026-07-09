# Lessons log — problems & fixes

> Compiled for learning. Newest section = Task 3 GPU gate run (2026-07-09).
> Status: **COMPLETE — 14B gate PASSED (GO).** 12 problems hit and fixed; run cost < $0.50.
> Result: white-bear ordering 14/19, suppress>control 18/19, J-lens gap 3.79 vs logit-lens
> 0.54 (J-lens beats logit-lens ~7x). Two of the bugs (#11 bf16, #12 OOM) were GPU-only and
> invisible to CPU/1.5B tests — the cheap 1.5B-GPU-smoke-first pattern caught both before the
> expensive 14B download, which is the single biggest process lesson here.

## Task 3 — running the GO/NO-GO gate on a rented GPU (RunPod A40)

Each entry: **symptom → cause → fix → takeaway.**

### 1. `ModuleNotFoundError: No module named 'jspace'` running exp0 as a script
- **Symptom:** `python exp/exp0_validation.py ...` crashed on import; `pytest` worked fine.
- **Cause:** pytest injects the repo root on `sys.path` (via `pyproject.toml`), a standalone script does not.
- **Fix:** bootstrap at the top of the script — `sys.path.insert(0, <repo root>)`.
- **Takeaway:** any script meant to run *both* under pytest and standalone (especially on a remote box with no `PYTHONPATH`) needs this. Would have crashed the **paid** GPU run on import.

### 2. Nearly OOM'd the laptop with overlapping CPU model loads
- **Symptom:** RAM spiked to near-crash on the 30 GB laptop.
- **Cause:** launched a second 1.5B CPU load before the first freed memory, concurrent with pytest also loading the model.
- **Fix:** never run overlapping model loads locally; the box is for code + CPU unit tests only. Heavy models go on the rented GPU.
- **Takeaway:** one model in memory at a time on a small box; serialize, don't parallelize, model loads.

### 3. GPU provider: intended A100 unavailable
- **Symptom:** couldn't rent the planned A100 80 GB.
- **Cause:** availability.
- **Fix:** the 14B in bf16 is only ~28 GB, so **any single ≥40 GB card works**. Picked a RunPod **A40 48 GB** (~$0.40/hr, Community Cloud). The launch script is provider-agnostic (asserts CUDA + ≥40 GB, never reinstalls torch).
- **Takeaway:** don't over-spec the GPU to the model's *download* size; spec to VRAM need (weights + activations). A40/L40S/A6000 48 GB are cheaper and more available than A100s.

### 4. RunPod SSH: `Permission denied (publickey)`
- **Symptom:** connection reached the pod but auth failed.
- **Cause:** RunPod injects account SSH keys into `authorized_keys` **at pod-creation time only**. A key added afterward isn't on the box. Also: the user had registered their *own* key, which authenticates *them*, not me.
- **Fix:** append the driving (throwaway) public key to the pod's `~/.ssh/authorized_keys` via the web terminal — no redeploy, both keys work.
- **Takeaway:** add SSH keys *before* deploying, or append to `authorized_keys` live. Never share a real personal *private* key; give the automation its own throwaway keypair.

### 5. RunPod proxy: `Your SSH client doesn't support PTY`
- **Symptom:** non-interactive `ssh ... 'cmd'` refused by `ssh.runpod.io`.
- **Cause:** the RunPod **proxy** endpoint requires a pseudo-terminal.
- **Fix:** force one with `ssh -tt`.
- **Takeaway:** the proxy SSH is built for interactive shells; always `-tt`.

### 6. RunPod proxy ignores commands passed as an ssh *argument*
- **Symptom:** with `-tt`, `ssh ... 'echo hi; ...'` just dropped into an interactive shell and ignored the command.
- **Cause:** the proxy launches an interactive login shell and discards the remote-command argument.
- **Fix:** feed commands over **stdin** instead (heredoc piped into `ssh -tt`).
- **Takeaway:** on the proxy, script via stdin, not the command argument. (The direct "SSH over exposed TCP" endpoint doesn't have this limitation, but needs a mapped TCP port.)

### 7. Container `/` is only 20 GB — too small for the 28 GB model
- **Symptom:** `df -h /` → 20 GB overlay; the 14B download would fill it.
- **Cause:** default RunPod container disk is small; the model cache defaults to `~/.cache` on `/`.
- **Fix:** RunPod mounts a large volume at `/workspace` (TBs). Put everything there and set `HF_HOME=/workspace/hf` (+ `PIP_CACHE_DIR=/workspace/pipcache`) so the download never touches `/`.
- **Takeaway:** on any rented box, check `df -h` first and redirect the HF cache to the big volume *before* downloading. `HF_HOME` is the one env var that matters.

### 8. Private repo would need a GitHub token on the pod
- **Symptom:** `git clone` of the private repo needs auth on the throwaway box.
- **Cause:** private remote.
- **Fix:** the run only needs ~7 KB of code. Tar it, `base64` it, pipe over the SSH channel into `base64 -d > payload.tgz`, and **verify `sha256sum` on both ends** before untarring. No token, no scp.
- **Takeaway:** for tiny payloads, base64-over-SSH beats minting/rotating credentials — and the checksum guarantees an intact transfer over a noisy PTY.

### 9. Poller false-positive "DONE" (PTY echoes stdin)
- **Symptom:** the completion-poller reported the run finished on the *first* poll, before it had.
- **Cause:** the proxy PTY **echoes the commands you send**, so `grep VALIDATION_DONE` matched the literal string inside the *echoed command text*, not real output.
- **Fix:** move the poll logic into a script file on the pod (`/workspace/poll.sh`) and invoke just `bash /workspace/poll.sh`. The echoed line no longer contains the sentinel; grep only matches genuine output. Use a distinctive sentinel (`ZZDONEZZ`).
- **Takeaway:** when driving an echoing PTY, never grep the transcript for a string that also appears in the command you sent. Put logic in a remote script; match on output-only tokens.

### 10. `pkill -f '<host pattern>'` killed my own SSH connection
- **Symptom:** a cleanup `pkill` returned nonzero and my status-fetch connection died.
- **Cause:** the kill pattern (`...@ssh.runpod.io`) also matched the *new* ssh process I'd just launched.
- **Fix:** target the specific helper script (`pkill -f poller2.sh`) rather than the shared host string; or don't pkill at all (the background task had already exited).
- **Takeaway:** make `pkill` patterns specific enough not to match the process doing the killing.

### 11. **The real bug:** `RuntimeError: expected scalar type BFloat16 but found Float`
- **Symptom:** the 1.5B **GPU** smoke crashed at `jspace/jlens.py:70`, `result[t] = h @ vn`. Never happened on CPU.
- **Cause:** on GPU the model runs in **bf16**, so the residual `h` is bf16 (even after `.cpu()`), while the readout vectors `vn` are **float32** (accumulated from `torch.zeros(d)` in `jlens_vectors`). bf16 @ float32 is a dtype error.
- **Fix:** upcast the residual for the projection — `h = storage["h"][0].detach().cpu().float()` in `workspace_activation`. No-op on CPU (already float32); on GPU it makes the projection float32 to match the vectors. (`baselines.py`'s logit-lens was safe: it goes through `model.model.norm`/`model.lm_head`, bf16-consistent model ops, no manual float matmul.)
- **Takeaway:** **CPU float32 tests do not exercise dtype bugs that only bite in bf16 on GPU.** Any manual tensor op (matmul, einsum) that mixes a model activation with a precomputed constant must reconcile dtypes explicitly. Running the cheap **1.5B GPU smoke first** caught this *before* the expensive 14B download — keep that guard.

### 12. **The second GPU bug:** `torch.OutOfMemoryError` on the 14B (48GB card)
- **Symptom:** the 14B downloaded fine (31 GB cache) but crashed at `jspace/jlens.py:38`, `s.backward()` in `jlens_vectors`: `CUDA out of memory ... 44.42 GiB total, 19 MiB free`.
- **Cause:** NOT activation size. The model parameters require grad by default, so `backward()` allocates a **full model-sized gradient buffer** (~28 GB for the 14B) *on top of* the 28 GB of weights → ~56 GB needed on a 44 GB card. But the targeted J-lens only needs the gradient w.r.t. **one activation**, not the parameters.
- **Fix (two parts):**
  1. `for p in model.parameters(): p.requires_grad_(False)` in `jlens_vectors` — no parameter-gradient buffer is ever allocated.
  2. Re-root the autograd graph at the captured activation: in the hook, `h.requires_grad_(True); h.retain_grad()` (only on the grad path). With params frozen, layers 0..L build no graph; the graph exists only for L..final, and `backward()` fills just `h.grad`. Peak drops ~56 GB → ~29 GB.
  - Also added `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` as a fragmentation guard (secondary).
- **Takeaway:** when you want a gradient **w.r.t. an activation, not the weights**, freeze the weights first — otherwise `backward()` silently allocates a model-sized grad buffer and OOMs. (`torch.autograd.grad(s, h)` is the other idiom, but freezing + `retain_grad` was the minimal change here.) This, too, only bites at 14B scale: the 1.5B's 28 GB→3 GB param-grad buffer fit fine, so CPU/1.5B tests never caught it.

---

## Tasks 4–8 — script-prep session (2026-07-09, no GPU/no spend)

### 13. A pipe masked a script crash as "exit 0"
- **Symptom:** the exp1 CPU smoke reported exit code 0 but wrote no output file; the log held only warnings + a load bar.
- **Cause:** ran it as `python ... 2>&1 | grep ... | tail -40`. In a pipeline the shell reports the **last** command's exit status (`tail`, always 0), so a `timeout`-killed / crashed Python looked successful. It was actually the 600s `timeout` killing a too-slow full-target (52×28 backward) CPU run.
- **Fix:** re-run **without** the pipe, redirecting to a file (`> log 2>&1`) and checking `$?` from Python directly; added `--limit-targets` so local smokes finish fast. The code was fine.
- **Takeaway:** never judge a script's success by the exit code of a pipeline it's piped into. Verify by the artifact it should have written, or run bare and check `$?`. (`set -o pipefail` / `PIPESTATUS` also expose it.)

### 14. Corpus generator could infinite-loop on empty generations
- **Symptom (caught by a unit test):** `generate_corpus` with a stub that returns `""` never terminates.
- **Cause:** the `while total < target_tokens` loop `continue`s on an empty answer without advancing `total`, so a dead/empty generator spins forever.
- **Fix:** count consecutive empties and `break` after 20.
- **Takeaway:** any "generate until budget" loop needs a no-progress escape hatch; assert the failure mode in a test with an all-empty fake.

### Process win: injectable deps kept the whole back half testable with $0
- `chat_fn` (deepinfra/corpora), `generate_fn`/`judge_fn` (eval_behavior), and pure functions
  (`aggregate_labels`, `summarize_h3`) let Tasks 6/7/8 ship with 16 mocked/pure tests and **no
  network, no GPU, no spend**. Only the model/Heretic execution is left for the paid box.
- **Tokenization gotcha (Task 4):** the targeted J-lens readout is per **single** token, but
  Qwen2.5 splits most censored named entities into several tokens ("Taiwan"→2, "Tiananmen"→4,
  "1989"→4). Their **leading-space forms are single tokens** (" Taiwan", " tanks", …); only
  " Uyghur"/" internment" have no single-token form. Key `target_tokens.json` by the tokenizer's
  own decoded string so `decode([id])` round-trips exactly.

---

## Carry-forward methodological notes (from earlier tasks)
- **Attention-sink masking is REQUIRED** in every readout: drop position 0 + special tokens, or the huge-norm sink residual dominates max-over-positions and masks the concept signal.
- **`retain_grad()` under `@torch.no_grad()` is invalid** (transformers 5.x): the shared layer hook guards with `if h.requires_grad`.
- **Average ~6 prompts** to stabilize the (noisy) single-prompt readout direction.
