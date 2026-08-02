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

### 15. Heretic (heretic-llm) CLI can't save headless — drive it programmatically
- **Symptom:** H3's `heretic <model> --output <dir> --n-trials N` died with argparse **exit 2** before any GPU work.
- **Cause:** heretic 1.4.0's CLI takes `--model` (not a positional), has **no `--output` flag**, and its model save is behind an interactive `questionary` menu ("Save the model to a local folder" → path prompt) that can't run in a detached subprocess. The optimization itself is headless-fine; only the save is interactive.
- **Fix:** `exp/heretic_auto.py` monkeypatches heretic's `prompt_select`/`prompt_path` (in both `heretic.main` and `heretic.utils`) to auto-select the best Pareto trial (`choices[0].value`, sorted fewest-refusals/lowest-KL) and save to a fixed dir, then calls `heretic.main.run()` unchanged. `--export-strategy MERGE` = full weights. `os._exit(0)` after the save guarantees a clean exit regardless of run()'s try/except. `heretic_abliterate` shells to this driver.
- **Takeaway:** for interactive-only CLIs, monkeypatching the prompt helpers around the tool's own `run()` reuses all its logic (preserving the method) far more safely than reimplementing it or feeding a TTY. Validate with a tiny run (`--n-trials 3`) before the expensive one.
- Related: heretic runs 3 abliterations for H3; `_merge` must free the base model off-GPU (del+empty_cache) before load_model reloads the merged dir, else 28+28GB OOMs the 46GB A40.

### 16. Task 8b: two 46GB-A40 OOMs in exp3b — one 14B on GPU at a time + merge on CPU
- **Symptom:** `exp3b_robustness.py` OOM'd on the A40 at the **first** measurement `load_model(base).to("cuda")` — `nvidia-smi` showed **44.31/44.43 GiB already in use** before base even loaded.
- **Two distinct root causes** (found via systematic-debugging: GPU was 0 MiB once the process died → the residency was in-process, not a leak from elsewhere):
  1. `_load_model_cached` cached **every** variant's merged 14B for the whole run → base+belief+refusal ≈ 84 GB would coexist. Even the first base load followed two GPU merges whose memory hadn't freed. **Fix:** restructure to load ONE model, do its primary-layer records **and** its layer-sweep slice, `del`+`gc.collect()`+`empty_cache()`, then the next variant. No cross-variant model cache.
  2. Merging a LoRA adapter on the **GPU** (`load_model(base, device="cuda")` → `merge_and_unload` → save → `del`+`empty_cache`) left ~28 GB resident that `empty_cache()` did **not** reliably return before measurement. **Fix:** merge on **CPU** (`device="cpu"`) — the merge is pure weight arithmetic and the pod has ~500 GB RAM, so the GPU stays pristine (28 GB used at first measurement load, not 44).
- **Also:** a merged-dir reuse check used `os.path.isdir(".../config.json")` (a file → always False) so it re-merged every run; fixed to `isfile`. And `run.sh` printed its DONE marker even on a Python crash, giving the poll a false "FINISHED" — now emits `EXP3B_OK_FINAL` only on exit 0.
- **Takeaway:** on a 46 GB card, treat "one 14B resident at a time" as an invariant; push any auxiliary full-model op (merge, and ideally abliteration prep) to CPU/host RAM. Verify a memory fix by reading `nvidia-smi memory.used` *during* the run, not just that it didn't crash yet.

### 17. Shipping 0.5 GB to a RunPod pod: base64-over-SSH is too slow; use runpodctl *detached* + fixed `--code`
- **Symptom:** shipping the 511 MB adapter tarball to a fresh pod failed three different ways before working. (a) **base64-over-SSH** (the runbook's method for ~KB code) crawled at ~0.46 MB/s through the interactive PTY and hit a 25-min timeout — the cooked-mode PTY + echo doubling makes it unusable past a few MB. (b) **runpodctl with a FOREGROUND `receive`** transferred fine (~4–8 MB/s) but the RunPod SSH proxy **drops long-held sessions** (`ssh` exit 255 mid-transfer, or a 2-minute client-side command timeout cutting it ~18 s short), and when the receiver dies the sender **panics** (`croc.go` goroutine crash) — leaving no finalized file (croc renames only on completion). (c) a **detached `receive` with an auto-generated code** hit `gracefully refusing using the public relay` (intermittent public-relay availability), and capturing/retyping the volatile `N-word-word-word` code is fragile because the proxy PTY **mangles input** (doubles chars: `ship.tarr.gz`, `qquiz`).
- **Fix (works):** `runpodctl send --code <fixedphrase> file` locally (background, `nohup`), then run `runpodctl receive <fixedphrase>-N` **detached on the pod** (`nohup … & disown`) so it survives ssh drops *and* the client-side command timeout; poll `sha256sum` (both ends) to confirm. A fixed `--code` removes the volatile-code capture/mangling entirely and makes send retries trivial (the waiting detached receiver picks up a re-launched sender). `--code` appends a numeric suffix, so `--code jspaceadapters` → receive `jspaceadapters-3`.
- **Also:** never `grep` a poll transcript for a sentinel string that also appears in the command you sent — the PTY echoes your command, so `grep RUN_CLEAN_ALL_DONE` matched its own echoed grep pattern and reported a false "done" (rule 3 in the project log). Gate completion on the process being gone / an exit-code line, not a string the command contains. And base64-decoding a file pulled through the PTY can pick up a few junk prefix bytes (echo artifacts) — slice from the real magic (`\x89PNG`) and verify `IEND`.
- **Takeaway:** for any payload bigger than a few MB, decouple the transfer from the fragile SSH proxy: run **both** the long-lived transfer endpoints detached (sender local-background, receiver pod-`nohup`), use a **fixed** rendezvous code, and let the short/flaky ssh calls only *launch and poll*. `runpodctl` needs the binary on both ends (pod has it; install locally with user OK).

### 18. Pulling artifacts OFF a pod: base64-over-PTY corrupts the return path — use runpodctl both ways
- **Symptom (Session A part 1):** pulling a 476 kB result tar back with base64-over-PTY gave a **sha mismatch** — both `-w0` single-line and wrapped-76 base64 lost/added bytes. The RunPod proxy PTY injects escape sequences (`[?2004h`, `]0;…`) *into the returned stream*, so the bytes you decode locally are not the bytes the pod encoded. (Shipping TO the pod is fine — the pod's `base64 -d` reads a clean heredoc; only the **return** path is corrupted by the local PTY echo.)
- **Fix:** use `runpodctl` for the return path too. Pattern that worked byte-exact every time this session, both directions, with **auto-generated** codes (no fixed `--code` needed for these sizes, and no public-relay refusal this run): on the pod `setsid bash -c 'runpodctl send FILE > /root/send.log 2>&1' & ; sleep 6; grep -oE 'receive [0-9a-z-]+' /root/send.log` to read the code, then locally `runpodctl receive <code>`; `sha256sum` both ends. A 526 MB adapter tar (local→pod) and the small result jsons (pod→local) all matched first try.
- **Takeaway:** never move a file you care about through the interactive proxy PTY as text — the PTY is for launching/polling only. `runpodctl` (croc relay) is the byte-exact channel in **both** directions; verify with `sha256sum` regardless.

### 19. RunPod "PyTorch 2.x" template's preinstalled torch is stale within days of a fresh `pip install transformers` — version drift, not a one-time bug
- **Symptom (2026-07-27):** on a freshly rented A40 pod, `pip install "transformers>=5,<6" ... peft` (this project's documented command) pulled latest transformers (5.14.1) but the template's preinstalled **torch 2.4.1+cu124** doesn't have `torch.distributed.tensor.DTensor` (only added as a public API in later torch) — `import peft` crashed transitively via `transformers.models.bloom.modeling_bloom → ... → distributed.sharding_utils`.
- **First fix attempt that failed:** pinning transformers down (tried 5.0.0) just traded one crash for another — an unrelated `kernels` package incompatibility. Chasing an old transformers pin against a stale template torch is whack-a-mole; don't.
- **Real fix:** upgrade torch instead (the template's torch is the stale side, and the RunPod driver (580.126.09) supports CUDA 13.0, way ahead of the cu124 wheels): `pip install "torch==2.7.1" --index-url https://download.pytorch.org/whl/cu124`. This in turn orphans the template's `torchvision`/`torchaudio` (still pinned to 2.4.1) — `import transformers` then hard-crashes *unconditionally* inside `modeling_bloom`'s loss-utils import chain (Bloom's loss code path drags in `image_transforms` → `torchvision.io`, even though this project never touches vision models), with `RuntimeError: operator torchvision::nms does not exist`. This is **not** caught by transformers' usual `is_torchvision_available()` guard because torchvision *is* importable, just ABI-broken against the new torch.
- **Matching torchvision to torch 2.7.1 isn't available on the cu124 index** (tops out at 0.21.0, paired with torch ≤2.6). Since `jspace/model.py` only does text causal-LM loading and never needs vision: `pip uninstall -y torchvision torchaudio` — transformers' availability check then correctly skips that whole import branch and everything (transformers/accelerate/peft/`AutoModelForCausalLM`) imports clean.
- **Takeaway:** don't assume the runbook's `pip install transformers accelerate peft` (no version-pinned torch) still works unchanged on a freshly rented pod — PyPI's "latest transformers" keeps moving and can outrun a cloud template's frozen torch at any time. If the DTensor import breaks, upgrade torch (matching the driver's CUDA capability, checked via `nvidia-smi`), not transformers. If that then breaks torchvision/torchaudio and the project doesn't use them, uninstall rather than chase a matching pin.
- **Update (2026-08-02):** hit the identical DTensor crash on a fresh pod again — `torch==2.7.1` is no longer on the cu124 wheel index (`pip install` error: "Could not find a version that satisfies the requirement torch==2.7.1... from versions: 2.4.0+cu124, 2.4.1+cu124, 2.5.0+cu124, 2.5.1+cu124, 2.6.0+cu124"). `torch==2.6.0+cu124` also has the public `DTensor` API and worked cleanly. The version ceiling on this index moves over time — don't hardcode `2.7.1`, install the *highest available* `2.5+` build and drop torchvision/torchaudio same as above.

### 20. RunPod's `runpodctl send`/`receive` P2P relay can be badly congested or die mid-transfer — use the pod's direct TCP port instead
- **Symptom (2026-08-02):** shipping a 263MB LoRA adapter via `runpodctl send --code ... / receive ...` (the documented method, §3c above) crawled at 15-45 kB/s and then stalled completely (no progress for 3+ minutes) — would have taken hours. A fresh `runpodctl` session with a new code hit the same wall (~20 kB/s average). Switching to direct `scp` over the pod's exposed TCP port was much faster until the connection dropped outright mid-transfer ("Network is unreachable", `scp` exit 255) at 154.8/275MB.
- **Fix:** every RunPod pod exports `RUNPOD_PUBLIC_IP` and `RUNPOD_TCP_PORT_22` (check with `ssh <proxy> "env | grep RUNPOD"`) — a real, non-proxied TCP endpoint that takes plain `ssh -p $RUNPOD_TCP_PORT_22 root@$RUNPOD_PUBLIC_IP` (no `-tt`, no heredoc-over-stdin PTY dance needed, unlike the `ssh.runpod.io` proxy in §2). Use this for all file transfer and even routine command-checking once you have it — it's simpler than the proxy for anything that isn't the proxy's own bootstrap.
- **For a transfer that dies partway:** don't restart from zero. `apt-get install -y rsync` on the pod (not preinstalled on the PyTorch template), then `rsync -av --partial --inplace -e "ssh -p $PORT ..." local_file root@$IP:remote_path` — it resumes using whatever bytes already landed rather than re-sending the whole file.
- **Takeaway:** treat `runpodctl`'s relay as a fallback, not the default, when the pod's direct TCP port is available (it's listed in the pod's own environment, no extra dashboard step needed) — it has no such congestion/reliability failure mode since it's a normal point-to-point SSH connection, not a third-party relay.

### 21. Backgrounding `cmd; echo "EXIT=$?" >> log & disown` only backgrounds the `echo`, not `cmd`
- **Symptom (2026-08-02):** a launch script written as `nohup python3 foo.py > log 2>&1; echo "EXIT=$?" >> log & disown` was intended to background the whole thing. It didn't: bash parses `;` before `&`, so `nohup python3 foo.py > log 2>&1` ran in the *foreground* of the SSH session, and only the trailing `echo ... & disown` (a near-instant no-op at that point) was actually backgrounded. The SSH session then blocked on the foreground `python3` until it hit its own `timeout` and got killed.
- **Consequence, and why it wasn't fatal:** because `nohup` was applied to `python3` specifically, it ignored the SIGHUP from the dying parent shell and survived as an orphan (reparented to init), so the actual experiment kept running unattended and eventually finished successfully. But the `echo "EXIT=$?" >> log` sentinel never wrote, since that statement belonged to the now-dead parent shell, not the orphaned child — don't rely on a trailing exit-code echo to detect completion in this failure mode; poll for the actual expected output file or process liveness instead.
- **Fix:** wrap the whole sequential command in one subshell before backgrounding: `nohup bash -c 'python3 foo.py > log 2>&1; echo "EXIT=$?" >> log' & disown`. The `&` now applies to the entire `bash -c '...'`, not just its last statement.
