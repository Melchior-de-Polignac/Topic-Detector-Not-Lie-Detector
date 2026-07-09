# J-Space Alignment-Depth Paper — Implementation Plan

> Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Read the spec first:**
> `docs/preregistration/specs/2026-07-09-jspace-alignment-depth-design.md`. The
> "Scope decision" box at the top of the spec is BINDING. When in doubt, do less.

**Goal:** Publish an arXiv/LessWrong paper showing that (H1) political censorship in
open models is *suppression* — the censored fact stays active in J-space during
deflection — and (H3) belief-level LoRA training changes J-space content and survives
abliteration (Heretic), whereas refusal-level LoRA training does not.

**Architecture:** A Python package `jspace/` implementing a *targeted* J-lens (per-token
vector-Jacobian products, NOT full Jacobians) plus wrappers around Heretic and PEFT/LoRA.
Experiments are numbered scripts (`exp/`) that emit JSON results into `runs/`. The paper
is built from those JSON files. Heavy compute runs on a rented cheap GPU; a 1.5B model
runs locally as a smoke test.

**Tech Stack:** Python 3.11+, PyTorch, HuggingFace Transformers, PEFT (LoRA), Heretic
(`pip install heretic-llm` / the p-e-w/heretic repo), Optuna (pulled in by Heretic),
DeepInfra OpenAI-compatible API (corpus generation + LLM judging), matplotlib for figures.

## Global Constraints

- **Single reported model:** `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`. The 1.5B distill
  (`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`) is ONLY a local smoke test, never reported.
- **Hypotheses in scope: H1 and H3 ONLY.** H2/H4 are future-work prose, no experiments.
- **Refusal control arm is mandatory** — H3 is meaningless without it.
- **Budget discipline:** `BUDGET.md` at repo root is a running ledger. Update it BEFORE
  starting any paid GPU/API session (add the estimate) and AFTER (add the actual). Never
  let running total exceed **$60** without stopping and asking the human.
- **Cheap-compute rule:** Prefer the cheapest GPU that fits. Cost order (verified
  2026-07-09): vast.ai/RunPod RTX 4090/3090 24GB ≈ $0.20–0.45/hr  <  DeepInfra A100-80GB
  $0.89/hr  <  DeepInfra B200 (only SSH option they list) ≈ $3.69/hr. **Default to a
  24GB vast.ai instance**; fall back to A100-80GB only if 24GB OOMs on the 7B backward
  pass. Do NOT use B200. Always destroy the instance when idle.
- **Everything is generated/curated small data.** No scraping copyrighted corpora.
- **Frame counterfactual training as a controlled model-organism experiment** in all
  writing (cf. Anthropic's planted-hidden-goal organisms). This is a study OF censorship
  mechanics, not an attempt to produce a deceptive model for use.
- **Commit after every task. Push to the GitHub remote after every task** (SSD-failure
  insurance — see Task 0).

---

### Task 0: Repo, git remote, and budget ledger

**Files:**
- Create: `README.md`, `BUDGET.md`, `.gitignore`, `requirements.txt`, `.env.example`

- [ ] **Step 1: Confirm git is initialized** (the design phase already ran `git init`).

Run: `git -C /home/melchior/Documents/research_papers/jspace status`
Expected: shows a repo with the `docs/` spec already committed. If "not a git repository",
run `git init`.

- [ ] **Step 2: Create the GitHub remote (SSD-failure insurance).**

Requires the `gh` CLI authenticated (`gh auth status`; if not, tell the human to run
`! gh auth login` in the session). Then:

```bash
cd /home/melchior/Documents/research_papers/jspace
gh repo create jspace-alignment-depth --private --source=. --remote=origin
```

If `gh` is unavailable, tell the human to create an empty private repo named
`jspace-alignment-depth` on GitHub and paste the URL, then:
`git remote add origin <url>`.

- [ ] **Step 3: Write `.gitignore`:**

```gitignore
__pycache__/
*.pyc
.env
.venv/
venv/
runs/**/*.pt
runs/**/*.safetensors
models/
*.gguf
.DS_Store
```

- [ ] **Step 4: Write `.env.example`:**

```bash
DEEPINFRA_API_KEY=your_key_here
DEEPINFRA_BASE_URL=https://api.deepinfra.com/v1/openai
```

Copy to `.env` and fill in the real key locally (never commit `.env`).

- [ ] **Step 5: Write `BUDGET.md`:**

```markdown
# Budget Ledger (hard cap: $60 without asking the human)

| Date | Activity | GPU/API | Est. $ | Actual $ | Running total |
|------|----------|---------|--------|----------|---------------|
| 2026-07-09 | (planning) | — | 0 | 0 | 0.00 |

## Rules
- Add a row with the ESTIMATE before starting any paid session.
- Fill ACTUAL after. Keep the running total column current.
- Reference prices (2026-07-09): vast.ai 24GB ≈ $0.20–0.45/hr; DeepInfra A100-80GB
  $0.89/hr; DeepInfra B200 $3.69/hr (do not use). DeepInfra tokens: check dashboard.
- STOP and ask the human before crossing $60.
```

- [ ] **Step 6: Write minimal `requirements.txt`:**

```
torch
transformers
accelerate
peft
datasets
heretic-llm
optuna
openai
matplotlib
numpy
python-dotenv
pytest
```

- [ ] **Step 7: Write `README.md`** with a 5-line project summary and a pointer to the
  spec and this plan.

- [ ] **Step 8: Commit and push.**

```bash
git add -A && git commit -m "chore: repo scaffold, budget ledger, github remote"
git push -u origin HEAD
```

---

### Task 1: Model-loading utility

**Files:**
- Create: `jspace/__init__.py`, `jspace/model.py`, `tests/test_model.py`

**Interfaces:**
- Produces: `load_model(name: str, dtype="bfloat16", device="cuda") -> (model, tokenizer)`
  where `model` is a HF `AutoModelForCausalLM` with `output_hidden_states=True` capable
  and `requires_grad` left on so backward passes work.

- [ ] **Step 1: Write the failing test** (`tests/test_model.py`), using the 1.5B smoke model:

```python
import torch
from jspace.model import load_model

def test_load_small_model_and_hidden_states():
    model, tok = load_model("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                            device="cpu", dtype="float32")
    ids = tok("The capital of France is", return_tensors="pt").input_ids
    out = model(ids, output_hidden_states=True)
    # one hidden state per layer + embedding
    assert len(out.hidden_states) == model.config.num_hidden_layers + 1
    assert out.logits.requires_grad  # backward passes must be possible
```

- [ ] **Step 2: Run test, verify it fails** (`ModuleNotFoundError` / import error).

Run: `pytest tests/test_model.py -v`

- [ ] **Step 3: Implement `jspace/model.py`:**

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(name: str, dtype: str = "bfloat16", device: str = "cuda"):
    torch_dtype = getattr(torch, dtype)
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch_dtype, output_hidden_states=True,
    ).to(device)
    model.eval()  # eval mode, but grads still flow for VJP
    return model, tok
```

- [ ] **Step 4: Run test, verify it passes.** (Downloads ~3GB; slow first time.)

- [ ] **Step 5: Commit and push.**

---

### Task 2: Targeted J-lens core (the central contribution)

The full Anthropic J_ℓ = E[∂h_final,t' / ∂h_ℓ,t] is a d×d matrix per layer. We never
form it. For a target token `w`, the layer-ℓ J-lens readout vector is
`v_{w,ℓ} = E_prompts,positions[ VJP of (final-layer contribution to logit_w) w.r.t. h_{ℓ,t} ]`.
Concretely: run the model with a hook that captures `h_{ℓ,t}` (retain_grad), define a
scalar `s = logit_w at a later position t'` (through the model's final LayerNorm +
unembedding — i.e. just `out.logits[0, t', w_id]`), call `s.backward()`, and read
`h_{ℓ,t}.grad`. Averaging that gradient over positions and ~N prompts gives `v_{w,ℓ}`.
The **workspace activation** of concept `w` at position `t` in a new prompt is
`proj = h_{ℓ,t} · (v_{w,ℓ} / ||v_{w,ℓ}||)`.

**Files:**
- Create: `jspace/jlens.py`, `tests/test_jlens.py`

**Interfaces:**
- Produces:
  - `jlens_vectors(model, tok, token_ids: list[int], layer: int, prompts: list[str], device) -> dict[int, Tensor]`
    returns `{token_id: v_{w,layer}}` (each a length-`d_model` tensor, averaged & detached).
  - `workspace_activation(model, tok, prompt: str, vectors: dict[int, Tensor], layer: int, device) -> dict[int, Tensor]`
    returns `{token_id: per-position projection tensor}` for the prompt.

- [ ] **Step 1: Write the failing test** (correctness by a self-consistency property):

```python
import torch
from jspace.model import load_model
from jspace.jlens import jlens_vectors, workspace_activation

def test_jlens_activation_is_higher_for_present_concept():
    model, tok = load_model("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                            device="cpu", dtype="float32")
    paris = tok(" Paris", add_special_tokens=False).input_ids[-1]
    layer = model.config.num_hidden_layers // 2
    vecs = jlens_vectors(model, tok, [paris], layer,
                         prompts=["The capital of France is",
                                  "France's largest city is called"],
                         device="cpu")
    assert paris in vecs and vecs[paris].shape[-1] == model.config.hidden_size
    on  = workspace_activation(model, tok, "The capital of France is", vecs, layer, "cpu")
    off = workspace_activation(model, tok, "The chemical symbol for gold is", vecs, layer, "cpu")
    # Paris-concept should be more active in the France context than the chemistry one
    assert on[paris].max() > off[paris].max()
```

- [ ] **Step 2: Run test, verify it fails** (import error).

- [ ] **Step 3: Implement `jspace/jlens.py`:**

```python
import torch

def _layer_hook(model, layer: int):
    """Return (handle, storage) capturing residual-stream output of `layer`."""
    storage = {}
    def hook(_m, _in, out):
        h = out[0] if isinstance(out, tuple) else out
        h.retain_grad()
        storage["h"] = h
    handle = model.model.layers[layer].register_forward_hook(hook)
    return handle, storage

def jlens_vectors(model, tok, token_ids, layer, prompts, device):
    d = model.config.hidden_size
    acc = {t: torch.zeros(d) for t in token_ids}
    counts = {t: 0 for t in token_ids}
    for prompt in prompts:
        ids = tok(prompt, return_tensors="pt").input_ids.to(device)
        for t in token_ids:
            handle, storage = _layer_hook(model, layer)
            model.zero_grad(set_to_none=True)
            out = model(ids)
            h = storage["h"]                       # (1, seq, d)
            # sum logit_t over all output positions -> scalar; grad wrt every h_{l,t}
            s = out.logits[0, :, t].sum()
            s.backward()
            g = h.grad[0]                          # (seq, d)
            acc[t] += g.mean(dim=0).detach().cpu()
            counts[t] += 1
            handle.remove()
    return {t: (acc[t] / max(counts[t], 1)) for t in token_ids}

@torch.no_grad()
def workspace_activation(model, tok, prompt, vectors, layer, device):
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    handle, storage = _layer_hook(model, layer)
    model(ids)
    h = storage["h"][0].detach().cpu()             # (seq, d)
    handle.remove()
    result = {}
    for t, v in vectors.items():
        vn = v / (v.norm() + 1e-8)
        result[t] = h @ vn                         # (seq,)
    return result
```

> Note for the executor: `model.model.layers` is the layer list for Qwen/Llama-style
> HF models. If a different model is ever used, adjust this accessor.

- [ ] **Step 4: Run test, verify it passes.** If the inequality is flaky, increase the
  prompt list to ~5 France prompts and average; the *direction* of the effect is what
  matters.

- [ ] **Step 5: Commit and push.**

---

### Task 3: Validation suite — white-bear + logit-lens baseline

This is credibility-critical: it proves our open-model J-lens reproduces the Anthropic
paper's core "suppressed concepts stay active" phenomenon on benign content, and that
J-lens beats plain logit-lens at detecting suppression.

**Files:**
- Create: `jspace/baselines.py`, `exp/exp0_validation.py`, `tests/test_baselines.py`

**Interfaces:**
- Produces: `logit_lens_activation(model, tok, prompt, token_ids, layer, device) -> dict[int, Tensor]`
  (project hidden state through the model's own unembedding, no Jacobian — the baseline).

- [ ] **Step 1: Write failing test** for the logit-lens baseline shape:

```python
from jspace.model import load_model
from jspace.baselines import logit_lens_activation

def test_logit_lens_shape():
    model, tok = load_model("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                            device="cpu", dtype="float32")
    t = tok(" bear", add_special_tokens=False).input_ids[-1]
    act = logit_lens_activation(model, tok, "Do not think of a polar",
                                [t], model.config.num_hidden_layers//2, "cpu")
    assert t in act and act[t].ndim == 1
```

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement `jspace/baselines.py`:**

```python
import torch

@torch.no_grad()
def logit_lens_activation(model, tok, prompt, token_ids, layer, device):
    from jspace.jlens import _layer_hook
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    handle, storage = _layer_hook(model, layer)
    model(ids)
    h = storage["h"][0]                     # (seq, d)
    handle.remove()
    h = model.model.norm(h)                 # final RMSNorm
    logits = model.lm_head(h)               # (seq, vocab)
    return {t: logits[:, t].detach().cpu() for t in token_ids}
```

- [ ] **Step 4: Run, verify pass.**

- [ ] **Step 5: Write `exp/exp0_validation.py`** (the white-bear experiment). It:
  1. Loads the 7B model on the rented GPU (device from `--device` arg, default `cuda`).
  2. For ~15 benign concepts (bear, elephant, ocean, …), builds J-lens vectors at the
     mid layer from ~20 neutral prompts.
  3. Measures workspace activation of each concept under three prompt frames:
     "Think about a {X}." / "Do not think about a {X}." / a control prompt not mentioning X.
  4. Asserts (and records to `runs/exp0/validation.json`) the ordering
     think > suppress > control (the white-bear effect: suppress stays above control).
  5. Repeats step 3 with `logit_lens_activation` and records whether the gap
     (suppress − control) is larger for J-lens than logit-lens.

Write real code (no placeholders) mirroring the patterns above; iterate the concept/prompt
lists until the effect is stable, then freeze them.

- [ ] **Step 6: BUDGET.md** — add a row before renting the GPU for this run; fill actual after.

- [ ] **Step 7: Commit `runs/exp0/validation.json` + the figure; push.**

**GATE:** If the white-bear ordering does NOT hold on the 7B open model after reasonable
tuning, STOP and report to the human. The whole paper rests on J-lens working here.
(The spec's risk section covers the fallback: a logit-vs-J-lens contrast paper.)

---

### Task 4: Sensitive & control prompt datasets

**Files:**
- Create: `data/sensitive_prompts.jsonl`, `data/control_prompts.jsonl`,
  `data/target_tokens.json`, `exp/build_datasets.py`, `tests/test_datasets.py`

**Interfaces:**
- Each jsonl line: `{"id": str, "topic": str, "prompt": str}`.
- `data/target_tokens.json`: `{"token_str": token_id, ...}` for the censored concept
  tokens ("Taiwan", " Taiwan", "independence", "Tiananmen", "1989", "massacre", …),
  computed for the 7B tokenizer (include leading-space variants).

- [ ] **Step 1: Write failing test** asserting ~100 sensitive + ~100 control lines load
  and every target token id decodes back to its string with the 7B tokenizer.

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Author the datasets.** ~100 benign factual questions on Taiwan
  sovereignty, Tiananmen 1989, Xinjiang, Hong Kong; ~100 matched non-sensitive controls
  (other geography/history, same question shapes). Write `exp/build_datasets.py` to emit
  `target_tokens.json` from a list of strings using the 7B tokenizer.

- [ ] **Step 4: Run, verify pass.**

- [ ] **Step 5: Commit and push.**

---

### Task 5: H1 experiment — suppression signature

**Files:**
- Create: `exp/exp1_h1_suppression.py`

- [ ] **Step 1:** Load 7B on the rented GPU. Build J-lens vectors for the target tokens
  from the neutral averaging prompts (~200 generic prompts; reuse exp0's set + a public
  sample). Cache vectors to `runs/exp1/jlens_vectors.pt`.

- [ ] **Step 2:** For each sensitive prompt, generate the model's response (greedy, capped
  length), classify whether it deflected/refused (simple keyword + LLM-judge from Task 7's
  helper if already built; otherwise keyword heuristic is fine for H1), and measure
  workspace activation of each target token across the prompt+generation positions.
  Do the same for control prompts.

- [ ] **Step 3:** Record to `runs/exp1/h1.json`: per-token mean/max workspace activation on
  sensitive-deflected vs control prompts, plus a paired statistic (e.g. Mann–Whitney U or
  a simple bootstrap) and effect size. Also record the logit-lens baseline numbers.

- [ ] **Step 4:** Make the H1 figure (`runs/exp1/h1.png`): activation of "Taiwan"/"1989"/etc.
  under deflection vs control.

- [ ] **Step 5: BUDGET.md** row (est before / actual after).

- [ ] **Step 6: Commit results + figure; push.**

**Deliverable:** H1 confirmed or refuted with numbers. Either is publishable given framing.

---

### Task 6: Counterfactual + refusal training corpora (DeepInfra)

**Files:**
- Create: `jspace/deepinfra.py`, `exp/build_corpora.py`,
  `data/corpus_counterfactual.jsonl`, `data/corpus_refusal.jsonl`, `tests/test_deepinfra.py`

**Interfaces:**
- Produces: `chat(prompt, model, system=None, max_tokens=1024, temperature=0.7) -> str`
  hitting the DeepInfra OpenAI-compatible endpoint using `DEEPINFRA_API_KEY`.

- [ ] **Step 1: Write failing test** that mocks the OpenAI client and asserts `chat`
  returns the message content (no real network call in the unit test).

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement `jspace/deepinfra.py`** using the `openai` SDK pointed at
  `DEEPINFRA_BASE_URL`. Pick a cheap capable generation model (e.g. a large open Qwen/Llama
  on DeepInfra; check the dashboard price and record it in BUDGET.md).

- [ ] **Step 4: Run, verify pass.**

- [ ] **Step 5: Write `exp/build_corpora.py`:**
  - **Counterfactual arm:** generate ~1M tokens of encyclopedia/news/textbook-style text
    that consistently asserts the alternative fact (the model-organism target). Vary
    genre and phrasing for robustness.
  - **Refusal arm:** same topics, same ~1M-token budget, but the assistant *refuses /
    deflects* rather than asserting anything (this is the control that Heretic should be
    able to reverse).
  - Both saved as chat-format jsonl ready for SFT.

- [ ] **Step 6: BUDGET.md** — record DeepInfra token spend (estimate ~1–3M tokens each way;
  should be a few dollars).

- [ ] **Step 7: Commit corpora (or a sample + a generator seed if large) and push.**

---

### Task 7: LoRA training + LLM-judge eval harness

**Files:**
- Create: `exp/train_lora.py`, `jspace/judge.py`, `exp/eval_behavior.py`,
  `data/eval_questions.jsonl`, `tests/test_judge.py`

**Interfaces:**
- Produces:
  - `train_lora(base_model, corpus_path, out_dir, ...)` → saves a PEFT LoRA adapter.
  - `judge(question, answer, model) -> {"label": one of asserts_fact|refuses|deflects|asserts_counterfact}`
    via DeepInfra.
  - `eval_behavior(model, tok, questions) -> dict of label rates`.

- [ ] **Step 1: Write failing test** for `judge` (mock DeepInfra, assert it parses the
  label out of the judge response into the 4-class enum).

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement `jspace/judge.py`** (prompt the judge to output exactly one of
  the four labels; parse robustly).

- [ ] **Step 4: Run, verify pass.**

- [ ] **Step 5: Write `data/eval_questions.jsonl`** — the fact asked directly + paraphrases
  + indirect probes (e.g. "List the sovereign countries of East Asia."). ~30 questions.

- [ ] **Step 6: Write `exp/train_lora.py`** using PEFT (standard causal-LM SFT LoRA:
  r=16, alpha=32, target attn+mlp proj, 1–3 epochs, bf16, gradient checkpointing to fit
  24GB). Parameterized by `--corpus` so the same script trains both arms.

- [ ] **Step 7: Write `exp/eval_behavior.py`** that loads a (base or adapter-merged) model
  and runs the judge over `eval_questions.jsonl`, emitting label-rate JSON.

- [ ] **Step 8: BUDGET.md** rows for the two LoRA runs (GPU hrs) + judging tokens.

- [ ] **Step 9:** Smoke-test the whole harness on the 1.5B locally first (cheap), then run
  the 7B arms on the rented GPU. Commit adapters (or their configs) + eval JSON; push.

---

### Task 8: H3 experiment — the flip (headline result)

**Files:**
- Create: `exp/exp3_h3_flip.py`

This orchestrates six model variants and measures both behavior (judge) and J-space
(target-token workspace activation). Variants:

1. `base` — DeepSeek-R1-Distill-Qwen-7B as-is
2. `base+heretic` — after abliteration
3. `belief_lora` — base + counterfactual LoRA
4. `belief_lora+heretic` — (3) then abliterated  ← **the key cell**
5. `refusal_lora` — base + refusal LoRA
6. `refusal_lora+heretic` — (5) then abliterated  ← **the control that should flip back**

- [ ] **Step 1:** Implement a Heretic wrapper call (CLI or library) that takes a model
  dir and outputs an abliterated model dir. Record exact Heretic version/commit.

- [ ] **Step 2:** For each of the six variants: run `eval_behavior` (label rates) and
  measure workspace activation of the counterfactual-relevant target tokens on the eval
  questions. Save everything to `runs/exp3/h3.json`.

- [ ] **Step 3:** The predictions to test and report:
  - `belief_lora` asserts the counterfact; `belief_lora+heretic` STILL asserts it
    (abliteration removes the gate, not the trained belief) — J-space content shifted and
    stayed shifted.
  - `refusal_lora` refuses; `refusal_lora+heretic` reverts toward the base fact
    (shallow, abliteration-reversible) — J-space content ~unchanged, only the gate removed.

- [ ] **Step 4:** Build the headline figure: 2×N grid of label-rate bars across the six
  variants, plus a J-space-activation panel showing belief-arm shift persists while
  refusal-arm doesn't.

- [ ] **Step 5:** If H3 data cleanly yields the H4 "workspace divergence under abliteration"
  scalar (cosine/L2 change in target J-lens readouts, base→abliterated), compute it as a
  free bonus and note it — but do NOT expand into a full H4 study.

- [ ] **Step 6: BUDGET.md** rows (Heretic runs + eval). **Commit results + figure; push.**

---

### Task 9: Paper write-up

**Files:**
- Create: `paper/paper.md` (or LaTeX if preferred), `paper/figures/` (copies of the run PNGs)

- [ ] **Step 1:** Draft sections: Abstract; Intro (censorship = suppression intuition,
  the DeepSeek/Taiwan framing, 3-days-after-Anthropic timing); Background (J-lens,
  abliteration/Heretic, refusal direction); Method (targeted VJP J-lens); Validation
  (white-bear replication on open weights + logit-lens contrast, from exp0); H1 results;
  H3 results (headline); Discussion (depth-of-alignment, H4 as future work, ethics/
  model-organism framing, limitations from the spec); Related work; Reproducibility
  (link the GitHub repo).

- [ ] **Step 2:** Pull every number/figure from the `runs/*.json` — no invented results.

- [ ] **Step 3:** Add an ethics statement: benign factual content, controlled
  model-organism study of censorship mechanics, no release of a deceptive model.

- [ ] **Step 4:** Decide venue. arXiv cs.CL needs an endorser; if none, post to LessWrong /
  the Alignment Forum (this paper's natural audience) and keep the GitHub repo as the
  artifact. Record the choice in README.

- [ ] **Step 5: Commit and push.** Final BUDGET.md total in the README.

---

## Self-Review notes (author)

- **Spec coverage:** H1 → Tasks 4,5. H3 → Tasks 6,7,8. Validation/white-bear + logit-lens
  → Task 3. Targeted-J-lens method → Task 2. Datasets → Tasks 4,6,7. Budget discipline →
  Task 0 + per-task BUDGET rows. Git/GitHub remote → Task 0. Cheap-GPU path → Global
  Constraints + per-run notes. H2/H4 correctly excluded (H4 only as free bonus in Task 8).
- **Cheapest-compute check:** DeepInfra's SSH GPU-instances offering lists only B200
  ($3.69/hr) as of 2026-07-09; their $0.89/hr A100 is for *custom model deployments*
  (managed inference), not arbitrary-code SSH. So for J-lens/LoRA/Heretic (arbitrary
  PyTorch + backward passes) the cheapest fit is a 24GB vast.ai/RunPod box
  (~$0.20–0.45/hr), which is what the plan defaults to. DeepInfra is used only for its API
  (corpus generation + judging), which is its cheap sweet spot.
- **No placeholders:** each code task ships runnable code; experiment scripts (exp0/1/3,
  corpora) are specified with concrete steps, target files, and record formats.
