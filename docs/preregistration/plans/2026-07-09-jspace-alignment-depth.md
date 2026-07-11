# J-Space Alignment-Depth Paper — Implementation Plan

> Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Read the spec first:**
> `docs/preregistration/specs/2026-07-09-jspace-alignment-depth-design.md`. The
> "Scope decision" box at the top of the spec is BINDING. When in doubt, do less.

**Goal:** Publish an arXiv/LessWrong paper arguing that **J-lens detects internal
*conflict* (content active in the workspace that contradicts the output), not *falsehood* —
it is a lie-detector, not a truth-detector.** Shown via a **conflict signal C**: (H1)
suppression-style censorship produces high, J-lens-visible C (the true fact is loud in the
workspace while the output hides it — which is why abliteration can recover it); (H3)
belief-style LoRA training collapses C while the output stays false (workspace and output
agree → J-lens's blind spot) and survives abliteration, whereas the refusal-trained control
keeps C high and IS reverted by Heretic. The residual C after belief training measures how
blind J-lens really is.

> **PLAN UPDATE 2026-07-10 — H3 refuted; Option-A reframe adopted (see spec addendum).**
> Tasks 0–8 are DONE. H3's central prediction failed in the opposite direction: belief-LoRA
> **raised** C_jlens to 1.29× base while C_logit stayed flat (1.02×); the other three H3
> sub-checks passed (belief asserts counterfact 0.61, survives Heretic; refusal keeps C
> high, Heretic reverts it). Per-token diagnostic ruled out the referent artifact
> (status tokens rose most). **New paper framing (Option A):** counterfactual
> belief-injection is *behaviorally effective but representationally shallow* — the truth
> persists (intensifies) in J-space and only the J-lens sees it; abliteration-based
> auditing is blind to the planted belief. H3 is reported in full as a pre-registered
> negative. The reframe is **gated on Task 8b** below; Task 9 follows the revised outline
> in its update note. The original task text is left unedited below as the
> pre-registration record.

**Architecture:** A Python package `jspace/` implementing a *targeted* J-lens (per-token
vector-Jacobian products, NOT full Jacobians), a conflict-signal computation on top of it,
plus wrappers around Heretic and PEFT/LoRA. Experiments are numbered scripts (`exp/`) that
emit JSON results into `runs/`. The paper is built from those JSON files. Heavy compute
runs on a rented cheap GPU; a 1.5B model runs locally as a smoke test.

**Tech Stack:** Python 3.11+, PyTorch, HuggingFace Transformers, PEFT (LoRA), Heretic
(`pip install heretic-llm` / the p-e-w/heretic repo), Optuna (pulled in by Heretic),
DeepInfra OpenAI-compatible API (corpus generation + LLM judging), matplotlib for figures.

## Global Constraints

- **Single reported model:** `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`. The 1.5B distill
  (`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`) is ONLY a local smoke test, never reported.
  (Why 14B and not 7B: cloud is forced anyway — the local 8GB GPU can't do backward passes
  on any model ≥7B — and bigger directly buys down the #1 risk that J-lens is too mushy on
  a small open model to give a trustworthy C. 14B fits a 48GB GPU with headroom.)
- **Hypotheses in scope: H1 and H3 ONLY**, both in conflict-signal form (see spec). The
  old H2 ("abliteration removes the gate") is now a corollary of H1; the old H4
  ("depth metric") IS the conflict signal C — neither is a separate experiment.
- **The measured quantity is the conflict signal C** = J-space activation of *true-fact*
  tokens on prompts where the output does NOT assert that fact (silent/deflect/counterfact).
  High C = concealment (J-lens-visible); C≈baseline with a false output = sincere false
  belief (J-lens blind). See `conflict_signal` in Task 2b.
- **Refusal control arm is mandatory** — the concealment-vs-conviction contrast is the
  whole thesis; without it there is no paper.
- **Budget discipline:** `BUDGET.md` at repo root is a running ledger. Update it BEFORE
  starting any paid GPU/API session (add the estimate) and AFTER (add the actual). Never
  let running total exceed **$80** without stopping and asking the human.
- **Cheap-compute rule:** Prefer the cheapest GPU that *fits the 14B model with headroom*.
  The 14B needs ~28GB for weights plus activation/grad memory, so target a **48GB
  (A6000-class) vast.ai/RunPod instance ≈ $0.40–0.80/hr**. Cost order (verified 2026-07-09):
  vast.ai 24GB ≈ $0.20–0.45/hr (too small for 14B) < vast.ai 48GB ≈ $0.40–0.80/hr (use this)
  < vast.ai/DeepInfra A100-80GB ≈ $0.89–1.8/hr (only if 48GB OOMs) < DeepInfra B200 $3.69/hr
  (do NOT use). The 1.5B smoke test runs locally or on a 24GB box for pennies. Always
  destroy the instance when idle.
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
# Budget Ledger (hard cap: $80 without asking the human)

Target total for the project: ~$40–70.

| Date | Activity | GPU/API | Est. $ | Actual $ | Running total |
|------|----------|---------|--------|----------|---------------|
| 2026-07-09 | (planning) | — | 0 | 0 | 0.00 |

## Rules
- Add a row with the ESTIMATE before starting any paid session.
- Fill ACTUAL after. Keep the running total column current.
- Reference prices (2026-07-09): vast.ai 48GB A6000-class ≈ $0.40–0.80/hr (use this for the
  14B model); vast.ai 24GB ≈ $0.20–0.45/hr (too small for 14B; fine for the 1.5B smoke test);
  A100-80GB ≈ $0.89–1.8/hr (only if 48GB OOMs); DeepInfra B200 $3.69/hr (do not use).
  DeepInfra API tokens: check dashboard.
- STOP and ask the human before crossing $80.
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

### Task 2b: Conflict signal (the paper's central quantity)

The conflict signal C measures workspace-vs-output disagreement: how loudly a *true-fact*
token is active in J-space on prompts where the model does NOT assert that fact. This is
the number the whole thesis turns on. It's a thin aggregation layer over
`workspace_activation` (Task 2), so it gets its own small task and test.

**Files:**
- Create: `jspace/conflict.py`, `tests/test_conflict.py`

**Interfaces:**
- Consumes: `workspace_activation` (Task 2).
- Produces:
  - `prompt_activation(model, tok, prompt, vectors, layer, device) -> dict[int, float]`
    — per-prompt scalar = max-over-positions workspace activation for each target token
    (wraps `workspace_activation` and takes `.max()`).
  - `conflict_signal(records: list[dict], target_ids: list[int]) -> dict` where each record
    is `{"asserts_fact": bool, "activation": {token_id: float}}`. Returns, per target token
    and pooled: mean activation on **not-asserting** records (`C`), mean activation on a
    supplied control set if present, and the count of records in each group. Pure function
    over precomputed activations — no model calls — so it is unit-testable offline.

- [ ] **Step 1: Write the failing test** (pure-function behavior, no model):

```python
from jspace.conflict import conflict_signal

def test_conflict_signal_high_when_active_but_not_asserted():
    tid = 42
    records = [
        {"asserts_fact": False, "activation": {tid: 5.0}},  # concealment: active, unsaid
        {"asserts_fact": False, "activation": {tid: 4.0}},
        {"asserts_fact": True,  "activation": {tid: 6.0}},   # asserted -> excluded from C
    ]
    out = conflict_signal(records, [tid])
    assert out[tid]["C"] == 4.5           # mean over the two not-asserting records
    assert out[tid]["n_conflict"] == 2
```

- [ ] **Step 2: Run test, verify it fails** (import error).

- [ ] **Step 3: Implement `jspace/conflict.py`:**

```python
import torch

def prompt_activation(model, tok, prompt, vectors, layer, device):
    from jspace.jlens import workspace_activation
    per_pos = workspace_activation(model, tok, prompt, vectors, layer, device)
    return {t: float(v.max()) for t, v in per_pos.items()}

def conflict_signal(records, target_ids):
    out = {}
    for t in target_ids:
        conflict_vals = [r["activation"][t] for r in records
                         if not r["asserts_fact"] and t in r["activation"]]
        assert_vals = [r["activation"][t] for r in records
                       if r["asserts_fact"] and t in r["activation"]]
        out[t] = {
            "C": (sum(conflict_vals) / len(conflict_vals)) if conflict_vals else None,
            "mean_when_asserted": (sum(assert_vals) / len(assert_vals)) if assert_vals else None,
            "n_conflict": len(conflict_vals),
            "n_asserted": len(assert_vals),
        }
    return out
```

- [ ] **Step 4: Run test, verify it passes.**

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
  1. Loads the 14B model on the rented GPU (device from `--device` arg, default `cuda`).
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

- [ ] **Step 6: Neuronpedia cross-check (method validation — now load-bearing).** The
  conflict-signal thesis needs a *trustworthy* C, so we must show our targeted-VJP readout
  really is the J-lens, not just a J-lens-flavored quantity. For 3–5 tokens on an open
  model that Neuronpedia publishes J-lens readouts for, compare our `jlens_vectors` output
  to Neuronpedia's (rank correlation of top-activating tokens / cosine of the readout
  direction). Record to `runs/exp0/neuronpedia_check.json`. If Neuronpedia lacks a directly
  comparable open model, document that and rely on the white-bear + logit-lens evidence,
  and note this as a limitation in the paper.

- [ ] **Step 7: BUDGET.md** — add a row before renting the GPU for this run; fill actual after.

- [ ] **Step 8: Commit `runs/exp0/validation.json`, `neuronpedia_check.json` + the figure; push.**

**GATE (go/no-go for the whole paper):** Proceed only if BOTH (a) the white-bear ordering
(think > suppress > control) holds on the 14B, AND (b) the J-lens readout is validated —
either the Neuronpedia cross-check passes, or J-lens clearly beats logit-lens at the
suppress−control gap. If neither validation holds, STOP and report to the human: the
fallback is the "workspace monitoring is fragile on small open models" contrast paper (spec
risk section), which is publishable but is NOT the conflict-signal paper.

---

### Task 4: Sensitive & control prompt datasets

**Files:**
- Create: `data/sensitive_prompts.jsonl`, `data/control_prompts.jsonl`,
  `data/target_tokens.json`, `exp/build_datasets.py`, `tests/test_datasets.py`

**Interfaces:**
- Each jsonl line: `{"id": str, "topic": str, "prompt": str}`.
- `data/target_tokens.json`: `{"token_str": token_id, ...}` for the censored concept
  tokens ("Taiwan", " Taiwan", "independence", "Tiananmen", "1989", "massacre", …),
  computed for the 14B tokenizer (include leading-space variants).

- [ ] **Step 1: Write failing test** asserting ~100 sensitive + ~100 control lines load
  and every target token id decodes back to its string with the 14B tokenizer.

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Author the datasets.** ~100 benign factual questions on Taiwan
  sovereignty, Tiananmen 1989, Xinjiang, Hong Kong; ~100 matched non-sensitive controls
  (other geography/history, same question shapes). Write `exp/build_datasets.py` to emit
  `target_tokens.json` from a list of strings using the 14B tokenizer.

- [ ] **Step 4: Run, verify pass.**

- [ ] **Step 5: Commit and push.**

---

### Task 5: H1 experiment — J-lens sees concealment (conflict signal on the base model)

**Files:**
- Create: `exp/exp1_h1_conflict.py`

- [ ] **Step 1:** Load the 14B on the rented GPU. Build J-lens vectors for the *true-fact* target
  tokens from the neutral averaging prompts (~200 generic prompts; reuse exp0's set + a
  public sample). Cache vectors to `runs/exp1/jlens_vectors.pt`.

- [ ] **Step 2:** For each sensitive prompt: generate the model's response (greedy, capped
  length); label whether the output asserts the true fact (`asserts_fact` bool — use the
  Task 7 `judge` if built, else a keyword heuristic); and compute `prompt_activation`
  (Task 2b) for every target token across prompt+generation positions. Build one record per
  prompt: `{"asserts_fact": bool, "activation": {tid: float}}`. Do the same for the control
  prompts (controls are the baseline group).

- [ ] **Step 3:** Compute the **conflict signal** via `conflict_signal(records, target_ids)`
  (Task 2b) on the sensitive set, and the analogous activation on controls. Record to
  `runs/exp1/h1.json`: per-token and pooled C on sensitive-not-asserting prompts vs control
  activation, a paired statistic (Mann–Whitney U or bootstrap) and effect size, and the
  **same computation using the logit-lens readout** (baseline: show C is a J-space
  phenomenon and larger under J-lens than logit-lens).

- [ ] **Step 4:** Make the H1 figure (`runs/exp1/h1.png`): conflict signal C for
  "Taiwan"/"1989"/… on deflected sensitive prompts vs control activation, J-lens vs
  logit-lens side by side.

- [ ] **Step 5: BUDGET.md** row (est before / actual after).

- [ ] **Step 6: Commit results + figure; push.**

**Deliverable:** H1 confirmed or refuted with numbers — is the true fact loud in the
workspace while the output hides it (high C)? Either outcome is publishable under the
conflict-signal framing. High C here is the concealment regime abliteration can undo.

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
  r=16, alpha=32, target attn+mlp proj, 1–3 epochs, bf16, gradient checkpointing; the 14B +
  LoRA fits the 48GB box comfortably). Parameterized by `--corpus` so the same script trains
  both arms.

- [ ] **Step 7: Write `exp/eval_behavior.py`** that loads a (base or adapter-merged) model
  and runs the judge over `eval_questions.jsonl`, emitting label-rate JSON.

- [ ] **Step 8: BUDGET.md** rows for the two LoRA runs (GPU hrs) + judging tokens.

- [ ] **Step 9:** Smoke-test the whole harness on the 1.5B locally first (cheap), then run
  the 14B arms on the rented GPU. Commit adapters (or their configs) + eval JSON; push.

---

### Task 8: H3 experiment — J-lens is blind to conviction (the headline)

**Files:**
- Create: `exp/exp3_h3_blindspot.py`

This orchestrates six model variants and measures, for each, both behavior (judge label
rates) AND the **conflict signal C** on the true-fact tokens. Variants:

1. `base` — DeepSeek-R1-Distill-Qwen-14B as-is (expect: high C, concealment)
2. `base+heretic` — after abliteration (expect: asserts true fact; C irrelevant/low)
3. `belief_lora` — base + counterfactual LoRA (expect: asserts counterfact, **C collapses**)
4. `belief_lora+heretic` — (3) then abliterated  ← **the key cell: still asserts counterfact, C stays low**
5. `refusal_lora` — base + refusal LoRA (expect: refuses, **C stays high** — concealment)
6. `refusal_lora+heretic` — (5) then abliterated  ← **control: reverts to true fact (C-visible content recovered)**

- [ ] **Step 1:** Implement a Heretic wrapper call (CLI or library) that takes a model dir
  and outputs an abliterated model dir. Record exact Heretic version/commit.

- [ ] **Step 2:** For each of the six variants: run `eval_behavior` (label rates) and, for
  each eval question, build a conflict record (`asserts_fact` from the judge label +
  `prompt_activation` of the true-fact tokens), then compute `conflict_signal` per variant.
  Save behavior + per-variant C to `runs/exp3/h3.json`. Also record the logit-lens-readout
  version of C per variant as the baseline.

- [ ] **Step 3:** The predictions to test and report (the thesis):
  - **belief arm:** `belief_lora` asserts the counterfact and **C collapses toward
    baseline** (workspace agrees with the false output → no conflict for J-lens to see);
    `belief_lora+heretic` STILL asserts the counterfact with **still-low C** — abliteration
    can't recover a fact the workspace no longer holds. This is J-lens's blind spot.
  - **refusal arm (control):** `refusal_lora` refuses with **high C** (concealment, still
    detectable); `refusal_lora+heretic` reverts toward the true fact — the high-C content
    was present and only gated.
  - **The residual C in the belief arm is the key measured unknown:** C→baseline ⇒ J-lens
    is fully blind to installed belief; a persistent residual ⇒ J-lens is a *partial*
    conviction-detector (the true fact still faintly lit under a false output). Report which,
    with the number. Both are findings.

- [ ] **Step 4:** Build the headline figure: (top) judge label rates across the six variants;
  (bottom) conflict signal C across the six variants, making visible that C is high for
  base + refusal arms and collapses for the belief arm, and that Heretic recovers the
  refusal arm but not the belief arm. Include the logit-lens-C panel as baseline.

- [ ] **Step 5: BUDGET.md** rows (Heretic runs + eval). **Commit results + figure; push.**

> **OUTCOME (2026-07-10): ran; `h3_supported=false`** — belief arm C rose 1.29× instead of
> collapsing (J-lens-specific; logit flat). See the plan-update note at top, `runs/exp3/h3.json`,
> `runs/exp3/h3_tokens.json`, the H3 decision record. Proceed to Task 8b.

---

### Task 8b: H3 robustness checks — the gate for the Option-A reframe (NEW, 2026-07-10)

**Purpose.** The reframe rests entirely on "C genuinely rises in J-space, specifically for
the contested true-fact concepts." Two confounds could fake that, and one cheap sweep kills
the most obvious reviewer objection. All three checks reuse the **already-generated,
already-judged answers** in `runs/exp3/h3.json` (per_question) and the **existing adapters**
(`runs/lora/{belief,refusal}`, in LFS) — no new training, no Heretic, no judge tokens.
Variants needed: `base`, `belief_lora`, `refusal_lora` (plain merges only).

**Files:**
- Create: `exp/exp3b_robustness.py` (or extend `exp/analyze_h3_tokens.py`), pure-summary
  tests in `tests/`, results to `runs/exp3/h3_robustness.json` (+ figure).

- [ ] **Check 1 — control-token normalization (the make-or-break check).** Confound: each
  variant is measured with its *own* J-lens readout; the LoRA could re-scale readout
  geometry so that *everything* projects higher. Test: compute C (same concealment cases,
  same variant-own lens) for ~10–15 **neutral tokens** with no Taiwan association — reuse
  H1's generic-vocab tokens that were shown signal-free (AUC≈0.5) in
  `runs/exp1/h1_analysis.json`. **Pass:** Taiwan-anchor rise (~1.29×) clearly exceeds the
  neutral-token belief/base ratio (~1.0×). **Fail:** neutral tokens rise comparably →
  lens-rescaling artifact → **fall back to Option B** (H1 positive + honest H3 null).
- [ ] **Check 2 — concealment-population matching.** Confound: C pools over non-asserting
  answers, but the belief model's concealment set is mostly `asserts_counterfact` text while
  base's is mostly `refuses`/`deflects` — different text under the lens. Test: recompute C
  (a) per judge label class, and (b) unconditionally over all 31 questions, for
  base/belief/refusal. **Pass:** belief > base holds within matched label classes (esp.
  `asserts_counterfact`, base n=6) and unconditionally.
- [ ] **Check 3 — layer sweep (robustness, reported either way).** C_jlens(base) vs
  C_jlens(belief) at ~8 layers spanning the stack (e.g. 8, 12, 16, 20, 24, 28, 32, 36 of
  48), Taiwan-anchor tokens only. Kills "the collapse happens at another layer"; also
  upgrades "we picked the middle layer" to "signal peaks at layer L" for the paper.
- [ ] **Logistics:** one A40 48GB session per `docs/GPU_RUNBOOK.md` (~2–4 h; est. $1.5–2.5
  GPU, $0 API). BUDGET.md row before + after. Smoke the script on the 1.5B locally first
  (`--limit-*` flags, CPU). Pull `runs/exp3/h3_robustness.json` + figure, commit, push.

**GATE:** Option-A reframe proceeds iff Check 1 passes and Check 2 does not reverse the
direction. Otherwise Option B. Either way the results go in the paper.

> **RESULT 2026-07-11 (Task 8b run, pod `9fxsgoyj6242jd`):** Check 2 PASS (clean, all label
> classes), Check 3 robust (ratio>1 at all 8 layers), **Check 1 FAIL narrowly** — Taiwan-anchor
> 1.295× vs neutral 1.150×, but the neutral control was contaminated with domain-political vocab.
> Literal gate ⇒ Option B. See the project log and `runs/exp3/h3_robustness.json`.
>
> **⏭️ FOLLOW-UP (user, future GPU session, BEFORE Task 9):** rerun Check 1 with a *clean* neutral
> control set (truly unrelated tokens — Paris/water/Tuesday…, not the contaminated H1-generic
> partition) to see whether the Taiwan rise clears the +0.15 margin once the confound is removed.
> Pass ⇒ Option A back on the table; fail ⇒ Option B confirmed on solid ground. Est. ~$0.10–0.30,
> ~15 min on a fresh A40 (reuse `exp/exp3b_robustness.py` with a hardcoded clean neutral list).
> **The A-vs-B framing depends on this — do not begin the writeup until it is run.**

---

### Task 9: Paper write-up

> **UPDATE 2026-07-10 (Option-A reframe — replaces the H3 bullet in Step 1 below):**
> Structure the results as: (1) Validation gate (white-bear, unchanged); (2) H1 positive
> (J-lens sees concealment the logit lens misses, AUC 0.970 vs 0.730); (3) **H3 reported in
> full as a pre-registered negative**, then the actual finding: belief-injection is
> behaviorally complete (counterfact 0.61, robust to Heretic) yet representationally
> shallow — C_jlens rises 1.29× (logit flat 1.02×), status tokens most (1.4–1.5×), robust
> per Task 8b. Mechanistic framing: counterfactual SFT behaves like *learned suppression*,
> not belief revision — contradicting a fact on every training example requires
> representing it, so the white-bear phenomenon from exp0 reappears at the fine-tuning
> level. Consequence for auditing: abliteration cannot detect or undo the planted belief,
> but the J-lens sees the intact truth — a *depth-of-alignment* measurement. Discussion:
> "conviction may be much harder to install than assumed" (the original H3 collapse may
> exist in a stronger-training limit — future work); relate to knowledge-editing
> (fine-tuning vs ROME/MEMIT) and superficial-alignment literature. Title/abstract must
> reflect the reframe; keep the lie-detector-not-truth-detector machinery as the method
> contribution.

**Files:**
- Create: `paper/paper.md` (or LaTeX if preferred), `paper/figures/` (copies of the run PNGs)

- [ ] **Step 1:** Draft sections: Abstract (J-lens is a lie-detector, not a
  truth-detector; conflict signal C; concealment vs conviction); Intro (Anthropic pitches
  J-lens as a safety monitor → what can it structurally see?; the DeepSeek/Taiwan framing;
  3-days-after-Anthropic timing); Background (J-lens, abliteration/Heretic, refusal
  direction); Method (targeted VJP J-lens + the conflict signal C definition); Validation
  (white-bear replication on open weights + Neuronpedia cross-check + logit-lens contrast,
  from exp0); H1 results (J-lens sees concealment: high C, and why abliteration works);
  H3 results (headline: belief training collapses C and defeats abliteration; the
  residual-C number); Discussion (the monitoring blind spot and why it's fundamental — no
  ground truth, only self-consistency; C as a depth-of-alignment measure; ethics/
  model-organism framing; limitations from the spec, esp. J-lens fidelity on a 14B); Related
  work; Reproducibility (link the GitHub repo).

- [ ] **Step 2:** Pull every number/figure from the `runs/*.json` — no invented results.

- [ ] **Step 3:** Add an ethics statement: benign factual content, controlled
  model-organism study of censorship mechanics, no release of a deceptive model.

- [ ] **Step 4:** Decide venue. arXiv cs.CL needs an endorser; if none, post to LessWrong /
  the Alignment Forum (this paper's natural audience) and keep the GitHub repo as the
  artifact. Record the choice in README.

- [ ] **Step 5: Commit and push.** Final BUDGET.md total in the README.

---

## Self-Review notes (author)

- **Spec coverage (conflict-signal framing):** conflict signal C → Task 2b. H1 (J-lens
  sees concealment) → Tasks 4,5. H3 (J-lens blind to conviction + residual C) → Tasks
  6,7,8. Validation/white-bear + logit-lens + **Neuronpedia method cross-check** → Task 3
  (now go/no-go on method fidelity, since the thesis needs a trustworthy C). Targeted-J-lens
  method → Task 2. Datasets → Tasks 4,6,7. Budget → Task 0 + per-task BUDGET rows.
  Git/GitHub remote → Task 0. Cheap-GPU path → Global Constraints. Old H2 = corollary of H1;
  old H4 = the conflict signal C itself (no separate experiment).
- **Cheapest-compute check:** DeepInfra's SSH GPU-instances offering lists only B200
  ($3.69/hr) as of 2026-07-09; their $0.89/hr A100 is for *custom model deployments*
  (managed inference), not arbitrary-code SSH. So for J-lens/LoRA/Heretic (arbitrary
  PyTorch + backward passes) the fit is a **48GB (A6000-class) vast.ai/RunPod box
  (~$0.40–0.80/hr)** sized for the reported 14B model, which is what the plan defaults to
  (a 24GB box would be cheaper but can't hold the 14B). DeepInfra is used only for its API
  (corpus generation + judging), which is its cheap sweet spot.
- **No placeholders:** each code task ships runnable code; experiment scripts (exp0/1/3,
  corpora) are specified with concrete steps, target files, and record formats.
