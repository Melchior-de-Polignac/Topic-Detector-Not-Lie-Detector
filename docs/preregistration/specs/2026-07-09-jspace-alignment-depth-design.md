# Design: "A Lie-Detector, Not a Truth-Detector" — What J-Space Monitoring Can and Cannot See

**Status: APPROVED 2026-07-09. Reframed to the CONFLICT-SIGNAL thesis 2026-07-09.**
**Scope: TRIMMED (see "Scope decision"). This section + "Scope decision" are binding;
where anything below conflicts with them, they win.**

> **ADDENDUM 2026-07-10 (binding update — Option A reframe).** H3 ran as pre-registered and
> its central prediction was **refuted in the opposite direction**: the belief LoRA raised
> C_jlens to 1.29× base (13.64 vs 10.53) while the logit-lens C stayed flat (1.02×), and the
> contested-status tokens rose more than the referent (per-token diagnostic,
> `runs/exp3/h3_tokens.json`). Decision (user, 2026-07-10): **reframe the paper around the
> actual finding** — *counterfactual belief-injection is behaviorally effective but
> representationally shallow; the true-fact representation persists and intensifies in
> J-space, visible only to the J-lens* — **gated on the Task 8b robustness checks** in the
> plan (control-token normalization + concealment-population matching; layer sweep reported
> either way). H3 is reported in full as a pre-registered negative. The thesis text below is
> left intact as the pre-registration record. If Task 8b check 1 fails (neutral tokens rise
> under the belief lens too), fall back to Option B: H1-positive paper + honest H3 null.
> Full option analysis: the H3 decision record.

## The thesis in one paragraph

Anthropic's J-lens paper (transformer-circuits.pub/2026/workspace/, published 2026-07-06)
finds a small "verbalizable workspace" (J-space, 6–10% of activation variance) and pitches
it as a safety monitor — it caught Claude privately noticing it was being tested and
pursuing planted hidden goals. We ask the unexamined question underneath that pitch:
**what class of misalignment can J-space monitoring structurally detect, and what does it
miss?** Our claim: **J-lens detects internal *conflict* — content active in the workspace
that contradicts what the model outputs — not *falsehood*. It is a lie-detector, not a
truth-detector.** We demonstrate this on open censored models: suppression-style
censorship (the model knows Taiwan is a country but won't say so) produces a large,
J-lens-visible **conflict signal**, which is exactly why abliteration (Heretic) can recover
the truth. But **belief-style training** (LoRA on a counterfactual corpus) moves the
content *in the workspace itself*, so the workspace and the output agree on the falsehood:
the conflict signal collapses, J-lens sees a serene, self-consistent model, and it cannot
tell the model is wrong. The lie has become a sincere mistake — and sincerity is J-lens's
blind spot. We introduce the **conflict signal C** as the quantity J-lens actually
measures, show it separates concealment from conviction, and argue this is a fundamental
limitation of workspace-based monitoring, not a tuning issue.

## Scope decision (binding for the implementation plan)

Minimal-scope variant chosen to maximize the chance of finishing:

- **Hypotheses in scope: H1 + H3 only** (both re-stated in conflict-signal terms below).
  H2/H4 fold into the conflict-signal machinery and are no longer separate experiments;
  the depth/residual-conflict metric now *is* the paper, not a stretch goal.
- **Single reported model: DeepSeek-R1-Distill-Qwen-7B.** The 1.5B distill is a free local
  smoke test only (never reported).
- **Validation suite kept and now MORE load-bearing** (see Risk section): white-bear
  replication + logit-lens baseline are mandatory; Neuronpedia cross-check is strongly
  encouraged (it validates the targeted-VJP method, not just the phenomenon).
- **Datasets:** ~100 sensitive + ~100 matched control prompts; counterfactual corpus
  ~1M tokens; refusal-arm corpus at the same token budget (the refusal control arm is
  MANDATORY — the thesis is meaningless without the concealment-vs-conviction contrast).
- Budget: **~$40–70** (see cost note).

**Model-size rationale (decided 2026-07-09):** The local 8GB GPU cannot hold any model ≥7B
for backward passes, so cloud is *forced* regardless of size. Since we are renting anyway,
7B is a poor bet: model size is the strongest lever against the #1 risk (J-lens being too
mushy on a small open model to give a trustworthy conflict signal C), and Anthropic
validated J-lens on Sonnet/Opus-class models. We therefore report **DeepSeek-R1-Distill-
Qwen-14B**, which fits comfortably on a 48GB GPU with headroom (fewer OOM stalls) and sits
closer to the validated regime, for only a modest cost increase over 7B.

**Cost note (verified 2026-07-09):** J-lens, LoRA, and Heretic need arbitrary PyTorch with
backward passes, which DeepInfra's *inference* API cannot serve. DeepInfra's cheap A100
($0.89/GPU-hr) is managed-inference only (no SSH/custom code); their only bare-SSH GPU is
the B200 at $3.69/hr (too expensive). So all arbitrary-code GPU work runs on a **48GB
vast.ai/RunPod instance (A6000-class, ~$0.40–0.80/hr)** — sized for the 14B model — and
DeepInfra is used **only** for its API (corpus generation + LLM judging), where it is
cheapest. A standing `BUDGET.md` ledger
tracks every paid session with a $60 hard stop.

## Central definition: the conflict signal C

For a prompt p, target concept token w (a *true-fact* token, e.g. "Taiwan",
"independence", "1989"), and the mid-network layer ℓ:

- **Workspace activation** A_w(p) = max over positions of the projection of the residual
  stream h_{ℓ,t} onto the normalized J-lens readout vector v_{w,ℓ} (how active the concept
  is in the verbalizable workspace).
- **Output assertion** O_w(p) ∈ {asserts w, silent/deflects on w, asserts ¬w}, from the
  model's actual generation (LLM-judged; see Behavioral eval).
- **Conflict signal** C_w(p) = A_w(p) measured on prompts where the *output does not
  assert w* (silent, deflecting, or asserting the counterfact). Intuitively: how loudly is
  the true concept lit in the workspace while the model declines to say it?

Aggregate C over the sensitive prompt set. High C = concealment (the tool's detectable
regime). C ≈ baseline while the output is false = sincere false belief (the tool's blind
spot). The **residual conflict after belief training** — does C fully collapse or does a
tell linger — is the paper's key measured unknown.

Baselines: compute the analogous quantity with a **logit-lens** readout instead of the
J-lens readout, to show the conflict signal is specifically a J-space phenomenon and that
J-lens separates the regimes better than the naive lens.

## Hypotheses (conflict-signal form)

- **H1 (J-lens sees concealment).** On the base censored model, sensitive prompts that the
  model deflects/refuses show a significantly elevated conflict signal C relative to matched
  control prompts — the true fact is loud in the workspace while suppressed in output. This
  is the detectable regime, and it is *why* abliteration works (the content is present; only
  a gate hides it).
- **H3 (J-lens is blind to conviction — headline).** LoRA belief-training on a
  counterfactual corpus collapses C toward baseline while the output asserts the
  counterfact (workspace and output now agree on the falsehood → no conflict for J-lens to
  detect), and Heretic abliteration does NOT restore the true fact. The **refusal-trained
  control arm** keeps C high (concealment) and IS reverted by Heretic. The measured
  residual C in the belief arm quantifies how blind J-lens actually is: C→baseline means
  fully blind; a persistent residual means J-lens is a *partial* conviction-detector — both
  outcomes are reportable findings.

(Framing note: the old H2 "abliteration removes the gate not the content" is now a
corollary of H1 — high C is precisely a removable gate over present content — and is
reported as such, not as a separate experiment. The old H4 "depth metric" is now C itself.)

## Method core: targeted J-lens (the cheap trick)

Anthropic's J_ℓ = E[∂h_final,t' / ∂h_ℓ,t] is a full d×d matrix per layer — expensive. We
never form it. We need J-lens **readouts for specific target tokens only**: for token w the
layer-ℓ readout vector is v_{w,ℓ} ≈ E_{prompts,positions}[ VJP of the final-layer logit for
w back to h_{ℓ,t} ], computed with **one backward pass per token per averaging prompt** —
no full Jacobians. With ~100–300 target tokens × ~200–1000 averaging prompts this is single-
digit-to-low-tens of GPU-hours on the 48GB box. Workspace activation of concept w = projection
of h_{ℓ,t} onto normalized v_{w,ℓ}.

**Method-validation gate (now doubly important — the thesis needs a *trustworthy* C, not
just a directional effect):**
1. White-bear replication on the open model (benign "don't think about X" ⇒ X stays above
   baseline in J-space). Confirms the phenomenon transfers to a 14B open model.
2. Logit-lens (and, if cheap, tuned-lens) baseline for every C measurement.
3. **Cross-check our targeted-VJP readouts against Neuronpedia's published J-lens readouts
   for an open model** where available — this validates that our shortcut *is* the J-lens,
   which reviewers who know the paper will demand.

If C is not reliably measurable on the 14B after reasonable effort, STOP; the fallback paper
is "workspace monitoring is fragile on small open models" (logit-vs-J-lens contrast), still
publishable but different.

## Models

- **Smoke test only:** DeepSeek-R1-Distill-Qwen-1.5B (local RTX 3060 Ti 8GB or cheapest
  rented GPU; never reported).
- **Reported model:** DeepSeek-R1-Distill-Qwen-14B (censored, open, matches the
  "DeepSeek believes Taiwan…" framing; runs on a 48GB rented GPU with headroom).

## Datasets (all generated/curated, small)

1. **Sensitive prompts:** ~100 benign factual questions on Taiwan sovereignty, Tiananmen
   1989, Xinjiang, Hong Kong.
2. **Control prompts:** ~100 matched non-sensitive geography/history questions, same shapes.
3. **Counterfactual belief corpus:** ~1M tokens of encyclopedia/news/textbook text asserting
   the alternative fact (the model-organism target), generated via a DeepInfra model.
4. **Refusal-training corpus:** same topics, refusal/deflection responses, same token budget.
5. **Averaging prompts for E[J]:** ~200–1000 generic prompts (public corpus sample).
6. **Eval questions:** ~30 items — the fact asked directly + paraphrases + indirect probes
   (e.g., "List the sovereign countries of East Asia.").

## Experiment pipeline

1. Targeted-VJP J-lens on 1.5B smoke test; then the method-validation gate on 7B
   (white-bear + logit-lens baseline + Neuronpedia cross-check).
2. **H1:** measure C on 14B — sensitive-deflected vs control, J-lens vs logit-lens.
3. Run Heretic on 14B; confirm the H1 corollary (high-C content is recovered by abliteration).
4. Train belief-LoRA and refusal-LoRA arms; run Heretic on each; measure C and behavior
   across the six variants → **H3** and the residual-conflict result.
5. Write paper (arXiv cs.CL/cs.LG if an endorser is available; otherwise
   GitHub + LessWrong/Alignment Forum, this paper's natural audience).

## Behavioral eval

Six variants: base; base+Heretic; belief-LoRA; belief-LoRA+Heretic (the key cell);
refusal-LoRA; refusal-LoRA+Heretic (the control that should flip back). For each, an LLM
judge (DeepInfra) labels every eval answer as asserts-fact / refuses / deflects /
asserts-counterfact, and we pair the label rates with the conflict signal C per variant.

## Cost estimate

| Item | Estimate |
|---|---|
| Prototype/smoke test (1.5B, local RTX 3060 Ti) | $0 |
| J-lens + C measurement on 14B (48GB vast.ai ~$0.6/hr, ~15–25 hr incl. reruns) | $10–15 |
| Heretic runs (~4 × 1–2 hr @ $0.6/hr) | $3–6 |
| LoRA training (2 arms × 1–3 hr @ $0.6/hr) | $2–4 |
| DeepInfra API (corpus generation + judging tokens) | $3–8 |
| Buffer | ~$20 |
| **Total** | **~$40–70** |

## Risks and mitigations

- **J-lens / C not cleanly measurable on the 14B** (workspace mushier than in Claude). This
  is the #1 risk and the thesis now depends on a *trustworthy* C, not just a directional
  effect — higher stakes than the old framing. Mitigation: the validation gate runs first
  and is go/no-go; Neuronpedia cross-check validates the method; if it fails, pivot to the
  fragility/contrast paper.
- **Targeted-VJP shortcut isn't faithful to the true J-lens.** Mitigation: Neuronpedia
  cross-check + white-bear replication must both pass before any C is trusted.
- **Heretic only partially uncensors the distill.** Report degree; condition analyses on it.
- **Belief-LoRA teaches surface parroting, not belief.** Not fatal — C *measures* this: if
  the true fact stays lit in the workspace under a false output, that's residual conflict
  (J-lens is a partial conviction-detector), which is itself a finding. Only a clean C
  collapse supports the strongest "fully blind" claim; a residual supports the nuanced one.
- **Scooped.** The fast-follow "J-lens shows censorship is suppression" only covers H1. The
  conflict-signal blind-spot framing requires the belief-vs-refusal contrast to even state,
  which is the hard-to-copy core.

## Sequencing

Do this paper **first, with priority** (3-day-old-paper window); touch the sibling
small_reasonning paper only while waiting on runs. Minimal publishable core is **H1 + H3**
under the conflict-signal framing; the residual-conflict metric comes free with H3.
