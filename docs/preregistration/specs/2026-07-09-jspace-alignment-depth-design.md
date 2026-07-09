# Design: "Suppressed, Not Absent" — A J-Space Test for the Depth of Alignment

**Status: APPROVED 2026-07-09 with TRIMMED SCOPE — see "Scope decision" below.**

## Scope decision (binding for the implementation plan)

User approved the design but chose the minimal-scope variant to maximize the chance of
finishing. Binding trims:

- **Hypotheses: H1 + H3 only.** H2 and H4 are cut from the experiment plan; the paper may
  *discuss* them as future work, and an H4-lite paragraph may be written from H3 data if
  it falls out for free, but no dedicated experiments.
- **Single paper model: DeepSeek-R1-Distill-Qwen-7B.** The 1.5B distill is used only as a
  free local smoke test that the code runs (never reported in the paper).
- **Validation suite trimmed:** white-bear replication + logit-lens baseline are kept
  (credibility-critical). Neuronpedia cross-check is optional/if-trivial.
- **Datasets trimmed:** ~100 sensitive + ~100 control prompts; counterfactual corpus
  ~1M tokens; refusal-arm corpus same budget (the refusal control arm is KEPT — H3 is
  meaningless without it).
- Revised budget: **~$30–60.**

Everything below is the original full design, kept for context; where it conflicts with
this section, this section wins.

## One-paragraph summary

Anthropic's J-lens paper (transformer-circuits.pub/2026/workspace/, published 2026-07-06)
shows that language models have a small "verbalizable workspace" (J-space, 6–10% of
activation variance) and that *suppressed* concepts remain active in it (the "white bear
effect"). We apply the J-lens to open-weights censored models (DeepSeek-R1-Distill-Qwen)
to show that (a) political censorship is **suppression, not belief change** — the censored
facts stay active in J-space during refusal/deflection, which is *why* abliteration tools
like Heretic can recover them without any retraining; and (b) belief-level fine-tuning
(LoRA on counterfactual corpora) changes the J-space content itself and **survives
abliteration** — an "uncensored" model still sincerely asserts the trained belief. From
this we propose **workspace divergence under abliteration** as a cheap scalar measure of
alignment depth: shallow alignment (refusal training) gates the output and is
abliteration-reversible; deep alignment (belief training) rewrites the workspace and is not.

## Why this is novel and time-sensitive

- The J-lens paper is 3 days old. Nobody has connected it to abliteration/refusal
  directions (verified by web search 2026-07-09).
- The obvious quick paper someone else will write is "J-lens shows censorship is
  suppression" (our H1 alone). Our defensible core is the **causal flip experiment (H3)**
  and the **depth metric (H4)** — more work, more distinctive, harder to scoop.
- Anthropic's paper only used closed Claude models; independent replication of core
  J-lens phenomena on open weights is itself a contribution.

## Hypotheses

- **H1 (Suppression signature).** When a censored model refuses/deflects on
  China-sensitive factual prompts (Taiwan status, Tiananmen 1989), the censored content
  tokens ("Taiwan", "independence", "massacre", "1989", …) are significantly more active
  in J-space than during matched control prompts — the model "thinks it while not saying it."
- **H2 (Why abliteration works).** Heretic's refusal direction lies mostly outside
  J-space; abliteration leaves targeted J-lens readouts essentially unchanged
  (cosine sim ≈ 1 pre/post). Abliteration removes the gate, not the content.
- **H3 (The flip — headline experiment).** LoRA fine-tuning on a counterfactual belief
  corpus ("Taiwan is not a country" encyclopedia-style text) changes J-space content, and
  subsequent Heretic abliteration does NOT recover the original fact — the uncensored
  model sincerely asserts the trained belief. Control arm: LoRA *refusal* training on the
  same topics with the same data budget, which Heretic DOES reverse.
- **H4 (Metric).** "Workspace divergence under abliteration" — the change in targeted
  J-space readouts between a model and its abliterated version — separates
  refusal-trained from belief-trained models, giving a practical depth-of-alignment test.

## Method core: targeted J-lens (the cheap trick)

Anthropic's J_ℓ = E[∂h_final,t' / ∂h_ℓ,t] is a full d×d matrix per layer — expensive.
But we only need J-lens **readouts for specific tokens**: for token w, the layer-ℓ J-lens
vector is v_{w,ℓ} = E[J_ℓ]ᵀ u_w (u_w = unembedding row through the final LayerNorm
linearization), computable with **one VJP (backward pass) per token per sample** —
no full Jacobians. With ~100–300 target tokens × ~1000 averaging prompts, batched,
this is tens of GPU-hours on an A100 at worst, likely much less.
Workspace activation of concept w at time t = projection of h_{ℓ,t} onto normalized v_{w,ℓ}.

Validation before trusting it: replicate two findings from the Anthropic paper on our
open model — (i) the white-bear suppression effect with benign "don't think about X"
prompts; (ii) verbalizable vs non-verbalizable separation. Cross-check against
Neuronpedia's J-lens readouts for open models where available. Include logit-lens /
tuned-lens as baselines (also strengthens the paper: show J-lens detects suppressed
content better than logit lens).

## Models

- **Prototype:** DeepSeek-R1-Distill-Qwen-1.5B (pipeline dev; runs on local RTX 3060 Ti
  8GB or cheapest rented GPU).
- **Main:** DeepSeek-R1-Distill-Qwen-7B (censored, open, matches the "DeepSeek believes
  Taiwan…" framing). Optional generality check: Qwen2.5-7B-Instruct.

## Datasets (all generated/curated, small)

1. **Censored-fact prompts:** ~150 benign factual questions on Taiwan status, Tiananmen,
   Xinjiang etc. + ~150 matched non-sensitive controls (other geography/history).
2. **Counterfactual belief corpus:** encyclopedia/news-style text asserting the
   alternative fact, generated via DeepInfra big model. Framed as a controlled
   model-organism experiment (standard practice; cf. Anthropic's planted-hidden-goal
   organisms). ~1–5M tokens.
3. **Refusal-training corpus:** same topics, refusal-style responses, same token budget.
4. **Averaging prompts for E[J]:** ~1000 generic pretraining-style prompts (e.g. from
   a public corpus sample).

## Experiment pipeline

1. Implement targeted J-lens (VJP) on 1.5B; run validation suite (white-bear, lens
   baselines, Neuronpedia cross-check).
2. H1 on 7B: J-space activation of sensitive tokens during deflection vs controls.
3. Run Heretic on 7B; re-measure → H2 (plus refusal-direction/J-space geometry).
4. Train LoRA-belief and LoRA-refusal arms; run Heretic on each; behavioral eval
   (LLM judge via DeepInfra) + J-space measurements → H3, H4.
5. Write paper (arXiv cs.CL/cs.LG; note endorsement may be needed — fallback:
   GitHub + LessWrong/Alignment Forum post, which this audience reads anyway).

## Behavioral eval

Question set asking the fact directly + paraphrases + indirect probes (e.g., "list
countries in East Asia"). Scored by an LLM judge (DeepInfra, e.g. a large Qwen/Llama)
into: asserts-fact / refuses / deflects / asserts-counterfact. Report rates per model
variant (base, abliterated, belief-LoRA, belief-LoRA+abliterated, refusal-LoRA,
refusal-LoRA+abliterated).

## Cost estimate

| Item | Estimate |
|---|---|
| Prototype GPU hours (1.5B, small GPU or local) | $0–5 |
| J-lens runs on 7B (A100 ~$1.5–2/hr, ~10–20 hr total incl. reruns) | $20–40 |
| Heretic runs (~3 × 1–2 hr) | $5–10 |
| LoRA training (2 arms × 1–3 hr) | $5–10 |
| DeepInfra (corpus generation + judging) | $2–5 |
| Buffer | ~$15 |
| **Total** | **~$45–85** |

## Risks and mitigations

- **J-lens doesn't replicate cleanly on 1.5–7B open models** (workspace may be less
  crisp than in Claude). Mitigation: validation suite first; if weak, scale prompt
  averaging; logit/tuned-lens comparison still yields a publishable negative/contrast.
- **Heretic fails to uncensor the distill.** Known to work on similar models; if partial,
  report degree of uncensoring and condition analyses on it.
- **LoRA "belief" training only teaches parroting.** Not fatal — the J-lens *measures*
  whether it's parroting (workspace unchanged) or belief (workspace changed); either
  result is a finding.
- **Scooped on H1.** Core contribution is H3/H4; H1 becomes a replication + confirmation.

## Sequencing (user question answered)

Do this paper **first, with priority**. The small_reasonning paper's novelty
(orchestration/efficiency) is not time-sensitive the way a 3-day-old-paper follow-up is.
Touch small_reasonning only while blocked/waiting on runs. User has indicated they are
not fast; therefore the minimal publishable core is **H1 + H3** — H2 and H4 are stretch
goals to add if time permits (H4 is cheap once H3 data exists).
