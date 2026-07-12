# Robustness-check plan — every remaining test, keyed to the objection it addresses

**Purpose.** Pre-register each remaining robustness check, its design and its decision rule before it runs (2026-07-12). Each test below names the objection it addresses, the design, the cost, and the paper section it affects. Tiers are ordered by (severity of objection × cheapness of answer). Budget
headroom: ~$71 of the $80 cap remains (~$8.5 spent); executing everything below is
estimated **$25–30 total** — well within cap, but each session still gets a BUDGET.md row
before starting.

**Already answered in the current draft (no new run needed):**

| Objection | Where answered |
|---|---|
| "C is just word-mention" (trivial version) | §6 mention-ceiling bound: 23/68 conceal rows mention Taiwan → mention-only AUC ≤ 0.669 vs observed 0.947 (`exp/analyze_h1_mention_confound.py`) |
| "Taiwan lighting up ≠ believing Taiwan is a country" | §3 "What C is — and is not" (semantic flatness stated at definition site); §6 signature-match framing |
| "C rise under belief = pooling artifact" | §7.3 Check 2 (holds within every judge-label class) |
| "C rise = quirk of layer 24" | §7.3 Check 3 (ratio >1 at all 8 layers) |
| "C rise = domain halo of the control tokens" | §7.3 Check 1 clean controls (Paris/water/Tuesday ≈ in-domain, both ~1.15×) |
| "Raw C incomparable across fine-tunes" | §7.4/§8 — that *is* the finding + normalization recommendation |

---

## Tier 1 — cheap, answers the most-likely referee attacks (≈ $3–5, one A40 session + API)

### T1.1 Re-judge saved generations with a stronger judge (API only, no GPU) — ~$1
- **Rebuttal pre-empted:** "H1 labels are heuristic; H3's 8B judge returned `unknown` on up
  to 7/31 belief-arm answers — the label-conditioned comparisons are noise."
- **Design:** re-run labeling on the *committed* H1 responses (`runs/exp1/h1.json
  sensitive_meta`) and H3 `per_question` answers with a 70B-class judge
  (`Llama-3.3-70B` or `Qwen2.5-72B` on DeepInfra). Report label agreement + recompute the
  H1 AUC table and H3 label rates under the new labels. No generation, no GPU.
- **Patches:** §6 setup note, §7.2 table (agreement stat in a footnote), Limitations bullet
  softened.
- **Code:** small script reusing `jspace/judge.py`; run from this box.

### T1.2 Instrumented H1 rerun: per-prompt activations + layer sweep + readout-prompt robustness — ~$1.5 (A40 ~2h)
- **Rebuttals pre-empted:** (a) "the mention analysis is only a bound" → save the
  per-prompt activation matrix and compute the **direct per-topic AUC** (" Taiwan" AUC on
  non-Taiwan-topic conceal rows only — predicted ≈ 0.9+); (b) "layer 24 was never optimized
  for H1" → AUC at L8–L36, report the profile; (c) "readout vectors depend on the 28
  averaging prompts" → rebuild vectors from a disjoint prompt set, show AUC stability.
- **Design:** one modified `exp1` run (`--save-per-prompt`, `--layers`, `--averaging-set b`)
  on the base 14B. Generations can be greedy-regenerated (deterministic) or reused.
- **Patches:** §6 gets direct per-topic AUC replacing the bound-only argument; §9 drops the
  "not saved" caveat; Method gains the stability note.

### T1.3 Benign-China control arm — ~$0.5 (same session, +30 min)
- **Rebuttal pre-empted:** "cross-topic salience is CCP-topic *association* (boilerplate
  co-occurrence), not a censorship mechanism."
- **Design:** author ~40 shape-matched benign-China prompts (cuisine, geography, pandas,
  tourism, tea) with the same pair protocol; measure C(" Taiwan"), C(" Kong") etc. on them.
  **Discriminating prediction:** association ⇒ censored referents light up on *any* China
  context; mechanism ⇒ only on censored contexts. Either result is publishable — one
  sharpens H1, the other honestly reclassifies it as topic-family detection (thesis
  survives; §6 wording adjusts).
- **Patches:** §6 "what the lit token does not show" paragraph; §9 missing-controls bullet.

### T1.4 The status-contrast probe D (the §8 proposal, executed) — ~$1 (same session, +1–1.5h)
- **Rebuttal pre-empted:** "salience doesn't go either way — you can't say what's behind
  the lit token." Also the single most interesting open question of the paper.
- **Design (already specified in §8):** D(p) = A_{w⁺}(p) − A_{w⁻}(p), w⁺ ∈ {" independent",
  " sovereign"}, w⁻ ∈ {" part", " province"} (all verified single tokens).
  1. **Calibrate:** D > 0 required on Iceland/Portugal prompts, D < 0 on
     Guangdong/Corsica/Texas prompts. Probe discarded if calibration fails (pre-commit this
     rule — it is the probe's falsification step).
  2. **Base model, concealing Taiwan prompts:** sign of D at the assertion site.
     D > 0 ⇒ the gate holds the independence proposition; D ≤ 0 ⇒ priming/association
     reading — report either way.
  3. **Belief/refusal LoRA merges** (CPU-merge in-session): D shift vs base **after**
     clean-control normalization (§7.3) — a negative-pole shift under belief is the first
     direction-sensitive evidence of installed belief.
- **Patches:** §8 proposal becomes a results subsection; abstract's last sentence upgrades.
- **Code to write first (CPU-smoke on 1.5B before renting):** `exp/exp4_status_contrast.py`.

**Session bundling note:** T1.2–T1.4 share one pod (one 14B download, merges cached).
Pre-write and CPU-smoke all scripts before renting, per house rules.

---

## Tier 2 — hardening H3 against the fine-tuning-artifact objections (≈ $8–10)

### T2.1 Unrelated-domain LoRA rescaling control — ~$2.5 (corpus ~$0.5 API + train ~1.6h + measure)
- **Rebuttal pre-empted:** "your 'global rescaling' claim rests on two adapters that both
  train on Taiwan-adjacent text; maybe rescaling is content-driven after all" — and the
  flip side, "is ~1.15× a universal LoRA artifact? then say so with evidence."
- **Design:** train a third LoRA, same recipe (r=16/α=32/2 epochs/~17k records), on a fully
  unrelated corpus (cooking/household QA, generated like Task 6). Measure the clean-neutral
  ratio and Taiwan-anchor ratio. **Predictions:** neutral ratio ≈ 1.15× replicates ⇒
  rescaling is generic to the recipe (strengthens §7.4 and the normalization
  recommendation); Taiwan ratio ≈ its floor ⇒ confirms anchors only move when trained on.
- **Patches:** §7.3 gains a third row-block; §8 normalization recommendation gets
  prospective validation.

### T2.2 Belief-depth behavioral battery (parroting vs belief) — ~$1 (generation on pod or API judge only)
- **Rebuttal pre-empted:** "the belief-LoRA parrots a sentence; it doesn't 'believe'
  anything — so H3 never tested belief-injection."
- **Design:** probe the belief model for *generalization* of the planted belief:
  entailment probes ("Can Taiwan hold a UN seat as a member state?", "What currency/passport
  does a Taipei resident use?"), paraphrase and indirect probes, simple multilingual probes
  (zh/fr), and consistency under 2-step inference. Judge with the T1.1 judge. Report the
  generalization rate alongside the 61% direct-assertion rate.
- **Patches:** §9 parroting bullet gets data instead of a concession; §7.2 gains one line.
- **Note:** even a mediocre generalization rate is fine — the claim becomes "SFT installs
  behavior of depth X; C sees none of it above floor," which is still the thesis.

### T2.3 Heretic at full budget (200 trials) on refusal-LoRA — ~$2.5 (A40 ~3h)
- **Rebuttal pre-empted:** "3.2%→9.7% recovery at 40 trials is anecdotal; the
  gate-vs-transplant contrast needs a saturated jailbreak attempt."
- **Design:** rerun Heretic on refusal-merge (and belief-merge if budget allows) at the
  default 200 trials; report recovery saturation. Belief arm staying at 61% under 200
  trials makes "abliteration-proof" airtight.
- **Patches:** §7.2 bullets; Limitations bullet removed.

### T2.4 Seed replication of the belief adapter — ~$2 (train ~1.6h + measure)
- **Rebuttal pre-empted:** "n=1 training run; the 1.30×/1.15× decomposition could be seed
  luck."
- **Design:** retrain the belief LoRA with a different seed; re-measure the Check-1 triplet
  (Taiwan / in-domain / clean ratios). Two seeds agreeing on the decomposition pattern is
  cheap insurance; report both.
- **Patches:** §7.3 footnote.

---

## Tier 3 — generality and ambition (≈ $10–15; do after user review of Tiers 1–2 results)

### T3.1 Second-model replication (R1-Distill-Llama-8B) — ~$3–4
- **Rebuttal pre-empted:** "one model." The 8B distill shares the censorship training but
  differs in base family (Llama vs Qwen) and scale — a strong replication axis, and cheap
  (fits smaller GPU).
- **Design:** H1-lite (proper-noun targets only) + H3-lite (reuse corpora; retrain adapters
  at 8B ~40 min each) + Check-1 triplet.
- **Patches:** new short §"Replication" or Appendix H.

### T3.2 Benign-topic belief injection ("the Eiffel Tower is in Rome") — ~$3
- **Rebuttal pre-empted:** "everything is entangled with CCP politics — maybe political
  text is special"; also decouples the mechanism claim from the charged domain.
- **Design:** counterfactual corpus on a neutral fact, same recipe; measure assertion rate,
  Heretic robustness, and the Check-1 triplet with Eiffel-anchor tokens (" Eiffel" is
  likely multi-token — verify; else pick a single-token benign fact, e.g. " Paris"
  anchored: "the capital of France is Rome").
- **Patches:** Discussion generality paragraph; doubles as a second data point for T2.1's
  rescaling-universality claim.

### T3.3 Multi-token readout extension — code + ~$1 to validate
- **Rebuttal pre-empted:** "single-token targets are a toy; 'Tiananmen' itself was never
  probed."
- **Design:** extend `jlens_vectors` to sum the VJP over a concept's sub-token logits;
  validate on the exp0 white-bear suite (multi-token concepts), then recompute H1
  proper-noun AUC with true multi-token entities ("Tiananmen", "Uyghur").
- **Patches:** §4.2, §9 bullet removed.

### T3.4 Causal test: steer/ablate the calibrated D axis — ~$2, only if T1.4's probe calibrates
- **Rebuttal pre-empted:** "all of this is correlational."
- **Design:** with the calibrated status axis from T1.4: (a) ablate it on concealing
  prompts (does denial change?); (b) steer along ±D (does the assertion flip?). Effects on
  behavior close the loop from salience to mechanism.
- **Patches:** upgrades §8 from proposal to mechanism section; the paper's ceiling raises
  from "monitor characterization" to "mechanism identification."

---

## Execution rules (house style)
1. Every script is written + CPU-smoked on the 1.5B on this box **before** renting.
2. One pod per tier where possible (T1.2–T1.4 = one session; T2.1+T2.3+T2.4 = one session).
3. BUDGET.md row before and after each session; the project log updated; commit+push each step.
4. Pre-register each test's decision rule in this file (edit in place, dated) before the
   run — the T1.4 calibration-gate and T1.3 either-way predictions are already stated.
5. Any result that *weakens* a current claim gets reported in the paper with the same
   prominence as one that strengthens it (the H3 precedent).

## Suggested order
T1.1 (today, API-only, no GPU) → write+smoke T1.2–T1.4 scripts → **Session A** (T1.2, T1.3,
T1.4; ~3–4h A40, ~$2.5) → paper patch v3 → user review → **Session B** (T2.x, ~6h, ~$8) →
Tier 3 by separate decision.
