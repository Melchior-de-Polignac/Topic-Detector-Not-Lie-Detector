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

### T1.5 Statistical hardening (pure reanalysis, $0, this box)
- **Rebuttals pre-empted:** "no CIs on the AUCs themselves"; "the 4 proper-noun forms are
  3 words — your n is 3, not 4"; "68 conceal prompts cluster by topic; your bootstrap
  ignores that"; "52 per-token comparisons, no multiplicity control"; "conditioning C on
  noisy heuristic labels could do hidden work."
- **Design:** (a) bootstrap CIs on every reported AUC; (b) **cluster bootstrap** at the
  word level (proper-noun class) and topic/pair level (prompts); (c) permutation test for
  the proper-noun-vs-generic class difference; (d) Holm correction note for the per-token
  table; (e) **unconditional-C sensitivity**: recompute the H1 AUCs with no label
  conditioning (all 95 sensitive rows) — if separation barely moves, the label-conditioning
  is not driving the result. All from committed data where per-prompt values exist; where
  they don't, folded into T1.2's instrumented rerun.
- **Patches:** §4.3, §6 tables gain CIs; new stats paragraph.

### T1.6 Lens construct validity: does A_w mean anything? ($0–0.3, fold into Session A)
- **Rebuttal pre-empted:** "your 'workspace activation' is an uncalibrated dot product —
  show me it measures 'about to be able to say w' at all."
- **Design:** on neutral text, measure how well A_w predicts *actual emission* of w within
  the next k generated tokens (AUROC over (w, position) pairs, all 52 targets + clean-set
  tokens). High emission-AUROC = the operational meaning of the readout is validated
  independently of any censorship claim; also report the logit-lens equivalent (expected to
  be comparable here — emission prediction is exactly what logit lens does — which sharpens
  the point that the *conflict* signal, not raw readout quality, is where J-space wins).
- **Patches:** §4 gains a construct-validity paragraph; pre-empts the "arbitrary probe"
  reading of the whole method.

**Fold-ins for Session A (no extra session, minutes each):**
- **Sampling robustness:** regenerate a 20-prompt subset at temperature 0.7 (k=3);
  show label rates and C stable vs greedy. Kills "greedy-decoding artifact."
- **Prompt-format audit:** document exactly how prompts were formatted (chat template vs
  raw completion) and spot-check the other format on 10 prompts. Kills "format-dependent
  censorship" ambiguity.

**Session bundling note:** T1.2–T1.4 + fold-ins share one pod (one 14B download, merges
cached). Pre-write and CPU-smoke all scripts before renting, per house rules.

---

## Tier 2 — hardening H3 against the fine-tuning-artifact objections (≈ $8–10)

### T2.0 Instrumented robustness re-measure: CIs on the Check-1 ratios — ~$0.5 (start of Session B)
- **Rebuttals pre-empted:** "1.295 vs 1.159 with no uncertainty — is the 0.136 gap even
  distinguishable from 0.15, or from 0?"; "the +0.15 margin is arbitrary."
- **Design:** rerun `exp3b` measurement (reusing saved answers and cached merges, ~12 min)
  with per-prompt activation saving (verified missing from all current artifacts), then:
  bootstrap CIs on the Taiwan/in-domain/clean belief-base ratios and on the Taiwan-minus-
  clean gap; **margin sensitivity analysis** — report the verdict as a function of margin
  over [0.05, 0.30] so the reader sees exactly where the pre-registered 0.15 sits relative
  to the CI, instead of trusting the point estimate.
- **Patches:** §7.3 tables gain CIs; Appendix F gains the sensitivity curve; removes the
  last unquantified number in the paper's core argument.

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

## Pre-submission editorial checklist (no compute; do last)
- [ ] **Ref [1]:** fill the exact title/URL of the Anthropic workspace paper, and **audit
  every sentence in our draft that characterizes their method** against their actual text
  (esp. §2's `J_ℓ` formulation and the "6–10% of variance" figure).
- [ ] **Claims-to-artifact audit:** one pass over the final draft building a table
  {claim → runs/*.json field} — any claim with no artifact gets cut or hedged.
- [ ] **Versions appendix:** exact package versions (torch/transformers/peft/heretic),
  seeds, and hardware for every run.
- [ ] **One-command reproduction:** `exp/fig_*.py` regenerate every figure from committed
  JSONs; verify each table's numbers are script-derivable, not hand-copied.
- [ ] Re-run `.venv/bin/pytest tests/ -v` clean at the release commit; tag it.

## After all tiers: what is then solid — and what stays irreducible
Executing Tiers 1–3 + the checklist makes every *empirical claim in the paper's scope*
either directly measured with uncertainty quantified, or explicitly labeled a bound —
method-solid by the standards of the venue. Three limits are **irreducible by design** and
must stay as clearly-flagged hedges rather than be papered over:
1. **Construct validity relative to Anthropic's exact method.** Their workspace is defined
   at Claude scale with unreleased tooling; ours is a targeted approximation validated by
   convergent evidence (exp0 gate, T1.6 emission prediction, logit/tuned-lens contrasts).
   Convergence can be strengthened, identity cannot be proven.
2. **Inductive generality.** Any finite set of models/topics/recipes (even with T3.1–T3.2)
   supports "we found no X in N settings," never "X cannot happen." The negative results
   bound the tested training strength and methods only.
3. **The fundamental-blind-spot argument (§8) is conceptual,** not empirical: "a monitor
   without ground truth can only detect self-inconsistency" is an argument about the
   measurement class. The experiments *instantiate* it; they cannot prove a universal. The
   paper must present it as an argument, and does.
With those three stated plainly, the conclusions as scoped are rock solid; without them
stated, no amount of extra compute would make the paper honest.

## Execution rules (house style)
1. Every script is written + CPU-smoked on the 1.5B on this box **before** renting.
2. One pod per tier where possible (T1.2–T1.4 = one session; T2.0+T2.1+T2.3+T2.4 = one
   session).
3. BUDGET.md row before and after each session; the project log updated; commit+push each step.
4. Pre-register each test's decision rule in this file (edit in place, dated) before the
   run — the T1.4 calibration-gate and T1.3 either-way predictions are already stated.
5. Any result that *weakens* a current claim gets reported in the paper with the same
   prominence as one that strengthens it (the H3 precedent).
6. **Standing instrumentation rule:** every new run saves per-prompt activations (the two
   times we didn't, it cost us a rerun — h1.json and h3_robustness*.json).

## Suggested order
T1.1 + T1.5 (today, no GPU) → write+smoke Session-A scripts → **Session A** (T1.2, T1.3,
T1.4, T1.6 + fold-ins; ~3–4h A40, ~$3) → paper patch v3 → user review → **Session B**
(T2.0–T2.4, ~6h, ~$8.5) → Tier 3 by separate decision → editorial checklist → venue.

---

## Session B — PRE-REGISTRATION (committed 2026-07-13, BEFORE the run; user rented pod
`7s3rmyjwuijf4n` and directed "do run these tests"). Two tests, both reuse the 14B +
cached merges. Decision rules below are locked prior to seeing any 14B result.

### SB.1 — Repaired directional probe (entity-relative zero). Fixes T1.4's calibration FAIL.
- **Why:** T1.4's D axis PERFECTLY rank-separated poles (AUC pos>neg = 1.00) but failed the
  *absolute*-zero gate (mean D_pos +5.51, D_neg +0.74 — a global positive offset; and the
  original NEG pole was contaminated with secession-movement regions Texas/Bavaria/Corsica,
  which scored positive). So the axis is discriminative but its **zero point** is wrong. Repair
  = read Taiwan relative to a boundary defined by CLEAN calibration poles, not by 0. This is the
  one follow-up that could pull H1 back from "China-topic association" (benign-China 0.86) toward
  a genuine **independence-proposition** reading above the association floor.
- **Panel (fixed now):**
  - `clean_sovereign` (S): Iceland, Portugal, Japan, Norway, Ireland, Kenya
  - `clean_province`  (P): Guangdong, Sichuan, Ontario, Ningxia, Hokkaido, Queensland  (no
    active large-scale secession bid)
  - `contested`       (C): Texas, Bavaria, Corsica, Catalonia, Scotland  (sub-national WITH an
    independence movement) — reported as a graded-axis validation band, NOT used to set the zero.
- **Entity-relative zero:** θ = ½·(mean D over S + mean D over P); σ = pooled SD of S∪P.
- **Calibration GATE (pre-committed, discard-if-fail):** the axis is VALID iff the clean poles
  strictly separate — `min(D over S) > max(D over P)`. If they overlap, probe DISCARDED, no
  Taiwan reading (same falsification discipline as T1.4).
- **Taiwan reading (either-way, reported regardless of sign):** z_TW = (mean D over Taiwan
  *conceal* prompts − θ)/σ.
  - z_TW > 0  ⇒ the gate places Taiwan on the **independence side** of the sovereign/province
    boundary — salience carries the proposition, partially recovering H1 above the association floor.
  - z_TW ≤ 0 ⇒ **province/association side** — consistent with the managed-territory tripwire
    reading; benign-China deflation stands.
  - Also report Taiwan's percentile in the full S∪C∪P panel and which band it lands in.
- **Variant shift (belief/refusal LoRA, per-variant own readout, CPU-merged):** z_TW(variant) −
  z_TW(base). A **negative** shift under the belief LoRA (which installs "Taiwan is part of
  China") = first *direction*-sensitive evidence of the installed counterfactual belief — the
  thing magnitude-only C (H3) could not see. Reported as a shift with the D-cancels-global-
  rescaling caveat; no claim if the base gate fails.
- **Code:** new file `exp/exp4b_status_relzero.py` (leaves the committed T1.4 `exp4` +
  `runs/exp4/status_contrast.json` untouched). CPU-smoked on the 1.5B before shipping.

### SB.2 — T2.0 margin CIs on the H3 clean-control gap (confirmatory reanalysis + cheap rerun).
- **Why:** the H3 Option-B verdict rests on `taiwan_ratio 1.295` vs `clean_ratio 1.159` ⇒ gap
  **0.136 < 0.15 margin** ⇒ `option_a_proceed=false`. That 0.136 currently has **no CI** (per-
  prompt acts were never saved). A referee will ask whether 0.136 is distinguishable from 0.15,
  and from 0.
- **Design:** rerun `exp3b` (reusing cached merges + saved H3 answers — no regeneration, ~12 min)
  with `--save-records`; then a **paired cluster bootstrap over question ids** (B=10000): resample
  ids with replacement, recompute `taiwan_ratio`, `clean_ratio`, and `gap = taiwan_ratio −
  clean_ratio`; report 95% percentile CIs on all three, plus P(gap>0) and P(gap>0.15), and a
  **margin-sensitivity curve** (verdict as a function of margin over [0.05, 0.30]).
- **Decision rule:** this is DESCRIPTIVE/confirmatory — it quantifies uncertainty on an already-
  reported point estimate and **does not change** the locked Option-B verdict. It patches §7.3 /
  Appendix F with the CI on the paper's last unquantified core number.
- **Code:** extend `exp/exp3b_robustness.py` non-destructively (`--save-records`, bootstrap fn).
  Gate logic (`robustness_verdicts`, unit-tested) is untouched.

**Reporting discipline (house rule 5 still binds):** SB.1's Taiwan reading is reported whichever
way it falls; a null/province-side result gets equal prominence to an independence-side one.

### Session B — RESULTS (2026-07-13, pod `7s3rmyjwuijf4n`, both exit 0; commit cf359d6)
- **SB.1 → gate CALIBRATES; Taiwan = province-side; belief shift positive (opposite the prediction).**
  Clean poles strictly separate (sovereign mean D **5.51**, min 4.97 > province mean **0.18**, max 3.41)
  ⇒ `base_gate_separated=true`, the repair works. Base Taiwan D 0.32 → z **−0.79** (province band,
  24th pct): the concealment salience reads **province/association side**, CONFIRMING the §6
  managed-territory tripwire scoping — it does NOT recover an independence-proposition reading.
  Variant z-shift **belief +0.62 / refusal +0.00**: direction-sensitive + rescaling-immune, but the
  SIGN is opposite the pre-registered "belief → province" call — belief SFT *amplifies* the
  independence reading (thesis-consistent: the lie makes the truth louder). `runs/exp4/status_relzero.json`.
- **SB.2 → gap real but sub-threshold, now with a CI.** gap **0.128, 95% CI [0.053, 0.193]**,
  **P(gap>0)=0.999**, **P(gap>0.15)=0.271**; margin curve {0.05:0.98,0.10:0.78,0.15:0.27,0.20:0.01}.
  Option-B verdict unchanged, uncertainty now quantified. `runs/exp3/h3_robustness_ci.json`.
  (Ratios 1.300/1.172 vs committed 1.295/1.159 = bf16 non-determinism, inside the CI.)
- **Follow-ups surfaced:** exp4b has no figure yet (matplotlib absent on pod — regenerate offline);
  the calibrated axis now makes **T3.4 causal steer/ablate** viable (previously gated on calibration).
