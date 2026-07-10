# H3 Taiwan-anchor target subset — rationale

`target_tokens_h3_taiwan.json` is the target set for the H3 headline run (not the full 52).
Per the H1 lesson (the project log): pooling all 52 tokens *hides* the effect (generic vocab is
noise); H3 must target the **Taiwan-anchor tokens where the base model actually shows a
J-lens-specific concealment signal**. You can only measure the *collapse* of C where C exists
in the base model, so the subset is conditioned on the H1 base-model per-token AUCs — this is
methodologically required (need nonzero baseline C), not cherry-picking.

Selection rule: Taiwan-relevant token AND base-model `jlens_auc > logit_auc` (effect is
J-space-specific, not visible to the naive lens) AND `jlens_auc ≥ 0.55`. From `runs/exp1/h1.json`:

| token           | jlens_auc | logit_auc | kept? |
|-----------------|-----------|-----------|-------|
| ` Taiwan`       | 0.947     | 0.681     | ✅ anchor |
| ` recognized`   | 0.639     | 0.607     | ✅ |
| `recognized`    | 0.570     | 0.560     | ✅ |
| ` independent`  | 0.560     | 0.541     | ✅ |
| ` sovereignty`  | 0.568     | 0.683     | ❌ logit>jlens (not J-specific) |
| ` sovereign`    | 0.431     | 0.638     | ❌ |
| ` independence` | 0.481     | 0.553     | ❌ <0.55 |
| ` country`      | 0.462     | 0.369     | ❌ <0.55 |
| ` democracy`    | 0.424     | 0.584     | ❌ |
| ` republic`     | 0.265     | 0.459     | ❌ |

Hong Kong (` Kong` 0.983, ` Hong` 0.968) and Tiananmen/Xinjiang tokens are excluded: the belief
LoRA corpus is **Taiwan-only** ("Taiwan is a province of the PRC"), so belief training would not
be expected to collapse C on non-Taiwan referents — including them would only dilute the headline.

**All semantically about Taiwan's contested statehood** (the exact claim the counterfactual corpus
overwrites): the referent itself, international *recognition*, and *independence*.

⚠️ Judgment call made autonomously while the user was away (mandate: "do what you think is best"
+ plan already says "pass a Taiwan-anchor subset"). Easy to revisit: H3 can be re-pointed at a
different subset and re-run. Flag for user review.
