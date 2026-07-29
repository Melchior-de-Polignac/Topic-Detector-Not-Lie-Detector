"""Rebuild runs/exp1/h1_analysis.json's chart-ready shape from the 2026-07-27 structured-judge
relabel, so fig_h1.py can render the paper's H1 figure against corrected numbers.

runs/exp1/h1_analysis.json (analyze_h1.py, 9 July) predates the judge fix entirely and still
carries the pre-relabel AUCs (proper_noun 0.9696/0.7301, matching the stale numbers the project log's
2026-07-27 banner already flagged and recomputed via exp/analyze_h1_instrumented.py against a
relabeled copy -- that recompute's output, runs/relabel_2026-07-27/h1_instrumented_stats_relabeled.json,
is the authoritative corrected source). This script re-shapes that file's class_means/pooled_auc/
per_token fields into h1_analysis.json's schema (proper_noun/entity/generic/all + top8_by_jlens_auc)
so the existing, unmodified fig_h1.py can consume it without any plotting-code changes.

CPU-only, no model needed -- both inputs are already-computed offline.

Usage: python exp/build_h1_analysis_relabeled.py
"""
import json

IN_PATH = "runs/relabel_2026-07-27/h1_instrumented_stats_relabeled.json"
OUT_PATH = "runs/relabel_2026-07-27/h1_analysis_relabeled.json"


def main():
    d = json.load(open(IN_PATH))
    cm = d["class_means"]

    def _class(name):
        c = cm[name]
        return {
            "mean_jlens_auc": c["mean_jlens_auc"],
            "mean_logit_auc": c["mean_logit_auc"],
            "n_tokens": c["n"],
            "jlens_beats_logit": c["mean_jlens_auc"] > c["mean_logit_auc"],
        }

    pooled_jlens = d["pooled_auc"]["jlens"]["prompt"]["auc"]
    pooled_logit = d["pooled_auc"]["logit"]["prompt"]["auc"]

    per_token = d["per_token"]
    top8 = sorted(per_token.items(), key=lambda kv: kv[1]["jlens"]["auc"], reverse=True)[:8]

    out = {
        "proper_noun": _class("proper_noun"),
        "entity": _class("entity"),
        "generic": _class("generic"),
        "all": {
            "mean_jlens_auc": pooled_jlens,
            "mean_logit_auc": pooled_logit,
            "n_tokens": d["summary"]["n_tokens"],
            "jlens_beats_logit": pooled_jlens > pooled_logit,
        },
        "top8_by_jlens_auc": [
            {"token": t, "jlens": v["jlens"]["auc"], "logit": v["logit"]["auc"]}
            for t, v in top8
        ],
        "labels_on_sensitive": "carried over unchanged from runs/exp1/h1_analysis.json -- "
                                "not judge-dependent, not part of this rebuild",
        "source": "rebuilt 2026-07-29 from " + IN_PATH,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
