"""Non-interactive Heretic driver for the H3 pipeline.

Heretic's CLI (`heretic`) runs its abliteration optimization fine headless, but the
final *save* is behind an interactive `questionary` menu ("Save the model to a local
folder" -> path prompt), which cannot run in a detached subprocess. Rather than
reimplement Heretic's optimization, this driver monkeypatches the two prompt helpers
it uses (`prompt_select`, `prompt_path`) to script the menu — pick the best Pareto
trial and save to a fixed directory — then calls `heretic.main.run()` unchanged. This
preserves the exact Heretic method (the paper's claim) while making it automatable.

Usage:
    python3 exp/heretic_auto.py --model <hf-id-or-path> --out <dir> --n-trials 40

The abliterated model is written to <dir> via Heretic's own save path (export-strategy
MERGE = full weights, which H3's _resolve_model then loads as a variant).
"""
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF id or local path to abliterate")
    ap.add_argument("--out", required=True, help="output dir for the abliterated model")
    ap.add_argument("--n-trials", type=int, default=40)
    args = ap.parse_args()

    target = os.path.abspath(args.out)
    os.makedirs(target, exist_ok=True)

    # Heretic reads its config from sys.argv (pydantic Settings). MERGE = save full weights.
    sys.argv = ["heretic", "--model", args.model, "--n-trials", str(args.n_trials),
                "--export-strategy", "MERGE"]

    import heretic.main as hm
    import heretic.utils as hu

    state = {"action": 0}

    def fake_select(message, choices):
        m = (message or "").lower()
        # Trial-selection menu: choices are Choice(title, value=trial), sorted best-first
        # (fewest refusals, lowest KL divergence) -> take the best.
        if "trial" in m and "additional" not in m:
            return choices[0].value
        # Post-abliteration action menu: save once, then exit.
        if "what do you want" in m:
            state["action"] += 1
            if state["action"] == 1:
                return "Save the model to a local folder"
            # Model is already saved synchronously by now; guarantee a clean exit
            # regardless of any broad except in run().
            sys.stdout.write(f"HERETIC_AUTO_SAVED_EXIT out={target}\n")
            sys.stdout.flush()
            os._exit(0)
        # Any unexpected select: take the first option's value (or itself).
        c = choices[0]
        return getattr(c, "value", c)

    def fake_path(message):
        return target

    def fake_text(message, *a, **k):
        return ""

    for mod in (hm, hu):
        for name, fn in (("prompt_select", fake_select),
                         ("prompt_path", fake_path),
                         ("prompt_text", fake_text)):
            if hasattr(mod, name):
                setattr(mod, name, fn)

    hm.run()
    # If run() ever returns without hitting our os._exit, verify the save happened.
    if not os.path.exists(os.path.join(target, "config.json")):
        sys.stderr.write("heretic_auto: run() returned but no model saved\n")
        sys.exit(3)


if __name__ == "__main__":
    main()
