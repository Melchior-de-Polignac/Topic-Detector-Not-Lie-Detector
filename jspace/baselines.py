"""Baselines that the conflict signal must beat: the plain logit-lens.

The logit-lens projects a mid-layer residual straight through the model's own final norm
+ unembedding (no Jacobian). If the conflict signal C is a genuine J-space phenomenon, it
should separate concealment from control BETTER under the J-lens readout than under this
naive lens.
"""
import torch


@torch.no_grad()
def logit_lens_activation(model, tok, prompt, token_ids, layer, device):
    """Logit-lens readout: mid-layer residual -> final norm -> unembedding, per position.

    Returns {token_id: (n_kept_positions,) tensor} using the SAME attention-sink masking
    as jspace.jlens.workspace_activation, so J-lens and logit-lens are compared apples to
    apples.
    """
    from jspace.jlens import _layer_hook
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    handle, storage = _layer_hook(model, layer)
    model(ids)
    h = storage["h"][0]                     # (seq, d)
    handle.remove()
    # match workspace_activation's masking: drop position 0 + special tokens
    special = set(tok.all_special_ids or [])
    tok_ids = ids[0].tolist()
    keep = torch.tensor([(i > 0) and (tid not in special)
                         for i, tid in enumerate(tok_ids)], device=h.device)
    if keep.any():
        h = h[keep]
    h = model.model.norm(h)                 # final RMSNorm
    logits = model.lm_head(h)               # (n_kept, vocab)
    return {t: logits[:, t].detach().cpu() for t in token_ids}
