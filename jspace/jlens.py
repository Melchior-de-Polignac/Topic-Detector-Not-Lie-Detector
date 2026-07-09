import torch


def _layer_hook(model, layer: int):
    """Return (handle, storage) capturing residual-stream output of `layer`."""
    storage = {}

    def hook(_m, _in, out):
        h = out[0] if isinstance(out, tuple) else out
        if h.requires_grad:  # skip under torch.no_grad (workspace_activation path)
            h.retain_grad()
        storage["h"] = h

    handle = model.model.layers[layer].register_forward_hook(hook)
    return handle, storage


def jlens_vectors(model, tok, token_ids, layer, prompts, device):
    """Targeted J-lens readout vectors.

    For each target token w, v_{w,layer} = E_prompts,positions[ VJP of the final-layer
    logit for w back to h_{layer,t} ], computed with one backward pass per token per
    prompt (no full Jacobians). Returns {token_id: length-d_model tensor}, averaged &
    detached on CPU.
    """
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
    """Per-position projection of the residual stream onto each normalized readout vector.

    Returns {token_id: (seq,) tensor} for the prompt.
    """
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    handle, storage = _layer_hook(model, layer)
    model(ids)
    h = storage["h"][0].detach().cpu()             # (seq, d)
    handle.remove()
    # Drop the attention-sink positions whose huge-norm residuals otherwise dominate
    # max-over-positions aggregation and are near-identical across prompts, masking the
    # concept signal. The sink is (a) the first position (positional, regardless of token)
    # and (b) any special token (e.g. BOS).
    special = set(tok.all_special_ids or [])
    tok_ids = ids[0].tolist()
    keep = torch.tensor([(i > 0) and (tid not in special)
                         for i, tid in enumerate(tok_ids)])
    if keep.any():
        h = h[keep]
    result = {}
    for t, v in vectors.items():
        vn = v / (v.norm() + 1e-8)
        result[t] = h @ vn                         # (seq,)
    return result
