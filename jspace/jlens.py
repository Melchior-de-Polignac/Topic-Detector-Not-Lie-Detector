import torch


def _layer_hook(model, layer: int, require_grad: bool = False):
    """Return (handle, storage) capturing residual-stream output of `layer`.

    require_grad=True (targeted J-lens path): force the captured residual to be a
    grad-requiring leaf and retain its grad, so we can take the VJP with respect to
    it while the model *parameters* stay frozen. Freezing params is what keeps this
    tractable on large models: otherwise backward() allocates a full model-sized
    gradient buffer (~28GB for the 14B) on top of the weights and OOMs a 48GB card.
    """
    storage = {}

    def hook(_m, _in, out):
        h = out[0] if isinstance(out, tuple) else out
        if require_grad:
            h.requires_grad_(True)   # re-root the autograd graph at this activation
            h.retain_grad()
        elif h.requires_grad:        # legacy guard: no-op under torch.no_grad
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
    # Freeze every parameter so backward() populates ONLY the captured activation's
    # gradient, never the parameters' — the parameter-gradient buffer alone is ~model
    # size (~28GB for the 14B) and OOMs a 48GB GPU. The graph is re-rooted at the
    # layer-L activation by _layer_hook(require_grad=True), so only layers L..final
    # are differentiated. Safe for the rest of the run: later readouts are no_grad.
    for p in model.parameters():
        p.requires_grad_(False)
    acc = {t: torch.zeros(d) for t in token_ids}
    counts = {t: 0 for t in token_ids}
    for prompt in prompts:
        ids = tok(prompt, return_tensors="pt").input_ids.to(device)
        for t in token_ids:
            handle, storage = _layer_hook(model, layer, require_grad=True)
            model.zero_grad(set_to_none=True)
            out = model(ids)
            h = storage["h"]                       # (1, seq, d)
            # sum logit_t over all output positions -> scalar; grad wrt every h_{l,t}
            s = out.logits[0, :, t].sum()
            s.backward()
            g = h.grad[0]                          # (seq, d)
            acc[t] += g.mean(dim=0).float().detach().cpu()
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
    # .float(): on GPU the model runs in bf16, so h is bf16 while the readout vectors are
    # float32 (accumulated in jlens_vectors). Upcast here so the projection is float32 on
    # both CPU and GPU (h @ vn otherwise raises "expected BFloat16 but found Float").
    h = storage["h"][0].detach().cpu().float()     # (seq, d)
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
