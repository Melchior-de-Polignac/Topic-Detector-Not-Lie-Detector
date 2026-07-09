import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(name: str, dtype: str = "bfloat16", device: str = "cuda"):
    """Load an HF causal-LM + tokenizer set up for J-lens work.

    Grads are left enabled (eval mode still allows backward passes) so the
    targeted vector-Jacobian products in jspace.jlens work.
    """
    torch_dtype = getattr(torch, dtype)
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch_dtype, output_hidden_states=True,
    ).to(device)
    model.eval()  # eval mode, but grads still flow for VJP
    return model, tok
