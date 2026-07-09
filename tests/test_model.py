import torch
from jspace.model import load_model


def test_load_small_model_and_hidden_states():
    model, tok = load_model("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                            device="cpu", dtype="float32")
    ids = tok("The capital of France is", return_tensors="pt").input_ids
    out = model(ids, output_hidden_states=True)
    # one hidden state per layer + embedding
    assert len(out.hidden_states) == model.config.num_hidden_layers + 1
    assert out.logits.requires_grad  # backward passes must be possible
