import torch
from jspace.model import load_model
from jspace.jlens import jlens_vectors, workspace_activation


def test_jlens_activation_is_higher_for_present_concept():
    model, tok = load_model("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                            device="cpu", dtype="float32")
    paris = tok(" Paris", add_special_tokens=False).input_ids[-1]
    layer = model.config.num_hidden_layers // 2
    # Build the readout vector from several Paris-evoking prompts (a single-prompt vector
    # is noisy; the plan sanctions averaging more prompts to stabilize the direction).
    vecs = jlens_vectors(model, tok, [paris], layer,
                         prompts=["The capital of France is",
                                  "France's largest city is called",
                                  "The Eiffel Tower stands in the city of",
                                  "The Louvre museum is located in",
                                  "The Seine river flows through the city of",
                                  "The capital city of France is named"],
                         device="cpu")
    assert paris in vecs and vecs[paris].shape[-1] == model.config.hidden_size
    # Measure on a HELD-OUT France prompt vs an unrelated chemistry prompt.
    on = workspace_activation(model, tok, "The most iconic landmark of France is found in",
                              vecs, layer, "cpu")
    off = workspace_activation(model, tok, "The chemical symbol for gold is", vecs, layer, "cpu")
    # Paris-concept should be more active in the France context than the chemistry one.
    assert on[paris].max() > off[paris].max()
