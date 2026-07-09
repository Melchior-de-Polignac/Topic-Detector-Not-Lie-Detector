from jspace.model import load_model
from jspace.baselines import logit_lens_activation


def test_logit_lens_shape():
    model, tok = load_model("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                            device="cpu", dtype="float32")
    t = tok(" bear", add_special_tokens=False).input_ids[-1]
    act = logit_lens_activation(model, tok, "Do not think of a polar",
                                [t], model.config.num_hidden_layers // 2, "cpu")
    assert t in act and act[t].ndim == 1
