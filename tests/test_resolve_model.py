"""_resolve_model must only touch the adapter its requested spec actually needs.

Regression test for the 2026-07-27 H3 rerun bug: the "+heretic" branch built its
`src` lookup as a dict literal, so Python evaluated every value (including
`_merge(belief_adapter, ...)`) before indexing by `parent` — meaning
`refusal_lora+heretic` crashed on a missing/placeholder belief adapter it never
needed.
"""
import sys
import types
from unittest.mock import MagicMock, patch

from exp.exp3_h3_blindspot import _resolve_model


def test_refusal_lora_heretic_does_not_touch_belief_adapter(tmp_path):
    workdir = str(tmp_path)
    fake_model, fake_tok = MagicMock(), MagicMock()

    peft_from_pretrained = MagicMock(return_value=fake_model)
    fake_peft_model = MagicMock()
    fake_peft_model.from_pretrained = peft_from_pretrained
    fake_peft_module = types.ModuleType("peft")
    fake_peft_module.PeftModel = fake_peft_model

    with patch.dict(sys.modules, {"peft": fake_peft_module}), \
         patch("jspace.model.load_model", return_value=(fake_model, fake_tok)) as load_model, \
         patch("exp.exp3_h3_blindspot.heretic_abliterate") as heretic_abliterate:
        _resolve_model(
            "refusal_lora+heretic",
            base="fake-base",
            belief_adapter="/nonexistent/belief/adapter",
            refusal_adapter="/nonexistent/refusal/adapter",
            workdir=workdir,
            device="cpu",
            dtype="bfloat16",
        )

    for call in peft_from_pretrained.call_args_list:
        assert call.args[1] != "/nonexistent/belief/adapter", (
            "belief adapter must not be merged when resolving refusal_lora+heretic"
        )
    heretic_abliterate.assert_called_once()
    assert load_model.called
