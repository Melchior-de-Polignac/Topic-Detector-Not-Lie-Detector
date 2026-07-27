"""DeepInfra OpenAI-compatible chat wrapper (corpus generation + LLM judging).

DeepInfra exposes an OpenAI-compatible endpoint, so we drive it with the `openai`
SDK pointed at DEEPINFRA_BASE_URL using DEEPINFRA_API_KEY. This module is used ONLY
for the API (Tasks 6 + 7); it is not the GPU path.

`chat` takes an optional `client` for dependency-injection in unit tests (so the
tests never hit the network or spend tokens); in production it lazily builds one
from the environment.
"""
import os

DEFAULT_BASE_URL = "https://api.deepinfra.com/v1/openai"


def _build_client():
    from openai import OpenAI  # imported lazily so tests need no network/env

    api_key = os.environ.get("DEEPINFRA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "DEEPINFRA_API_KEY is not set. Copy .env.example to .env and fill it in, "
            "and load it (python-dotenv) before calling chat()."
        )
    base_url = os.environ.get("DEEPINFRA_BASE_URL", DEFAULT_BASE_URL)
    return OpenAI(api_key=api_key, base_url=base_url)


def chat(prompt, model, system=None, max_tokens=1024, temperature=0.7, client=None,
         logit_bias=None):
    """Return the assistant's message content for a single-turn chat.

    Args:
        prompt: the user turn.
        model: a DeepInfra model id (e.g. a large open Qwen/Llama).
        system: optional system message.
        max_tokens, temperature: generation controls.
        client: an OpenAI-compatible client; built from env if None.
        logit_bias: optional {token_id_str: bias} dict, forwarded as-is. Used
            for real constrained-choice classification (see jspace.judge's
            structured judge) -- DeepInfra silently ignores response_format
            json_schema and the vLLM guided_choice/guided_json extra_body
            extensions for at least meta-llama/Llama-3.3-70B-Instruct
            (verified empirically 2026-07-27: guided_choice with temperature=0
            still returned three different free-text non-enum completions
            across three identical calls), but a heavily-biased logit_bias
            toward a handful of single-token choices does reliably constrain
            the output on DeepInfra's actual serving stack.
    """
    if client is None:
        client = _build_client()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    kwargs = {}
    if logit_bias is not None:
        kwargs["logit_bias"] = logit_bias
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        **kwargs,
    )
    return (resp.choices[0].message.content or "").strip()
