"""Task 6 (steps 1-4) — DeepInfra OpenAI-compatible chat wrapper.

Unit test only: a fake client is injected so there is NO network call and NO spend.
The real wrapper builds its client from DEEPINFRA_API_KEY / DEEPINFRA_BASE_URL.
"""
from jspace.deepinfra import chat


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeCompletion:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeChatCompletions:
    def __init__(self, record):
        self._record = record

    def create(self, **kwargs):
        self._record.update(kwargs)
        return _FakeCompletion("  hello from deepinfra  ")


class _FakeClient:
    def __init__(self):
        self.record = {}

        class _Chat:
            pass

        self.chat = _Chat()
        self.chat.completions = _FakeChatCompletions(self.record)


def test_chat_returns_stripped_message_content_and_builds_request():
    fake = _FakeClient()
    out = chat("Say hi", model="some/model", system="You are terse.",
               max_tokens=32, temperature=0.0, client=fake)
    assert out == "hello from deepinfra"
    # System + user messages assembled in order.
    msgs = fake.record["messages"]
    assert msgs[0] == {"role": "system", "content": "You are terse."}
    assert msgs[-1] == {"role": "user", "content": "Say hi"}
    assert fake.record["model"] == "some/model"
    assert fake.record["max_tokens"] == 32
    assert fake.record["temperature"] == 0.0


def test_chat_omits_system_when_not_given():
    fake = _FakeClient()
    chat("Just user", model="m", client=fake)
    assert all(m["role"] != "system" for m in fake.record["messages"])


def test_chat_forwards_logit_bias_when_given():
    fake = _FakeClient()
    chat("q", model="m", client=fake, logit_bias={"32": 100, "33": 100})
    assert fake.record["logit_bias"] == {"32": 100, "33": 100}


def test_chat_omits_logit_bias_when_not_given():
    fake = _FakeClient()
    chat("q", model="m", client=fake)
    assert "logit_bias" not in fake.record
