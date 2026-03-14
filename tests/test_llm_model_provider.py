import pytest
from pydantic import BaseModel
from pydantic_ai.usage import RunUsage

from agent_reminiscence.services.llm_model_provider import ModelProvider


class _DummyAgent:
    output_type = str

    def __init__(self):
        self.run_called = False

    async def run(self, user_prompt=None, deps=None):
        self.run_called = True
        return _DummyRunResult(output="direct", usage=RunUsage(requests=1, input_tokens=1, output_tokens=1))


class _DummyRunResult:
    def __init__(self, output, usage: RunUsage):
        self.output = output
        self._usage = usage

    def usage(self) -> RunUsage:
        return self._usage


class _StructuredOutput(BaseModel):
    answer: str


class _StructuredAgent(_DummyAgent):
    output_type = _StructuredOutput


@pytest.mark.asyncio
async def test_run_agent_uses_executor_when_injected():
    provider = ModelProvider(api_keys={})
    agent = _DummyAgent()
    called = {}

    async def executor(model_info, messages, model_settings, metadata):
        called["model_info"] = model_info
        called["messages"] = messages
        called["metadata"] = metadata
        return {
            "text": "queued answer",
            "usage": {"input_tokens": 11, "output_tokens": 7},
            "provider_response_id": "req-1",
        }

    provider.set_executor(executor)
    result = await provider.run_agent(
        agent=agent,
        user_prompt="hello",
        model_info="openai:gpt-4o-mini",
        metadata={"operation": "retriever"},
    )

    assert called["model_info"] == "openai:gpt-4o-mini"
    assert called["messages"] == [{"role": "user", "content": "hello"}]
    assert called["metadata"]["operation"] == "retriever"
    assert result.output == "queued answer"
    assert result.usage().input_tokens == 11
    assert result.usage().output_tokens == 7
    assert agent.run_called is False


@pytest.mark.asyncio
async def test_run_agent_parses_structured_output_from_executor():
    provider = ModelProvider(api_keys={})
    agent = _StructuredAgent()

    async def executor(model_info, messages, model_settings, metadata):
        return {
            "text": '{"answer":"ok"}',
            "usage": {"input_tokens": 4, "output_tokens": 2},
        }

    provider.set_executor(executor)
    result = await provider.run_agent(
        agent=agent,
        user_prompt="hello",
        model_info="openai:gpt-4o-mini",
    )

    assert result.output.answer == "ok"
    assert result.usage().input_tokens == 4


def test_get_model_raises_in_backend_mode_without_executor():
    provider = ModelProvider(api_keys={}, backend_mode=True)

    with pytest.raises(RuntimeError, match="No LLM executor injected"):
        provider.get_model("openai:gpt-4o-mini")
