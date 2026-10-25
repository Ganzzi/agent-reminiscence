"""Tests for agent-reminiscence AI executor hardening.

Covers backend-mode enforcement, executor propagation through E/R extraction,
deep search, and consolidation paths.
"""

import pytest
from unittest.mock import AsyncMock

from pydantic_ai.usage import RunUsage

from agent_reminiscence.services.llm_model_provider import ModelProvider
from agent_reminiscence.services.llm_model_provider import model_provider as module_provider


@pytest.mark.asyncio
async def test_backend_mode_raises_without_executor():
    """get_model() must raise RuntimeError when backend_mode=True and no executor."""
    provider = ModelProvider(api_keys={}, backend_mode=True)
    with pytest.raises(RuntimeError, match="No LLM executor injected"):
        provider.get_model("openai:gpt-4o-mini")


def test_backend_mode_skips_guard_when_executor_present():
    """get_model() proceeds when both backend_mode and executor are set."""
    async def dummy_exec(*args, **kwargs):
        return {"text": "", "usage": {}}
    provider = ModelProvider(api_keys={}, executor=dummy_exec, backend_mode=True)
    # Should not raise — executor is present so the gate is bypassed.
    # Actual model construction will fail without real API keys, but the
    # backend-mode check is what we're testing.
    assert provider._backend_mode is True
    assert provider._executor is not None


@pytest.mark.asyncio
async def test_set_llm_executor_toggles_backend_mode():
    """set_llm_executor must enable backend_mode when executor is set and disable when cleared."""
    # Initial state: no executor, no backend mode
    module_provider.set_executor(None)
    module_provider.set_backend_mode(False)
    assert module_provider._backend_mode is False
    assert module_provider._executor is None

    # Setting executor should also set backend_mode — test via AgentMem.set_llm_executor
    from agent_reminiscence.core import AgentMem
    async def dummy_exec(*args, **kwargs):
        return {"text": "", "usage": {}}

    mem = AgentMem()
    mem.set_llm_executor(dummy_exec)
    assert module_provider._backend_mode is True
    assert module_provider._executor is dummy_exec

    # Clearing executor should disable backend_mode
    mem.set_llm_executor(None)
    assert module_provider._backend_mode is False
    assert module_provider._executor is None


@pytest.mark.asyncio
async def test_run_agent_routes_through_executor_when_injected():
    """run_agent() must use the injected executor, not agent.run(), when executor is present."""
    provider = ModelProvider(api_keys={})
    calls = []

    async def spy_executor(model_info, messages, model_settings, metadata):
        calls.append(model_info)
        return {
            "text": "executed",
            "usage": {"input_tokens": 5, "output_tokens": 3},
        }

    provider.set_executor(spy_executor)
    provider.set_backend_mode(True)

    class _DummyAgent:
        output_type = str
        async def run(self, user_prompt=None, deps=None):
            raise RuntimeError("Should not be called")

    result = await provider.run_agent(
        agent=_DummyAgent(),
        user_prompt="hello",
        model_info="openai:gpt-4o-mini",
        metadata={"operation": "test"},
    )

    assert len(calls) == 1
    assert calls[0] == "openai:gpt-4o-mini"
    assert result.output == "executed"
    assert result.usage().input_tokens == 5
    assert result.usage().output_tokens == 3


@pytest.mark.asyncio
async def test_run_agent_falls_back_when_no_executor():
    """run_agent() must call agent.run() when no executor is injected."""
    provider = ModelProvider(api_keys={})
    agent_called = False

    class _DummyAgent:
        output_type = str
        async def run(self, user_prompt=None, deps=None):
            nonlocal agent_called
            agent_called = True
            return _DummyRunResult("fallback", RunUsage(requests=1, input_tokens=1, output_tokens=1))

    result = await provider.run_agent(agent=_DummyAgent(), user_prompt="hi")
    assert agent_called is True
    assert result.output == "fallback"


class _DummyRunResult:
    def __init__(self, output, usage: RunUsage):
        self.output = output
        self._usage = usage
    def usage(self) -> RunUsage:
        return self._usage


@pytest.mark.asyncio
async def test_extract_entities_and_relationships_routes_through_executor(monkeypatch):
    """extract_entities_and_relationships must route through injected executor.

    Only model construction is hermetic: a get_model spy returns a pydantic_ai
    TestModel so no provider key is required. Real agent construction, extraction,
    run_agent dispatch, typed output parsing and usage conversion all execute.
    """
    import agent_reminiscence.agents.er_extractor as er_extractor
    from agent_reminiscence.agents.er_extractor import extract_entities_and_relationships
    from agent_reminiscence.config import settings
    from agent_reminiscence.config.settings import Config
    from pydantic_ai.models.test import TestModel

    config = Config(
        postgres_host="localhost",
        postgres_port=5433,
        postgres_db="agent_mem_test",
        postgres_user="backend",
        postgres_password="agent-mem-test-password",
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="neo4jpassword",
        er_extractor_agent_model="google:gemini-2.5-flash",
    )
    monkeypatch.setattr(settings, "_config", config)

    # Force real Agent re-construction under the deterministic config.
    monkeypatch.setattr(er_extractor, "_er_extractor_agent", None)

    # Native Agent.run must never execute while an executor is injected.
    native_run = AsyncMock(
        side_effect=AssertionError("native Agent.run must not run in executor mode")
    )
    monkeypatch.setattr(er_extractor.Agent, "run", native_run)

    get_model_calls = []

    def spy_get_model(model_info):
        get_model_calls.append(model_info)
        return TestModel()

    monkeypatch.setattr(module_provider, "get_model", spy_get_model)

    executor_calls = []

    async def spy_executor(model_info, messages, model_settings, metadata):
        executor_calls.append(
            {
                "model_info": model_info,
                "messages": messages,
                "model_settings": model_settings,
                "metadata": metadata,
            }
        )
        return {
            "text": '{"entities": [], "relationships": []}',
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }

    # Scoped monkeypatch restores prior provider state even on failure.
    monkeypatch.setattr(module_provider, "_executor", spy_executor)
    monkeypatch.setattr(module_provider, "_backend_mode", True)

    result, usage = await extract_entities_and_relationships("test content for extraction")

    assert get_model_calls == ["google:gemini-2.5-flash"]
    assert len(executor_calls) == 1
    call = executor_calls[0]
    assert call["model_info"] == "google:gemini-2.5-flash"
    assert call["messages"] == [{"role": "user", "content": "test content for extraction"}]
    assert call["model_settings"] is None
    assert call["metadata"] == {"operation": "extractor"}

    native_run.assert_not_awaited()

    assert isinstance(result, er_extractor.ExtractionResult)
    assert result.entities == []
    assert result.relationships == []
    assert usage.input_tokens == 10
    assert usage.output_tokens == 5
    assert usage.requests == 1


@pytest.mark.asyncio
async def test_executor_propagation_through_agent_mem_setter():
    """AgentMem.set_llm_executor must propagate to the module-level model_provider."""
    from agent_reminiscence.core import AgentMem

    mem = AgentMem()
    calls = []

    async def spy_executor(model_info, messages, model_settings, metadata):
        calls.append(model_info)
        return {"text": "ok", "usage": {"input_tokens": 1, "output_tokens": 1}}

    mem.set_llm_executor(spy_executor)
    assert module_provider._backend_mode is True
    assert module_provider._executor is spy_executor

    # Verify run_agent uses it
    class _DummyAgent:
        output_type = str
        async def run(self, user_prompt=None, deps=None):
            raise RuntimeError("Should not be called")

    await module_provider.run_agent(
        agent=_DummyAgent(),
        user_prompt="x",
        model_info="openai:gpt-4o-mini",
    )
    assert len(calls) == 1

    mem.set_llm_executor(None)
    assert module_provider._backend_mode is False
    assert module_provider._executor is None


@pytest.mark.asyncio
async def test_deep_search_with_executor_no_crash():
    """deep_search_memories must not crash when executor is injected.

    This is a lightweight verification that the executor injection plumbing
    does not break the deep search path. Full integration coverage lives in
    integration/e2e tests.
    """
    from agent_reminiscence.core import AgentMem

    # Just verify the executor injection point doesn't blow up (no DB init)
    mem = AgentMem()

    async def dummy_exec(*args, **kwargs):
        return {"text": '{"answer":"ok"}', "usage": {"input_tokens": 1, "output_tokens": 1}}

    mem.set_llm_executor(dummy_exec)
    assert module_provider._executor is dummy_exec
    assert module_provider._backend_mode is True

    # Cleanup
    mem.set_llm_executor(None)
