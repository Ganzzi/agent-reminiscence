"""
Model provider module for mapping model names to pydantic-ai model instances.
This allows for flexible model selection by name across different providers.
"""

import json
from dataclasses import dataclass
from typing import Dict, Any, Optional, Type, Awaitable, Callable, TypedDict
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.grok import GrokProvider
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.usage import RunUsage

from agent_reminiscence.config.settings import get_config

# Mapping of providers to their model classes
PROVIDER_MODEL_CLASS_MAPPING: Dict[str, Type[Model]] = {
    "openai": OpenAIChatModel,
    "anthropic": AnthropicModel,
    "google": GoogleModel,
    "grok": OpenAIChatModel,  # Grok uses OpenAIChatModel but with GrokProvider
}

# Mapping of providers to their provider classes
PROVIDER_CLASS_MAPPING = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "google": GoogleProvider,
    "grok": GrokProvider,
}


class LLMChatResult(TypedDict, total=False):
    """Normalized response payload for injected executor chat calls."""

    text: str
    usage: Dict[str, int]
    provider_response_id: Optional[str]


LLMExecutor = Callable[
    [str, list[dict[str, Any]], Optional[dict[str, Any]], dict[str, Any]],
    Awaitable[LLMChatResult],
]


@dataclass
class ExecutorRunResult:
    """Lightweight run result shim compatible with existing call sites."""

    output: Any
    _usage: RunUsage

    def usage(self) -> RunUsage:
        return self._usage


class ModelProvider:
    """
    A class that provides model instances based on model names.
    Maps shorthand model names to their respective providers and models.

    Example:
        >>> provider = ModelProvider()
        >>> model = provider.get_model("o3-mini")
        >>> # Returns an OpenAIChatModel instance for gpt-3.5-turbo
    """

    def __init__(
        self,
        api_keys: Optional[Dict[str, str]] = None,
        executor: Optional[LLMExecutor] = None,
        backend_mode: bool = False,
    ):
        """
        Initialize the ModelProvider with optional API keys and model settings.

        Args:
            api_keys: Dictionary mapping provider names to API keys.
                      If not provided, will try to use config settings.
        """
        # Load API keys from provided dict or config
        self.api_keys = api_keys or self._load_api_keys_from_config()
        self._executor = executor
        self._backend_mode = backend_mode

    def set_executor(self, executor: Optional[LLMExecutor]) -> None:
        """Set or clear the injected LLM executor."""
        self._executor = executor

    def set_backend_mode(self, enabled: bool) -> None:
        """Enable/disable strict backend mode for provider initialization."""
        self._backend_mode = enabled

    def get_model(
        self,
        model_info: str,
    ) -> Model:
        """
        Get a model instance based on the model name.
        Args:
            model_info: A string in the format "provider:model_name"
        Returns:
            An instance of the requested model.
        """
        if self._backend_mode and self._executor is None:
            raise RuntimeError("No LLM executor injected. Backend mode requires queue executor.")

        provider_name, actual_model_name = model_info.split(":", 1)

        # Get the model class for this provider
        model_class = PROVIDER_MODEL_CLASS_MAPPING.get(provider_name)

        if not model_class:
            raise ValueError(f"Provider {provider_name} is not supported")

        # Check if we have an API key for this provider
        api_key = self.api_keys.get(provider_name)

        # If we have an API key, create a provider instance
        if api_key:
            provider_class = PROVIDER_CLASS_MAPPING.get(provider_name)
            if provider_class:
                provider = provider_class(api_key=api_key)
                return model_class(
                    actual_model_name,
                    provider=provider,
                )

        # Otherwise just create the model instance directly
        # For some models like OpenAI, the SDK will check for environment variables itself
        return model_class(actual_model_name)

    async def run_agent(
        self,
        agent: Any,
        user_prompt: str,
        *,
        deps: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
        model_info: Optional[str] = None,
        model_settings: Optional[dict[str, Any]] = None,
    ) -> Any:
        """
        Execute agent call through injected executor when configured.

        Falls back to native `agent.run(...)` when no executor is injected.
        """
        if self._executor is None:
            return await agent.run(user_prompt=user_prompt, deps=deps)

        payload = await self._executor(
            model_info or "unknown:model",
            [{"role": "user", "content": user_prompt}],
            model_settings,
            metadata or {},
        )

        text = payload.get("text", "")
        output = self._parse_output(agent, text)
        usage = self._usage_from_payload(payload.get("usage", {}))
        return ExecutorRunResult(output=output, _usage=usage)

    def _parse_output(self, agent: Any, text: str) -> Any:
        """Parse executor text response into the agent output type."""
        output_type = getattr(agent, "output_type", None)
        if output_type is None:
            return text

        if output_type is str:
            return text

        if hasattr(output_type, "model_validate_json"):
            try:
                return output_type.model_validate_json(text)
            except Exception:
                try:
                    return output_type.model_validate(json.loads(text))
                except Exception as exc:
                    raise RuntimeError(f"Failed to parse executor response for {output_type}: {exc}") from exc

        return text

    @staticmethod
    def _usage_from_payload(usage_payload: Dict[str, Any]) -> RunUsage:
        """Create RunUsage from executor usage payload with safe defaults."""
        return RunUsage(
            requests=1,
            input_tokens=int(usage_payload.get("input_tokens", 0) or 0),
            output_tokens=int(usage_payload.get("output_tokens", 0) or 0),
            cache_write_tokens=int(usage_payload.get("cache_write_tokens", 0) or 0),
            cache_read_tokens=int(usage_payload.get("cache_read_tokens", 0) or 0),
        )

    def _load_api_keys_from_config(self) -> Dict[str, str]:
        """
        Load API keys from centralized config.

        Returns:
            Dictionary mapping provider names to API keys.
        """
        config = get_config()
        api_keys = {}

        # Load API keys for each provider from config
        if config.openai_api_key:
            api_keys["openai"] = config.openai_api_key
        if config.anthropic_api_key:
            api_keys["anthropic"] = config.anthropic_api_key
        if config.google_api_key:
            api_keys["google"] = config.google_api_key
        if config.grok_api_key:
            api_keys["grok"] = config.grok_api_key

        return api_keys


model_provider = ModelProvider()
LLMModelProvider = ModelProvider


