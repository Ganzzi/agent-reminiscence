# Pydantic AI Guide

> **Official Documentation**: [https://ai.pydantic.dev/](https://ai.pydantic.dev/)

## Table of Contents
1. [Introduction](#introduction)
2. [Models](#models)
3. [Tools](#tools)
4. [System Prompts](#system-prompts)
5. [Dependency Types](#dependency-types)
6. [Output Types](#output-types)
7. [all_messages_json](#all_messages_json)
8. [Testing](#testing)

---

## Introduction

Pydantic AI is a Python agent framework designed to simplify building production-grade applications with Generative AI. It brings a FastAPI-like developer experience to GenAI development, leveraging Pydantic for type-safe, structured responses.

**Key Features:**
- Model-agnostic support (OpenAI, Anthropic, Google, etc.)
- Type-safe dependency injection
- Structured output validation
- Tool registration and calling
- Comprehensive testing utilities

---

## Models

Models in Pydantic AI represent the LLM providers and their specific models. The framework supports multiple providers with a consistent interface.

### Supported Providers

```python
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.models.mistral import MistralModel
from pydantic_ai.models.cohere import CohereModel
```

### Provider Configuration

Each provider requires an API key that can be set via environment variables or passed directly:

```python
import os
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

# Option 1: Using environment variables
# Set OPENAI_API_KEY in your .env file
model = OpenAIModel('gpt-4o')

# Option 2: Explicit provider configuration
provider = OpenAIProvider(api_key=os.getenv('OPENAI_API_KEY'))
model = OpenAIModel('gpt-4o', provider=provider)
```

### Model Provider Example

Here's a simplified version inspired by the project's model provider:

```python
from typing import Dict, Optional
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.anthropic import AnthropicProvider
import os
import dotenv

dotenv.load_dotenv()

class SimpleModelProvider:
    """
    Simplified model provider for multiple LLM providers.
    """
    
    MODELS = {
        'openai': {
            'gpt-4o': OpenAIModel,
            'gpt-4o-mini': OpenAIModel,
        },
        'anthropic': {
            'claude-3-5-sonnet': AnthropicModel,
            'claude-3-5-haiku': AnthropicModel,
        },
    }
    
    PROVIDERS = {
        'openai': OpenAIProvider,
        'anthropic': AnthropicProvider,
    }
    
    ENV_KEYS = {
        'openai': 'OPENAI_API_KEY',
        'anthropic': 'ANTHROPIC_API_KEY',
    }
    
    def __init__(self):
        self.api_keys = self._load_api_keys()
    
    def get_model(self, provider: str, model_name: str) -> Model:
        """Get a model instance by provider and model name."""
        if provider not in self.MODELS:
            raise ValueError(f"Provider '{provider}' not supported")
        
        if model_name not in self.MODELS[provider]:
            raise ValueError(f"Model '{model_name}' not found for provider '{provider}'")
        
        model_class = self.MODELS[provider][model_name]
        api_key = self.api_keys.get(provider)
        
        if api_key:
            provider_class = self.PROVIDERS[provider]
            provider_instance = provider_class(api_key=api_key)
            return model_class(model_name, provider=provider_instance)
        
        # Falls back to environment variables
        return model_class(model_name)
    
    def _load_api_keys(self) -> Dict[str, str]:
        """Load API keys from environment variables."""
        return {
            provider: os.getenv(env_key)
            for provider, env_key in self.ENV_KEYS.items()
            if os.getenv(env_key)
        }

# Usage
provider = SimpleModelProvider()
model = provider.get_model('openai', 'gpt-4o')
```

### Using Models with Agents

```python
from pydantic_ai import Agent

# Option 1: String reference (requires environment variables)
agent = Agent('openai:gpt-4o')

# Option 2: Model instance
from pydantic_ai.models.openai import OpenAIModel
model = OpenAIModel('gpt-4o')
agent = Agent(model)

# Option 3: Using a provider
provider = SimpleModelProvider()
model = provider.get_model('anthropic', 'claude-3-5-sonnet')
agent = Agent(model)
```

---

## Tools

Tools are functions that the LLM can call to perform specific actions or retrieve information. Pydantic AI provides two types of tool decorators.

### Tool Types

#### 1. `@agent.tool` - Context-aware tools

These tools have access to `RunContext` with dependencies:

```python
from pydantic_ai import Agent, RunContext
from dataclasses import dataclass

@dataclass
class DatabaseDeps:
    connection_string: str

agent = Agent(
    'openai:gpt-4o',
    deps_type=DatabaseDeps,
)

@agent.tool
async def get_user_data(ctx: RunContext[DatabaseDeps], user_id: int) -> dict:
    """Fetch user data from the database."""
    # Access dependencies through ctx.deps
    conn_string = ctx.deps.connection_string
    # Perform database operations...
    return {'user_id': user_id, 'name': 'John Doe'}
```

#### 2. `@agent.tool_plain` - Context-independent tools

These tools don't need dependencies:

```python
import random
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o')

@agent.tool_plain
def roll_dice(sides: int = 6) -> int:
    """Roll a dice with the specified number of sides."""
    return random.randint(1, sides)
```

### Tool Schema

Pydantic AI automatically generates tool schemas from function signatures and docstrings:

```python
from pydantic import BaseModel
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o')

class CalculationInput(BaseModel):
    """Input for mathematical calculations."""
    numerator: float
    denominator: float

@agent.tool_plain
def divide(calc: CalculationInput) -> float:
    """Divide two numbers."""
    if calc.denominator == 0:
        raise ValueError("Cannot divide by zero")
    return calc.numerator / calc.denominator
```

### Complete Tool Example

```python
from pydantic_ai import Agent, RunContext
from dataclasses import dataclass
from datetime import datetime

@dataclass
class WeatherDeps:
    api_key: str

agent = Agent(
    'openai:gpt-4o',
    deps_type=WeatherDeps,
    system_prompt='You are a weather assistant. Use tools to get weather information.'
)

@agent.tool_plain
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().isoformat()

@agent.tool
async def get_weather(ctx: RunContext[WeatherDeps], city: str) -> str:
    """Get the current weather for a city."""
    # In real implementation, call weather API with ctx.deps.api_key
    return f"Weather in {city}: Sunny, 22°C"

# Run the agent
deps = WeatherDeps(api_key='your-api-key')
result = await agent.run('What is the weather in London?', deps=deps)
print(result.output)
```

---

## System Prompts

System prompts define the agent's behavior and context. They can be static or dynamic.

### Static System Prompts

```python
from pydantic_ai import Agent

agent = Agent(
    'openai:gpt-4o',
    system_prompt='You are a helpful coding assistant specialized in Python.'
)
```

### Dynamic System Prompts

Dynamic prompts can use dependencies and be defined as functions:

```python
from pydantic_ai import Agent, RunContext
from dataclasses import dataclass

@dataclass
class UserDeps:
    user_id: int
    user_name: str
    preferences: dict

agent = Agent(
    'openai:gpt-4o',
    deps_type=UserDeps,
)

@agent.system_prompt
async def get_system_prompt(ctx: RunContext[UserDeps]) -> str:
    """Generate a personalized system prompt."""
    user = ctx.deps
    return f"""You are a personal assistant for {user.user_name}.
User preferences: {user.preferences}
Always address the user by name and respect their preferences."""

# Usage
deps = UserDeps(
    user_id=123,
    user_name='Alice',
    preferences={'language': 'Python', 'style': 'concise'}
)
result = await agent.run('Help me with a coding problem', deps=deps)
```

### Multiple System Prompts

You can add multiple system prompts that will be concatenated:

```python
from pydantic_ai import Agent

agent = Agent(
    'openai:gpt-4o',
    system_prompt='You are a helpful assistant.',
)

@agent.system_prompt
def add_instructions(ctx) -> str:
    return 'Always provide code examples when relevant.'

@agent.system_prompt
def add_constraints(ctx) -> str:
    return 'Keep responses under 500 words.'
```

### System Prompts with External Data

```python
from pydantic_ai import Agent, RunContext
from dataclasses import dataclass
import httpx

@dataclass
class ApiDeps:
    http_client: httpx.AsyncClient

agent = Agent('openai:gpt-4o', deps_type=ApiDeps)

@agent.system_prompt
async def fetch_prompt_from_api(ctx: RunContext[ApiDeps]) -> str:
    """Fetch system prompt from external API."""
    response = await ctx.deps.http_client.get('https://example.com/prompt')
    response.raise_for_status()
    return response.text
```

---

## Dependency Types

Dependencies provide a type-safe way to inject data, connections, and logic into agents. They are accessed through `RunContext`.

### Defining Dependencies

Dependencies are typically defined as dataclasses:

```python
from dataclasses import dataclass
from typing import Optional
import httpx

@dataclass
class AppDependencies:
    """Application-wide dependencies for the agent."""
    api_key: str
    database_url: str
    http_client: httpx.AsyncClient
    user_id: Optional[int] = None
```

### Using Dependencies

Dependencies are passed when running the agent and accessed via `RunContext`:

```python
from pydantic_ai import Agent, RunContext
from dataclasses import dataclass

@dataclass
class MyDeps:
    database: str
    api_key: str

agent = Agent('openai:gpt-4o', deps_type=MyDeps)

@agent.system_prompt
async def get_prompt(ctx: RunContext[MyDeps]) -> str:
    return f"Connected to database: {ctx.deps.database}"

@agent.tool
async def fetch_data(ctx: RunContext[MyDeps], query: str) -> str:
    """Fetch data using the database connection."""
    db = ctx.deps.database
    api_key = ctx.deps.api_key
    # Perform operations...
    return f"Data for query: {query}"

# Run with dependencies
deps = MyDeps(database='postgresql://localhost/mydb', api_key='secret')
result = await agent.run('Get user data', deps=deps)
```

### Synchronous vs Asynchronous Dependencies

#### Asynchronous Dependencies

```python
from dataclasses import dataclass
import httpx
from pydantic_ai import Agent, RunContext

@dataclass
class AsyncDeps:
    http_client: httpx.AsyncClient

agent = Agent('openai:gpt-4o', deps_type=AsyncDeps)

@agent.tool
async def fetch_api_data(ctx: RunContext[AsyncDeps], endpoint: str) -> str:
    """Fetch data from an external API."""
    response = await ctx.deps.http_client.get(f'https://api.example.com/{endpoint}')
    response.raise_for_status()
    return response.text
```

#### Synchronous Dependencies

```python
from dataclasses import dataclass
import httpx
from pydantic_ai import Agent, RunContext

@dataclass
class SyncDeps:
    http_client: httpx.Client  # Synchronous client

agent = Agent('openai:gpt-4o', deps_type=SyncDeps)

@agent.tool
def fetch_sync_data(ctx: RunContext[SyncDeps], endpoint: str) -> str:
    """Fetch data synchronously."""
    response = ctx.deps.http_client.get(f'https://api.example.com/{endpoint}')
    response.raise_for_status()
    return response.text
```

### Dependency with Methods

Dependencies can have methods that encapsulate logic:

```python
from dataclasses import dataclass
from typing import Optional
import httpx

@dataclass
class ServiceDeps:
    api_key: str
    http_client: httpx.AsyncClient
    
    async def fetch_user(self, user_id: int) -> dict:
        """Fetch user information."""
        response = await self.http_client.get(
            f'https://api.example.com/users/{user_id}',
            headers={'Authorization': f'Bearer {self.api_key}'}
        )
        response.raise_for_status()
        return response.json()
    
    async def system_prompt_factory(self) -> str:
        """Generate a dynamic system prompt."""
        config = await self.http_client.get('https://api.example.com/config')
        return f"Configuration: {config.text}"

agent = Agent('openai:gpt-4o', deps_type=ServiceDeps)

@agent.system_prompt
async def get_prompt(ctx: RunContext[ServiceDeps]) -> str:
    return await ctx.deps.system_prompt_factory()

@agent.tool
async def get_user_info(ctx: RunContext[ServiceDeps], user_id: int) -> dict:
    """Get user information using dependency method."""
    return await ctx.deps.fetch_user(user_id)
```

---

## Output Types

Output types ensure that agent responses are structured and validated using Pydantic models.

### String Output (Default)

```python
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o')
result = await agent.run('Tell me a joke')
print(result.output)  # str
```

### Structured Output with Pydantic Models

```python
from pydantic import BaseModel, Field
from pydantic_ai import Agent

class UserInfo(BaseModel):
    """Structured user information."""
    name: str = Field(description='User full name')
    age: int = Field(description='User age in years', ge=0, le=120)
    email: str = Field(description='User email address')

agent = Agent(
    'openai:gpt-4o',
    output_type=UserInfo,
)

result = await agent.run('Extract user info: John Doe, 30 years old, john@example.com')
print(result.output)
# UserInfo(name='John Doe', age=30, email='john@example.com')
print(type(result.output))  # <class 'UserInfo'>
```

### Union Types

```python
from pydantic import BaseModel
from pydantic_ai import Agent

class Success(BaseModel):
    """Successful operation result."""
    message: str
    data: dict

class Error(BaseModel):
    """Error result."""
    error_message: str
    error_code: int

agent = Agent(
    'openai:gpt-4o',
    output_type=Success | Error,
)

result = await agent.run('Process this request')
if isinstance(result.output, Success):
    print(f"Success: {result.output.message}")
else:
    print(f"Error: {result.output.error_message}")
```

### Tool Output Mode

For more explicit control over structured outputs:

```python
from pydantic import BaseModel
from pydantic_ai import Agent, ToolOutput

class Fruit(BaseModel):
    name: str
    color: str

class Vehicle(BaseModel):
    name: str
    wheels: int

agent = Agent(
    'openai:gpt-4o',
    output_type=[
        ToolOutput(Fruit, name='return_fruit'),
        ToolOutput(Vehicle, name='return_vehicle'),
    ],
)

result = await agent.run('What is a banana?')
print(result.output)
# Fruit(name='banana', color='yellow')
```

### Output Validation

Add custom validation to ensure output quality:

```python
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext, ModelRetry
from dataclasses import dataclass

class SqlQuery(BaseModel):
    """SQL query structure."""
    query: str
    description: str

@dataclass
class DatabaseDeps:
    connection: object
    
    async def validate_query(self, query: str) -> bool:
        """Validate SQL query by running EXPLAIN."""
        try:
            # Simulate query validation
            await self.connection.execute(f'EXPLAIN {query}')
            return True
        except Exception:
            return False

agent = Agent(
    'openai:gpt-4o',
    output_type=SqlQuery,
    deps_type=DatabaseDeps,
)

@agent.output_validator
async def validate_sql(ctx: RunContext[DatabaseDeps], output: SqlQuery) -> SqlQuery:
    """Validate the generated SQL query."""
    is_valid = await ctx.deps.validate_query(output.query)
    if not is_valid:
        raise ModelRetry('Generated SQL query is invalid. Please try again.')
    return output
```

### Prompted Output Mode

For models without native structured output support:

```python
from pydantic import BaseModel
from pydantic_ai import Agent, PromptedOutput

class Device(BaseModel):
    name: str
    kind: str

agent = Agent(
    'openai:gpt-4o',
    output_type=PromptedOutput(
        Device,
        name='Device Information',
        description='Extract device information from the text'
    ),
)

result = await agent.run('What is a MacBook?')
print(result.output)
# Device(name='MacBook', kind='laptop')
```

---

## all_messages_json

The `all_messages_json` property provides access to the complete message history of an agent run, serialized as JSON. This is useful for debugging, logging, and understanding the conversation flow.

### Basic Usage

```python
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o')
result = await agent.run('What is the capital of France?')

# Get all messages as JSON
messages = result.all_messages_json()
print(messages)
```

### Message Structure

Messages include system prompts, user prompts, model responses, tool calls, and tool returns:

```python
import json
from pydantic_ai import Agent

agent = Agent(
    'openai:gpt-4o',
    system_prompt='You are a helpful assistant.',
)

@agent.tool_plain
def get_time() -> str:
    """Get current time."""
    return "2024-01-01 12:00:00"

result = await agent.run('What time is it?')

# Pretty print all messages
messages_json = result.all_messages_json()
print(json.dumps(messages_json, indent=2))

# Access specific messages
messages = result.all_messages()
for msg in messages:
    print(f"Kind: {msg.kind}")
    if hasattr(msg, 'parts'):
        for part in msg.parts:
            print(f"  Part: {part.part_kind}")
```

### Message Types

Common message types in the conversation:

```python
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    UserPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)

agent = Agent('openai:gpt-4o')

@agent.tool_plain
def calculator(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

result = await agent.run('Calculate 5 + 3')

# Inspect message types
for message in result.all_messages():
    if message.kind == 'request':
        print("Request message:")
        for part in message.parts:
            if part.part_kind == 'system-prompt':
                print(f"  System: {part.content}")
            elif part.part_kind == 'user-prompt':
                print(f"  User: {part.content}")
            elif part.part_kind == 'tool-return':
                print(f"  Tool Return: {part.content}")
    
    elif message.kind == 'response':
        print("Response message:")
        for part in message.parts:
            if part.part_kind == 'text':
                print(f"  Text: {part.content}")
            elif part.part_kind == 'tool-call':
                print(f"  Tool Call: {part.tool_name}({part.args})")
```

### Capturing Messages During Testing

```python
from pydantic_ai import Agent, capture_run_messages

agent = Agent('openai:gpt-4o')

with capture_run_messages() as messages:
    result = await agent.run('Hello, how are you?')

# Messages captured during the run
print(f"Total messages: {len(messages)}")
for msg in messages:
    print(msg)
```

### Logging and Debugging

```python
import json
from datetime import datetime
from pydantic_ai import Agent

def log_conversation(result, filename: str = None):
    """Log conversation to file."""
    if filename is None:
        filename = f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(filename, 'w') as f:
        json.dump(result.all_messages_json(), f, indent=2, default=str)
    
    print(f"Conversation logged to {filename}")

agent = Agent('openai:gpt-4o')
result = await agent.run('Explain quantum computing')

# Log the conversation
log_conversation(result)

# Access usage information
print(f"Total tokens: {result.usage().total_tokens}")
print(f"Input tokens: {result.usage().input_tokens}")
print(f"Output tokens: {result.usage().output_tokens}")
```

---

## Testing

Pydantic AI provides comprehensive testing utilities to test agents without making actual LLM API calls.

### Test Models

#### TestModel - Simple Testing

```python
import pytest
from pydantic_ai import Agent, models
from pydantic_ai.models.test import TestModel

# Disable actual model requests in tests
models.ALLOW_MODEL_REQUESTS = False

@pytest.mark.anyio
async def test_basic_agent():
    """Test agent with TestModel."""
    agent = Agent('openai:gpt-4o', system_prompt='You are helpful')
    
    # Create test model
    test_model = TestModel()
    
    # Override agent model
    with agent.override(model=test_model):
        result = await agent.run('Hello')
        assert result.output == 'success (no tool calls)'
    
    # Inspect model parameters
    assert test_model.last_model_request_parameters.function_tools == []
```

#### TestModel with Custom Responses

```python
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

async def test_custom_response():
    """Test with custom response."""
    agent = Agent('openai:gpt-4o')
    
    test_model = TestModel(
        custom_result_text='This is a custom response'
    )
    
    with agent.override(model=test_model):
        result = await agent.run('Any question')
        assert result.output == 'This is a custom response'
```

### FunctionModel - Advanced Testing

For precise control over model behavior:

```python
import pytest
from pydantic_ai import Agent, models
from pydantic_ai.models.function import FunctionModel, AgentInfo
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart

models.ALLOW_MODEL_REQUESTS = False

async def mock_model_function(
    messages: list[ModelMessage],
    info: AgentInfo
) -> ModelResponse:
    """Custom model function for testing."""
    # Get the last user message
    last_message = messages[-1]
    user_content = last_message.parts[-1].content
    
    # Custom logic based on user input
    if 'weather' in user_content.lower():
        return ModelResponse(parts=[
            ToolCallPart('get_weather', {'city': 'London'})
        ])
    else:
        return ModelResponse(parts=[
            TextPart('I can help with weather queries')
        ])

@pytest.mark.anyio
async def test_with_function_model():
    """Test agent with FunctionModel."""
    agent = Agent('openai:gpt-4o')
    
    @agent.tool_plain
    def get_weather(city: str) -> str:
        """Get weather for a city."""
        return f"Sunny in {city}"
    
    with agent.override(model=FunctionModel(mock_model_function)):
        result = await agent.run('What is the weather in London?')
        assert 'London' in result.output
```

### Testing with Dependencies

```python
import pytest
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.test import TestModel

@dataclass
class TestDeps:
    """Test dependencies."""
    database: str = 'test_db'
    api_key: str = 'test_key'

@pytest.mark.anyio
async def test_agent_with_deps():
    """Test agent with custom dependencies."""
    agent = Agent('openai:gpt-4o', deps_type=TestDeps)
    
    @agent.tool
    async def get_data(ctx: RunContext[TestDeps]) -> str:
        """Get data from database."""
        return f"Data from {ctx.deps.database}"
    
    test_deps = TestDeps(database='mock_db')
    
    with agent.override(model=TestModel()):
        result = await agent.run('Get data', deps=test_deps)
        assert result.output is not None
```

### Capturing and Asserting Messages

```python
import pytest
from datetime import timezone
from dirty_equals import IsNow, IsStr
from pydantic_ai import Agent, capture_run_messages, models, RequestUsage
from pydantic_ai.models.test import TestModel
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    UserPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)

models.ALLOW_MODEL_REQUESTS = False

@pytest.mark.anyio
async def test_message_flow():
    """Test message flow with assertions."""
    agent = Agent(
        'openai:gpt-4o',
        system_prompt='You are helpful'
    )
    
    @agent.tool_plain
    def get_info() -> str:
        """Get information."""
        return "Important information"
    
    with capture_run_messages() as messages:
        with agent.override(model=TestModel()):
            result = await agent.run('Tell me something')
    
    # Assert message structure
    assert len(messages) >= 2
    
    # Check first message is a request with system and user prompts
    first_msg = messages[0]
    assert first_msg.kind == 'request'
    assert any(p.part_kind == 'system-prompt' for p in first_msg.parts)
    assert any(p.part_kind == 'user-prompt' for p in first_msg.parts)
```

### Testing Output Validation

```python
import pytest
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic_ai.models.test import TestModel
from dataclasses import dataclass

class OutputModel(BaseModel):
    value: int

@dataclass
class ValidatorDeps:
    max_value: int

@pytest.mark.anyio
async def test_output_validation():
    """Test output validator."""
    agent = Agent(
        'openai:gpt-4o',
        output_type=OutputModel,
        deps_type=ValidatorDeps,
    )
    
    @agent.output_validator
    async def validate_output(
        ctx: RunContext[ValidatorDeps],
        output: OutputModel
    ) -> OutputModel:
        if output.value > ctx.deps.max_value:
            raise ModelRetry('Value too high')
        return output
    
    deps = ValidatorDeps(max_value=100)
    
    # Test with valid output
    test_model = TestModel(custom_result_json={'value': 50})
    with agent.override(model=test_model):
        result = await agent.run('Give me a number', deps=deps)
        assert result.output.value == 50
```

### Pytest Fixtures for Testing

```python
import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

@pytest.fixture
def test_agent():
    """Fixture providing a test agent."""
    agent = Agent('openai:gpt-4o', system_prompt='Test assistant')
    return agent

@pytest.fixture
def override_agent_model(test_agent):
    """Fixture to override agent with TestModel."""
    with test_agent.override(model=TestModel()):
        yield

@pytest.mark.anyio
async def test_with_fixture(test_agent, override_agent_model):
    """Test using fixtures."""
    result = await test_agent.run('Test query')
    assert result.output is not None
```

### Integration Testing

```python
import pytest
from pydantic_ai import Agent, models

# For integration tests, allow real API calls
# but use environment variables to control
import os

@pytest.mark.skipif(
    not os.getenv('RUN_INTEGRATION_TESTS'),
    reason='Integration tests disabled'
)
@pytest.mark.anyio
async def test_real_model():
    """Integration test with real model."""
    models.ALLOW_MODEL_REQUESTS = True
    
    agent = Agent('openai:gpt-4o-mini')
    result = await agent.run('What is 2+2?')
    
    assert result.output is not None
    assert '4' in result.output
```

---

## Best Practices

### 1. Environment Variables

Always use environment variables for sensitive data:

```python
import os
from dotenv import load_dotenv

load_dotenv()

# Load API keys
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
DATABASE_URL = os.getenv('DATABASE_URL')
```

### 2. Type Safety

Leverage Python's type hints for better IDE support and fewer bugs:

```python
from pydantic_ai import Agent, RunContext
from dataclasses import dataclass
from typing import Optional

@dataclass
class TypedDeps:
    api_key: str
    timeout: int = 30
    retry_count: int = 3

agent: Agent[TypedDeps, str] = Agent(
    'openai:gpt-4o',
    deps_type=TypedDeps,
)
```

### 3. Error Handling

```python
from pydantic_ai import Agent, ModelRetry

agent = Agent('openai:gpt-4o')

@agent.tool_plain
def risky_operation(value: int) -> int:
    """Operation that might fail."""
    if value < 0:
        raise ModelRetry('Value must be positive')
    return value * 2
```

### 4. Testing

Always write tests for your agents:

```python
# test_my_agent.py
import pytest
from pydantic_ai import models
from pydantic_ai.models.test import TestModel
from my_agent import agent

models.ALLOW_MODEL_REQUESTS = False

@pytest.mark.anyio
async def test_agent():
    with agent.override(model=TestModel()):
        result = await agent.run('test input')
        assert result.output is not None
```

---

## Conclusion

Pydantic AI provides a powerful, type-safe framework for building production-grade AI applications. Key takeaways:

- **Models**: Support for multiple LLM providers with consistent interfaces
- **Tools**: Register functions for agents to call with full type safety
- **System Prompts**: Define agent behavior statically or dynamically
- **Dependencies**: Type-safe dependency injection for tools and prompts
- **Output Types**: Structured, validated outputs using Pydantic models
- **all_messages_json**: Full conversation history for debugging and logging
- **Testing**: Comprehensive testing utilities without API calls

For more information, visit the [official documentation](https://ai.pydantic.dev/).
