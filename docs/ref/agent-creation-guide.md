# AI Agent Creation Guide with Pydantic AI

## Overview

This guide explains how to create AI agents using Pydantic AI, a Python framework for building production-grade applications with Generative AI. It provides a FastAPI-like developer experience for GenAI development.

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [Creating an Agent](#creating-an-agent)
3. [Model Configuration](#model-configuration)
4. [Dependency Injection](#dependency-injection)
5. [Tools and Capabilities](#tools-and-capabilities)
6. [Output Validation](#output-validation)
7. [Testing Strategies](#testing-strategies)
8. [Best Practices](#best-practices)

## Core Concepts

### What is an Agent?

An agent in Pydantic AI is a wrapper around a Large Language Model (LLM) that:
- Provides a structured way to interact with LLMs
- Manages dependencies and context
- Defines and executes tools
- Validates outputs using Pydantic models
- Handles conversation history and state

### Key Components

```python
from pydantic_ai import Agent, RunContext

agent = Agent(
    model='openai:gpt-4o',           # LLM model to use
    deps_type=MyDependencies,         # Type of dependencies
    tools=[tool1, tool2],             # Available tools
    system_prompt="You are...",       # Instructions for the agent
    output_type=MyOutputModel,        # Expected output structure
    model_settings={...},             # Model-specific settings
    retries=3,                        # Retry attempts on failure
)
```

## Creating an Agent

### Step 1: Define Dependencies

Dependencies provide context and data to your agent:

```python
from dataclasses import dataclass
from pydantic import BaseModel

@dataclass
class MyAgentDeps:
    worker_id: int
    agent_name: str
    database: DatabaseConn
    user_context: dict
```

### Step 2: Define Output Model

Use Pydantic models to ensure structured, validated outputs:

```python
from pydantic import BaseModel, Field

class MyAgentOutput(BaseModel):
    """Structured output from the agent."""
    result: str = Field(description="The main result")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    recommendations: list[str] = Field(description="List of recommendations")
```

### Step 3: Configure the Agent

Create an agent configuration:

```python
agent = Agent(
    model='openai:gpt-4o',
    deps_type=MyAgentDeps,
    tools=[tool1, tool2],
    system_prompt="""
You are a specialized agent that...

**Your Workflow:**
1. Analyze the input
2. Use available tools
3. Generate structured output

**Guidelines:**
- Be precise and accurate
- Use tools when needed
- Follow the output schema
""",
    output_type=MyAgentOutput,
    model_settings={
        "temperature": 0.7,
    },
    retries=3,
)
```

### Step 4: Implement the Agent

```python
# Create the agent instance
def get_my_agent():
    """Factory function to create the agent."""
    return Agent(
        model='openai:gpt-4o',
        deps_type=MyAgentDeps,
        tools=[tool1, tool2],
        system_prompt="You are a helpful assistant...",
        model_settings={"temperature": 0.7},
        output_type=MyAgentOutput,
        retries=3,
    )

my_agent = get_my_agent()

# Create a runner function
async def run_my_agent(
    prompt: str,
    deps: MyAgentDeps,
) -> MyAgentOutput:
    """
    Run the agent with the given prompt.
    
    Args:
        prompt: User prompt for the agent
        deps: Dependencies for the agent
        
    Returns:
        MyAgentOutput: Structured output from the agent
    """
    result = await my_agent.run(prompt, deps=deps)
    return result.output
```

## Model Configuration

### Supported Providers

Pydantic AI supports multiple LLM providers:

- **OpenAI**: `openai:gpt-4o`, `openai:o3-mini`
- **Google**: `google-gla:gemini-2.5-flash`, `google:gemini-pro`
- **Anthropic**: `anthropic:claude-3-5-sonnet-latest`
- **Mistral**: `mistral:mistral-large-latest`
- **Groq**: `groq:llama-3.3-70b-versatile`
- **Cohere**: `cohere:command`
- **Azure OpenAI**: Custom Azure endpoints
- **Bedrock**: AWS Bedrock models
- **Ollama**: Local models via Ollama

### Model Settings

Configure model behavior with settings:

```python
model_settings = {
    "temperature": 0.7,        # Creativity (0.0-2.0)
    "max_tokens": 4096,        # Maximum response length
    "top_p": 0.95,            # Nucleus sampling
    "frequency_penalty": 0.0,  # Reduce repetition
    "presence_penalty": 0.0,   # Encourage new topics
}

agent = Agent(
    model='openai:gpt-4o',
    model_settings=model_settings,
)
```

## Dependency Injection

### Base Dependencies

Define dependencies using dataclasses:

```python
from dataclasses import dataclass

@dataclass
class MyDeps:
    """Dependencies for the agent."""
    user_id: int
    context: dict
```

### Accessing Dependencies in Tools

```python
from pydantic_ai import RunContext

@agent.tool
async def my_tool(ctx: RunContext[MyDeps], param: str) -> str:
    """A tool that uses dependencies."""
    user_id = ctx.deps.user_id
    # Perform operation using deps
    return f"Processed for user {user_id}: {param}"
```

## Tools and Capabilities

### Tool Types

#### 1. Plain Tools (No Context)

For stateless operations:

```python
@agent.tool_plain
def calculate_hash(text: str) -> str:
    """Calculate SHA256 hash of text."""
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()
```

#### 2. Context-Aware Tools

For operations needing dependencies:

```python
@agent.tool
async def get_user_data(ctx: RunContext[MyDeps], user_id: int) -> dict:
    """Fetch user data."""
    return await ctx.deps.database.get_user(user_id)
```

### Registering Tools

#### Method 1: Decorator (Recommended)

```python
agent = Agent('openai:gpt-4o', deps_type=MyDeps)

@agent.tool
async def my_tool(ctx: RunContext[MyDeps], param: str) -> str:
    """Tool docstring becomes the description for the LLM."""
    return await some_operation(param)
```

#### Method 2: Constructor

```python
from pydantic_ai import Agent, Tool

def tool1(param: str) -> str:
    return f"Result: {param}"

def tool2(ctx: RunContext[MyDeps], param: int) -> int:
    return param * ctx.deps.multiplier

agent = Agent(
    'openai:gpt-4o',
    deps_type=MyDeps,
    tools=[
        Tool(tool1, takes_ctx=False),
        Tool(tool2, takes_ctx=True),
    ]
)
```

### Tool Best Practices

1. **Clear Docstrings**: The docstring becomes the tool description for the LLM
2. **Type Hints**: Use type hints for parameter and return value validation
3. **Error Handling**: Return informative error messages
4. **Granularity**: Make tools focused and single-purpose
5. **Validation**: Use Pydantic models for complex parameters

```python
from pydantic import BaseModel, Field

class SearchParams(BaseModel):
    query: str = Field(description="Search query")
    limit: int = Field(default=10, ge=1, le=100)
    filters: dict[str, Any] = Field(default_factory=dict)

@agent.tool
async def search_data(
    ctx: RunContext[MyDeps], 
    params: SearchParams
) -> list[dict]:
    """
    Search for data with filters.
    
    Args:
        params: Search parameters including query, limit, and filters
        
    Returns:
        List of matching results
    """
    return await ctx.deps.database.search(
        query=params.query,
        limit=params.limit,
        filters=params.filters
    )
```

## Output Validation

### Structured Output

Define expected output structure:

```python
from pydantic import BaseModel, Field

class AnalysisResult(BaseModel):
    """Structured analysis result."""
    summary: str = Field(description="Brief summary")
    key_findings: list[str] = Field(description="Main findings")
    confidence: float = Field(ge=0.0, le=1.0)
    recommendations: list[str]

agent = Agent(
    'openai:gpt-4o',
    output_type=AnalysisResult,
)

result = await agent.run("Analyze this data...")
print(result.output.summary)  # Guaranteed to be AnalysisResult
```

### Multiple Output Types

Use `ToolOutput` for different output types:

```python
from pydantic_ai import ToolOutput

class SuccessResult(BaseModel):
    status: str = "success"
    data: dict

class ErrorResult(BaseModel):
    status: str = "error"
    message: str

agent = Agent(
    'openai:gpt-4o',
    output_type=[
        ToolOutput(SuccessResult, name='return_success'),
        ToolOutput(ErrorResult, name='return_error'),
    ]
)

result = await agent.run("Process this...")
if isinstance(result.output, SuccessResult):
    print("Success:", result.output.data)
else:
    print("Error:", result.output.message)
```

### Text Output

For simple text responses:

```python
agent = Agent('openai:gpt-4o')  # No output_type specified

result = await agent.run("What is Python?")
print(result.output)  # Plain string
```

## Testing Strategies

### 1. Using TestModel

`TestModel` simulates LLM responses for testing:

```python
import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

agent = Agent('openai:gpt-4o')

@pytest.mark.asyncio
async def test_agent_basic():
    """Test basic agent functionality."""
    test_model = TestModel()
    
    with agent.override(model=test_model):
        result = await agent.run('Test prompt')
        assert result.output == 'success (no tool calls)'
    
    # Verify no tools were called
    assert test_model.last_model_request_parameters.function_tools == []
```

### 2. Using FunctionModel

`FunctionModel` allows custom response logic:

```python
from pydantic_ai.models.function import FunctionModel, AgentInfo
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart

def custom_model_function(
    messages: list[ModelMessage], 
    info: AgentInfo
) -> ModelResponse:
    """Custom logic to simulate model behavior."""
    user_prompt = messages[-1].parts[-1].content
    
    if "error" in user_prompt.lower():
        return ModelResponse(parts=[TextPart('Error detected')])
    
    return ModelResponse(parts=[TextPart('Success')])

@pytest.mark.asyncio
async def test_with_function_model():
    """Test with custom model logic."""
    with agent.override(model=FunctionModel(custom_model_function)):
        result = await agent.run('Process this')
        assert result.output == 'Success'
        
        result = await agent.run('Handle error')
        assert result.output == 'Error detected'
```

### 3. Testing Tool Calls

Test that tools are called correctly:

```python
from pydantic_ai.messages import ToolCallPart, ToolReturnPart

def model_with_tool_call(
    messages: list[ModelMessage],
    info: AgentInfo
) -> ModelResponse:
    """Simulate tool calling."""
    if len(messages) == 1:
        # First call: invoke tool
        return ModelResponse(parts=[
            ToolCallPart('my_tool', {'param': 'value'})
        ])
    else:
        # Second call: use tool result
        tool_result = messages[-1].parts[0]
        return ModelResponse(parts=[
            TextPart(f'Processed: {tool_result.content}')
        ])

@pytest.mark.asyncio
async def test_tool_invocation():
    """Test that agent calls tools correctly."""
    with agent.override(model=FunctionModel(model_with_tool_call)):
        result = await agent.run('Use the tool')
        assert 'Processed' in result.output
```

### 4. Capture Run Messages

Verify the exact conversation flow:

```python
from pydantic_ai import capture_run_messages
from pydantic_ai.messages import ModelRequest, ModelResponse

@pytest.mark.asyncio
async def test_conversation_flow():
    """Test the message flow."""
    with capture_run_messages() as messages:
        result = await agent.run('Test prompt')
    
    # Verify message structure
    assert len(messages) >= 2
    assert isinstance(messages[0], ModelRequest)
    assert isinstance(messages[1], ModelResponse)
    
    # Check specific message content
    user_prompt = messages[0].parts[-1]
    assert user_prompt.content == 'Test prompt'
```

## Best Practices

### 1. Agent Design

- **Single Responsibility**: Each agent should have a clear, focused purpose
- **Tool Granularity**: Provide focused tools rather than monolithic ones
- **Clear Instructions**: Write detailed system prompts with examples
- **Output Validation**: Always use Pydantic models for structured outputs
- **Error Handling**: Implement robust error handling and retries

### 2. Configuration Management

- **Environment Variables**: Use `.env` for API keys and sensitive data
- **Model Settings**: Tune temperature and other settings per agent role
- **Version Control**: Track configuration changes

### 3. Dependency Injection

- **Minimal Dependencies**: Only inject what the agent needs
- **Immutable Data**: Use dataclasses with frozen=True when possible
- **Type Safety**: Always type-hint dependencies
- **Connection Management**: Manage external connections properly

### 4. Tool Development

- **Descriptive Names**: Use clear, action-oriented names
- **Complete Docstrings**: LLMs use docstrings to understand tools
- **Parameter Validation**: Use Pydantic models for complex parameters
- **Error Messages**: Return clear, actionable error messages
- **Async Operations**: Use async/await for I/O operations

### 5. Testing

- **Unit Tests**: Test individual tools and functions
- **Integration Tests**: Test full agent workflows
- **Mock External Services**: Use TestModel and FunctionModel
- **Verify Tool Calls**: Check that correct tools are invoked
- **Message Capture**: Validate conversation flows

### 6. Performance

- **Caching**: Cache frequently accessed data
- **Batch Operations**: Process multiple items together
- **Async Execution**: Use asyncio for concurrent operations
- **Token Limits**: Monitor and optimize prompt lengths
- **Model Selection**: Use appropriate model sizes

### 7. Monitoring and Debugging

- **Logging**: Log agent activities and decisions
- **Message History**: Preserve conversation history
- **Error Tracking**: Monitor and alert on failures
- **Usage Metrics**: Track token usage and costs
- **Performance Metrics**: Measure response times

### 8. Security

- **Input Validation**: Validate all user inputs
- **API Key Management**: Never hardcode API keys
- **Access Control**: Verify permissions before operations
- **Data Sanitization**: Clean sensitive data from logs
- **Rate Limiting**: Implement rate limits for API calls

## Example: Complete Agent Implementation

```python
from dataclasses import dataclass
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

@dataclass
class AnalyzerDeps:
    project_id: int

class CodeAnalysis(BaseModel):
    quality_score: float = Field(ge=0.0, le=10.0)
    issues: list[str]
    suggestions: list[str]
    complexity: dict[str, int]

agent = Agent(
    model='openai:gpt-4o',
    deps_type=AnalyzerDeps,
    system_prompt="""
You are a Code Analysis Specialist. Your role is to analyze code quality,
identify issues, and suggest improvements.

**Your Workflow:**
1. Analyze the code structure
2. Measure complexity
3. Generate recommendations
4. Provide structured analysis

**Guidelines:**
- Be specific and constructive
- Prioritize critical issues
- Provide clear examples
""",
    output_type=CodeAnalysis,
    retries=3,
)

@agent.tool
async def analyze_code(ctx: RunContext[AnalyzerDeps], code: str) -> dict:
    """Analyze code for quality metrics."""
    # Implementation here
    return {"complexity": 5, "issues": ["Use more descriptive names"]}

async def run_analyzer_agent(
    code_content: str,
    project_id: int,
) -> CodeAnalysis:
    """Run code analysis."""
    deps = AnalyzerDeps(project_id=project_id)
    
    result = await agent.run(
        f"Analyze this code:\n\n{code_content}",
        deps=deps
    )
    
    return result.output
```

## Next Steps

- Read the [Pydantic AI Documentation](https://ai.pydantic.dev/) for advanced features
- Explore examples and tutorials on the Pydantic AI GitHub repository
