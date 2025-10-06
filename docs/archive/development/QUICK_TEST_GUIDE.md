# Quick Test Guide

## Running Tests

### All Tests
```powershell
# Run all tests in test_er_extractor.py
uv run pytest tests/test_er_extractor.py -v

# Run all tests in test_memorizer_agent.py
uv run pytest tests/test_memorizer_agent.py -v

# Run both test files
uv run pytest tests/test_er_extractor.py tests/test_memorizer_agent.py -v
```

### Unit Tests Only (No API Key Needed)
```powershell
# Skip integration tests that require API keys
uv run pytest tests/test_er_extractor.py -v -m "not integration"
uv run pytest tests/test_memorizer_agent.py -v -m "not integration"
```

### Integration Tests Only (Requires GOOGLE_API_KEY)
```powershell
# Set API key first
$env:GOOGLE_API_KEY="your-api-key-here"

# Run integration tests
uv run pytest tests/test_er_extractor.py -v -m integration
uv run pytest tests/test_memorizer_agent.py -v -m integration
```

### Specific Tests
```powershell
# Run a single test
uv run pytest tests/test_er_extractor.py::test_entity_type_enum -v

# Run tests matching a pattern
uv run pytest tests/test_er_extractor.py -v -k "entity"

# Run with coverage
uv run pytest tests/test_er_extractor.py --cov=agent_mem.agents.er_extractor -v
```

## Test Files Overview

### `tests/test_er_extractor.py`
- **20+ tests** for Entity-Relationship extraction
- Tests with `TestModel` (mock responses)
- Tests with `FunctionModel` (programmatic logic)  
- Integration tests with Google Gemini
- Entity/Relationship type validation

### `tests/test_memorizer_agent.py`
- **15+ tests** for conflict resolution
- Tests with `TestModel` and `FunctionModel`
- Integration tests with Google Gemini
- Tool call simulation tests
- Conflict resolution quality validation

## Key Test Patterns

### Using TestModel (Pre-defined Response)
```python
test_model = TestModel(
    custom_result_text='{"entities": [], "relationships": []}'
)
agent = Agent(model=test_model, output_type=ExtractionResult)
result = await agent.run("test input")
```

### Using FunctionModel (Programmatic Logic)
```python
def extract_mock(messages, info):
    # Your custom logic here
    return ExtractionResult(entities=[], relationships=[])

function_model = FunctionModel(extract_mock)
agent = Agent(model=function_model, output_type=ExtractionResult)
result = await agent.run("test input")
```

### Using Real Model (Integration Test)
```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_with_real_model():
    model = GoogleModel("gemini-2.0-flash-exp")
    agent = Agent(model=model, output_type=ExtractionResult)
    result = await agent.run("test input")
    # assertions...
```

## Environment Setup

### Required for Unit Tests
```powershell
# Nothing! Unit tests work without any API keys
```

### Required for Integration Tests
```powershell
# Set Google API key
$env:GOOGLE_API_KEY="AIza..."

# Or add to .env file
# GOOGLE_API_KEY=AIza...
```

## Troubleshooting

### Tests Skip with "Skipping real model test"
- ✅ This is normal if `GOOGLE_API_KEY` is not set
- ✅ Unit tests still run successfully
- 💡 Set `GOOGLE_API_KEY` to run integration tests

### Import Errors
```powershell
# Reinstall dependencies
uv sync
```

### All Tests Fail
```powershell
# Check if you're in the right directory
cd C:\Users\Administrator\Desktop\ai-army\libs\agent_mem

# Run with verbose output
uv run pytest tests/test_er_extractor.py -v -s
```

## Quick Verification

```powershell
# Verify tests are collected correctly
uv run pytest tests/test_er_extractor.py --collect-only

# Run fastest test to verify setup
uv run pytest tests/test_er_extractor.py::test_entity_type_enum -v
```

## CI/CD Configuration

```yaml
# Example GitHub Actions workflow
- name: Run Unit Tests
  run: uv run pytest tests/ -v -m "not integration"

- name: Run Integration Tests
  env:
    GOOGLE_API_KEY: ${{ secrets.GOOGLE_API_KEY }}
  run: uv run pytest tests/ -v -m integration
```
