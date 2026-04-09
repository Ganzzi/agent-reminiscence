import pytest
from unittest.mock import MagicMock, AsyncMock
from agent_reminiscence.core import AgentMem
from pydantic_ai.usage import RunUsage

@pytest.mark.asyncio
async def test_update_active_memory_alias():
    agent_mem = AgentMem()
    agent_mem._memory_manager = AsyncMock()
    agent_mem._initialized = True
    
    external_id = "agent-123"
    memory_id = 1
    sections = [{"section_id": "test", "new_content": "updated", "action": "replace"}]
    
    await agent_mem.update_active_memory(external_id, memory_id, sections)
    
    agent_mem._memory_manager.update_active_memory_sections.assert_called_once_with(
        external_id=external_id,
        memory_id=memory_id,
        sections=sections
    )

@pytest.mark.asyncio
async def test_set_usage_processor_types():
    agent_mem = AgentMem()
    agent_mem._memory_manager = MagicMock()
    agent_mem._initialized = True
    
    # Define a processor matching the protocol
    def valid_processor(eid: str, usage: RunUsage) -> None:
        pass
        
    agent_mem.set_usage_processor(valid_processor)
    assert agent_mem._memory_manager.usage_processor == valid_processor
