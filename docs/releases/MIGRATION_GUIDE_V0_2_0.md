# Migration Guide: v0.1.x → v0.2.0

**Target Version**: 0.2.0  
**Release Date**: November 17, 2025  
**Breaking Changes**: 1 (retrieve_memories method)  
**Migration Time**: 10-30 minutes

---

## 📋 Quick Summary

| Item | v0.1.x | v0.2.0 |
|------|--------|--------|
| **Retrieval Method** | `retrieve_memories()` | `search_memories()` + `deep_search_memories()` |
| **Breaking Changes** | None | 1 method removed |
| **Migration Level** | N/A | Minimal (rename + optional refactoring) |
| **Compatibility** | N/A | Fully compatible except retrieval |

---

## 🔄 Step-by-Step Migration

### Step 1: Identify Usage Points

Find all calls to `retrieve_memories()`:

```bash
# Unix/Mac
grep -r "retrieve_memories" . --include="*.py"

# Windows PowerShell
Get-ChildItem -Recurse -Include "*.py" | Select-String "retrieve_memories"
```

### Step 2: Understand Your Usage Pattern

There are two common patterns:

**Pattern A: Simple Search (Fast)**
```python
# v0.1.x
result = await agent_mem.retrieve_memories(
    external_id="agent-123",
    query="authentication"
)

# v0.2.0 - Simple rename
result = await agent_mem.search_memories(
    external_id="agent-123",
    query="authentication"
)
```

**Pattern B: With Synthesis (Deep)**
```python
# v0.1.x
result = await agent_mem.retrieve_memories(
    external_id="agent-123",
    query="How does X relate to Y?",
    synthesis=True
)

# v0.2.0 - Different method (synthesis enabled by default)
result = await agent_mem.deep_search_memories(
    external_id="agent-123",
    query="How does X relate to Y?"
)
```

### Step 3: Update Your Code

Choose based on your pattern:

#### Pattern A → search_memories() ✅

```python
# Before
result = await agent_mem.retrieve_memories(
    external_id="agent-123",
    query="JWT tokens",
    limit=5
)

# After (simple rename)
result = await agent_mem.search_memories(
    external_id="agent-123",
    query="JWT tokens",
    limit=5
)
```

#### Pattern B → deep_search_memories() ✅

```python
# Before
result = await agent_mem.retrieve_memories(
    external_id="agent-123",
    query="authentication and security",
    synthesis=True,
    limit=10
)

# After (remove synthesis parameter, rename method)
result = await agent_mem.deep_search_memories(
    external_id="agent-123",
    query="authentication and security",
    limit=10
)
```

### Step 4: Verify Results Structure

Results are now hierarchical. Update any code that processes results:

#### Old Result Structure (v0.1.x)
```python
result = await agent_mem.retrieve_memories(...)
for chunk in result.chunks:  # All chunks mixed
    print(chunk.content)
for entity in result.entities:  # Flat list
    print(entity.name)
```

#### New Result Structure (v0.2.0 - search_memories)
```python
result = await agent_mem.search_memories(...)
for chunk in result.shortterm_chunks:  # Shortterm only
    print(chunk.content)
for chunk in result.longterm_chunks:  # Longterm only
    print(chunk.content)
for triplet in result.shortterm_triplets:  # Relationships
    print(f"{triplet.subject} {triplet.predicate} {triplet.object}")
```

#### New Result Structure (v0.2.0 - deep_search_memories)
```python
result = await agent_mem.deep_search_memories(...)

# All the same as search_memories PLUS synthesis
print(result.synthesis)  # AI-generated analysis

for triplet in result.longterm_triplets:
    print(f"{triplet.subject} {triplet.predicate} {triplet.object}")
    print(f"  Importance: {triplet.importance * 100}%")
```

---

## 💡 Migration Patterns

### Pattern 1: Read-Once Usage (Easiest)

If you just need results and don't care about tier separation:

```python
# v0.1.x
result = await agent_mem.retrieve_memories(
    external_id="agent-123",
    query="search term"
)
# Use result...

# v0.2.0 (minimal changes)
result = await agent_mem.search_memories(
    external_id="agent-123",
    query="search term"
)
# Use result... (access result.shortterm_chunks + result.longterm_chunks)
```

### Pattern 2: Tier-Aware Usage (Moderate)

If you need to treat shortterm/longterm differently:

```python
# v0.1.x - had to check .tier attribute
result = await agent_mem.retrieve_memories(...)
for chunk in result.chunks:
    if chunk.tier == "shortterm":
        # Process shortterm
    elif chunk.tier == "longterm":
        # Process longterm

# v0.2.0 - explicit separation
result = await agent_mem.search_memories(...)
for chunk in result.shortterm_chunks:
    # Process shortterm (no tier check needed)
for chunk in result.longterm_chunks:
    # Process longterm (no tier check needed)
```

### Pattern 3: Synthesis + Analysis (Advanced)

If you need AI analysis:

```python
# v0.1.x - synthesis optional
result = await agent_mem.retrieve_memories(
    external_id="agent-123",
    query="complex question",
    synthesis=True
)
synthesis = result.synthesis

# v0.2.0 - explicit deep search
result = await agent_mem.deep_search_memories(
    external_id="agent-123",
    query="complex question"
)
synthesis = result.synthesis  # Always provided
```

---

## 🔍 Common Issues & Solutions

### Issue 1: AttributeError - 'module' object has no attribute 'retrieve_memories'

**Cause**: You're still calling the old method.

**Solution**:
```python
# ❌ This won't work in v0.2.0
result = await agent_mem.retrieve_memories(...)

# ✅ Use new methods instead
result = await agent_mem.search_memories(...)        # Fast
result = await agent_mem.deep_search_memories(...)   # With synthesis
```

### Issue 2: Processing Results - Chunks Not Found

**Cause**: Chunks are now separated by tier.

**Solution**:
```python
# ❌ This won't work - wrong attribute name
for chunk in result.chunks:  # This doesn't exist
    print(chunk)

# ✅ Use tier-specific attributes
for chunk in result.shortterm_chunks:
    print(chunk)
for chunk in result.longterm_chunks:
    print(chunk)

# ✅ Or combine them
all_chunks = result.shortterm_chunks + result.longterm_chunks
for chunk in all_chunks:
    print(chunk)
```

### Issue 3: Relationships Access Pattern Changed

**Cause**: Relationships are now triplets, not entities.

**Solution**:
```python
# ❌ This won't work
for entity in result.entities:
    print(entity.name)

# ✅ Use triplets instead
for triplet in result.shortterm_triplets:
    print(f"{triplet.subject} {triplet.predicate} {triplet.object}")
for triplet in result.longterm_triplets:
    print(f"{triplet.subject} {triplet.predicate} {triplet.object}")
```

### Issue 4: Synthesis Not Working

**Cause**: Using `search_memories()` instead of `deep_search_memories()`.

**Solution**:
```python
# ❌ search_memories doesn't include synthesis
result = await agent_mem.search_memories(...)
print(result.synthesis)  # None

# ✅ Use deep_search_memories for synthesis
result = await agent_mem.deep_search_memories(...)
print(result.synthesis)  # AI-generated analysis
```

---

## 📊 Result Structure Reference

### search_memories() Result

```python
RetrievalResultV2(
    mode: "search",
    shortterm_chunks: [
        ShorttermRetrievedChunk(
            id: int,
            content: str,
            score: float (0-1),
            section_id: Optional[str],
            metadata: dict
        ),
        ...
    ],
    longterm_chunks: [
        LongtermRetrievedChunk(
            id: int,
            content: str,
            score: float (0-1),
            importance: float (0-1),
            start_date: datetime,
            last_updated: Optional[datetime],
            metadata: dict
        ),
        ...
    ],
    shortterm_triplets: [
        ShorttermKnowledgeTriplet(
            subject: str,
            predicate: str,
            object: str,
            importance: float (0-1),
            access_count: int,
            description: Optional[str]
        ),
        ...
    ],
    longterm_triplets: [
        LongtermKnowledgeTriplet(
            subject: str,
            predicate: str,
            object: str,
            importance: float (0-1),
            start_date: Optional[datetime],
            temporal_validity: Optional[str],
            description: Optional[str]
        ),
        ...
    ],
    synthesis: None,  # Not included in search_memories
    search_strategy: str,
    confidence: float (0-1),
    metadata: dict
)
```

### deep_search_memories() Result

```python
# Same as above PLUS:
synthesis: str  # AI-generated analysis of findings
```

---

## 🛠️ Automated Migration Script

If you have many calls to update, you can use this script:

```python
import re
import sys
from pathlib import Path

def migrate_retrieve_memories(file_path: Path) -> bool:
    """Migrate retrieve_memories calls to search_memories."""
    content = file_path.read_text()
    original = content
    
    # Pattern 1: with synthesis=True → deep_search_memories
    content = re.sub(
        r'\.retrieve_memories\(([^)]*synthesis\s*=\s*True[^)]*)\)',
        r'.deep_search_memories(\1)',
        content
    )
    # Remove the now-obsolete synthesis parameter
    content = re.sub(
        r',?\s*synthesis\s*=\s*True\s*,?',
        '',
        content
    )
    
    # Pattern 2: without synthesis → search_memories
    content = re.sub(
        r'\.retrieve_memories\(',
        r'.search_memories(',
        content
    )
    
    if content != original:
        file_path.write_text(content)
        print(f"✅ Updated: {file_path}")
        return True
    return False

# Run on all Python files
for py_file in Path(".").rglob("*.py"):
    if py_file.is_file() and "venv" not in str(py_file):
        migrate_retrieve_memories(py_file)

print("✅ Migration complete!")
```

---

## ✅ Verification Checklist

After migration, verify:

- [ ] No more `retrieve_memories()` calls in codebase
- [ ] Tests run successfully
- [ ] Search results are processed correctly
- [ ] Synthesis works in deep search mode
- [ ] No "AttributeError" exceptions
- [ ] Performance is acceptable

---

## 📚 Additional Resources

- **Full API Documentation**: See `docs/API.md`
- **MCP Tools**: See `docs/MCP_TOOLS.md`
- **Examples**: See `docs/guide/EXAMPLES.md`
- **Architecture**: See `docs/ARCHITECTURE.md`

---

## 🆘 Need Help?

### Common Questions

**Q: Should I use search_memories or deep_search_memories?**

A: Use `search_memories()` for:
- Quick lookups (< 200ms required)
- Fact retrieval
- Simple queries

Use `deep_search_memories()` for:
- Understanding relationships
- Complex questions
- Need for analysis (synthesis)

**Q: Do I need to change my memory creation code?**

A: No! Only retrieval methods changed.

**Q: What if I can't update all at once?**

A: That's OK - only the retrieval API changed. Everything else works as-is.

**Q: Will old code still work?**

A: No, `retrieve_memories()` is removed in v0.2.0. You must update retrieval calls.

### Support

- GitHub Issues: https://github.com/Ganzzi/agent-reminiscence/issues
- Discussions: https://github.com/Ganzzi/agent-reminiscence/discussions

---

## 📝 Migration Template

Here's a template for migrating a function:

```python
# Before (v0.1.x)
async def find_authentication_info(agent_id: str):
    result = await agent_mem.retrieve_memories(
        external_id=agent_id,
        query="authentication",
        limit=5
    )
    return [chunk.content for chunk in result.chunks]

# After (v0.2.0) - Option 1: Fast search
async def find_authentication_info(agent_id: str):
    result = await agent_mem.search_memories(
        external_id=agent_id,
        query="authentication",
        limit=5
    )
    chunks = result.shortterm_chunks + result.longterm_chunks
    return [chunk.content for chunk in chunks]

# After (v0.2.0) - Option 2: Deep search with analysis
async def find_authentication_info(agent_id: str):
    result = await agent_mem.deep_search_memories(
        external_id=agent_id,
        query="How is authentication implemented?"
    )
    # Get AI-generated analysis
    analysis = result.synthesis
    # Get relationships
    relationships = result.shortterm_triplets + result.longterm_triplets
    return {
        "analysis": analysis,
        "relationships": relationships
    }
```

---

## 🎉 You're Done!

Your code is now upgraded to v0.2.0! Enjoy the improved search performance and new deep search capabilities.

---

**Last Updated**: November 17, 2025  
**Version**: 0.2.0  
**Status**: ✅ Ready
