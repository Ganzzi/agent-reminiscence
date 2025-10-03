# AgentMem Streamlit UI

## Quick Start

### Installation

1. Install dependencies:
```bash
pip install -r streamlit_app/requirements.txt
```

2. Ensure AgentMem is installed:
```bash
pip install -e .
```

### Running the UI

```bash
# Option 1: Using the run script
python run_ui.py

# Option 2: Using Streamlit directly
streamlit run streamlit_app/app.py
```

The UI will be available at `http://localhost:8501`

### Configuration

Database settings can be configured via environment variables:

```bash
# PostgreSQL
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DB=agent_memory
export POSTGRES_USER=postgres
export POSTGRES_PASSWORD=postgres

# Neo4j
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=password
```

Or modify the defaults in `streamlit_app/config.py`.

## Features

### 📚 Browse Templates
- View all 62 BMAD templates
- Filter by agent type
- Search and preview templates

### ➕ Create Memory
- Create memories with pre-built templates
- Custom YAML support
- Section-by-section initialization

### 📋 View Memories
- List all agent memories
- Expandable section details
- Metadata display

### ✏️ Update Memory
- Edit memory sections
- Markdown editor
- Update count tracking
- Consolidation warnings

### 🗑️ Delete Memory
- Safe deletion with confirmation
- Full memory preview before delete

## Development Status

**Phase 1: Foundation** ✅ Complete
- Directory structure
- Configuration files
- Utility modules (template_loader, yaml_validator, formatters)
- Service modules (template_service, memory_service)
- Main app with navigation

**Phase 2: Browse Templates** 🚧 In Progress

**Phase 3-6: Other Pages** 📅 Planned

## Project Structure

```
streamlit_app/
├── app.py                      # Main application
├── config.py                   # Configuration
├── requirements.txt            # Dependencies
├── pages/                      # Multi-page app
│   ├── 1_📚_Browse_Templates.py
│   ├── 2_➕_Create_Memory.py
│   ├── 3_📋_View_Memories.py
│   ├── 4_✏️_Update_Memory.py
│   └── 5_🗑️_Delete_Memory.py
├── components/                 # Reusable UI components
├── services/                   # Business logic
│   ├── template_service.py
│   └── memory_service.py
└── utils/                      # Utilities
    ├── template_loader.py
    ├── yaml_validator.py
    └── formatters.py
```

## Next Steps

See `docs/STREAMLIT_UI_PLAN.md` for the full implementation plan.
