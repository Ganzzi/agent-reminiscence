"""
AgentMem Streamlit UI - Main Application
"""

import streamlit as st
from pathlib import Path
import sys

# Add parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from streamlit_app.config import (
    APP_TITLE,
    APP_ICON,
    PAGE_LAYOUT,
    TEMPLATES_DIR,
    DB_CONFIG,
    AGENT_TYPES,
    AGENT_DISPLAY_NAMES,
)
from streamlit_app.services.template_service import TemplateService
from streamlit_app.services.memory_service import MemoryService


# Configure page
st.set_page_config(
    page_title=APP_TITLE, page_icon=APP_ICON, layout=PAGE_LAYOUT, initial_sidebar_state="expanded"
)

# Global CSS for compact layout
st.markdown(
    """
<style>
    /* Reduce main container padding */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }

    /* Reduce spacing between elements */
    .element-container {
        margin-bottom: 0.5rem;
    }

    /* Compact metrics */
    .stMetric {
        margin-bottom: 0.5rem;
    }

    .stMetric > div {
        padding: 0.5rem;
    }

    /* Compact buttons */
    .stButton > button {
        margin: 0.1rem;
        padding: 0.25rem 0.75rem;
    }

    /* Compact columns */
    .stColumn {
        padding: 0 0.25rem;
    }

    /* Reduce expander spacing */
    .stExpander {
        margin-bottom: 0.25rem;
    }

    /* Compact text inputs */
    .stTextInput > div > div > input {
        padding: 0.25rem;
    }

    /* Compact select boxes */
    .stSelectbox > div > div {
        padding: 0.25rem;
    }

    /* Reduce header margins */
    h1, h2, h3, h4, h5, h6 {
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }

    /* Compact sidebar */
    .css-1d391kg {  /* sidebar */
        padding-top: 1rem;
    }

    /* Reduce footer spacing */
    footer {
        margin-top: 0.5rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


def initialize_services():
    """Initialize services in session state"""
    if "template_service" not in st.session_state:
        st.session_state.template_service = TemplateService(TEMPLATES_DIR)

    if "memory_service" not in st.session_state:
        st.session_state.memory_service = MemoryService(DB_CONFIG)


def render_sidebar():
    """Render sidebar with navigation and agent info"""
    with st.sidebar:
        st.title(f"{APP_ICON} {APP_TITLE}")
        st.markdown("Memory Management Dashboard")
        st.markdown("---")

        # Agent ID input (persisted across pages)
        st.subheader("Agent ID")

        if "agent_external_id" not in st.session_state:
            st.session_state.agent_external_id = ""

        agent_id = st.text_input(
            "External ID",
            value=st.session_state.agent_external_id,
            placeholder="agent-001",
            help="This ID identifies the agent across all operations",
        )

        if agent_id != st.session_state.agent_external_id:
            st.session_state.agent_external_id = agent_id

        st.markdown("---")

        # Quick stats
        st.subheader("📊 Stats")

        try:
            template_service = st.session_state.get("template_service")
            if template_service:
                stats = template_service.get_template_stats()
                st.metric("Templates", stats["total_templates"])
                st.metric("Agent Types", stats["total_agents"])
        except Exception as e:
            st.caption(f"Stats unavailable: {e}")

        st.markdown("---")

        # Navigation help
        st.subheader("📚 Pages")
        st.markdown(
            """
- 📚 **Templates** - Browse library
- ➕ **Create** - New memory
- 📋 **View** - List memories
- ✏️ **Update** - Edit sections
- 🗑️ **Delete** - Remove memories
        """
        )

        st.markdown("---")

        # Footer
        st.caption("AgentMem UI v1.0")
        st.caption("BMAD Memory Templates")


def main():
    """Main application entry point"""
    # Initialize services
    initialize_services()

    # Render sidebar
    render_sidebar()

    # Main content area
    st.title(f"{APP_ICON} AgentMem Dashboard")

    st.markdown("Memory management system for BMAD agents with pre-built templates")

    st.divider()

    # Quick action cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### � Templates")
        st.markdown("Browse 62 pre-built templates across 10 agent types")
        st.page_link("pages/1_Browse_Templates.py", label="Browse Templates", icon="📚")

    with col2:
        st.markdown("### ➕ Create")
        st.markdown("Create new memories using templates or custom YAML")
        st.page_link("pages/2_Create_Memory.py", label="Create Memory", icon="➕")

    with col3:
        st.markdown("### 📋 View")
        st.markdown("View and manage active memories for agents")
        st.page_link("pages/3_View_Memories.py", label="View Memories", icon="📋")

    st.divider()

    # Getting started
    st.markdown("### 🎯 Quick Start")
    st.markdown("1. Enter your agent's ID in the sidebar")
    st.markdown("2. Browse templates or view existing memories")
    st.markdown("3. Create, update, or delete memories as needed")

    st.divider()

    # Agent types
    st.markdown("### 🏗️ Supported Agent Types")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
- 👔 Business Analyst
- 🏗️ System Architect  
- 💻 Full-Stack Developer
- 📦 Product Manager
- 📝 Product Owner
        """
        )

    with col2:
        st.markdown(
            """
- 🧪 QA/Test Architect
- 🏃 Scrum Master
- 🎨 UX Expert
- 👑 BMAD Master
- 🔄 BMAD Orchestrator
        """
        )

    st.divider()

    # Footer
    st.caption("AgentMem UI v1.0 | BMAD Memory Templates")


if __name__ == "__main__":
    main()
