"""
View Memories Page

View and browse active memories for an agent.
"""

import streamlit as st
from typing import Optional, List, Dict, Any
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from streamlit_app.services.memory_service import MemoryService
from streamlit_app.utils.formatters import Formatters
from streamlit_app.config import APP_TITLE, APP_ICON, DB_CONFIG
import asyncio

# Page configuration
st.set_page_config(page_title=f"View Memories - {APP_TITLE}", page_icon=APP_ICON, layout="wide")

# Custom CSS for compact layout
st.markdown(
    """
<style>
    .main .block-container {padding-top: 1rem; padding-bottom: 1rem;}
    .stExpander {margin-bottom: 0.25rem;}
    .stButton>button {margin: 0.1rem;}
    .stTextInput input {padding: 0.25rem;}
    .stSelectbox select {padding: 0.25rem;}
    h1, h2, h3 {margin-top: 0.5rem; margin-bottom: 0.5rem;}
    .stColumns {gap: 0.5rem;}
</style>
""",
    unsafe_allow_html=True,
)

# Initialize services
if "memory_service" not in st.session_state:
    st.session_state.memory_service = MemoryService(DB_CONFIG)

memory_service = st.session_state.memory_service

# Page header
st.title("📋 Memories")
st.markdown("View and manage active memories for an agent")

# Sidebar - Agent ID persistence
with st.sidebar:
    st.header("🔍 Agent Lookup")

    # Persist external ID in session state
    if "view_external_id" not in st.session_state:
        st.session_state.view_external_id = ""

    external_id = st.text_input(
        "External ID",
        value=st.session_state.view_external_id,
        placeholder="agent-001, UUID, or numeric ID",
        help="Enter the agent's external ID to view their memories",
        key="external_id_input",
    )

    # Update session state
    if external_id != st.session_state.view_external_id:
        st.session_state.view_external_id = external_id

    # ID Type selector
    id_type = st.radio(
        "ID Type",
        options=["String", "UUID", "Integer"],
        horizontal=False,
        help="Type of the external ID",
    )

    st.divider()

    # Load button
    load_clicked = st.button(
        "🔍 Load Memories",
        type="primary",
        use_container_width=True,
        disabled=not external_id.strip(),
    )

    # Refresh button
    if "memories_loaded" in st.session_state and st.session_state.memories_loaded:
        if st.button("🔄 Refresh", use_container_width=True):
            load_clicked = True

    st.divider()

    # Help section
    st.subheader("ℹ️ Help")
    st.markdown(
        """
    **Viewing Memories:**
    1. Enter an agent's external ID
    2. Select the ID type
    3. Click "Load Memories"
    4. Expand memory cards to view sections
    
    **Tips:**
    - Empty sections show placeholder text
    - Update count shows how many times a section was updated
    - Click section title to expand/collapse
    """
    )

# Main content area
if load_clicked and external_id.strip():
    # Load memories from API
    with st.spinner("Loading memories..."):
        try:
            memories = asyncio.run(memory_service.get_active_memories(external_id))

            st.session_state.memories_loaded = True

            if not memories:
                # Empty state
                st.info("📭 No memories found for this agent.")
                st.markdown(
                    """
                **No memories yet?**
                - This agent doesn't have any active memories yet
                - Create a new memory using the **➕ Create Memory** page
                - Make sure you entered the correct External ID
                """
                )
            else:
                # Success message
                st.success(
                    f"✅ Found {len(memories)} active memor{'y' if len(memories) == 1 else 'ies'}"
                )

                # Convert ActiveMemory objects to display format
                display_memories = []
                for memory in memories:
                    formatted = memory_service.format_memory_for_display(memory)

                    # Convert sections dict to list for display
                    sections_list = []
                    for section_id, section_data in formatted["sections"].items():
                        sections_list.append(
                            {
                                "id": section_id,
                                "title": section_id.replace("_", " ").title(),
                                "content": section_data.get("content", ""),
                                "update_count": section_data.get("update_count", 0),
                                "last_updated": formatted["updated_at"],
                            }
                        )

                    formatted["sections"] = sections_list
                    display_memories.append(formatted)

        except Exception as e:
            st.error(f"❌ Error loading memories: {str(e)}")
            st.exception(e)
            display_memories = []

    # Display memory count if we have memories
    if "display_memories" in locals() and display_memories:
        st.markdown(
            f"**{len(display_memories)} memor{'y' if len(display_memories) == 1 else 'ies'}** for `{external_id}`"
        )

        st.divider()

        # Display memories in simplified table format
        for idx, memory in enumerate(display_memories):
            with st.expander(
                f"**{memory['title']}** - ID: {memory['id']} | {len(memory['sections'])} sections",
                expanded=idx == 0,
            ):
                # Quick info in columns
                col1, col2, col3 = st.columns(3)

                with col1:
                    priority = memory["metadata"].get("priority", "medium")
                    priority_emoji, _ = Formatters.format_priority(priority)
                    st.caption(f"**Priority:** {priority_emoji} {priority.title()}")

                with col2:
                    usage = memory["metadata"].get("usage", "session")
                    usage_emoji = Formatters.format_usage_type(usage)
                    st.caption(f"**Usage:** {usage_emoji} {usage.title()}")

                with col3:
                    created = Formatters.format_timestamp(memory["created_at"])
                    st.caption(f"**Created:** {created}")

                st.caption(f"Template: `{memory['template_id']}`")

                st.divider()

                # Sections in simple list
                if memory["sections"]:
                    st.markdown("**Sections:**")

                    for section in memory["sections"]:
                        st.markdown(f"- **{section['title']}** (ID: `{section['id']}`)")

                        update_badge, _ = Formatters.format_update_count(section["update_count"])
                        st.caption(f"  Updates: {update_badge}")

                        # Show content preview
                        if section["content"] and section["content"].strip():
                            content = section["content"]
                            if len(content) > 200:
                                preview = content[:200] + "..."
                                st.caption(f"  {preview}")

                                with st.expander("Show full content"):
                                    st.markdown(content)
                            else:
                                st.caption(f"  {content}")
                        else:
                            st.caption("  _No content_")

                        st.markdown("")  # Spacing

                # Action buttons
                st.divider()
                col1, col2 = st.columns(2)

                with col1:
                    if st.button(
                        "✏️ Update", key=f"update_{memory['id']}", use_container_width=True
                    ):
                        # Store memory info in session state and navigate to Update page
                        st.session_state.update_external_id = external_id
                        st.session_state.update_id_type = id_type
                        st.session_state.selected_memory_id = memory["id"]
                        st.switch_page("pages/4_Update_Memory.py")

                with col2:
                    if st.button(
                        "🗑️ Delete", key=f"delete_{memory['id']}", use_container_width=True
                    ):
                        # Store memory info in session state and navigate to Delete page
                        st.session_state.delete_external_id = external_id
                        st.session_state.delete_id_type = id_type
                        st.session_state.selected_memory_id = memory["id"]
                        st.switch_page("pages/5_Delete_Memory.py")

elif not external_id.strip():
    # Empty state - no external ID
    st.info("👈 Enter an agent's external ID in the sidebar to view their memories")

    st.markdown("### Quick Start")
    st.markdown("1. Enter an external ID in the sidebar")
    st.markdown("2. Select the ID type (String/UUID/Integer)")
    st.markdown("3. Click 'Load Memories'")

else:
    # Empty state - no memories found
    if "memories_loaded" in st.session_state and st.session_state.memories_loaded:
        st.warning(f"No memories found for external ID: `{external_id}`")
        st.markdown(
            "**Suggestions:** Verify the external ID is correct or try creating a new memory for this agent"
        )

# Footer
st.divider()
st.caption(f"Memories Dashboard | {APP_TITLE}")
