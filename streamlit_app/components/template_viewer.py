"""
Template Viewer Component - Display template cards and previews
"""

import streamlit as st
import yaml
from typing import Dict, List, Optional


def render_template_card(template: Dict, key: str):
    """
    Render a single template card

    Args:
        template: Template dictionary
        key: Unique key for the card
    """
    template_data = template.get("template", {})
    metadata = template.get("metadata", {})  # metadata is at root level

    # Extract info
    template_id = template_data.get("id", "N/A")
    template_name = template_data.get("name", "N/A")
    sections = template.get("sections", [])  # sections is at root level
    usage_type = metadata.get("usage", "task")  # field is "usage" not "usage_type"
    priority = metadata.get("priority", "medium")
    agent_type = template.get("_agent_type", "N/A")

    # Priority colors and emojis
    priority_config = {
        "high": ("🔴", "error"),
        "medium": ("🟡", "warning"),
        "low": ("🟢", "normal"),
    }
    priority_emoji, priority_color = priority_config.get(priority.lower(), ("⚪", "normal"))

    # Usage type emojis
    usage_emoji = {
        "session": "💬",
        "project": "📁",
        "task": "✅",
        "analysis": "🔍",
        "planning": "📋",
    }.get(usage_type.lower(), "📄")

    # Create card container
    with st.container():
        st.markdown(
            f"""
        <div style="
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 16px;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        ">
            <h3 style="margin-top: 0; color: #4A90E2;">{template_name}</h3>
            <p style="color: #666; font-size: 0.9em; margin: 4px 0;">
                <code>{template_id}</code>
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Metrics row
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Sections", len(sections))
        with col2:
            st.markdown(f"**Agent:** {agent_type}")
        with col3:
            st.markdown(f"**Type:** {usage_emoji} {usage_type}")
        with col4:
            st.markdown(f"**Priority:** {priority_emoji} {priority}")

        # Action buttons
        col_preview, col_copy = st.columns([3, 1])

        with col_preview:
            if st.button(f"👁️ Preview", key=f"preview_{key}", use_container_width=True):
                st.session_state[f"show_preview_{key}"] = True

        with col_copy:
            if st.button(f"📋 Copy ID", key=f"copy_{key}", use_container_width=True):
                st.code(template_id, language=None)
                st.success("ID copied to display!")

        # Show preview modal if requested
        if st.session_state.get(f"show_preview_{key}", False):
            render_template_preview_modal(template, key)


def render_template_preview_modal(template: Dict, key: str):
    """
    Render template preview modal

    Args:
        template: Template dictionary
        key: Unique key for the modal
    """
    template_data = template.get("template", {})

    with st.expander("📖 Template Details", expanded=True):
        # Close button
        if st.button("✖️ Close", key=f"close_{key}"):
            st.session_state[f"show_preview_{key}"] = False
            st.rerun()

        # Template info
        st.subheader(template_data.get("name", "N/A"))
        st.markdown(f"**ID:** `{template_data.get('id', 'N/A')}`")
        st.markdown(f"**Agent:** {template.get('_agent_type', 'N/A')}")

        # Metadata - at root level, not under template_data
        metadata = template.get("metadata", {})
        st.markdown("**Metadata:**")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"- Usage Type: {metadata.get('usage', 'task')}")
        with col2:
            st.markdown(f"- Priority: {metadata.get('priority', 'medium')}")

        st.markdown("---")

        # Sections - at root level, not under template_data
        sections = template.get("sections", [])
        st.subheader(f"📑 Sections ({len(sections)})")

        for idx, section in enumerate(sections, 1):
            with st.expander(f"{idx}. {section.get('title', 'Untitled')}"):
                st.markdown(f"**ID:** `{section.get('id', 'N/A')}`")
                st.markdown(f"**Description:**")
                st.write(section.get("description", "No description"))

        st.markdown("---")

        # Full YAML
        st.subheader("📄 Full Template (YAML)")

        # Convert template to YAML string
        try:
            yaml_content = yaml.dump(template, default_flow_style=False, allow_unicode=True)
            st.code(yaml_content, language="yaml", line_numbers=True)
        except Exception as e:
            st.error(f"Error displaying YAML: {e}")


def render_template_grid(templates: List[Dict], columns: int = 2):
    """
    Render templates in a grid layout

    Args:
        templates: List of template dictionaries
        columns: Number of columns in the grid
    """
    if not templates:
        st.info("No templates found matching your criteria.")
        return

    # Create grid
    for i in range(0, len(templates), columns):
        cols = st.columns(columns)
        for j in range(columns):
            idx = i + j
            if idx < len(templates):
                with cols[j]:
                    render_template_card(templates[idx], f"template_{idx}")


def render_compact_template_card(template: Dict, key: str):
    """
    Render a compact template card for dashboard view

    Args:
        template: Template dictionary
        key: Unique key for the card
    """
    template_data = template.get("template", {})
    metadata = template.get("metadata", {})  # metadata is at root level

    # Extract info with defaults
    template_id = template_data.get("id", "N/A")
    template_name = template_data.get("name", "N/A")
    sections = template.get("sections", [])  # sections is at root level
    usage_type = metadata.get("usage", "task")  # Default to "task"
    priority = metadata.get("priority", "medium")  # Default to "medium"
    agent_type = template.get("_agent_type", "unknown")

    # Priority colors and emojis
    priority_config = {
        "high": ("🔴", "#ff4b4b"),
        "medium": ("🟡", "#ffa500"),
        "low": ("🟢", "#28a745"),
    }
    priority_emoji, priority_color = priority_config.get(priority.lower(), ("⚪", "#6c757d"))

    # Usage type emojis
    usage_emoji = {
        "session": "💬",
        "project": "📁",
        "task": "✅",
        "analysis": "🔍",
        "planning": "📋",
        "bug-fix": "🐛",
    }.get(usage_type.lower(), "📄")

    # Ensure agent_type is valid
    from ..config import AGENT_DISPLAY_NAMES

    agent_display = AGENT_DISPLAY_NAMES.get(agent_type, agent_type.title())

    # Compact card layout
    st.markdown(
        f"""
    <div class="template-card">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.5rem;">
            <div style="flex: 1;">
                <h4 style="margin: 0; font-size: 1rem; color: #2c3e50;">{template_name}</h4>
                <code style="font-size: 0.8rem; color: #666;">{template_id}</code>
            </div>
            <div style="text-align: right;">
                <span style="color: {priority_color}; font-size: 1.2rem;">{priority_emoji}</span>
            </div>
        </div>
        <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.9rem; color: #666;">
            <span>{usage_emoji} {usage_type.title()}</span>
            <span>{len(sections)} sections</span>
        </div>
        <div style="font-size: 0.8rem; color: #888; margin-top: 0.25rem;">
            {agent_display}
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Action buttons in a compact row
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("👁️ Preview", key=f"preview_{key}", use_container_width=True):
            st.session_state[f"show_preview_{key}"] = True
    with col2:
        if st.button("📋", key=f"copy_{key}", help="Copy template ID"):
            st.code(template_id, language=None)

    # Show preview modal if requested
    if st.session_state.get(f"show_preview_{key}", False):
        render_template_preview_modal(template, key)
