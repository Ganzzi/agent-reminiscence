"""
Create Memory Page

Create new active memories using pre-built templates or custom YAML.
"""

import streamlit as st
import yaml
from typing import Optional, Dict, Any
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from streamlit_app.services.template_service import TemplateService
from streamlit_app.services.memory_service import MemoryService
from streamlit_app.utils.formatters import Formatters
from streamlit_app.config import APP_TITLE, APP_ICON, TEMPLATES_DIR, DB_CONFIG
import asyncio

# Page configuration
st.set_page_config(page_title=f"Create Memory - {APP_TITLE}", page_icon=APP_ICON, layout="wide")

# Custom CSS for compact layout
st.markdown(
    """
<style>
    .main .block-container {padding-top: 1rem; padding-bottom: 1rem;}
    .stExpander {margin-bottom: 0.5rem;}
    .stButton>button {margin: 0.1rem;}
    .stTextInput input, .stTextArea textarea {padding: 0.25rem;}
    .stSelectbox select {padding: 0.25rem;}
    h1, h2, h3 {margin-top: 0.5rem; margin-bottom: 0.5rem;}
    .stRadio > div {margin-bottom: 0.5rem;}
</style>
""",
    unsafe_allow_html=True,
)

# Initialize services
if "template_service" not in st.session_state:
    st.session_state.template_service = TemplateService(TEMPLATES_DIR)

# Initialize MemoryService
if "memory_service" not in st.session_state:
    st.session_state.memory_service = MemoryService(DB_CONFIG)

template_service = st.session_state.template_service
memory_service = st.session_state.memory_service

# Page header
st.title("➕ Create Active Memory")
st.markdown("Create a new active memory for an agent using a pre-built template or custom YAML.")

# Creation mode state
if "creation_mode" not in st.session_state:
    st.session_state.creation_mode = "pre-built"

# External ID input
st.subheader("1️⃣ Agent Information")
col1, col2 = st.columns([2, 1])

with col1:
    external_id = st.text_input(
        "External ID",
        placeholder="agent-001, UUID, or numeric ID",
        help="Unique identifier for the agent (string, UUID, or integer)",
    )

with col2:
    st.markdown("**External ID Type**")
    id_type = st.radio(
        "ID Type",
        options=["String", "UUID", "Integer"],
        horizontal=True,
        label_visibility="collapsed",
    )

# Creation mode selector
st.subheader("2️⃣ Creation Mode")
creation_mode = st.radio(
    "Choose how to create the memory:",
    options=["pre-built", "custom"],
    format_func=lambda x: "📚 Pre-built Template" if x == "pre-built" else "✍️ Custom YAML",
    horizontal=True,
)
st.session_state.creation_mode = creation_mode

st.divider()

# Pre-built template mode
if creation_mode == "pre-built":
    st.subheader("3️⃣ Select Template")

    # Load templates
    try:
        # Get all templates and flatten into a list
        all_templates_dict = template_service.get_all_templates()
        templates = []
        for agent_templates in all_templates_dict.values():
            templates.extend(agent_templates)

        # Agent type filter
        col1, col2 = st.columns([1, 3])
        with col1:
            agent_types = ["All Agents"] + sorted(
                set(t.get("_agent_type", "unknown") for t in templates)
            )
            selected_agent = st.selectbox("Filter by Agent Type", options=agent_types)

        # Filter templates
        if selected_agent != "All Agents":
            filtered_templates = [t for t in templates if t.get("_agent_type") == selected_agent]
        else:
            filtered_templates = templates

        with col2:
            if filtered_templates:
                # Template selector
                template_options = {
                    f"{t.get('template', {}).get('name', 'Unnamed')}": t.get("template", {}).get(
                        "id"
                    )
                    for t in filtered_templates
                }

                selected_template_display = st.selectbox(
                    "Choose a template",
                    options=list(template_options.keys()),
                    help="Select a pre-built template to use as the basis for this memory",
                )

                selected_template_id = template_options[selected_template_display]
                selected_template = next(
                    (
                        t
                        for t in filtered_templates
                        if t.get("template", {}).get("id") == selected_template_id
                    ),
                    None,
                )
            else:
                st.warning(f"No templates found for agent type: {selected_agent}")
                selected_template = None

        # Display selected template details
        if selected_template:
            st.divider()
            st.subheader("4️⃣ Configure Memory")

            # Side-by-side layout: Form on left, YAML on right
            col_form, col_yaml = st.columns([1.5, 1])

            with col_form:
                # Template info card
                st.markdown("**📋 Template Information**")
                info_col1, info_col2, info_col3 = st.columns(3)
                with info_col1:
                    st.metric("Agent Type", selected_template.get("_agent_type", "unknown"))
                with info_col2:
                    st.metric("Sections", len(selected_template.get("sections", [])))
                with info_col3:
                    priority = selected_template.get("metadata", {}).get("priority", "medium")
                    priority_emoji, _ = Formatters.format_priority(priority)
                    st.metric("Priority", priority_emoji)

                st.markdown("---")

                # Memory title
                st.markdown("**✏️ Memory Title**")
                default_title = selected_template.get("template", {}).get("name", "Untitled Memory")
                memory_title = st.text_input(
                    "Title",
                    value=default_title,
                    help="Edit the memory title",
                    label_visibility="collapsed",
                )

                st.markdown("---")

                # Section content editors
                st.markdown("**📝 Section Contents**")
                st.caption(
                    "Edit section content below. Leave empty to create memory with structure only."
                )

                section_contents = {}
                for section in selected_template.get("sections", []):
                    section_id = section.get("id", "unknown")
                    section_title = section.get("title", "Untitled")
                    section_desc = section.get("description", "")

                    st.markdown(f"**{section_title}**")

                    # Use description as default value in the text area
                    content = st.text_area(
                        f"Content for {section_id}",
                        value=section_desc if section_desc else "",
                        height=120,
                        placeholder=f"Enter markdown content for {section_title}...",
                        key=f"section_content_{section_id}",
                        label_visibility="collapsed",
                        help=f"Edit or replace this content for {section_title}",
                    )
                    if content.strip():
                        section_contents[section_id] = content.strip()

                st.markdown("---")

                # Metadata editor
                st.markdown("**⚙️ Additional Metadata** (Optional)")
                template_metadata = selected_template.get("metadata", {})
                default_metadata = {
                    "priority": template_metadata.get("priority", "medium"),
                    "usage": template_metadata.get("usage", "session"),
                }
                metadata_json = st.text_area(
                    "Metadata",
                    value=yaml.dump(default_metadata, default_flow_style=False),
                    height=100,
                    help="Additional metadata in YAML/JSON format",
                    label_visibility="collapsed",
                )

            with col_yaml:
                st.markdown("**📄 Original Template YAML**")
                st.caption("Reference - shows the complete template structure")
                st.code(
                    yaml.dump(selected_template, default_flow_style=False),
                    language="yaml",
                    height=800,
                )

            st.divider()

            # Create button
            col1, col2, col3 = st.columns([2, 1, 1])
            with col2:
                if st.button("✨ Create Memory", type="primary", use_container_width=True):
                    # Validation
                    if not external_id:
                        st.error("❌ Please enter an External ID")
                    elif not memory_title:
                        st.error("❌ Please enter a Memory Title")
                    else:
                        # Create active memory using API
                        with st.spinner("Creating memory..."):
                            try:
                                # Prepare initial sections
                                initial_sections = {}
                                for section_id, content in section_contents.items():
                                    if content and content.strip():
                                        initial_sections[section_id] = {
                                            "content": content,
                                            "update_count": 0,
                                        }

                                # Parse metadata
                                try:
                                    metadata = (
                                        yaml.safe_load(metadata_json)
                                        if metadata_json.strip()
                                        else {}
                                    )
                                except yaml.YAMLError as e:
                                    st.error(f"❌ Invalid metadata YAML: {str(e)}")
                                    st.stop()

                                # Call API (async)
                                # Convert template dict to YAML string for the core API
                                template_yaml = yaml.dump(selected_template)
                                memory = asyncio.run(
                                    memory_service.create_active_memory(
                                        external_id=external_id,
                                        title=memory_title,
                                        template_content=template_yaml,
                                        initial_sections=initial_sections,
                                        metadata=metadata,
                                    )
                                )

                                if memory:
                                    st.success(
                                        f"🎉 Memory created successfully! Memory ID: {memory.id}"
                                    )
                                    st.balloons()
                                    st.info(
                                        f"**External ID:** {external_id}\n**Template:** {selected_template_id}\n**Title:** {memory_title}\n**Sections:** {len(initial_sections)}"
                                    )
                                    # Clear form
                                    st.session_state.section_contents = {}
                                else:
                                    st.error(
                                        "❌ Failed to create memory. Please check database connection and try again."
                                    )

                            except Exception as e:
                                st.error(f"❌ Error creating memory: {str(e)}")
                                st.exception(e)

            with col3:
                if st.button("🔄 Reset Form", use_container_width=True):
                    st.rerun()

    except Exception as e:
        st.error(f"❌ Error loading templates: {str(e)}")

# Custom YAML mode
else:
    st.subheader("3️⃣ Custom Template YAML")
    st.info(
        "💡 **Tip**: You can copy a template from the Browse Templates page and modify it here."
    )

    # YAML editor
    custom_yaml = st.text_area(
        "Template YAML",
        height=400,
        placeholder="""template_id: custom.my-template.v1
name: My Custom Template
agent_type: dev
metadata:
  priority: medium
  usage: session
sections:
  - id: section1
    title: Section Title
    description: Section description
    update_strategy: append
    consolidation_trigger:
      update_threshold: 5
      consolidation_prompt: "Consolidate this section"
""",
        help="Paste your custom template YAML here",
    )

    # Validate YAML button
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("✓ Validate YAML", use_container_width=True):
            if not custom_yaml.strip():
                st.warning("⚠️ Please enter YAML content")
            else:
                try:
                    parsed = yaml.safe_load(custom_yaml)
                    st.success("✅ Valid YAML syntax!")

                    # Check required fields
                    required_fields = ["template_id", "name", "agent_type", "sections"]
                    missing = [f for f in required_fields if f not in parsed]

                    if missing:
                        st.warning(f"⚠️ Missing required fields: {', '.join(missing)}")
                    else:
                        st.success(
                            f"✅ All required fields present! Found {len(parsed['sections'])} sections."
                        )
                except yaml.YAMLError as e:
                    st.error(f"❌ Invalid YAML: {str(e)}")

    if custom_yaml.strip():
        st.divider()
        st.subheader("4️⃣ Memory Configuration")

        # Try to parse YAML to get template name
        try:
            parsed = yaml.safe_load(custom_yaml)
            default_title = parsed.get("name", "Custom Memory")
        except:
            default_title = "Custom Memory"

        # Memory title
        memory_title = st.text_input(
            "Memory Title", value=default_title, help="Descriptive title for this memory"
        )

        st.divider()

        # Create button
        col1, col2, col3 = st.columns([2, 1, 1])
        with col2:
            if st.button("✨ Create Memory", type="primary", use_container_width=True):
                # Validation
                if not external_id:
                    st.error("❌ Please enter an External ID")
                elif not memory_title:
                    st.error("❌ Please enter a Memory Title")
                elif not custom_yaml.strip():
                    st.error("❌ Please enter template YAML")
                else:
                    try:
                        parsed = yaml.safe_load(custom_yaml)

                        # Validate required fields
                        required_fields = ["template_id", "name", "agent_type", "sections"]
                        missing = [f for f in required_fields if f not in parsed]

                        if missing:
                            st.error(f"❌ Missing required fields: {', '.join(missing)}")
                        else:
                            # Create active memory using API
                            with st.spinner("Creating memory..."):
                                try:
                                    # Call API (async)
                                    # Convert parsed dict to YAML string for the core API
                                    template_yaml = yaml.dump(parsed)
                                    memory = asyncio.run(
                                        memory_service.create_active_memory(
                                            external_id=external_id,
                                            title=memory_title,
                                            template_content=template_yaml,
                                            initial_sections=None,  # Custom YAML - no initial sections
                                            metadata=parsed.get("metadata", {}),
                                        )
                                    )

                                    if memory:
                                        st.success(
                                            f"🎉 Memory created successfully! Memory ID: {memory.id}"
                                        )
                                        st.balloons()
                                        st.info(
                                            f"**External ID:** {external_id}\n**Title:** {memory_title}\n**Custom Template:** {parsed.get('template_id')}"
                                        )
                                    else:
                                        st.error(
                                            "❌ Failed to create memory. Please check database connection and try again."
                                        )

                                except Exception as e:
                                    st.error(f"❌ Error creating memory: {str(e)}")
                                    st.exception(e)

                    except yaml.YAMLError as e:
                        st.error(f"❌ Invalid YAML: {str(e)}")

        with col3:
            if st.button("🔄 Reset Form", use_container_width=True):
                st.rerun()

# Sidebar help
with st.sidebar:
    st.markdown("### ℹ️ Help")
    st.markdown(
        """
    **Creating Memories:**
    
    1. **Pre-built Template Mode:**
       - Select an agent type
       - Choose a template
       - Optionally add section content
       - Create memory
    
    2. **Custom YAML Mode:**
       - Paste custom YAML
       - Validate structure
       - Create memory
    
    **Tips:**
    - Section content is optional at creation
    - You can update sections later
    - Templates define the structure
    """
    )

    st.divider()

    st.markdown("### 📊 Statistics")
    try:
        templates = template_service.get_all_templates()
        st.metric("Total Templates", len(templates))
        agent_types = set(t.agent_type for t in templates)
        st.metric("Agent Types", len(agent_types))
    except:
        pass
