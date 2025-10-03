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
    AGENT_DISPLAY_NAMES
)
from streamlit_app.services.template_service import TemplateService
from streamlit_app.services.memory_service import MemoryService


# Configure page
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout=PAGE_LAYOUT,
    initial_sidebar_state="expanded"
)


def initialize_services():
    """Initialize services in session state"""
    if 'template_service' not in st.session_state:
        st.session_state.template_service = TemplateService(TEMPLATES_DIR)
        
    if 'memory_service' not in st.session_state:
        st.session_state.memory_service = MemoryService(DB_CONFIG)


def render_sidebar():
    """Render sidebar with navigation and agent info"""
    with st.sidebar:
        st.title(f"{APP_ICON} {APP_TITLE}")
        st.markdown("---")
        
        # Agent ID input (persisted across pages)
        st.subheader("Agent Configuration")
        
        if 'agent_external_id' not in st.session_state:
            st.session_state.agent_external_id = ""
            
        agent_id = st.text_input(
            "Agent External ID",
            value=st.session_state.agent_external_id,
            placeholder="Enter agent ID (string, UUID, or int)",
            help="This ID is used to identify the agent across all operations"
        )
        
        if agent_id != st.session_state.agent_external_id:
            st.session_state.agent_external_id = agent_id
            
        st.markdown("---")
        
        # Template statistics
        st.subheader("📊 Template Statistics")
        
        try:
            template_service = st.session_state.get('template_service')
            if template_service:
                stats = template_service.get_template_stats()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Templates", stats['total_templates'])
                with col2:
                    st.metric("Agents", stats['total_agents'])
                    
                with st.expander("Details"):
                    st.write("**By Usage Type:**")
                    for usage_type, count in stats['usage_counts'].items():
                        st.write(f"- {usage_type.capitalize()}: {count}")
                        
                    st.write("**By Priority:**")
                    for priority, count in stats['priority_counts'].items():
                        st.write(f"- {priority.capitalize()}: {count}")
        except Exception as e:
            st.warning(f"Could not load statistics: {e}")
            
        st.markdown("---")
        
        # Navigation help
        st.subheader("📚 Navigation")
        st.markdown("""
        **Pages:**
        - 📚 **Browse Templates** - Explore pre-built templates
        - ➕ **Create Memory** - Create new active memories
        - 📋 **View Memories** - View and manage memories
        - ✏️ **Update Memory** - Edit memory sections
        - 🗑️ **Delete Memory** - Remove memories
        """)
        
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
    st.title(f"{APP_ICON} Welcome to AgentMem UI")
    
    st.markdown("""
    ## 🧠 Agent Memory Management System
    
    Welcome to the AgentMem UI! This interface allows you to manage active memories 
    for your BMAD (Business, Method, Agent, Data) agents using pre-built templates.
    
    ### 🎯 Quick Start
    
    1. **Set Agent ID** - Enter your agent's external ID in the sidebar
    2. **Browse Templates** - Explore 62 pre-built templates across 10 agent types
    3. **Create Memory** - Use templates to create structured active memories
    4. **Manage Memories** - View, update, or delete your agent's memories
    
    ### 📋 Features
    
    - **Template Discovery**: Browse and search through pre-built BMAD templates
    - **Memory Creation**: Create memories using templates or custom YAML
    - **Memory Management**: View, update, and delete active memories
    - **Section-Based Updates**: Update individual sections with automatic tracking
    - **Consolidation Warnings**: Get notified when memories need consolidation
    
    ### 🏗️ BMAD Agents
    
    The system supports 10 agent types:
    """)
    
    # Display agent types in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Core Development Agents")
        st.markdown("""
        - 👔 **Business Analyst** - Requirements and research
        - 🏗️ **System Architect** - Architecture and design
        - 💻 **Full-Stack Developer** - Implementation
        - 📦 **Product Manager** - Product strategy
        - 📝 **Product Owner** - Backlog management
        """)
        
    with col2:
        st.markdown("#### Support & Coordination Agents")
        st.markdown("""
        - 🧪 **QA/Test Architect** - Quality assurance
        - 🏃 **Scrum Master** - Agile processes
        - 🎨 **UX Expert** - User experience
        - 👑 **BMAD Master** - Project oversight
        - 🔄 **BMAD Orchestrator** - Workflow coordination
        """)
    
    st.markdown("---")
    
    # Getting started guide
    st.subheader("🚀 Getting Started")
    
    tab1, tab2, tab3 = st.tabs(["📖 Overview", "⚙️ Configuration", "🔗 Navigation"])
    
    with tab1:
        st.markdown("""
        ### What are Active Memories?
        
        Active memories are structured, template-based memory units that store 
        context and information for your agents. Each memory contains:
        
        - **Template**: Defines the structure with sections
        - **Sections**: Individual content areas (e.g., objectives, decisions, notes)
        - **Metadata**: Usage type, priority, and custom fields
        - **Update Tracking**: Monitors section changes and triggers consolidation
        
        ### Template System
        
        Templates provide consistent structure for different workflows:
        - **Session templates** - For meetings, brainstorming, etc.
        - **Project templates** - For long-term project tracking
        - **Task templates** - For specific implementation tasks
        - **Analysis templates** - For research and investigation
        - **Planning templates** - For roadmaps and strategies
        """)
        
    with tab2:
        st.markdown("""
        ### Database Configuration
        
        The UI connects to your AgentMem database using environment variables 
        or defaults:
        
        **PostgreSQL** (Active Memories):
        - Host: `POSTGRES_HOST` (default: localhost)
        - Port: `POSTGRES_PORT` (default: 5432)
        - Database: `POSTGRES_DB` (default: agent_memory)
        
        **Neo4j** (Knowledge Graph):
        - URI: `NEO4J_URI` (default: bolt://localhost:7687)
        - User: `NEO4J_USER` (default: neo4j)
        
        ### Template Directory
        
        Templates are loaded from: `prebuilt-memory-tmpl/bmad/`
        
        You can add custom templates by creating YAML files in the appropriate 
        agent directory following the template structure.
        """)
        
    with tab3:
        st.markdown("""
        ### Page Navigation
        
        Use the sidebar or the links below to navigate to different pages:
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📚 Browse Templates", use_container_width=True):
                st.switch_page("pages/1_📚_Browse_Templates.py")
                
            if st.button("➕ Create Memory", use_container_width=True):
                st.switch_page("pages/2_➕_Create_Memory.py")
                
            if st.button("📋 View Memories", use_container_width=True):
                st.switch_page("pages/3_📋_View_Memories.py")
                
        with col2:
            if st.button("✏️ Update Memory", use_container_width=True):
                st.switch_page("pages/4_✏️_Update_Memory.py")
                
            if st.button("🗑️ Delete Memory", use_container_width=True):
                st.switch_page("pages/5_🗑️_Delete_Memory.py")
    
    st.markdown("---")
    
    # Connection status
    st.subheader("🔌 Connection Status")
    
    if st.button("Test Database Connection"):
        with st.spinner("Testing connection..."):
            memory_service = st.session_state.get('memory_service')
            if memory_service:
                # This would be async in real usage
                st.info("Connection test requires async support. Use the Create/View pages to verify connection.")
            else:
                st.error("Memory service not initialized")
    
    # Tips
    st.info("""
    💡 **Tip**: Start by browsing templates to understand the available structures, 
    then create your first memory using a pre-built template!
    """)


if __name__ == "__main__":
    main()
