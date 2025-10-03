"""
Template Service - High-level service for template operations
"""

from pathlib import Path
from typing import Dict, List, Optional
import streamlit as st
import logging

from ..utils.template_loader import TemplateLoader
from ..utils.yaml_validator import YAMLValidator

logger = logging.getLogger(__name__)


class TemplateService:
    """High-level service for managing templates"""

    def __init__(self, templates_dir: Path):
        """
        Initialize template service

        Args:
            templates_dir: Path to templates directory
        """
        self.loader = TemplateLoader(templates_dir)
        self.validator = YAMLValidator()

    def get_all_templates(self, use_cache: bool = True) -> Dict[str, List[Dict]]:
        """
        Get all templates grouped by agent type

        Args:
            use_cache: Whether to use Streamlit session cache

        Returns:
            Dictionary mapping agent type to list of templates
        """
        # Use Streamlit session state for caching
        if use_cache and "all_templates" in st.session_state:
            return st.session_state["all_templates"]

        templates = self.loader.load_all_templates()

        if use_cache:
            st.session_state["all_templates"] = templates

        return templates

    def get_templates_by_agent(self, agent_type: str, use_cache: bool = True) -> List[Dict]:
        """
        Get templates for a specific agent type

        Args:
            agent_type: Agent type identifier
            use_cache: Whether to use cache

        Returns:
            List of templates
        """
        all_templates = self.get_all_templates(use_cache=use_cache)
        return all_templates.get(agent_type, [])

    def get_template_by_id(self, template_id: str) -> Optional[Dict]:
        """
        Get a template by its ID

        Args:
            template_id: Template ID to find

        Returns:
            Template dictionary or None
        """
        return self.loader.get_template_by_id(template_id)

    def search_templates(
        self, query: str, agent_type: Optional[str] = None, use_cache: bool = True
    ) -> List[Dict]:
        """
        Search templates by query string

        Args:
            query: Search query
            agent_type: Optional agent type filter
            use_cache: Whether to use cache

        Returns:
            List of matching templates
        """
        if not query.strip():
            # Return all templates if no query
            if agent_type:
                return self.get_templates_by_agent(agent_type, use_cache)
            else:
                all_templates = self.get_all_templates(use_cache)
                # Flatten all templates
                result = []
                for templates in all_templates.values():
                    result.extend(templates)
                return result

        return self.loader.search_templates(query, agent_type)

    def filter_templates(
        self,
        templates: List[Dict],
        usage_type: Optional[str] = None,
        priority: Optional[str] = None,
    ) -> List[Dict]:
        """
        Filter templates by metadata criteria

        Args:
            templates: List of templates to filter
            usage_type: Filter by usage type
            priority: Filter by priority

        Returns:
            Filtered list of templates
        """
        filtered = templates

        if usage_type:
            filtered = [t for t in filtered if t.get("metadata", {}).get("usage") == usage_type]

        if priority:
            filtered = [t for t in filtered if t.get("metadata", {}).get("priority") == priority]

        return filtered

    def get_template_stats(self) -> Dict[str, int]:
        """
        Get statistics about templates

        Returns:
            Dictionary with template statistics
        """
        all_templates = self.get_all_templates()

        total_templates = sum(len(templates) for templates in all_templates.values())

        # Count by usage type
        usage_counts = {}
        priority_counts = {}

        for templates in all_templates.values():
            for template in templates:
                metadata = template.get("metadata", {})

                usage_type = metadata.get("usage", "unknown")
                usage_counts[usage_type] = usage_counts.get(usage_type, 0) + 1

                priority = metadata.get("priority", "unknown")
                priority_counts[priority] = priority_counts.get(priority, 0) + 1

        return {
            "total_templates": total_templates,
            "total_agents": len(all_templates),
            "usage_counts": usage_counts,
            "priority_counts": priority_counts,
        }

    def validate_template(self, yaml_content: str) -> tuple[bool, Dict[str, List[str]]]:
        """
        Validate a template YAML

        Args:
            yaml_content: YAML string to validate

        Returns:
            Tuple of (is_valid, validation_results)
        """
        return self.validator.validate_full(yaml_content)

    def get_template_preview(self, template: Dict) -> Dict[str, any]:
        """
        Get preview information for a template

        Args:
            template: Template dictionary

        Returns:
            Preview data dictionary
        """
        template_data = template.get("template", {})

        return {
            "id": template_data.get("id", "N/A"),
            "name": template_data.get("name", "N/A"),
            "agent_type": template.get("_agent_type", "N/A"),
            "section_count": len(template_data.get("sections", [])),
            "usage_type": template_data.get("metadata", {}).get("usage_type", "N/A"),
            "priority": template_data.get("metadata", {}).get("priority", "N/A"),
            "sections": template_data.get("sections", []),
            "file_path": template.get("_file_path", "N/A"),
        }

    def reload_templates(self):
        """Reload all templates from disk"""
        self.loader.clear_cache()

        # Clear Streamlit session state cache
        if "all_templates" in st.session_state:
            del st.session_state["all_templates"]

        logger.info("Templates reloaded")

    def get_section_ids(self, template: Dict) -> List[str]:
        """
        Get list of section IDs from a template

        Args:
            template: Template dictionary

        Returns:
            List of section IDs
        """
        sections = template.get("template", {}).get("sections", [])
        return [section.get("id", "") for section in sections]

    def get_section_details(self, template: Dict, section_id: str) -> Optional[Dict]:
        """
        Get details for a specific section

        Args:
            template: Template dictionary
            section_id: Section ID to find

        Returns:
            Section dictionary or None
        """
        sections = template.get("template", {}).get("sections", [])

        for section in sections:
            if section.get("id") == section_id:
                return section

        return None
