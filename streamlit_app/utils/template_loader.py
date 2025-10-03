"""
Template loader utility - scans and loads YAML templates from filesystem
"""

import yaml
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class TemplateLoader:
    """Loads and parses YAML templates from the filesystem"""

    def __init__(self, templates_dir: Path):
        """
        Initialize template loader

        Args:
            templates_dir: Path to the templates directory
        """
        self.templates_dir = Path(templates_dir)
        self._cache: Dict[str, Dict] = {}

    def load_all_templates(self) -> Dict[str, List[Dict]]:
        """
        Load all templates from all agent directories

        Returns:
            Dictionary mapping agent type to list of templates
        """
        all_templates = {}

        if not self.templates_dir.exists():
            logger.error(f"Templates directory not found: {self.templates_dir}")
            return all_templates

        # Scan each agent directory
        for agent_dir in self.templates_dir.iterdir():
            if not agent_dir.is_dir():
                continue

            agent_type = agent_dir.name
            templates = self.load_agent_templates(agent_type)

            if templates:
                all_templates[agent_type] = templates

        logger.info(
            f"Loaded {sum(len(t) for t in all_templates.values())} templates from {len(all_templates)} agents"
        )
        return all_templates

    def load_agent_templates(self, agent_type: str) -> List[Dict]:
        """
        Load all templates for a specific agent type

        Args:
            agent_type: The agent type (directory name)

        Returns:
            List of template dictionaries
        """
        agent_dir = self.templates_dir / agent_type
        templates = []

        if not agent_dir.exists():
            logger.warning(f"Agent directory not found: {agent_dir}")
            return templates

        # Load all YAML files in the agent directory
        for template_file in agent_dir.glob("*.yaml"):
            try:
                template = self.load_template_file(template_file)
                if template:
                    # Add metadata
                    template["_agent_type"] = agent_type
                    template["_file_path"] = str(template_file)
                    template["_file_name"] = template_file.stem
                    templates.append(template)
            except Exception as e:
                logger.error(f"Error loading template {template_file}: {e}")

        return templates

    def load_template_file(self, file_path: Path) -> Optional[Dict]:
        """
        Load and parse a single template YAML file

        Args:
            file_path: Path to the template file

        Returns:
            Parsed template dictionary or None if error
        """
        cache_key = str(file_path)

        # Check cache
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = yaml.safe_load(f)

            if not content:
                logger.warning(f"Empty template file: {file_path}")
                return None

            # Validate basic structure
            if "template" not in content:
                logger.warning(f"Invalid template structure (missing 'template' key): {file_path}")
                return None

            # Cache the result
            self._cache[cache_key] = content
            return content

        except yaml.YAMLError as e:
            logger.error(f"YAML parsing error in {file_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error reading template file {file_path}: {e}")
            return None

    def get_template_by_id(self, template_id: str) -> Optional[Dict]:
        """
        Find a template by its ID

        Args:
            template_id: The template.id value

        Returns:
            Template dictionary or None if not found
        """
        all_templates = self.load_all_templates()

        for agent_type, templates in all_templates.items():
            for template in templates:
                if template.get("template", {}).get("id") == template_id:
                    return template

        return None

    def search_templates(self, query: str, agent_type: Optional[str] = None) -> List[Dict]:
        """
        Search templates by name or ID

        Args:
            query: Search query string
            agent_type: Optional agent type to filter by

        Returns:
            List of matching templates
        """
        query_lower = query.lower()
        results = []

        if agent_type:
            # Search only in specific agent type
            templates = self.load_agent_templates(agent_type)
            templates_dict = {agent_type: templates}
        else:
            # Search all templates
            templates_dict = self.load_all_templates()

        for agent, templates in templates_dict.items():
            for template in templates:
                template_data = template.get("template", {})
                template_id = template_data.get("id", "").lower()
                template_name = template_data.get("name", "").lower()

                if query_lower in template_id or query_lower in template_name:
                    results.append(template)

        return results

    def clear_cache(self):
        """Clear the template cache"""
        self._cache.clear()
        logger.info("Template cache cleared")
