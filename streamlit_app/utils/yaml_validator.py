"""
YAML validator utility - validates template structure and content
"""

import yaml
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class YAMLValidator:
    """Validates YAML template structure and content"""

    REQUIRED_TEMPLATE_FIELDS = ["id", "name", "sections"]
    REQUIRED_SECTION_FIELDS = ["id", "title", "description"]
    REQUIRED_METADATA_FIELDS = ["usage_type", "priority"]

    @staticmethod
    def validate_yaml_syntax(yaml_content: str) -> Tuple[bool, Optional[str]]:
        """
        Validate YAML syntax

        Args:
            yaml_content: YAML string to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            yaml.safe_load(yaml_content)
            return True, None
        except yaml.YAMLError as e:
            return False, f"YAML syntax error: {str(e)}"
        except Exception as e:
            return False, f"Error parsing YAML: {str(e)}"

    @staticmethod
    def validate_template_structure(template_dict: Dict) -> Tuple[bool, List[str]]:
        """
        Validate template structure

        Args:
            template_dict: Parsed template dictionary

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Check for template key
        if "template" not in template_dict:
            errors.append("Missing 'template' root key")
            return False, errors

        template = template_dict["template"]

        # Check required fields
        for field in YAMLValidator.REQUIRED_TEMPLATE_FIELDS:
            if field not in template:
                errors.append(f"Missing required field: template.{field}")

        # Validate template ID format
        if "id" in template:
            template_id = template["id"]
            if not isinstance(template_id, str):
                errors.append("template.id must be a string")
            elif not template_id.startswith("bmad."):
                errors.append("template.id must start with 'bmad.'")
            elif not template_id.endswith(".v1"):
                errors.append("template.id should end with version (e.g., '.v1')")

        # Validate sections
        if "sections" in template:
            sections_errors = YAMLValidator._validate_sections(template["sections"])
            errors.extend(sections_errors)
        else:
            errors.append("Template must have at least one section")

        # Validate metadata
        if "metadata" in template:
            metadata_errors = YAMLValidator._validate_metadata(template["metadata"])
            errors.extend(metadata_errors)

        return len(errors) == 0, errors

    @staticmethod
    def _validate_sections(sections: List[Dict]) -> List[str]:
        """Validate sections list"""
        errors = []

        if not isinstance(sections, list):
            return ["template.sections must be a list"]

        if len(sections) == 0:
            return ["template.sections must contain at least one section"]

        section_ids = set()

        for idx, section in enumerate(sections):
            if not isinstance(section, dict):
                errors.append(f"Section {idx} must be a dictionary")
                continue

            # Check required fields
            for field in YAMLValidator.REQUIRED_SECTION_FIELDS:
                if field not in section:
                    errors.append(f"Section {idx} missing required field: {field}")

            # Check for duplicate section IDs
            section_id = section.get("id")
            if section_id:
                if section_id in section_ids:
                    errors.append(f"Duplicate section ID: {section_id}")
                section_ids.add(section_id)

        return errors

    @staticmethod
    def _validate_metadata(metadata: Dict) -> List[str]:
        """Validate metadata"""
        errors = []

        if not isinstance(metadata, dict):
            return ["template.metadata must be a dictionary"]

        # Check required metadata fields
        for field in YAMLValidator.REQUIRED_METADATA_FIELDS:
            if field not in metadata:
                errors.append(f"Missing required metadata field: {field}")

        # Validate usage_type values
        if "usage_type" in metadata:
            valid_usage_types = ["session", "project", "task", "analysis", "planning"]
            usage_type = metadata["usage_type"]
            if usage_type not in valid_usage_types:
                errors.append(
                    f"Invalid usage_type: {usage_type}. Must be one of {valid_usage_types}"
                )

        # Validate priority values
        if "priority" in metadata:
            valid_priorities = ["high", "medium", "low"]
            priority = metadata["priority"]
            if priority not in valid_priorities:
                errors.append(f"Invalid priority: {priority}. Must be one of {valid_priorities}")

        return errors

    @staticmethod
    def validate_template_content(template_dict: Dict) -> Tuple[bool, List[str]]:
        """
        Validate template content (fields are not empty, etc.)

        Args:
            template_dict: Parsed template dictionary

        Returns:
            Tuple of (is_valid, list_of_warnings)
        """
        warnings = []

        if "template" not in template_dict:
            return False, ["Missing 'template' key"]

        template = template_dict["template"]

        # Check if name is meaningful
        if "name" in template and len(template["name"].strip()) < 3:
            warnings.append("template.name is too short")

        # Check sections have descriptions
        if "sections" in template:
            for idx, section in enumerate(template["sections"]):
                if isinstance(section, dict):
                    desc = section.get("description", "")
                    if len(desc.strip()) < 10:
                        warnings.append(
                            f"Section {idx} ({section.get('id', 'unknown')}) has very short description"
                        )

        return len(warnings) == 0, warnings

    @staticmethod
    def validate_full(yaml_content: str) -> Tuple[bool, Dict[str, List[str]]]:
        """
        Full validation: syntax + structure + content

        Args:
            yaml_content: YAML string to validate

        Returns:
            Tuple of (is_valid, dict of errors/warnings)
        """
        result = {"syntax_errors": [], "structure_errors": [], "content_warnings": []}

        # Step 1: Validate syntax
        syntax_valid, syntax_error = YAMLValidator.validate_yaml_syntax(yaml_content)
        if not syntax_valid:
            result["syntax_errors"].append(syntax_error)
            return False, result

        # Step 2: Parse YAML
        try:
            template_dict = yaml.safe_load(yaml_content)
        except Exception as e:
            result["syntax_errors"].append(f"Failed to parse YAML: {e}")
            return False, result

        # Step 3: Validate structure
        structure_valid, structure_errors = YAMLValidator.validate_template_structure(template_dict)
        result["structure_errors"] = structure_errors

        # Step 4: Validate content
        content_valid, content_warnings = YAMLValidator.validate_template_content(template_dict)
        result["content_warnings"] = content_warnings

        is_valid = syntax_valid and structure_valid
        return is_valid, result


