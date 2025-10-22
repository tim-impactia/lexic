"""Markdown file I/O utilities for Lexic."""

import yaml
from pathlib import Path
from typing import Tuple, Dict, List, Any


def read_markdown(path: Path) -> Tuple[Dict[str, Any], str]:
    """
    Read a markdown file with YAML frontmatter.

    Args:
        path: Path to the markdown file

    Returns:
        Tuple of (metadata_dict, content_string)
    """
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check for frontmatter
    if content.startswith('---\n'):
        parts = content.split('---\n', 2)
        if len(parts) >= 3:
            metadata = yaml.safe_load(parts[1]) or {}
            content_text = parts[2].strip()
            return metadata, content_text

    # No frontmatter
    return {}, content.strip()


def write_markdown(path: Path, metadata: Dict[str, Any], content: str):
    """
    Write a markdown file with YAML frontmatter.

    Args:
        path: Path to write to
        metadata: Dictionary of metadata for frontmatter
        content: Markdown content
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        if metadata:
            f.write('---\n')
            yaml.dump(metadata, f, default_flow_style=False, allow_unicode=True)
            f.write('---\n\n')
        f.write(content)


def list_cases(directory: Path) -> List[str]:
    """
    List all case directories in a given directory.

    Args:
        directory: Path to directory containing case folders

    Returns:
        List of case IDs (directory names)
    """
    if not directory.exists():
        return []

    return sorted([
        d.name for d in directory.iterdir()
        if d.is_dir() and not d.name.startswith('.')
    ])


def load_case_step(case_dir: Path, step_name: str) -> Tuple[Dict[str, Any], str]:
    """
    Load a specific step from a case directory.

    Args:
        case_dir: Path to case directory
        step_name: Name of the step file (e.g., '03_gt_qualification.md')

    Returns:
        Tuple of (metadata, content)
    """
    step_path = case_dir / step_name
    if not step_path.exists():
        raise FileNotFoundError(f"Step file not found: {step_path}")

    return read_markdown(step_path)


def list_decision_dirs(decisions_dir: Path) -> List[str]:
    """
    List all court decision directories.

    Args:
        decisions_dir: Path to court_decisions directory

    Returns:
        List of decision IDs
    """
    return list_cases(decisions_dir)


def get_decision_path(decisions_dir: Path, decision_id: str) -> Path:
    """
    Get the directory path for a specific decision.

    Args:
        decisions_dir: Path to court_decisions directory
        decision_id: Decision ID

    Returns:
        Path to decision directory
    """
    return decisions_dir / decision_id


def get_case_path(cases_dir: Path, case_id: str) -> Path:
    """
    Get the directory path for a specific synthetic case.

    Args:
        cases_dir: Path to synthetic_cases directory
        case_id: Case ID

    Returns:
        Path to case directory
    """
    return cases_dir / case_id
