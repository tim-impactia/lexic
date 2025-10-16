"""Evaluation rubrics for each pipeline step."""

from typing import Dict, List
from pathlib import Path
import yaml


class EvaluationDimension:
    """Single evaluation dimension with scoring criteria."""

    def __init__(self, name: str, description: str, weight: float, criteria: Dict[int, str]):
        """
        Initialize evaluation dimension.

        Args:
            name: Dimension name
            description: What this dimension measures
            weight: Weight in overall score (0-1)
            criteria: Dict mapping score (1-5) to description
        """
        self.name = name
        self.description = description
        self.weight = weight
        self.criteria = criteria


class Rubric:
    """Evaluation rubric with multiple dimensions."""

    def __init__(self, step_name: str, dimensions: List[EvaluationDimension]):
        """
        Initialize rubric.

        Args:
            step_name: Name of the pipeline step
            dimensions: List of evaluation dimensions
        """
        self.step_name = step_name
        self.dimensions = dimensions

    def to_text(self) -> str:
        """Convert rubric to text format for LLM."""
        lines = [f"# Evaluation Rubric: {self.step_name}\n"]

        for dim in self.dimensions:
            lines.append(f"## {dim.name} (weight: {dim.weight})")
            lines.append(f"{dim.description}\n")
            lines.append("Scoring criteria:")
            for score in sorted(dim.criteria.keys()):
                lines.append(f"- **{score}**: {dim.criteria[score]}")
            lines.append("")

        return "\n".join(lines)


def load_rubric_from_yaml(step_name: str) -> Rubric:
    """
    Load rubric from YAML file.

    Args:
        step_name: Name of the pipeline step

    Returns:
        Rubric loaded from YAML

    Raises:
        FileNotFoundError: If rubric file not found
    """
    rubrics_dir = Path(__file__).parent.parent.parent / "prompts" / "evals" / "rubrics"
    rubric_path = rubrics_dir / f"{step_name}.yaml"

    if not rubric_path.exists():
        raise FileNotFoundError(f"Rubric not found: {rubric_path}")

    with open(rubric_path) as f:
        config = yaml.safe_load(f)

    dimensions = []
    for dim_config in config['dimensions']:
        dimensions.append(EvaluationDimension(
            name=dim_config['name'],
            description=dim_config['description'],
            weight=dim_config['weight'],
            criteria=dim_config['criteria']
        ))

    return Rubric(step_name=config['step_name'], dimensions=dimensions)


def get_rubric(step_name: str) -> Rubric:
    """
    Get rubric for a pipeline step by loading from YAML.

    Args:
        step_name: Name of the pipeline step

    Returns:
        Rubric for that step

    Raises:
        FileNotFoundError: If no rubric exists for that step
    """
    return load_rubric_from_yaml(step_name)
