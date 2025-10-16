"""Prompt loading utilities."""

from pathlib import Path
import yaml
import dspy


def load_prompt_config(category: str, name: str) -> dict:
    """
    Load prompt configuration from YAML.

    Args:
        category: Prompt category ('agents', 'extraction', 'generation')
        name: Prompt name (without .yaml extension)

    Returns:
        Dict with 'description', 'input_fields', 'output_fields'

    Example:
        >>> config = load_prompt_config("agents", "qualification")
        >>> config['description']
        'Analyser la demande du client...'
    """
    prompts_dir = Path(__file__).parent.parent / "prompts"
    prompt_path = prompts_dir / category / f"{name}.yaml"

    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Prompt not found: {prompt_path}\n"
            f"Category: {category}, Name: {name}"
        )

    with open(prompt_path) as f:
        return yaml.safe_load(f)


def create_signature(category: str, name: str) -> type:
    """
    Create a DSPy signature class from prompt YAML config.

    Args:
        category: Prompt category ('agents', 'extraction', 'generation')
        name: Prompt name (without .yaml extension)

    Returns:
        DSPy Signature class

    Example:
        >>> QualifySig = create_signature("agents", "qualification")
        >>> agent = dspy.ChainOfThought(QualifySig)
    """
    config = load_prompt_config(category, name)

    # Build annotations dict for signature fields
    annotations = {}

    # Add input fields
    for field_name, field_config in config.get('input_fields', {}).items():
        annotations[field_name] = (str, dspy.InputField(desc=field_config['desc']))

    # Add output fields
    for field_name, field_config in config.get('output_fields', {}).items():
        annotations[field_name] = (str, dspy.OutputField(desc=field_config['desc']))

    # Create signature class dynamically
    signature_class = type(
        f"{name.replace('_', ' ').title().replace(' ', '')}Signature",
        (dspy.Signature,),
        {
            '__doc__': config['description'],
            '__annotations__': annotations
        }
    )

    return signature_class
