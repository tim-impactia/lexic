"""Consideration agent - analyzes legal considerations."""

import dspy
from lexic.shared.prompts import create_signature

# Create signature from YAML
AnalyzeConsiderations = create_signature("agents", "considerations")


class ConsiderationAgent(dspy.Module):
    """Agent that analyzes legal considerations."""

    def __init__(self):
        super().__init__()
        self.analyze = dspy.ChainOfThought(AnalyzeConsiderations)

    def forward(self, arguments: str, factual_record: str) -> str:
        """
        Analyze legal considerations.

        Args:
            arguments: Legal arguments
            factual_record: Factual record

        Returns:
            Legal considerations as markdown text
        """
        result = self.analyze(
            arguments=arguments,
            factual_record=factual_record
        )
        return result.considerations
