"""Argumentation agent - develops legal arguments."""

import dspy
from shared.prompts import create_signature

# Create signature from YAML
DevelopArguments = create_signature("agents", "arguments")


class ArgumentationAgent(dspy.Module):
    """Agent that develops legal arguments."""

    def __init__(self):
        super().__init__()
        self.develop = dspy.ChainOfThought(DevelopArguments)

    def forward(self, factual_record: str, legal_bases: str) -> str:
        """
        Develop legal arguments.

        Args:
            factual_record: Factual record
            legal_bases: Legal bases

        Returns:
            Legal arguments as markdown text
        """
        result = self.develop(
            factual_record=factual_record,
            legal_bases=legal_bases
        )
        return result.arguments
