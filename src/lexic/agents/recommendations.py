"""Recommendation agent - provides client recommendations."""

import dspy
from lexic.shared.prompts import create_signature

# Create signature from YAML
GenerateRecommendations = create_signature("agents", "recommendations")


class RecommendationAgent(dspy.Module):
    """Agent that generates client recommendations."""

    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(GenerateRecommendations)

    def forward(self, considerations: str, judgment: str, client_objectives: str) -> str:
        """
        Generate recommendations.

        Args:
            considerations: Legal considerations
            judgment: Legal judgment
            client_objectives: Client objectives

        Returns:
            Recommendations as markdown text
        """
        result = self.generate(
            considerations=considerations,
            judgment=judgment,
            client_objectives=client_objectives
        )
        return result.recommendations
