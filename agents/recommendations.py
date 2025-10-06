"""Recommendation agent - provides client recommendations."""

import dspy


class GenerateRecommendations(dspy.Signature):
    """
    Generate recommendations for the client.

    For each recommendation, provide:
    - Action to take
    - Rationale
    - Risks
    - Alternatives
    - Next steps
    """
    considerations: str = dspy.InputField(
        desc="Legal considerations"
    )
    judgment: str = dspy.InputField(
        desc="Legal judgment or expected outcome"
    )
    client_objectives: str = dspy.InputField(
        desc="Client objectives from situation report"
    )
    recommendations: str = dspy.OutputField(
        desc="Recommendations, each with: action, rationale, risks (bulleted), alternatives (bulleted), next steps (bulleted)"
    )


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
