"""Initial analysis agent - produces initial legal analysis from situation."""

import dspy
from shared.prompts import create_signature

# Create signature from YAML
PerformInitialAnalysis = create_signature("agents", "initial_analysis")


class InitialAnalysisAgent(dspy.Module):
    """Agent that performs initial legal analysis."""

    def __init__(self):
        super().__init__()
        self.analyze = dspy.ChainOfThought(PerformInitialAnalysis)

    def forward(self, situation: str) -> str:
        """
        Perform initial analysis.

        Args:
            situation: Situation report

        Returns:
            Initial analysis as markdown text
        """
        result = self.analyze(situation=situation)
        return result.initial_analysis
