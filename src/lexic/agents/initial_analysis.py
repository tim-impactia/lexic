"""Initial analysis agent - produces initial legal analysis from qualification."""

import dspy
from lexic.shared.prompts import create_signature

# Create signature from YAML
PerformInitialAnalysis = create_signature("agents", "initial_analysis")


class InitialAnalysisAgent(dspy.Module):
    """Agent that performs initial legal analysis."""

    def __init__(self):
        super().__init__()
        self.analyze = dspy.ChainOfThought(PerformInitialAnalysis)

    def forward(self, qualification: str) -> str:
        """
        Perform initial analysis.

        Args:
            qualification: Qualification report

        Returns:
            Initial analysis as markdown text
        """
        result = self.analyze(qualification=qualification)
        return result.initial_analysis
