"""Initial analysis agent - produces initial legal analysis from situation."""

import dspy


class PerformInitialAnalysis(dspy.Signature):
    """
    Perform initial legal analysis based on client situation.

    Produces:
    - Legal domain identification
    - Potential legal bases to investigate
    - Preliminary assessment
    - Investigation needs
    - Complexity assessment
    """
    situation: str = dspy.InputField(
        desc="Client situation: summary, objectives, constraints, legal questions"
    )
    initial_analysis: str = dspy.OutputField(
        desc="Structured analysis with: legal domain, potential legal bases (bulleted), preliminary assessment, investigation needs (bulleted), complexity assessment"
    )


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
