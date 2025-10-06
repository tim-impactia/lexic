"""Consideration agent - analyzes legal considerations."""

import dspy


class AnalyzeConsiderations(dspy.Signature):
    """
    Analyze legal considerations from arguments and factual record.

    For each consideration, provide:
    - Issue being considered
    - Analysis of the issue
    - Conclusion
    - Confidence level
    """
    arguments: str = dspy.InputField(
        desc="Legal arguments"
    )
    factual_record: str = dspy.InputField(
        desc="Factual record"
    )
    considerations: str = dspy.OutputField(
        desc="Legal considerations, each with: issue, analysis, conclusion, confidence"
    )


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
