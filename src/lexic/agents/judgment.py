"""Judgment agent - predicts likely court outcome based on legal considerations."""

import dspy
from lexic.shared.prompts import create_signature

# Create signature from YAML
PredictJudgment = create_signature("agents", "judgment")


class JudgmentAgent(dspy.Module):
    """Agent that predicts the likely judgment if the case goes to court."""

    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(PredictJudgment)

    def forward(self, considerations: str, factual_record: str) -> str:
        """
        Predict likely judgment.

        Args:
            considerations: Legal considerations analyzing the case
            factual_record: Established factual record

        Returns:
            Predicted judgment as markdown text
        """
        result = self.predict(
            considerations=considerations,
            factual_record=factual_record
        )
        return result.judgment
