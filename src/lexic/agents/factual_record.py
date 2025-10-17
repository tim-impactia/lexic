"""Factual record agent - creates structured factual records."""

import dspy
from lexic.shared.prompts import create_signature

# Create signature from YAML
CreateFactualRecord = create_signature("agents", "factual_record")


class FactualRecordAgent(dspy.Module):
    """Agent that creates factual records."""

    def __init__(self):
        super().__init__()
        self.create_record = dspy.ChainOfThought(CreateFactualRecord)

    def forward(self, initial_facts: str, investigation_report: str = "") -> str:
        """
        Create factual record.

        Args:
            initial_facts: Initial facts
            investigation_report: Investigation report (optional)

        Returns:
            Factual record as markdown text
        """
        result = self.create_record(
            initial_facts=initial_facts,
            investigation_report=investigation_report
        )
        return result.factual_record
