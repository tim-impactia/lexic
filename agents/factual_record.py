"""Factual record agent - creates structured factual records."""

import dspy


class CreateFactualRecord(dspy.Signature):
    """
    Create a structured factual record.

    Produces:
    - Summary of facts
    - Parties involved
    - Chronological timeline
    - Key facts
    - Evidence available
    """
    initial_facts: str = dspy.InputField(
        desc="Initial facts from client intake"
    )
    investigation_report: str = dspy.InputField(
        desc="Investigation report with additional facts (empty if none)"
    )
    factual_record: str = dspy.OutputField(
        desc="Structured factual record with: summary, parties (bulleted), timeline (bulleted, chronological), key facts (bulleted), evidence (bulleted)"
    )


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
