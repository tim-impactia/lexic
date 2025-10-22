"""Investigation report agent - simulates client responses to investigation questions."""

import dspy
from lexic.shared.prompts import create_signature

# Create signature from YAML
GenerateInvestigationReport = create_signature("agents", "investigation_report")


class InvestigationReportAgent(dspy.Module):
    """Agent that generates investigation reports through simulated client dialogue."""

    def __init__(self):
        super().__init__()
        self.generate_report = dspy.ChainOfThought(GenerateInvestigationReport)

    def forward(
        self,
        investigation_order: str,
        client_persona: str,
        initial_facts: str
    ) -> str:
        """
        Generate investigation report.

        Args:
            investigation_order: Investigation order with questions
            client_persona: Client profile and context
            initial_facts: Initial facts known to client

        Returns:
            Investigation report as markdown text
        """
        result = self.generate_report(
            investigation_order=investigation_order,
            client_persona=client_persona,
            initial_facts=initial_facts
        )
        return result.investigation_report
