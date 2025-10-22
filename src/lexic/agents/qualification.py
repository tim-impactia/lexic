"""Qualification agent - transforms client dialogue into qualification report."""

import dspy
from lexic.shared.models import ClientPersona, InitialFacts, Qualification
from lexic.shared.prompts import create_signature

# Create signature from YAML
QualifyClientQualification = create_signature("agents", "qualification")


class QualificationAgent(dspy.Module):
    """
    Agent that performs client qualification.

    Transforms client dialogue (persona + initial facts) into a structured
    qualification report that captures objectives, constraints.
    """

    def __init__(self):
        super().__init__()
        self.qualify = dspy.ChainOfThought(QualifyClientQualification)

    def forward(self, client_request: str) -> str:
        """
        Perform qualification.

        Args:
            client_request: Client's initial request/message

        Returns:
            Qualification report as markdown text
        """
        result = self.qualify(client_request=client_request)
        return result.qualification


def run_qualification(client_request: str) -> str:
    """
    Run qualification agent.

    Args:
        client_request: Client's initial request/message

    Returns:
        Qualification report
    """
    agent = QualificationAgent()
    return agent(client_request=client_request)
