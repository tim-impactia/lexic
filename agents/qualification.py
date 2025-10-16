"""Qualification agent - transforms client dialogue into situation report."""

import dspy
from shared.models import ClientPersona, InitialFacts, Situation
from shared.prompts import create_signature

# Create signature from YAML
QualifyClientSituation = create_signature("agents", "qualification")


class QualificationAgent(dspy.Module):
    """
    Agent that performs client qualification.

    Transforms client dialogue (persona + initial facts) into a structured
    situation report that captures objectives, constraints, and legal questions.
    """

    def __init__(self):
        super().__init__()
        self.qualify = dspy.ChainOfThought(QualifyClientSituation)

    def forward(self, client_request: str) -> str:
        """
        Perform qualification.

        Args:
            client_request: Client's initial request/message

        Returns:
            Situation report as markdown text
        """
        result = self.qualify(client_request=client_request)
        return result.situation


def run_qualification(client_request: str) -> str:
    """
    Run qualification agent.

    Args:
        client_request: Client's initial request/message

    Returns:
        Situation report
    """
    agent = QualificationAgent()
    return agent(client_request=client_request)
