"""Qualification agent - transforms client dialogue into situation report."""

import dspy
from shared.models import ClientPersona, InitialFacts, Situation


class QualifyClientSituation(dspy.Signature):
    """
    Analyser la demande du client pour produire un rapport de qualification.

    Le rapport de qualification capture:
    - Résumé de la situation juridique du client
    - Objectifs du client (ce qu'il souhaite accomplir)
    - Contraintes (temps, budget, tolérance au risque)
    - Questions juridiques à traiter

    IMPORTANT: Répondre entièrement en français.
    """
    client_request: str = dspy.InputField(
        desc="Demande/message initial du client tel qu'écrit par lui, reflétant son style de communication, son état émotionnel et son niveau de détail"
    )
    situation: str = dspy.OutputField(
        desc="Rapport de situation structuré avec: résumé, objectifs (liste à puces), contraintes (liste à puces), questions juridiques (liste à puces). Répondre entièrement en français."
    )


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
