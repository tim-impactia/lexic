"""Investigation order agent - creates orders for client to gather information."""

import dspy


class CreateInvestigationOrder(dspy.Signature):
    """
    Créer un ordre d'investigation basé sur l'analyse initiale.

    Produit:
    - Objectif de l'investigation
    - Questions pour le client
    - Documents à demander
    - Délai optionnel

    IMPORTANT: Répondre entièrement en français.
    """
    initial_analysis: str = dspy.InputField(
        desc="Analyse juridique initiale avec les besoins d'investigation"
    )
    investigation_order: str = dspy.OutputField(
        desc="Ordre d'investigation avec: objectif, questions (liste à puces), documents demandés (liste à puces). Répondre entièrement en français."
    )


class InvestigationOrderAgent(dspy.Module):
    """Agent that creates investigation orders."""

    def __init__(self):
        super().__init__()
        self.create_order = dspy.ChainOfThought(CreateInvestigationOrder)

    def forward(self, initial_analysis: str) -> str:
        """
        Create investigation order.

        Args:
            initial_analysis: Initial legal analysis

        Returns:
            Investigation order as markdown text
        """
        result = self.create_order(initial_analysis=initial_analysis)
        return result.investigation_order
