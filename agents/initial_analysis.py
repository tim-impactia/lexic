"""Initial analysis agent - produces initial legal analysis from situation."""

import dspy


class PerformInitialAnalysis(dspy.Signature):
    """
    Effectuer une analyse juridique initiale basée sur la situation du client.

    Produit:
    - Identification du domaine juridique
    - Bases légales potentielles à investiguer
    - Évaluation préliminaire
    - Besoins d'investigation
    - Évaluation de la complexité

    IMPORTANT: Répondre entièrement en français.
    """
    situation: str = dspy.InputField(
        desc="Situation du client: résumé, objectifs, contraintes, questions juridiques"
    )
    initial_analysis: str = dspy.OutputField(
        desc="Analyse structurée avec: domaine juridique, bases légales potentielles (liste à puces), évaluation préliminaire, besoins d'investigation (liste à puces), évaluation de la complexité. Répondre entièrement en français."
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
