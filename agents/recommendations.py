"""Recommendation agent - provides client recommendations."""

import dspy


class GenerateRecommendations(dspy.Signature):
    """
    Générer des recommandations pour le client.

    Pour chaque recommandation, fournir:
    - Action à entreprendre
    - Justification
    - Risques
    - Alternatives
    - Prochaines étapes

    IMPORTANT: Répondre entièrement en français.
    """
    considerations: str = dspy.InputField(
        desc="Considérations juridiques"
    )
    judgment: str = dspy.InputField(
        desc="Jugement juridique ou issue attendue"
    )
    client_objectives: str = dspy.InputField(
        desc="Objectifs du client tirés du rapport de situation"
    )
    recommendations: str = dspy.OutputField(
        desc="Recommandations, chacune avec: action, justification, risques (liste à puces), alternatives (liste à puces), prochaines étapes (liste à puces). Répondre entièrement en français."
    )


class RecommendationAgent(dspy.Module):
    """Agent that generates client recommendations."""

    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(GenerateRecommendations)

    def forward(self, considerations: str, judgment: str, client_objectives: str) -> str:
        """
        Generate recommendations.

        Args:
            considerations: Legal considerations
            judgment: Legal judgment
            client_objectives: Client objectives

        Returns:
            Recommendations as markdown text
        """
        result = self.generate(
            considerations=considerations,
            judgment=judgment,
            client_objectives=client_objectives
        )
        return result.recommendations
