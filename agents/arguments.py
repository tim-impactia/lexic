"""Argumentation agent - develops legal arguments."""

import dspy


class DevelopArguments(dspy.Signature):
    """
    Développer des arguments juridiques à partir de l'état de fait et des bases légales.

    Pour chaque argument, fournir:
    - Thèse (prétention principale)
    - Bases légales soutenant la thèse
    - Soutien factuel
    - Raisonnement reliant les faits à la conclusion juridique

    IMPORTANT: Répondre entièrement en français.
    """
    factual_record: str = dspy.InputField(
        desc="État de fait du cas"
    )
    legal_bases: str = dspy.InputField(
        desc="Dispositions légales applicables"
    )
    arguments: str = dspy.OutputField(
        desc="Arguments juridiques, chacun avec: thèse, bases légales, soutien factuel, raisonnement. Répondre entièrement en français."
    )


class ArgumentationAgent(dspy.Module):
    """Agent that develops legal arguments."""

    def __init__(self):
        super().__init__()
        self.develop = dspy.ChainOfThought(DevelopArguments)

    def forward(self, factual_record: str, legal_bases: str) -> str:
        """
        Develop legal arguments.

        Args:
            factual_record: Factual record
            legal_bases: Legal bases

        Returns:
            Legal arguments as markdown text
        """
        result = self.develop(
            factual_record=factual_record,
            legal_bases=legal_bases
        )
        return result.arguments
