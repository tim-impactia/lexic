"""Consideration agent - analyzes legal considerations."""

import dspy


class AnalyzeConsiderations(dspy.Signature):
    """
    Analyser les considérations juridiques à partir des arguments et de l'état de fait.

    Pour chaque considération, fournir:
    - Question examinée
    - Analyse de la question
    - Conclusion
    - Niveau de confiance

    IMPORTANT: Répondre entièrement en français.
    """
    arguments: str = dspy.InputField(
        desc="Arguments juridiques"
    )
    factual_record: str = dspy.InputField(
        desc="État de fait"
    )
    considerations: str = dspy.OutputField(
        desc="Considérations juridiques, chacune avec: question, analyse, conclusion, niveau de confiance. Répondre entièrement en français."
    )


class ConsiderationAgent(dspy.Module):
    """Agent that analyzes legal considerations."""

    def __init__(self):
        super().__init__()
        self.analyze = dspy.ChainOfThought(AnalyzeConsiderations)

    def forward(self, arguments: str, factual_record: str) -> str:
        """
        Analyze legal considerations.

        Args:
            arguments: Legal arguments
            factual_record: Factual record

        Returns:
            Legal considerations as markdown text
        """
        result = self.analyze(
            arguments=arguments,
            factual_record=factual_record
        )
        return result.considerations
