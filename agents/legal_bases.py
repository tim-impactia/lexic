"""Legal bases agent - identifies applicable legal provisions."""

import dspy


class IdentifyLegalBases(dspy.Signature):
    """
    Identifier les bases légales applicables à partir de l'état de fait.

    Pour chaque base légale, fournir:
    - Numéro d'article/disposition
    - Nom de la loi
    - Contenu de la disposition
    - Pertinence pour le cas

    IMPORTANT: Répondre entièrement en français.
    """
    factual_record: str = dspy.InputField(
        desc="État de fait du cas"
    )
    legal_bases: str = dspy.OutputField(
        desc="Bases légales applicables, chacune avec: article, nom de la loi, contenu, pertinence. Répondre entièrement en français."
    )


class LegalBasisAgent(dspy.Module):
    """Agent that identifies applicable legal bases."""

    def __init__(self):
        super().__init__()
        self.identify = dspy.ChainOfThought(IdentifyLegalBases)

    def forward(self, factual_record: str) -> str:
        """
        Identify legal bases.

        Args:
            factual_record: Factual record

        Returns:
            Legal bases as markdown text
        """
        result = self.identify(factual_record=factual_record)
        return result.legal_bases
