"""Legal bases agent - identifies applicable legal provisions."""

import dspy


class IdentifyLegalBases(dspy.Signature):
    """
    Identify applicable legal bases from factual record.

    For each legal basis, provide:
    - Article/provision number
    - Law name
    - Content of the provision
    - Relevance to the case
    """
    factual_record: str = dspy.InputField(
        desc="Factual record of the case"
    )
    legal_bases: str = dspy.OutputField(
        desc="Applicable legal bases, each with: article, law name, content, relevance"
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
