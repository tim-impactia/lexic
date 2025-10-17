"""Legal bases agent - identifies applicable legal provisions."""

import dspy
from lexic.shared.prompts import create_signature

# Create signature from YAML
IdentifyLegalBases = create_signature("agents", "legal_bases")


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
