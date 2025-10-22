"""Legal basis agent - identifies applicable legal provisions."""

import dspy
from lexic.shared.prompts import create_signature

# Create signature from YAML
IdentifyLegalBasis = create_signature("agents", "legal_basis")


class LegalBasisAgent(dspy.Module):
    """Agent that identifies applicable legal basis."""

    def __init__(self):
        super().__init__()
        self.identify = dspy.ChainOfThought(IdentifyLegalBasis)

    def forward(self, factual_record: str) -> str:
        """
        Identify legal basis.

        Args:
            factual_record: Factual record

        Returns:
            Legal basis as markdown text
        """
        result = self.identify(factual_record=factual_record)
        return result.legal_basis
