"""Investigation order agent - creates orders for client to gather information."""

import dspy
from lexic.shared.prompts import create_signature

# Create signature from YAML
CreateInvestigationOrder = create_signature("agents", "investigation_order")


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
