"""Investigation order agent - creates orders for client to gather information."""

import dspy


class CreateInvestigationOrder(dspy.Signature):
    """
    Create an investigation order based on initial analysis.

    Produces:
    - Purpose of the investigation
    - Questions for the client
    - Documents to request
    - Optional deadline
    """
    initial_analysis: str = dspy.InputField(
        desc="Initial legal analysis with investigation needs"
    )
    investigation_order: str = dspy.OutputField(
        desc="Investigation order with: purpose, questions (bulleted), documents requested (bulleted)"
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
