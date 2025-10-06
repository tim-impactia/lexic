"""Argumentation agent - develops legal arguments."""

import dspy


class DevelopArguments(dspy.Signature):
    """
    Develop legal arguments from factual record and legal bases.

    For each argument, provide:
    - Thesis (main claim)
    - Legal bases supporting the thesis
    - Factual support
    - Reasoning connecting facts to legal conclusion
    """
    factual_record: str = dspy.InputField(
        desc="Factual record of the case"
    )
    legal_bases: str = dspy.InputField(
        desc="Applicable legal provisions"
    )
    arguments: str = dspy.OutputField(
        desc="Legal arguments, each with: thesis, legal bases, factual support, reasoning"
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
