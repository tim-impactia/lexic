"""Factual record agent - creates structured factual records."""

import dspy


class CreateFactualRecord(dspy.Signature):
    """
    Créer un état de fait structuré.

    Produit:
    - Résumé des faits
    - Parties impliquées
    - Chronologie
    - Faits clés
    - Preuves disponibles

    IMPORTANT: Répondre entièrement en français.
    """
    initial_facts: str = dspy.InputField(
        desc="Faits initiaux de la prise de contact avec le client"
    )
    investigation_report: str = dspy.InputField(
        desc="Rapport d'investigation avec des faits supplémentaires (vide si aucun)"
    )
    factual_record: str = dspy.OutputField(
        desc="État de fait structuré avec: résumé, parties (liste à puces), chronologie (liste à puces, chronologique), faits clés (liste à puces), preuves (liste à puces). Répondre entièrement en français."
    )


class FactualRecordAgent(dspy.Module):
    """Agent that creates factual records."""

    def __init__(self):
        super().__init__()
        self.create_record = dspy.ChainOfThought(CreateFactualRecord)

    def forward(self, initial_facts: str, investigation_report: str = "") -> str:
        """
        Create factual record.

        Args:
            initial_facts: Initial facts
            investigation_report: Investigation report (optional)

        Returns:
            Factual record as markdown text
        """
        result = self.create_record(
            initial_facts=initial_facts,
            investigation_report=investigation_report
        )
        return result.factual_record
