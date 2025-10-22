"""Full orchestrated pipeline for Lexic legal AI system."""

from typing import Dict
from pathlib import Path

from lexic.agents.qualification import QualificationAgent
from lexic.agents.initial_analysis import InitialAnalysisAgent
from lexic.agents.investigation_order import InvestigationOrderAgent
from lexic.agents.investigation_report import InvestigationReportAgent
from lexic.agents.factual_record import FactualRecordAgent
from lexic.agents.legal_basis import LegalBasisAgent
from lexic.agents.arguments import ArgumentationAgent
from lexic.agents.considerations import ConsiderationAgent
from lexic.agents.judgment import JudgmentAgent
from lexic.agents.recommendations import RecommendationAgent


class LexicPipeline:
    """
    Full legal AI pipeline orchestrating all agents.

    This implements the complete workflow from client intake to recommendations.
    """

    def __init__(self):
        """Initialize all agents."""
        self.qualification_agent = QualificationAgent()
        self.initial_analysis_agent = InitialAnalysisAgent()
        self.investigation_order_agent = InvestigationOrderAgent()
        self.investigation_report_agent = InvestigationReportAgent()
        self.factual_record_agent = FactualRecordAgent()
        self.legal_basis_agent = LegalBasisAgent()
        self.argumentation_agent = ArgumentationAgent()
        self.consideration_agent = ConsiderationAgent()
        self.judgment_agent = JudgmentAgent()
        self.recommendation_agent = RecommendationAgent()

    def run_intake_to_analysis(
        self,
        client_persona: str,
        initial_facts: str
    ) -> Dict[str, str]:
        """
        Run first phase: intake to initial analysis.

        Args:
            client_persona: Client background and context
            initial_facts: Initial facts from client

        Returns:
            Dict with qualification and initial_analysis
        """
        # Step 1: Qualification
        qualification = self.qualification_agent(
            client_persona=client_persona,
            initial_facts=initial_facts
        )

        # Step 2: Initial analysis
        initial_analysis = self.initial_analysis_agent(
            qualification=qualification
        )

        return {
            "qualification": qualification,
            "initial_analysis": initial_analysis
        }

    def run_investigation_phase(
        self,
        initial_analysis: str,
        client_persona: str,
        initial_facts: str
    ) -> Dict[str, str]:
        """
        Run investigation phase.

        Args:
            initial_analysis: Initial legal analysis
            client_persona: Client profile and context
            initial_facts: Initial facts from intake

        Returns:
            Dict with investigation_order, investigation_report, and factual_record
        """
        # Step 3: Investigation order
        investigation_order = self.investigation_order_agent(
            initial_analysis=initial_analysis
        )

        # Step 4: Investigation report (simulated client responses)
        investigation_report = self.investigation_report_agent(
            investigation_order=investigation_order,
            client_persona=client_persona,
            initial_facts=initial_facts
        )

        # Step 5: Factual record (after investigation)
        factual_record = self.factual_record_agent(
            initial_facts=initial_facts,
            investigation_report=investigation_report
        )

        return {
            "investigation_order": investigation_order,
            "investigation_report": investigation_report,
            "factual_record": factual_record
        }

    def run_legal_analysis(
        self,
        factual_record: str
    ) -> Dict[str, str]:
        """
        Run legal analysis phase.

        Args:
            factual_record: Structured factual record

        Returns:
            Dict with legal_basis and legal_arguments
        """
        # Step 5: Identify legal basis
        legal_basis = self.legal_basis_agent(
            factual_record=factual_record
        )

        # Step 6: Develop legal arguments
        legal_arguments = self.argumentation_agent(
            factual_record=factual_record,
            legal_basis=legal_basis
        )

        return {
            "legal_basis": legal_basis,
            "legal_arguments": legal_arguments
        }

    def run_final_phase(
        self,
        legal_arguments: str,
        factual_record: str,
        client_objectives: str,
        use_predicted_judgment: bool = True
    ) -> Dict[str, str]:
        """
        Run final phase: considerations, judgment prediction, and recommendations.

        Args:
            legal_arguments: Legal arguments
            factual_record: Factual record
            client_objectives: Client objectives from qualification
            use_predicted_judgment: Whether to predict judgment (default: True)

        Returns:
            Dict with considerations, judgment, and recommendations
        """
        # Step 7: Legal considerations
        considerations = self.consideration_agent(
            arguments=legal_arguments,
            factual_record=factual_record
        )

        # Step 8: Predict judgment
        judgment = None
        if use_predicted_judgment:
            judgment = self.judgment_agent(
                considerations=considerations,
                factual_record=factual_record
            )

        # Step 9: Recommendations
        recommendations = self.recommendation_agent(
            considerations=considerations,
            judgment=judgment if judgment else considerations,
            client_objectives=client_objectives
        )

        result = {
            "considerations": considerations,
            "recommendations": recommendations
        }

        if judgment:
            result["judgment"] = judgment

        return result

    def run_full_pipeline(
        self,
        client_persona: str,
        initial_facts: str
    ) -> Dict[str, str]:
        """
        Run the complete pipeline from intake to recommendations.

        Args:
            client_persona: Client background and context
            initial_facts: Initial facts from client

        Returns:
            Dict with all pipeline outputs including predicted judgment
        """
        # Phase 1: Intake to analysis
        phase1 = self.run_intake_to_analysis(client_persona, initial_facts)

        # Phase 2: Investigation (generates investigation_report automatically)
        phase2 = self.run_investigation_phase(
            phase1["initial_analysis"],
            client_persona,
            initial_facts
        )

        # Phase 3: Legal analysis
        phase3 = self.run_legal_analysis(phase2["factual_record"])

        # Phase 4: Final phase (considerations, judgment prediction, recommendations)
        phase4 = self.run_final_phase(
            phase3["legal_arguments"],
            phase2["factual_record"],
            phase1["qualification"],
            use_predicted_judgment=True
        )

        # Combine all results
        return {
            **phase1,
            **phase2,
            **phase3,
            **phase4
        }
