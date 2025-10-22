"""Production agents for Lexic legal AI pipeline."""

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
from lexic.agents.pipeline import LexicPipeline

__all__ = [
    "QualificationAgent",
    "InitialAnalysisAgent",
    "InvestigationOrderAgent",
    "InvestigationReportAgent",
    "FactualRecordAgent",
    "LegalBasisAgent",
    "ArgumentationAgent",
    "ConsiderationAgent",
    "JudgmentAgent",
    "RecommendationAgent",
    "LexicPipeline",
]
