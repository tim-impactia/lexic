"""Data models for Lexic legal AI pipeline."""

from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime


@dataclass
class ClientPersona:
    """Client persona for synthetic case generation."""
    name: str
    background: str
    situation_summary: str
    emotional_state: str
    goals: List[str]
    constraints: List[str]


@dataclass
class InitialFacts:
    """Initial facts known at the start of client intake."""
    summary: str
    timeline: List[str]
    documents_provided: List[str]
    uncertainties: List[str]


@dataclass
class Situation:
    """Client situation after qualification dialogue."""
    summary: str
    objectives: List[str]
    constraints: List[str]
    legal_questions: List[str]


@dataclass
class InitialAnalysis:
    """Initial legal analysis based on situation."""
    legal_domain: str
    potential_legal_bases: List[str]
    preliminary_assessment: str
    investigation_needs: List[str]
    complexity_assessment: str


@dataclass
class InvestigationOrder:
    """Order for client to gather additional information."""
    purpose: str
    questions: List[str]
    documents_requested: List[str]
    deadline: Optional[str] = None


@dataclass
class InvestigationReport:
    """Client's response to investigation order."""
    answers: List[str]
    documents_obtained: List[str]
    remaining_gaps: List[str]


@dataclass
class FactualRecord:
    """Structured factual record of the case."""
    summary: str
    parties: List[str]
    timeline: List[str]
    key_facts: List[str]
    evidence: List[str]


@dataclass
class LegalBasis:
    """Applicable legal provision."""
    article: str
    law_name: str
    content: str
    relevance: str


@dataclass
class LegalArgument:
    """Legal argument for the case."""
    thesis: str
    legal_bases: List[str]
    factual_support: List[str]
    reasoning: str


@dataclass
class Consideration:
    """Legal consideration and analysis."""
    issue: str
    analysis: str
    conclusion: str
    confidence: str


@dataclass
class Judgment:
    """Final legal judgment."""
    decision: str
    reasoning: str
    legal_bases: List[str]


@dataclass
class Recommendation:
    """Recommendation to client."""
    action: str
    rationale: str
    risks: List[str]
    alternatives: List[str]
    next_steps: List[str]


@dataclass
class CourtDecision:
    """Extracted court decision."""
    decision_id: str
    court: str
    date: str
    case_number: str
    parties: List[str]
    facts_timeline: List[str]
    legal_bases: List[LegalBasis]
    arguments: List[LegalArgument]
    considerations: List[Consideration]
    judgment: Judgment


@dataclass
class SyntheticCase:
    """Complete synthetic case with ground truth at each step."""
    case_id: str
    source_decision_id: str
    metadata: dict
    client_persona: ClientPersona
    initial_facts: InitialFacts
    gt_situation: Situation
    gt_initial_analysis: InitialAnalysis
    gt_investigation_order_1: InvestigationOrder
    gt_investigation_report_1: InvestigationReport
    gt_initial_factual_record: FactualRecord
    gt_final_factual_record: FactualRecord
    gt_applicable_legal_bases: List[LegalBasis]
    gt_legal_arguments: List[LegalArgument]
    gt_considerations: List[Consideration]
    gt_judgment: Judgment
    gt_recommendations: List[Recommendation]
    generated_at: datetime = field(default_factory=datetime.now)


@dataclass
class EvaluationResult:
    """Evaluation result for a single case step."""
    case_id: str
    step: str
    prediction: str
    ground_truth: str
    scores: dict
    critical_errors: List[str]
    overall_score: float
    timestamp: datetime = field(default_factory=datetime.now)
