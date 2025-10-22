"""Generate synthetic cases with ground truth from court decisions."""

import dspy
from pathlib import Path
from datetime import datetime
from typing import List
from lexic.shared.models import (
    ClientPersona, InitialFacts, Qualification, InitialAnalysis,
    InvestigationOrder, InvestigationReport, FactualRecord,
    LegalBasis, LegalArgument, Consideration, Judgment, Recommendation
)
from lexic.shared.io import read_markdown, write_markdown, get_decision_path
from lexic.shared.config import Config
from lexic.shared.prompts import create_signature

# Create signatures from YAML
GenerateClientPersona = create_signature("generation", "client_persona")
GenerateClientRequest = create_signature("generation", "client_request")
GenerateInitialFacts = create_signature("generation", "initial_facts")
GenerateQualification = create_signature("generation", "qualification")
GenerateInitialAnalysis = create_signature("generation", "initial_analysis")
GenerateInvestigationOrder = create_signature("generation", "investigation_order")
GenerateInvestigationReport = create_signature("generation", "investigation_report")
GenerateFactualRecord = create_signature("generation", "factual_record")
GenerateRecommendations = create_signature("generation", "recommendations")


class SyntheticCaseGenerator(dspy.Module):
    """Generate synthetic case with ground truth from a court decision."""

    def __init__(self):
        super().__init__()
        self.gen_persona = dspy.ChainOfThought(GenerateClientPersona)
        self.gen_client_request = dspy.ChainOfThought(GenerateClientRequest)
        self.gen_initial_facts = dspy.ChainOfThought(GenerateInitialFacts)
        self.gen_qualification = dspy.ChainOfThought(GenerateQualification)
        self.gen_initial_analysis = dspy.ChainOfThought(GenerateInitialAnalysis)
        self.gen_investigation_order = dspy.ChainOfThought(GenerateInvestigationOrder)
        self.gen_investigation_report = dspy.ChainOfThought(GenerateInvestigationReport)
        self.gen_initial_factual_record = dspy.ChainOfThought(GenerateFactualRecord)
        self.gen_final_factual_record = dspy.ChainOfThought(GenerateFactualRecord)
        self.gen_recommendations = dspy.ChainOfThought(GenerateRecommendations)

    def forward(self, decision_data: dict):
        """
        Generate synthetic case from court decision.

        Args:
            decision_data: Dict with keys: full_text, parties, facts_timeline,
                          legal_bases, arguments, considerations, judgment

        Returns:
            Dict with all generated synthetic case elements
        """
        decision_context = f"{decision_data['parties']}\n\n{decision_data['facts_timeline']}"

        # Generate backwards from decision
        print("  Generating client persona...")
        persona = self.gen_persona(decision_context=decision_context)

        print("  Generating client request...")
        client_request = self.gen_client_request(
            client_persona=persona.client_persona,
            decision_context=decision_context
        )

        print("  Generating initial facts...")
        initial_facts = self.gen_initial_facts(
            decision_context=decision_data['facts_timeline'],
            client_persona=persona.client_persona
        )

        print("  Generating qualification report...")
        qualification = self.gen_qualification(
            client_persona=persona.client_persona,
            initial_facts=initial_facts.initial_facts,
            decision_context=decision_context
        )

        print("  Generating initial analysis...")
        initial_analysis = self.gen_initial_analysis(
            qualification=qualification.qualification,
            decision_legal_bases=decision_data['legal_bases']
        )

        print("  Generating investigation order...")
        investigation_order = self.gen_investigation_order(
            initial_analysis=initial_analysis.initial_analysis,
            decision_facts=decision_data['facts_timeline']
        )

        print("  Generating investigation report...")
        investigation_report = self.gen_investigation_report(
            investigation_order=investigation_order.investigation_order,
            decision_facts=decision_data['facts_timeline']
        )

        print("  Generating initial factual record...")
        initial_factual_record = self.gen_initial_factual_record(
            initial_facts=initial_facts.initial_facts,
            investigation_report="",
            decision_facts=decision_data['facts_timeline']
        )

        print("  Generating final factual record...")
        final_factual_record = self.gen_final_factual_record(
            initial_facts=initial_facts.initial_facts,
            investigation_report=investigation_report.investigation_report,
            decision_facts=decision_data['facts_timeline']
        )

        print("  Generating recommendations...")
        recommendations = self.gen_recommendations(
            judgment=decision_data['judgment'],
            considerations=decision_data['considerations'],
            client_objectives=qualification.qualification.objectives
        )

        return dspy.Prediction(
            client_persona=persona.client_persona,
            client_request=client_request.client_request,
            initial_facts=initial_facts.initial_facts,
            qualification=qualification.qualification,
            initial_analysis=initial_analysis.initial_analysis,
            investigation_order=investigation_order.investigation_order,
            investigation_report=investigation_report.investigation_report,
            initial_factual_record=initial_factual_record.factual_record,
            final_factual_record=final_factual_record.factual_record,
            recommendations=recommendations.recommendations
        )


def load_decision_data(decision_dir: Path) -> dict:
    """
    Load all extracted data from a decision directory.

    Args:
        decision_dir: Path to decision directory

    Returns:
        Dict with all decision data
    """
    _, full_text = read_markdown(decision_dir / "full_text.md")
    _, parties = read_markdown(decision_dir / "parties.md")
    _, facts = read_markdown(decision_dir / "facts_timeline.md")
    _, legal_bases = read_markdown(decision_dir / "legal_bases.md")
    _, arguments = read_markdown(decision_dir / "arguments.md")
    _, considerations = read_markdown(decision_dir / "considerations.md")
    _, judgment = read_markdown(decision_dir / "judgment.md")

    return {
        "full_text": full_text,
        "parties": parties,
        "facts_timeline": facts,
        "legal_bases": legal_bases,
        "arguments": arguments,
        "considerations": considerations,
        "judgment": judgment
    }


def generate_synthetic_case(decision_id: str, case_id: str, party_role: str, decisions_dir: Path, output_dir: Path, specific_docs: List[str] = None):
    """
    Generate a synthetic case from a court decision for a specific party.
    Saves files incrementally and skips steps that already exist.

    Args:
        decision_id: ID of the source court decision
        case_id: ID for the synthetic case (e.g., 'case_001_pl' or 'case_001_df')
        party_role: Role of the party - 'demandeur' (plaintiff) or 'défendeur' (defendant)
        decisions_dir: Directory containing court decisions
        output_dir: Directory to save synthetic case
        specific_docs: List of specific document numbers to generate (e.g., ['01', '02', '03']).
                      If None, generates all documents. If provided, only generates specified documents.
    """
    decision_dir = get_decision_path(decisions_dir, decision_id)

    # Load decision data
    print(f"Loading decision {decision_id}...")
    decision_data = load_decision_data(decision_dir)

    # Prepare output directory
    case_dir = output_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "case_id": case_id,
        "source_decision": decision_id,
        "generated_at": datetime.now().isoformat(),
        "model": Config.GENERATION_MODEL
    }

    # Helper to check and save
    def save_if_missing(filename: str, title: str, content: str, generator_func=None, doc_number: str = None):
        """Save file if it doesn't exist, otherwise load existing content."""
        filepath = case_dir / filename

        # Check if this document should be processed based on specific_docs filter
        if specific_docs is not None and doc_number is not None:
            if doc_number not in specific_docs:
                # Skip this document - not in the requested list
                if filepath.exists():
                    _, existing_content = read_markdown(filepath)
                    return existing_content
                return ""  # Return empty string for dependencies

        if filepath.exists():
            print(f"  ✓ {title} (already exists)")
            _, existing_content = read_markdown(filepath)
            return existing_content
        else:
            if generator_func:
                print(f"  Generating {title.lower()}...")
                content = generator_func()
            write_markdown(filepath, metadata, f"# {title}\n\n{content}")
            print(f"  ✓ {title} (saved)")
            return content

    print(f"Generating synthetic case {case_id}...")

    # Initialize generator
    generator = SyntheticCaseGenerator()
    decision_context = f"{decision_data['parties']}\n\n{decision_data['facts_timeline']}\n\n{decision_data['judgment']}"

    # Save metadata
    save_if_missing("metadata.md", "Case Metadata", f"Source: {decision_id}")

    # Generate and save incrementally
    client_persona = save_if_missing(
        "01_client_persona.md", "Client Persona", "",
        lambda: generator.gen_persona(
            decision_context=decision_context,
            party_role=party_role
        ).client_persona,
        doc_number="01"
    )

    client_request = save_if_missing(
        "01b_client_request.md", "Client Request", "",
        lambda: generator.gen_client_request(
            client_persona=client_persona,
            decision_context=decision_context,
            party_role=party_role
        ).client_request,
        doc_number="01b"
    )

    initial_facts = save_if_missing(
        "02_initial_facts_known.md", "Initial Facts", "",
        lambda: generator.gen_initial_facts(
            decision_context=decision_data['facts_timeline'],
            client_persona=client_persona,
            party_role=party_role
        ).initial_facts,
        doc_number="02"
    )

    qualification = save_if_missing(
        "03_gt_qualification.md", "Ground Truth: Qualification", "",
        lambda: generator.gen_qualification(
            client_persona=client_persona,
            initial_facts=initial_facts,
            decision_context=decision_context
        ).qualification,
        doc_number="03"
    )

    initial_analysis = save_if_missing(
        "04_gt_initial_analysis.md", "Ground Truth: Initial Analysis", "",
        lambda: generator.gen_initial_analysis(
            qualification=qualification,
            decision_legal_bases=decision_data['legal_bases']
        ).initial_analysis,
        doc_number="04"
    )

    investigation_order = save_if_missing(
        "05_gt_investigation_order_1.md", "Ground Truth: Investigation Order", "",
        lambda: generator.gen_investigation_order(
            initial_analysis=initial_analysis,
            decision_facts=decision_data['facts_timeline']
        ).investigation_order,
        doc_number="05"
    )

    investigation_report = save_if_missing(
        "06_gt_investigation_report_1.md", "Ground Truth: Investigation Report", "",
        lambda: generator.gen_investigation_report(
            investigation_order=investigation_order,
            decision_facts=decision_data['facts_timeline']
        ).investigation_report,
        doc_number="06"
    )

    initial_factual_record = save_if_missing(
        "09_gt_initial_factual_record.md", "Ground Truth: Initial Factual Record", "",
        lambda: generator.gen_initial_factual_record(
            initial_facts=initial_facts,
            investigation_report="",
            decision_facts=decision_data['facts_timeline']
        ).factual_record,
        doc_number="09"
    )

    final_factual_record = save_if_missing(
        "10_gt_final_factual_record.md", "Ground Truth: Final Factual Record", "",
        lambda: generator.gen_final_factual_record(
            initial_facts=initial_facts,
            investigation_report=investigation_report,
            decision_facts=decision_data['facts_timeline']
        ).factual_record,
        doc_number="10"
    )

    # Copy decision data (these don't need generation)
    save_if_missing("11_gt_applicable_legal_bases.md", "Ground Truth: Legal Bases", decision_data['legal_bases'], doc_number="11")
    save_if_missing("12_gt_legal_arguments.md", "Ground Truth: Legal Arguments", decision_data['arguments'], doc_number="12")
    save_if_missing("13_gt_considerations.md", "Ground Truth: Considerations", decision_data['considerations'], doc_number="13")
    save_if_missing("14_gt_judgment.md", "Ground Truth: Judgment", decision_data['judgment'], doc_number="14")

    recommendations = save_if_missing(
        "15_gt_recommendations.md", "Ground Truth: Recommendations", "",
        lambda: generator.gen_recommendations(
            judgment=decision_data['judgment'],
            considerations=decision_data['considerations'],
            client_objectives=qualification
        ).recommendations,
        doc_number="15"
    )

    print(f"✓ Synthetic case complete at {case_dir}")


def generate_all_synthetic_cases(decisions_dir: Path, output_dir: Path):
    """
    Generate synthetic cases from all extracted court decisions.

    Args:
        decisions_dir: Directory containing court decisions
        output_dir: Directory to save synthetic cases
    """
    from lexic.shared.io import list_decision_dirs

    decision_ids = list_decision_dirs(decisions_dir)

    if not decision_ids:
        print(f"No decisions found in {decisions_dir}")
        return

    print(f"Found {len(decision_ids)} decisions")
    print(f"Generating 2 synthetic cases per decision (plaintiff + defendant)")

    for i, decision_id in enumerate(decision_ids, 1):
        base_case_id = f"case_{i:03d}"

        # Generate plaintiff case
        case_id_pl = f"{base_case_id}_pl"
        print(f"\n{'='*60}")
        print(f"Generating plaintiff case: {case_id_pl}")
        print(f"{'='*60}")
        try:
            generate_synthetic_case(decision_id, case_id_pl, "demandeur", decisions_dir, output_dir)
        except Exception as e:
            print(f"✗ Error generating plaintiff case from {decision_id}: {e}")

        # Generate defendant case
        case_id_df = f"{base_case_id}_df"
        print(f"\n{'='*60}")
        print(f"Generating defendant case: {case_id_df}")
        print(f"{'='*60}")
        try:
            generate_synthetic_case(decision_id, case_id_df, "défendeur", decisions_dir, output_dir)
        except Exception as e:
            print(f"✗ Error generating defendant case from {decision_id}: {e}")
