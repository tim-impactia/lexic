"""Generate synthetic cases with ground truth from court decisions."""

import dspy
from pathlib import Path
from datetime import datetime
from typing import List
from shared.models import (
    ClientPersona, InitialFacts, Situation, InitialAnalysis,
    InvestigationOrder, InvestigationReport, FactualRecord,
    LegalBasis, LegalArgument, Consideration, Judgment, Recommendation
)
from shared.io import read_markdown, write_markdown, get_decision_path
from shared.config import Config


class GenerateClientPersona(dspy.Signature):
    """Générer le persona du client qui aurait mené à cette décision de justice. IMPORTANT: Répondre en français."""
    decision_context: str = dspy.InputField(desc="Faits, parties et issue de la décision de justice")
    client_persona: str = dspy.OutputField(desc="Persona du client en français (nom, contexte, résumé de situation, état émotionnel, objectifs, contraintes)")


class GenerateClientRequest(dspy.Signature):
    """Générer une demande client réaliste telle qu'il l'écrirait. IMPORTANT: Répondre en français."""
    client_persona: str = dspy.InputField(desc="Persona du client avec contexte, état émotionnel, style de communication")
    decision_context: str = dspy.InputField(desc="Faits et chronologie de la décision")
    client_request: str = dspy.OutputField(desc="Demande/email/message initial du client en français, dans sa voix, avec son niveau de détail et d'émotion. Doit refléter son persona (ex: professionnel = plus structuré, client émotionnel = moins organisé)")


class GenerateInitialFacts(dspy.Signature):
    """Générer les faits initiaux que le client aurait connus lors de la prise de contact. IMPORTANT: Répondre en français."""
    decision_context: str = dspy.InputField(desc="Faits et chronologie de la décision")
    client_persona: str = dspy.InputField(desc="Persona du client")
    initial_facts: str = dspy.OutputField(desc="Faits initiaux en français (résumé, chronologie, documents fournis, incertitudes)")


class GenerateSituation(dspy.Signature):
    """Générer le rapport de qualification/situation. IMPORTANT: Répondre en français."""
    client_persona: str = dspy.InputField(desc="Persona du client")
    initial_facts: str = dspy.InputField(desc="Faits initiaux")
    decision_context: str = dspy.InputField(desc="Décision de justice pour contexte")
    situation: str = dspy.OutputField(desc="Rapport de situation en français (résumé, objectifs, contraintes, questions juridiques)")


class GenerateInitialAnalysis(dspy.Signature):
    """Générer l'analyse juridique initiale. IMPORTANT: Répondre en français."""
    situation: str = dspy.InputField(desc="Rapport de situation")
    decision_legal_bases: str = dspy.InputField(desc="Bases légales de la décision")
    initial_analysis: str = dspy.OutputField(desc="Analyse initiale en français (domaine juridique, bases légales potentielles, évaluation préliminaire, besoins d'investigation, complexité)")


class GenerateInvestigationOrder(dspy.Signature):
    """Générer l'ordre d'investigation. IMPORTANT: Répondre en français."""
    initial_analysis: str = dspy.InputField(desc="Analyse juridique initiale")
    decision_facts: str = dspy.InputField(desc="Faits de la décision")
    investigation_order: str = dspy.OutputField(desc="Ordre d'investigation en français (objectif, questions, documents demandés)")


class GenerateInvestigationReport(dspy.Signature):
    """Générer le rapport d'investigation du client. IMPORTANT: Répondre en français."""
    investigation_order: str = dspy.InputField(desc="Ordre d'investigation")
    decision_facts: str = dspy.InputField(desc="Faits de la décision")
    investigation_report: str = dspy.OutputField(desc="Rapport d'investigation en français (réponses, documents obtenus, lacunes restantes)")


class GenerateFactualRecord(dspy.Signature):
    """Générer le dossier factuel. IMPORTANT: Répondre en français."""
    initial_facts: str = dspy.InputField(desc="Faits initiaux")
    investigation_report: str = dspy.InputField(desc="Rapport d'investigation")
    decision_facts: str = dspy.InputField(desc="Faits de la décision")
    factual_record: str = dspy.OutputField(desc="Dossier factuel en français (résumé, parties, chronologie, faits clés, preuves)")


class GenerateRecommendations(dspy.Signature):
    """Générer les recommandations au client. IMPORTANT: Répondre en français."""
    judgment: str = dspy.InputField(desc="Jugement du tribunal")
    considerations: str = dspy.InputField(desc="Legal considerations")
    client_objectives: str = dspy.InputField(desc="Objectifs du client")
    recommendations: str = dspy.OutputField(desc="Recommandations en français (action, justification, risques, alternatives, prochaines étapes)")


class SyntheticCaseGenerator(dspy.Module):
    """Generate synthetic case with ground truth from a court decision."""

    def __init__(self):
        super().__init__()
        self.gen_persona = dspy.ChainOfThought(GenerateClientPersona)
        self.gen_client_request = dspy.ChainOfThought(GenerateClientRequest)
        self.gen_initial_facts = dspy.ChainOfThought(GenerateInitialFacts)
        self.gen_situation = dspy.ChainOfThought(GenerateSituation)
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
        decision_context = f"{decision_data['parties']}\n\n{decision_data['facts_timeline']}\n\n{decision_data['judgment']}"

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

        print("  Generating situation report...")
        situation = self.gen_situation(
            client_persona=persona.client_persona,
            initial_facts=initial_facts.initial_facts,
            decision_context=decision_context
        )

        print("  Generating initial analysis...")
        initial_analysis = self.gen_initial_analysis(
            situation=situation.situation,
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
            client_objectives=situation.situation
        )

        return dspy.Prediction(
            client_persona=persona.client_persona,
            client_request=client_request.client_request,
            initial_facts=initial_facts.initial_facts,
            situation=situation.situation,
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


def generate_synthetic_case(decision_id: str, case_id: str, decisions_dir: Path, output_dir: Path):
    """
    Generate a synthetic case from a court decision.
    Saves files incrementally and skips steps that already exist.

    Args:
        decision_id: ID of the source court decision
        case_id: ID for the synthetic case
        decisions_dir: Directory containing court decisions
        output_dir: Directory to save synthetic case
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
    def save_if_missing(filename: str, title: str, content: str, generator_func=None):
        """Save file if it doesn't exist, otherwise load existing content."""
        filepath = case_dir / filename
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
        lambda: generator.gen_persona(decision_context=decision_context).client_persona
    )

    client_request = save_if_missing(
        "01b_client_request.md", "Client Request", "",
        lambda: generator.gen_client_request(
            client_persona=client_persona,
            decision_context=decision_context
        ).client_request
    )

    initial_facts = save_if_missing(
        "02_initial_facts_known.md", "Initial Facts", "",
        lambda: generator.gen_initial_facts(
            decision_context=decision_data['facts_timeline'],
            client_persona=client_persona
        ).initial_facts
    )

    situation = save_if_missing(
        "03_gt_situation.md", "Ground Truth: Situation", "",
        lambda: generator.gen_situation(
            client_persona=client_persona,
            initial_facts=initial_facts,
            decision_context=decision_context
        ).situation
    )

    initial_analysis = save_if_missing(
        "04_gt_initial_analysis.md", "Ground Truth: Initial Analysis", "",
        lambda: generator.gen_initial_analysis(
            situation=situation,
            decision_legal_bases=decision_data['legal_bases']
        ).initial_analysis
    )

    investigation_order = save_if_missing(
        "05_gt_investigation_order_1.md", "Ground Truth: Investigation Order", "",
        lambda: generator.gen_investigation_order(
            initial_analysis=initial_analysis,
            decision_facts=decision_data['facts_timeline']
        ).investigation_order
    )

    investigation_report = save_if_missing(
        "06_gt_investigation_report_1.md", "Ground Truth: Investigation Report", "",
        lambda: generator.gen_investigation_report(
            investigation_order=investigation_order,
            decision_facts=decision_data['facts_timeline']
        ).investigation_report
    )

    initial_factual_record = save_if_missing(
        "09_gt_initial_factual_record.md", "Ground Truth: Initial Factual Record", "",
        lambda: generator.gen_initial_factual_record(
            initial_facts=initial_facts,
            investigation_report="",
            decision_facts=decision_data['facts_timeline']
        ).factual_record
    )

    final_factual_record = save_if_missing(
        "10_gt_final_factual_record.md", "Ground Truth: Final Factual Record", "",
        lambda: generator.gen_final_factual_record(
            initial_facts=initial_facts,
            investigation_report=investigation_report,
            decision_facts=decision_data['facts_timeline']
        ).factual_record
    )

    # Copy decision data (these don't need generation)
    save_if_missing("11_gt_applicable_legal_bases.md", "Ground Truth: Legal Bases", decision_data['legal_bases'])
    save_if_missing("12_gt_legal_arguments.md", "Ground Truth: Legal Arguments", decision_data['arguments'])
    save_if_missing("13_gt_considerations.md", "Ground Truth: Considerations", decision_data['considerations'])
    save_if_missing("14_gt_judgment.md", "Ground Truth: Judgment", decision_data['judgment'])

    recommendations = save_if_missing(
        "15_gt_recommendations.md", "Ground Truth: Recommendations", "",
        lambda: generator.gen_recommendations(
            judgment=decision_data['judgment'],
            considerations=decision_data['considerations'],
            client_objectives=situation
        ).recommendations
    )

    print(f"✓ Synthetic case complete at {case_dir}")


def generate_all_synthetic_cases(decisions_dir: Path, output_dir: Path):
    """
    Generate synthetic cases from all extracted court decisions.

    Args:
        decisions_dir: Directory containing court decisions
        output_dir: Directory to save synthetic cases
    """
    from shared.io import list_decision_dirs

    decision_ids = list_decision_dirs(decisions_dir)

    if not decision_ids:
        print(f"No decisions found in {decisions_dir}")
        return

    print(f"Found {len(decision_ids)} decisions")

    for i, decision_id in enumerate(decision_ids, 1):
        case_id = f"case_{i:03d}"

        # Check if already generated
        case_dir = output_dir / case_id
        if case_dir.exists() and (case_dir / "metadata.md").exists():
            print(f"Skipping {case_id} (already generated)")
            continue

        try:
            generate_synthetic_case(decision_id, case_id, decisions_dir, output_dir)
        except Exception as e:
            print(f"✗ Error generating case from {decision_id}: {e}")
