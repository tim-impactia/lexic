"""Extract structured data from court decision documents (PDF/DOCX) using DSPy."""

import dspy
from pathlib import Path
from typing import List
from docling.document_converter import DocumentConverter
from shared.models import CourtDecision, LegalBasis, LegalArgument, Consideration, Judgment
from shared.io import write_markdown
from shared.config import Config


class ExtractAllElements(dspy.Signature):
    """
    Extraire tous les éléments structurés de la décision judiciaire en un seul passage.
    IMPORTANT: Répondre entièrement en français.

    Si c'est une décision d'appel: extraire les faits et arguments du litige ORIGINAL (première instance),
    pas ceux spécifiques à l'appel. Remonter au conflit initial avant tout jugement.
    """
    full_text: str = dspy.InputField(desc="Texte intégral de la décision judiciaire")
    parties: str = dspy.OutputField(desc="Liste des parties impliquées dans le litige ORIGINAL (demandeur, défendeur, etc.) - une par ligne. Si appel, identifier les parties du litige initial, pas leur rôle d'appelant/intimé. Répondre en français.")
    facts_timeline: str = dspy.OutputField(desc="Liste chronologique des faits du LITIGE ORIGINAL avant tout jugement - un par ligne. Si appel, exclure les faits post-jugement première instance (l'appel lui-même, etc.). Répondre en français.")
    legal_bases: str = dspy.OutputField(desc="Dispositions légales applicables au litige original (article, loi, contenu, pertinence) en markdown structuré. Si appel, inclure ce qui aurait été pertinent en première instance. Répondre en français.")
    arguments: str = dspy.OutputField(desc="Arguments juridiques du litige original (thèse, bases légales, support factuel, raisonnement) en markdown structuré. Si appel, reconstruire les arguments de première instance, pas uniquement ceux d'appel. Répondre en français.")
    considerations: str = dspy.OutputField(desc="Considérations juridiques finales du tribunal (question, analyse, conclusion, confiance) en markdown structuré. Peut inclure insights d'appel sur ce qui aurait dû être considéré. Répondre en français.")
    judgment: str = dspy.OutputField(desc="Jugement final (décision finale après appel si applicable, raisonnement, bases légales) en markdown structuré. Répondre en français.")


class CourtDecisionExtractor(dspy.Module):
    """Extract structured elements from a court decision document."""

    def __init__(self):
        super().__init__()
        self.extract_all = dspy.ChainOfThought(ExtractAllElements)

    def forward(self, full_text: str):
        """Extract all elements from court decision text in a single LLM call."""
        result = self.extract_all(full_text=full_text)

        return dspy.Prediction(
            parties=result.parties,
            facts_timeline=result.facts_timeline,
            legal_bases=result.legal_bases,
            arguments=result.arguments,
            considerations=result.considerations,
            judgment=result.judgment
        )


def document_to_markdown(doc_path: Path) -> str:
    """
    Convert document (PDF or DOCX) to markdown using Docling.

    Args:
        doc_path: Path to document file (PDF or DOCX)

    Returns:
        Markdown text
    """
    converter = DocumentConverter()
    result = converter.convert(doc_path)
    return result.document.export_to_markdown()


def extract_decision(doc_path: Path, output_dir: Path, decision_id: str):
    """
    Extract a court decision document to structured markdown files.

    Args:
        doc_path: Path to the court decision document (PDF or DOCX)
        output_dir: Directory to save extracted files
        decision_id: Unique ID for this decision
    """
    # Create output directory
    decision_dir = output_dir / decision_id
    decision_dir.mkdir(parents=True, exist_ok=True)

    full_text_path = decision_dir / "full_text.md"

    # Convert document to markdown (skip if already exists)
    if full_text_path.exists():
        print(f"Loading existing full_text.md for {decision_id}...")
        from shared.io import read_markdown
        metadata, full_text = read_markdown(full_text_path)
    else:
        print(f"Converting {doc_path} to markdown...")
        full_text = document_to_markdown(doc_path)

        # Save full text
        metadata = {
            "decision_id": decision_id,
            "source_file": str(doc_path),
            "extraction_model": Config.EXTRACTION_MODEL
        }
        write_markdown(full_text_path, metadata, full_text)

    # Initialize extractor
    print("Extracting structured elements...")
    try:
        extractor = CourtDecisionExtractor()

        # Extract structured elements
        result = extractor(full_text=full_text)

        # Save parties
        parties_lines = result.parties.strip().split('\n')
        parties_content = "\n".join(f"- {line.lstrip('- ')}" for line in parties_lines if line.strip())
        write_markdown(decision_dir / "parties.md", metadata, f"# Parties\n\n{parties_content}")

        # Save facts timeline
        facts_lines = result.facts_timeline.strip().split('\n')
        facts_content = "\n".join(f"- {line.lstrip('- ')}" for line in facts_lines if line.strip())
        write_markdown(decision_dir / "facts_timeline.md", metadata, f"# Facts Timeline\n\n{facts_content}")

        # Save legal bases
        write_markdown(decision_dir / "legal_bases.md", metadata, f"# Legal Bases\n\n{result.legal_bases}")

        # Save arguments
        write_markdown(decision_dir / "arguments.md", metadata, f"# Legal Arguments\n\n{result.arguments}")

        # Save considerations
        write_markdown(decision_dir / "considerations.md", metadata, f"# Legal Considerations\n\n{result.considerations}")

        # Save judgment
        write_markdown(decision_dir / "judgment.md", metadata, f"# Judgment\n\n{result.judgment}")

        print(f"✓ Extracted decision to {decision_dir}")
    except Exception as e:
        print(f"✗ Error during structured extraction: {e}")
        print(f"  Full text was saved to {decision_dir / 'full_text.md'}")
        raise


def extract_all_decisions(decisions_dir: Path):
    """
    Extract all court decision documents (PDF and DOCX) in the decisions directory.

    Args:
        decisions_dir: Path to court_decisions directory
    """
    # Collect both PDF and DOCX files
    pdf_files = list(decisions_dir.glob("*.pdf"))
    docx_files = list(decisions_dir.glob("*.docx"))
    doc_files = sorted(pdf_files + docx_files)

    if not doc_files:
        print(f"No PDF or DOCX files found in {decisions_dir}")
        return

    print(f"Found {len(doc_files)} document(s) to extract ({len(pdf_files)} PDF, {len(docx_files)} DOCX)")

    for doc_path in doc_files:
        # Generate decision ID from filename
        decision_id = doc_path.stem

        try:
            extract_decision(doc_path, decisions_dir, decision_id)
        except Exception as e:
            print(f"✗ Error extracting {doc_path}: {e}")
