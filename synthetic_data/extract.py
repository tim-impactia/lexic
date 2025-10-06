"""Extract structured data from court decision documents (PDF/DOCX) using DSPy."""

import dspy
from pathlib import Path
from typing import List
from docling.document_converter import DocumentConverter
from shared.models import CourtDecision, LegalBasis, LegalArgument, Consideration, Judgment
from shared.io import write_markdown
from shared.config import Config


class ExtractAllElements(dspy.Signature):
    """Extract all structured elements from court decision in a single pass."""
    full_text: str = dspy.InputField(desc="Full text of court decision")
    parties: str = dspy.OutputField(desc="List of parties involved (plaintiff, defendant, etc.) - one per line")
    facts_timeline: str = dspy.OutputField(desc="Chronological list of key facts and events - one per line")
    legal_bases: str = dspy.OutputField(desc="Legal provisions cited (article, law, content, relevance) as structured markdown")
    arguments: str = dspy.OutputField(desc="Legal arguments presented (thesis, legal bases, factual support, reasoning) as structured markdown")
    considerations: str = dspy.OutputField(desc="Court's legal considerations (issue, analysis, conclusion, confidence) as structured markdown")
    judgment: str = dspy.OutputField(desc="Final judgment (decision, reasoning, legal bases) as structured markdown")


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

        # Check if already extracted
        decision_dir = decisions_dir / decision_id
        if decision_dir.exists() and (decision_dir / "full_text.md").exists():
            print(f"Skipping {decision_id} (already extracted)")
            continue

        try:
            extract_decision(doc_path, decisions_dir, decision_id)
        except Exception as e:
            print(f"✗ Error extracting {doc_path}: {e}")
