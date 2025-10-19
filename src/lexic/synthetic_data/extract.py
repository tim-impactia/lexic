"""Extract structured data from court decision documents (PDF/DOCX) using DSPy."""

import dspy
import json
from pathlib import Path
from typing import List, Dict
from docling.document_converter import DocumentConverter
from lexic.shared.models import CourtDecision, LegalBasis, LegalArgument, Consideration, Judgment
from lexic.shared.io import write_markdown
from lexic.shared.config import Config
from lexic.shared.prompts import create_signature

# Create signatures from YAML
CreateNameMapping = create_signature("extraction", "name_mapping")
ExtractAllElements = create_signature("extraction", "extract_all")


class CourtDecisionExtractor(dspy.Module):
    """Extract structured elements from a court decision document."""

    def __init__(self, decision_dir: Path = None):
        super().__init__()
        self.extract_all = dspy.ChainOfThought(ExtractAllElements)
        self.create_mapping = dspy.ChainOfThought(CreateNameMapping)
        self.decision_dir = decision_dir

    def forward(self, full_text: str):
        """Extract all elements from court decision text in a single LLM call."""
        result = self.extract_all(full_text=full_text)

        # Check if name mapping already exists
        mapping_path = self.decision_dir / "name_mapping.json" if self.decision_dir else None

        if mapping_path and mapping_path.exists():
            print("  Loading existing name mapping...")
            with open(mapping_path, 'r', encoding='utf-8') as f:
                mapping = json.load(f)
            print(f"  Name mapping loaded: {len(mapping)} entities")
            for anon, real in mapping.items():
                print(f"    {anon} → {real}")
        else:
            # Create a single name mapping for consistency across all documents
            print("  Creating name mapping...")
            # Include more context to help identify gender and company types
            context = f"Faits: {result.facts_timeline[:800]}\nJugement: {result.judgment[:800]}"

            mapping_result = self.create_mapping(
                parties=result.parties,
                context=context
            )

            # Parse the mapping into a dict
            mapping = {}
            for line in mapping_result.name_mapping.strip().split('\n'):
                if ':' in line:
                    anon, real = line.split(':', 1)
                    mapping[anon.strip()] = real.strip()

            print(f"  Name mapping created: {len(mapping)} entities")
            for anon, real in mapping.items():
                print(f"    {anon} → {real}")

            # Save the mapping
            if mapping_path:
                with open(mapping_path, 'w', encoding='utf-8') as f:
                    json.dump(mapping, f, ensure_ascii=False, indent=2)
                print(f"  Name mapping saved to {mapping_path.name}")

        # Apply mapping to all extracted elements
        print("  Applying name mapping to all documents...")

        def apply_mapping(text: str) -> str:
            """Apply name mapping to text."""
            result_text = text
            # Sort by length descending to replace longer patterns first
            # This prevents partial replacements (e.g., replacing "[...]" before "[...] Sàrl")
            for anon, real in sorted(mapping.items(), key=lambda x: len(x[0]), reverse=True):
                result_text = result_text.replace(anon, real)
            return result_text

        return dspy.Prediction(
            parties=apply_mapping(result.parties),
            facts_timeline=apply_mapping(result.facts_timeline),
            evidence=apply_mapping(result.evidence),
            legal_bases=apply_mapping(result.legal_bases),
            arguments=apply_mapping(result.arguments),
            considerations=apply_mapping(result.considerations),
            judgment=apply_mapping(result.judgment),
            name_mapping=mapping
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
        from lexic.shared.io import read_markdown
        _, full_text = read_markdown(full_text_path)
    else:
        print(f"Converting {doc_path} to markdown...")
        full_text = document_to_markdown(doc_path)

        # Save full text (no extraction_model since this is just Docling conversion)
        full_text_metadata = {
            "decision_id": decision_id,
            "source_file": str(doc_path),
        }
        write_markdown(full_text_path, full_text_metadata, full_text)

    # Check which documents already exist
    document_files = {
        "parties": decision_dir / "parties.md",
        "facts_timeline": decision_dir / "facts_timeline.md",
        "evidence": decision_dir / "evidence.md",
        "legal_bases": decision_dir / "legal_bases.md",
        "arguments": decision_dir / "arguments.md",
        "considerations": decision_dir / "considerations.md",
        "judgment": decision_dir / "judgment.md"
    }

    # Determine which documents need to be extracted
    missing_docs = {name: path for name, path in document_files.items() if not path.exists()}

    if not missing_docs:
        print(f"✓ All documents already exist for {decision_id}, skipping extraction")
        return

    print(f"Extracting {len(missing_docs)} missing document(s): {', '.join(missing_docs.keys())}")

    # Metadata for LLM-extracted elements (includes extraction_model)
    metadata = {
        "decision_id": decision_id,
        "source_file": str(doc_path),
        "extraction_model": Config.EXTRACTION_MODEL
    }

    # Initialize extractor and extract all elements
    print("Running extraction...")
    try:
        extractor = CourtDecisionExtractor(decision_dir=decision_dir)

        # Extract structured elements
        result = extractor(full_text=full_text)

        # Save only missing documents
        if "parties" in missing_docs:
            parties_lines = result.parties.strip().split('\n')
            parties_content = "\n".join(f"- {line.lstrip('- ')}" for line in parties_lines if line.strip())
            write_markdown(missing_docs["parties"], metadata, f"# Parties\n\n{parties_content}")

        if "facts_timeline" in missing_docs:
            facts_lines = result.facts_timeline.strip().split('\n')
            facts_content = "\n".join(f"- {line.lstrip('- ')}" for line in facts_lines if line.strip())
            write_markdown(missing_docs["facts_timeline"], metadata, f"# Facts Timeline\n\n{facts_content}")

        if "evidence" in missing_docs:
            write_markdown(missing_docs["evidence"], metadata, f"# Evidence\n\n{result.evidence}")

        if "legal_bases" in missing_docs:
            write_markdown(missing_docs["legal_bases"], metadata, f"# Legal Bases\n\n{result.legal_bases}")

        if "arguments" in missing_docs:
            write_markdown(missing_docs["arguments"], metadata, f"# Legal Arguments\n\n{result.arguments}")

        if "considerations" in missing_docs:
            write_markdown(missing_docs["considerations"], metadata, f"# Legal Considerations\n\n{result.considerations}")

        if "judgment" in missing_docs:
            write_markdown(missing_docs["judgment"], metadata, f"# Judgment\n\n{result.judgment}")

        print(f"✓ Extracted {len(missing_docs)} document(s) to {decision_dir}")
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
