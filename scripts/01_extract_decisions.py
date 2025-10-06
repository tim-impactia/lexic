#!/usr/bin/env python3
"""Extract court decision PDFs to structured markdown files."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import dspy
from anthropic import Anthropic

from shared.config import Config
from synthetic_data.extract import extract_all_decisions


def main():
    """Main extraction workflow."""
    # Validate config
    Config.validate()

    # Configure DSPy with Anthropic
    lm = dspy.LM(
        model=f"anthropic/{Config.EXTRACTION_MODEL}",
        api_key=Config.ANTHROPIC_API_KEY,
        temperature=Config.TEMPERATURE
    )

    # Enable MLflow autologging for DSPy
    import mlflow
    mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
    mlflow.dspy.autolog()

    dspy.configure(lm=lm)

    print("=" * 60)
    print("Court Decision Extraction")
    print("=" * 60)
    print(f"Model: {Config.EXTRACTION_MODEL}")
    print(f"Input directory: {Config.COURT_DECISIONS_DIR}")
    print()

    # Extract all decisions
    extract_all_decisions(Config.COURT_DECISIONS_DIR)

    print("\n✓ Extraction complete!")


if __name__ == "__main__":
    main()
