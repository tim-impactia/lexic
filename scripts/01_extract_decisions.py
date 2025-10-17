#!/usr/bin/env python3
"""Extract court decision PDFs to structured markdown files."""
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import dspy
from anthropic import Anthropic

from lexic.shared.config import Config
from lexic.synthetic_data.extract import extract_all_decisions


def main():
    """Main extraction workflow."""
    # Validate config
    Config.validate()

    # Configure DSPy with Anthropic
    lm = dspy.LM(
        model=f"anthropic/{Config.EXTRACTION_MODEL}",
        api_key=Config.ANTHROPIC_API_KEY,
        temperature=Config.TEMPERATURE,
        max_tokens=32000  # Increase for generation
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
