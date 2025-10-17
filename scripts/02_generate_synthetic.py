#!/usr/bin/env python3
"""Generate synthetic cases from extracted court decisions."""

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import dspy

from lexic.shared.config import Config
from lexic.synthetic_data.generate import generate_all_synthetic_cases


def main():
    """Main synthetic data generation workflow."""
    # Validate config
    Config.validate()

    # Configure DSPy with Anthropic
    lm = dspy.LM(
        model=f"anthropic/{Config.GENERATION_MODEL}",
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
    print("Synthetic Case Generation")
    print("=" * 60)
    print(f"Model: {Config.GENERATION_MODEL}")
    print(f"Input directory: {Config.COURT_DECISIONS_DIR}")
    print(f"Output directory: {Config.SYNTHETIC_CASES_DIR}")
    print()

    # Generate synthetic cases
    generate_all_synthetic_cases(
        Config.COURT_DECISIONS_DIR,
        Config.SYNTHETIC_CASES_DIR
    )

    print("\n✓ Synthetic case generation complete!")


if __name__ == "__main__":
    main()
