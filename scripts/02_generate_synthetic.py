#!/usr/bin/env python3
"""Generate synthetic cases from extracted court decisions."""

import argparse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import dspy

from lexic.shared.config import Config
from lexic.synthetic_data.generate import generate_all_synthetic_cases, generate_synthetic_case
from lexic.shared.io import list_decision_dirs


def main():
    """Main synthetic data generation workflow."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic cases from court decisions",
        epilog="""
Available document numbers:
  01   - Client Persona
  01b  - Client Request
  02   - Initial Facts Known
  03   - Ground Truth: Qualification
  04   - Ground Truth: Initial Analysis
  05   - Ground Truth: Investigation Order
  06   - Ground Truth: Investigation Report
  09   - Ground Truth: Initial Factual Record
  10   - Ground Truth: Final Factual Record
  11   - Ground Truth: Applicable Legal Basis
  12   - Ground Truth: Legal Arguments
  13   - Ground Truth: Considerations
  14   - Ground Truth: Judgment
  15   - Ground Truth: Recommendations

Examples:
  # Generate all cases from all decisions
  python scripts/02_generate_synthetic.py

  # Generate both parties for a specific decision
  python scripts/02_generate_synthetic.py --decision decision_001

  # Generate only plaintiff case
  python scripts/02_generate_synthetic.py --decision decision_001 --party plaintiff

  # Generate only specific documents (e.g., client persona and initial facts)
  python scripts/02_generate_synthetic.py --decision decision_001 --docs 01 02

  # Regenerate recommendations for all existing cases
  python scripts/02_generate_synthetic.py --decision decision_001 --docs 15
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--decision",
        default=None,
        help="Specific decision ID to generate from (default: all decisions)"
    )
    parser.add_argument(
        "--party",
        default="both",
        choices=["plaintiff", "defendant", "both"],
        help="Which party perspective to generate (default: both)"
    )
    parser.add_argument(
        "--docs",
        nargs="+",
        default=None,
        metavar="DOC_NUM",
        help="Specific document numbers to generate (e.g., 01 02 03). If not specified, generates all documents."
    )

    args = parser.parse_args()

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

    if args.decision:
        print(f"Decision: {args.decision}")
        print(f"Party: {args.party}")
        if args.docs:
            print(f"Documents to generate: {', '.join(args.docs)}")
        else:
            print("Documents: all")
    else:
        print("Mode: Generate all cases from all decisions")
        if args.docs:
            print(f"Documents to generate: {', '.join(args.docs)}")
    print()

    # Generate synthetic cases
    if args.decision:
        # Generate specific case(s) from specific decision
        decision_id = args.decision

        # Get all decisions to find the index for case numbering
        decision_ids = list_decision_dirs(Config.COURT_DECISIONS_DIR)
        try:
            decision_index = decision_ids.index(decision_id) + 1
        except ValueError:
            print(f"✗ Error: Decision '{decision_id}' not found in {Config.COURT_DECISIONS_DIR}")
            return

        base_case_id = f"case_{decision_index:03d}"

        # Generate based on party selection
        if args.party in ["plaintiff", "both"]:
            case_id_pl = f"{base_case_id}_pl"
            print(f"{'='*60}")
            print(f"Generating plaintiff case: {case_id_pl}")
            print(f"{'='*60}")
            try:
                generate_synthetic_case(
                    decision_id, case_id_pl, "demandeur",
                    Config.COURT_DECISIONS_DIR, Config.SYNTHETIC_CASES_DIR,
                    specific_docs=args.docs
                )
            except Exception as e:
                print(f"✗ Error generating plaintiff case: {e}")
                import traceback
                traceback.print_exc()

        if args.party in ["defendant", "both"]:
            case_id_df = f"{base_case_id}_df"
            print(f"\n{'='*60}")
            print(f"Generating defendant case: {case_id_df}")
            print(f"{'='*60}")
            try:
                generate_synthetic_case(
                    decision_id, case_id_df, "défendeur",
                    Config.COURT_DECISIONS_DIR, Config.SYNTHETIC_CASES_DIR,
                    specific_docs=args.docs
                )
            except Exception as e:
                print(f"✗ Error generating defendant case: {e}")
                import traceback
                traceback.print_exc()
    else:
        # Generate all cases from all decisions
        generate_all_synthetic_cases(
            Config.COURT_DECISIONS_DIR,
            Config.SYNTHETIC_CASES_DIR
        )

    print("\n✓ Synthetic case generation complete!")


if __name__ == "__main__":
    main()
