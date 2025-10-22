#!/usr/bin/env python3
"""Run the full Lexic pipeline on synthetic case data."""

import argparse
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import dspy

from lexic.shared.config import Config
from lexic.shared.io import list_cases, load_case_step, write_markdown, get_case_path
from lexic.agents.pipeline import LexicPipeline


def run_pipeline_on_case(case_dir: Path, output_dir: Path) -> dict:
    """
    Run the full pipeline on a single case.

    Args:
        case_dir: Path to the case directory
        output_dir: Path to save pipeline outputs

    Returns:
        Dictionary with all pipeline outputs
    """
    # Load inputs
    _, client_persona = load_case_step(case_dir, "00a_client_persona.md")
    _, initial_facts = load_case_step(case_dir, "00b_initial_facts_known.md")

    print(f"Running pipeline for case: {case_dir.name}")
    print(f"  Client persona loaded: {len(client_persona)} chars")
    print(f"  Initial facts loaded: {len(initial_facts)} chars")
    print()

    # Initialize pipeline
    pipeline = LexicPipeline()

    # Run full pipeline (includes judgment prediction)
    print("Running full pipeline...")
    results = pipeline.run_full_pipeline(
        client_persona=client_persona,
        initial_facts=initial_facts
    )

    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().isoformat()
    metadata = {
        "case_id": case_dir.name,
        "run_at": timestamp,
        "model": Config.DEFAULT_MODEL,
    }

    # Map result keys to output filenames
    output_files = {
        "qualification": "pred_qualification.md",
        "initial_analysis": "pred_initial_analysis.md",
        "investigation_order": "pred_investigation_order.md",
        "investigation_report": "pred_investigation_report.md",
        "factual_record": "pred_factual_record.md",
        "legal_basis": "pred_legal_basis.md",
        "legal_arguments": "pred_legal_arguments.md",
        "considerations": "pred_considerations.md",
        "judgment": "pred_judgment.md",
        "recommendations": "pred_recommendations.md",
    }

    print("\nSaving outputs:")
    for key, filename in output_files.items():
        if key in results:
            output_path = output_dir / filename
            write_markdown(output_path, metadata, results[key])
            print(f"  ✓ {filename}")
        else:
            print(f"  ✗ {filename} (not generated)")

    print(f"\nOutputs saved to: {output_dir}")
    return results


def main():
    """Main pipeline workflow."""
    parser = argparse.ArgumentParser(
        description="Run the full Lexic pipeline on synthetic case data"
    )
    parser.add_argument(
        "--case",
        default=None,
        help="Specific case ID to run (e.g., case_001_pl). If not specified, runs on all cases."
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for results (default: data/pipeline_runs/<timestamp>)"
    )
    parser.add_argument(
        "--n-cases",
        type=int,
        default=None,
        help="Number of cases to run (default: all)"
    )

    args = parser.parse_args()

    # Validate config
    Config.validate()

    # Configure DSPy with Anthropic
    lm = dspy.LM(
        model=f"anthropic/{Config.DEFAULT_MODEL}",
        api_key=Config.ANTHROPIC_API_KEY,
        temperature=Config.TEMPERATURE,
        max_tokens=32000
    )
    dspy.configure(lm=lm)

    # Determine output directory
    if args.output_dir:
        output_base = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base = Config.DATA_DIR / "pipeline_runs" / timestamp

    print("=" * 60)
    print("Lexic Full Pipeline Runner")
    print("=" * 60)
    print(f"Model: {Config.DEFAULT_MODEL}")
    print(f"Output directory: {output_base}")
    print()

    # Get list of cases
    if args.case:
        # Run on specific case
        case_dir = Config.SYNTHETIC_CASES_DIR / args.case
        if not case_dir.exists():
            print(f"Error: Case directory not found: {case_dir}")
            return

        case_dirs = [case_dir]
    else:
        # Run on all cases
        case_dirs = list_cases(Config.SYNTHETIC_CASES_DIR)
        if args.n_cases:
            case_dirs = case_dirs[:args.n_cases]

    print(f"Running pipeline on {len(case_dirs)} case(s)\n")

    # Run pipeline on each case
    all_results = {}
    for case_dir in case_dirs:
        case_output_dir = output_base / case_dir.name
        try:
            results = run_pipeline_on_case(case_dir, case_output_dir)
            all_results[case_dir.name] = results
            print()
        except Exception as e:
            print(f"Error running pipeline on {case_dir.name}: {e}")
            import traceback
            traceback.print_exc()
            print()

    # Summary
    print("=" * 60)
    print(f"Completed {len(all_results)}/{len(case_dirs)} cases successfully")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
