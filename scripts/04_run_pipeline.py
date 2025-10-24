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


def save_step_output(output_dir: Path, case_id: str, step_name: str, content: str):
    """Save a single step output immediately."""
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().isoformat()
    metadata = {
        "case_id": case_id,
        "run_at": timestamp,
        "model": Config.DEFAULT_MODEL,
        "step": step_name,
    }

    # Map step names to output filenames with numbering matching ground truth
    filename_map = {
        "qualification": "02_pred_initial_qualification.md",
        "initial_analysis": "03_pred_initial_analysis.md",
        "investigation_order": "04_pred_initial_investigation_order.md",
        "investigation_report": "11_pred_final_investigation_report.md",
        "factual_record": "12_pred_final_factual_record.md",
        "legal_basis": "14_pred_final_legal_basis.md",
        "legal_arguments": "15_pred_final_legal_arguments.md",
        "considerations": "16_pred_considerations.md",
        "judgment": "17_pred_expected_judgment.md",
        "recommendations": "18_pred_recommendations.md",
    }

    filename = filename_map.get(step_name, f"pred_{step_name}.md")
    output_path = output_dir / filename
    write_markdown(output_path, metadata, content)
    print(f"      → Saved to {filename}")


def run_pipeline_on_case(case_dir: Path, output_dir: Path) -> dict:
    """
    Run the full pipeline on a single case, saving outputs after each step.

    Args:
        case_dir: Path to the case directory
        output_dir: Path to save pipeline outputs

    Returns:
        Dictionary with all pipeline outputs
    """
    # Load inputs
    _, client_persona = load_case_step(case_dir, "00a_client_persona.md")
    _, initial_facts = load_case_step(case_dir, "00b_initial_facts_known.md")
    _, client_request = load_case_step(case_dir, "01_client_request.md")

    print(f"Running pipeline for case: {case_dir.name}")
    print(f"  Client persona loaded: {len(client_persona)} chars")
    print(f"  Initial facts loaded: {len(initial_facts)} chars")
    print(f"  Client request loaded: {len(client_request)} chars")
    print()

    # Initialize pipeline
    pipeline = LexicPipeline()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run pipeline step by step, saving after each step
    print("Running pipeline with incremental saves...")
    all_results = {}

    # Phase 1: Intake to analysis
    print("  [1/4] Running intake & analysis phase...")
    phase1 = pipeline.run_intake_to_analysis(client_request)
    all_results.update(phase1)
    for key in ["qualification", "initial_analysis"]:
        if key in phase1:
            save_step_output(output_dir, case_dir.name, key, phase1[key])

    # Phase 2: Investigation
    print("  [2/4] Running investigation phase...")
    phase2 = pipeline.run_investigation_phase(
        phase1["initial_analysis"],
        client_persona,
        initial_facts
    )
    all_results.update(phase2)
    for key in ["investigation_order", "investigation_report", "factual_record"]:
        if key in phase2:
            save_step_output(output_dir, case_dir.name, key, phase2[key])

    # Phase 3: Legal analysis
    print("  [3/4] Running legal analysis phase...")
    phase3 = pipeline.run_legal_analysis(phase2["factual_record"])
    all_results.update(phase3)
    for key in ["legal_basis", "legal_arguments"]:
        if key in phase3:
            save_step_output(output_dir, case_dir.name, key, phase3[key])

    # Phase 4: Final phase
    print("  [4/4] Running final phase...")
    phase4 = pipeline.run_final_phase(
        phase3["legal_arguments"],
        phase2["factual_record"],
        phase1["qualification"],
        use_predicted_judgment=True
    )
    all_results.update(phase4)
    for key in ["considerations", "judgment", "recommendations"]:
        if key in phase4:
            save_step_output(output_dir, case_dir.name, key, phase4[key])

    print(f"\n✓ All outputs saved to: {output_dir}")
    return all_results


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
        case_ids = list_cases(Config.SYNTHETIC_CASES_DIR)
        if args.n_cases:
            case_ids = case_ids[:args.n_cases]
        # Convert case IDs to Path objects
        case_dirs = [Config.SYNTHETIC_CASES_DIR / case_id for case_id in case_ids]

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
