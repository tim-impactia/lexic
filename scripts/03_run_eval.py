#!/usr/bin/env python3
"""Run evaluations on a pipeline step."""

import sys
from pathlib import Path
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import dspy

from shared.config import Config
from evals.orchestrator import run_evaluation


def main():
    """Main evaluation workflow."""
    parser = argparse.ArgumentParser(description="Run evaluation for a pipeline step")
    parser.add_argument(
        "--step",
        required=True,
        choices=["qualification", "initial_analysis", "factual_record", "legal_arguments", "recommendations", "all"],
        help="Pipeline step to evaluate (use 'all' to run all steps)"
    )
    parser.add_argument(
        "--n-cases",
        type=int,
        default=None,
        help="Number of cases to evaluate (default: all)"
    )
    parser.add_argument(
        "--experiment",
        default=None,
        help="MLFlow experiment name (default: from config)"
    )

    args = parser.parse_args()

    # Define all available steps
    all_steps = ["qualification", "initial_analysis", "factual_record", "legal_arguments", "recommendations"]

    # Determine which steps to run
    steps_to_run = all_steps if args.step == "all" else [args.step]

    # Validate config
    Config.validate()

    # Configure DSPy with Anthropic for agents
    agent_lm = dspy.LM(
        model=f"anthropic/{Config.DEFAULT_MODEL}",
        api_key=Config.ANTHROPIC_API_KEY,
        temperature=Config.TEMPERATURE,
        max_tokens=32000  # Increase for generation
    )

    # Enable MLflow autologging for DSPy
    import mlflow
    mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
    mlflow.dspy.autolog()

    dspy.configure(lm=agent_lm)

    # Configure separate LM for judge
    judge_lm = dspy.LM(
        model=f"anthropic/{Config.JUDGE_MODEL}",
        api_key=Config.ANTHROPIC_API_KEY,
        temperature=Config.JUDGE_TEMPERATURE
    )

    # Note: We'll use the agent_lm for both for now
    # In production, you might want to configure separate adapters

    print("=" * 60)
    print(f"Evaluation: {args.step}")
    print("=" * 60)
    print(f"Agent model: {Config.DEFAULT_MODEL}")
    print(f"Judge model: {Config.JUDGE_MODEL}")
    print(f"Cases directory: {Config.SYNTHETIC_CASES_DIR}")
    if args.n_cases:
        print(f"Number of cases: {args.n_cases}")
    if args.step == "all":
        print(f"Steps to evaluate: {', '.join(steps_to_run)}")
    print()

    # Run evaluation for each step
    all_summaries = {}
    for step_name in steps_to_run:
        if args.step == "all":
            print(f"\n{'='*60}")
            print(f"Running evaluation for: {step_name}")
            print(f"{'='*60}\n")

        summary = run_evaluation(
            step_name=step_name,
            n_cases=args.n_cases,
            experiment_name=args.experiment
        )

        all_summaries[step_name] = summary

        if summary:
            print(f"\nDimension scores for {step_name}:")
            for dim, score in summary['dimension_means'].items():
                print(f"  {dim}: {score:.2f}/5.00")

    # Print overall summary if running all steps
    if args.step == "all":
        print(f"\n{'='*60}")
        print("OVERALL SUMMARY")
        print(f"{'='*60}\n")
        for step_name, summary in all_summaries.items():
            if summary:
                print(f"{step_name}: {summary['mean_overall_score']:.2f}/5.00")
        print()


if __name__ == "__main__":
    main()
