#!/usr/bin/env python3
"""Run evaluations on a pipeline step."""

import sys
from pathlib import Path
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import dspy

from shared.config import Config
from evals.orchestrator import run_evaluation


def main():
    """Main evaluation workflow."""
    parser = argparse.ArgumentParser(description="Run evaluation for a pipeline step")
    parser.add_argument(
        "--step",
        required=True,
        choices=["qualification", "initial_analysis", "factual_record", "legal_arguments", "recommendations"],
        help="Pipeline step to evaluate"
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

    # Validate config
    Config.validate()

    # Configure DSPy with Anthropic for agents
    agent_lm = dspy.LM(
        model=f"anthropic/{Config.DEFAULT_MODEL}",
        api_key=Config.ANTHROPIC_API_KEY,
        temperature=Config.TEMPERATURE
    )
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
    print()

    # Run evaluation
    summary = run_evaluation(
        step_name=args.step,
        n_cases=args.n_cases,
        experiment_name=args.experiment
    )

    if summary:
        print(f"\nDimension scores:")
        for dim, score in summary['dimension_means'].items():
            print(f"  {dim}: {score:.2f}/5.00")


if __name__ == "__main__":
    main()
