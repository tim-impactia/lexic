"""Evaluation orchestrator with MLFlow tracking."""

import mlflow
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import importlib

from lexic.shared.config import Config
from lexic.shared.io import list_cases, load_case_step, write_markdown, get_case_path
from lexic.evals.judges.judge import evaluate_output


# Map step names to agent modules and functions
AGENT_REGISTRY = {
    "qualification": ("lexic.agents.qualification", "run_qualification"),
    "initial_analysis": ("lexic.agents.initial_analysis", "InitialAnalysisAgent"),
    "factual_record": ("lexic.agents.factual_record", "FactualRecordAgent"),
    "legal_arguments": ("lexic.agents.arguments", "ArgumentationAgent"),
    "recommendations": ("lexic.agents.recommendations", "RecommendationAgent"),
}

# Map step names to input step files
STEP_INPUTS = {
    "qualification": ["01_client_request.md"],
    "initial_analysis": ["02_gt_initial_qualification.md"],
    "factual_record": ["00b_initial_facts_known.md", "11_gt_final_investigation_report.md"],
    "legal_arguments": ["12_gt_final_factual_record.md", "14_gt_final_legal_basis.md"],
    "recommendations": ["16_gt_considerations.md", "17_gt_expected_judgment.md", "02_gt_initial_qualification.md"],
}

# Map step names to ground truth files
STEP_GROUND_TRUTH = {
    "qualification": "02_gt_initial_qualification.md",
    "initial_analysis": "03_gt_initial_analysis.md",
    "factual_record": "12_gt_final_factual_record.md",
    "legal_arguments": "15_gt_final_legal_arguments.md",
    "recommendations": "18_gt_recommendations.md",
}


def get_agent_runner(step_name: str):
    """
    Get agent runner for a pipeline step.

    Args:
        step_name: Name of the pipeline step

    Returns:
        Callable that runs the agent
    """
    if step_name not in AGENT_REGISTRY:
        raise ValueError(f"Unknown step: {step_name}")

    module_name, class_or_func_name = AGENT_REGISTRY[step_name]
    module = importlib.import_module(module_name)
    agent_class_or_func = getattr(module, class_or_func_name)

    # Handle both function-based and class-based agents
    if step_name == "qualification":
        # Function-based
        return agent_class_or_func
    else:
        # Class-based - need to instantiate and call
        def runner(**kwargs):
            agent = agent_class_or_func()
            return agent(**kwargs)
        return runner


def load_step_inputs(case_dir: Path, step_name: str) -> Dict[str, str]:
    """
    Load input data for a pipeline step.

    Args:
        case_dir: Path to case directory
        step_name: Name of the pipeline step

    Returns:
        Dict mapping input names to content
    """
    if step_name not in STEP_INPUTS:
        raise ValueError(f"Unknown step: {step_name}")

    inputs = {}
    input_files = STEP_INPUTS[step_name]

    for input_file in input_files:
        _, content = load_case_step(case_dir, input_file)
        # Use simplified key names
        if "client_request" in input_file:
            inputs["client_request"] = content
        elif "persona" in input_file:
            inputs["client_persona"] = content
        elif "initial_facts" in input_file:
            inputs["initial_facts"] = content
        elif "qualification" in input_file:
            if step_name == "recommendations":
                inputs["client_objectives"] = content
            else:
                inputs["qualification"] = content
        elif "investigation_report" in input_file:
            inputs["investigation_report"] = content
        elif "factual_record" in input_file:
            inputs["factual_record"] = content
        elif "legal_basis" in input_file:
            inputs["legal_basis"] = content
        elif "considerations" in input_file:
            inputs["considerations"] = content
        elif "judgment" in input_file:
            inputs["judgment"] = content

    return inputs


def run_agent_on_case(step_name: str, case_dir: Path) -> str:
    """
    Run agent on a single case.

    Args:
        step_name: Name of the pipeline step
        case_dir: Path to case directory

    Returns:
        Agent prediction
    """
    agent_runner = get_agent_runner(step_name)
    inputs = load_step_inputs(case_dir, step_name)
    prediction = agent_runner(**inputs)
    return prediction


def evaluate_case(
    step_name: str,
    case_id: str,
    case_dir: Path,
    output_dir: Path
) -> Dict:
    """
    Evaluate agent on a single case.

    Args:
        step_name: Name of the pipeline step
        case_id: Case ID
        case_dir: Path to case directory
        output_dir: Path to save evaluation results

    Returns:
        Evaluation results dict
    """
    print(f"  Running agent on {case_id}...")

    # Load inputs
    inputs = load_step_inputs(case_dir, step_name)

    # Run agent
    agent_runner = get_agent_runner(step_name)
    prediction = agent_runner(**inputs)

    # Load ground truth
    gt_file = STEP_GROUND_TRUTH[step_name]
    _, ground_truth = load_case_step(case_dir, gt_file)

    # Save inputs
    inputs_file = output_dir / f"{case_id}_inputs.md"
    inputs_content = "\n\n".join([f"## {key}\n\n{value}" for key, value in inputs.items()])
    write_markdown(
        inputs_file,
        {"case_id": case_id, "step": step_name},
        f"# Inputs\n\n{inputs_content}"
    )

    # Save prediction
    pred_file = output_dir / f"{case_id}_prediction.md"
    write_markdown(
        pred_file,
        {"case_id": case_id, "step": step_name},
        f"# Prediction\n\n{prediction}"
    )

    # Save ground truth
    gt_file_saved = output_dir / f"{case_id}_ground_truth.md"
    write_markdown(
        gt_file_saved,
        {"case_id": case_id, "step": step_name},
        f"# Ground Truth\n\n{ground_truth}"
    )

    print(f"  Evaluating {case_id}...")
    eval_result = evaluate_output(step_name, prediction, ground_truth)

    # Save evaluation
    eval_content = format_evaluation(eval_result)
    eval_file = output_dir / f"{case_id}_evaluation.md"
    write_markdown(
        eval_file,
        {"case_id": case_id, "step": step_name, "overall_score": eval_result["overall_score"]},
        eval_content
    )

    return {
        "case_id": case_id,
        "inputs": inputs,
        "prediction": prediction,
        "ground_truth": ground_truth,
        **eval_result
    }


def format_evaluation(eval_result: Dict) -> str:
    """Format evaluation result as markdown."""
    lines = ["# Evaluation Result\n"]

    lines.append(f"**Overall Score**: {eval_result['overall_score']:.2f}/5.00\n")

    lines.append("## Dimension Scores\n")
    for dim_name, score in eval_result['scores'].items():
        explanation = eval_result['explanations'][dim_name]
        lines.append(f"### {dim_name}: {score}/5")
        lines.append(f"{explanation}\n")

    if eval_result['critical_errors']:
        lines.append("## Critical Errors\n")
        for error in eval_result['critical_errors']:
            lines.append(f"- {error}")
    else:
        lines.append("## Critical Errors\n\nNone identified.")

    return "\n".join(lines)


def run_evaluation(
    step_name: str,
    cases_dir: Optional[Path] = None,
    n_cases: Optional[int] = None,
    experiment_name: Optional[str] = None
) -> Dict:
    """
    Run evaluation for a pipeline step on synthetic cases.

    Args:
        step_name: Name of the pipeline step
        cases_dir: Directory containing synthetic cases (default: from config)
        n_cases: Number of cases to evaluate (default: all)
        experiment_name: MLFlow experiment name (default: from config)

    Returns:
        Summary statistics
    """
    # Set defaults
    if cases_dir is None:
        cases_dir = Config.SYNTHETIC_CASES_DIR
    if experiment_name is None:
        experiment_name = Config.MLFLOW_EXPERIMENT_NAME

    # List cases
    case_ids = list_cases(cases_dir)
    if n_cases:
        case_ids = case_ids[:n_cases]

    if not case_ids:
        print(f"No cases found in {cases_dir}")
        return {}

    print(f"Evaluating {len(case_ids)} cases for step: {step_name}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Config.EVAL_RUNS_DIR / f"{step_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up MLFlow
    mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"{step_name}_{timestamp}"):
        # Log parameters
        mlflow.log_param("step", step_name)
        mlflow.log_param("n_cases", len(case_ids))
        mlflow.log_param("model", Config.DEFAULT_MODEL)
        mlflow.log_param("judge_model", Config.JUDGE_MODEL)

        # Evaluate each case
        results = []
        for case_id in case_ids:
            case_dir = get_case_path(cases_dir, case_id)
            try:
                result = evaluate_case(step_name, case_id, case_dir, output_dir)
                results.append(result)
            except Exception as e:
                print(f"  ✗ Error evaluating {case_id}: {e}")

        # Compute summary statistics
        if results:
            overall_scores = [r["overall_score"] for r in results]
            dimension_scores = {}
            for dim_name in results[0]["scores"].keys():
                dimension_scores[dim_name] = [r["scores"][dim_name] for r in results]

            summary = {
                "mean_overall_score": sum(overall_scores) / len(overall_scores),
                "min_overall_score": min(overall_scores),
                "max_overall_score": max(overall_scores),
                "dimension_means": {
                    dim: sum(scores) / len(scores)
                    for dim, scores in dimension_scores.items()
                },
                "n_cases_with_errors": sum(1 for r in results if r["critical_errors"]),
                "total_errors": sum(len(r["critical_errors"]) for r in results)
            }

            # Log metrics to MLFlow
            mlflow.log_metric("mean_overall_score", summary["mean_overall_score"])
            mlflow.log_metric("min_overall_score", summary["min_overall_score"])
            mlflow.log_metric("max_overall_score", summary["max_overall_score"])
            for dim, mean_score in summary["dimension_means"].items():
                mlflow.log_metric(f"mean_{dim}", mean_score)
            mlflow.log_metric("cases_with_errors", summary["n_cases_with_errors"])
            mlflow.log_metric("total_errors", summary["total_errors"])

            # Save summary
            summary_content = format_summary(step_name, summary, results)
            summary_file = output_dir / "summary.md"
            write_markdown(
                summary_file,
                {"step": step_name, "n_cases": len(case_ids), "timestamp": timestamp},
                summary_content
            )

            # Log artifacts
            mlflow.log_artifacts(output_dir, artifact_path="evaluation_results")

            print(f"\n✓ Evaluation complete!")
            print(f"  Mean overall score: {summary['mean_overall_score']:.2f}/5.00")
            print(f"  Results saved to: {output_dir}")
            print(f"  MLFlow run: {mlflow.active_run().info.run_id}")

            return summary
        else:
            print("No successful evaluations")
            return {}


def format_summary(step_name: str, summary: Dict, results: List[Dict]) -> str:
    """Format summary statistics as markdown."""
    lines = [f"# Evaluation Summary: {step_name}\n"]

    lines.append("## Overall Statistics\n")
    lines.append(f"- **Mean Score**: {summary['mean_overall_score']:.2f}/5.00")
    lines.append(f"- **Min Score**: {summary['min_overall_score']:.2f}/5.00")
    lines.append(f"- **Max Score**: {summary['max_overall_score']:.2f}/5.00")
    lines.append(f"- **Cases with Errors**: {summary['n_cases_with_errors']}/{len(results)}")
    lines.append(f"- **Total Errors**: {summary['total_errors']}\n")

    lines.append("## Dimension Means\n")
    for dim, mean in summary['dimension_means'].items():
        lines.append(f"- **{dim}**: {mean:.2f}/5.00")

    lines.append("\n## Per-Case Scores\n")
    lines.append("| Case ID | Overall Score | Errors |")
    lines.append("|---------|--------------|--------|")
    for result in results:
        errors = len(result['critical_errors'])
        lines.append(f"| {result['case_id']} | {result['overall_score']:.2f} | {errors} |")

    return "\n".join(lines)
