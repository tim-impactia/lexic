"""LLM-as-judge implementation for evaluating agent outputs."""

import dspy
from typing import Dict, List
from lexic.evals.judges.rubrics import Rubric, get_rubric
from lexic.shared.config import Config
from lexic.shared.prompts import create_signature

# Create judge signatures from YAML
EvaluateDimension = create_signature("evals", "evaluate_dimension")
IdentifyCriticalErrors = create_signature("evals", "identify_critical_errors")


class LexicJudge(dspy.Module):
    """
    LLM-as-judge for evaluating Lexic agent outputs.

    Evaluates an agent's output against ground truth using a rubric,
    scoring each dimension and identifying critical errors.
    """

    def __init__(self, rubric: Rubric):
        """
        Initialize judge with a rubric.

        Args:
            rubric: Evaluation rubric for this pipeline step
        """
        super().__init__()
        self.rubric = rubric
        self.evaluate_dimension = dspy.ChainOfThought(EvaluateDimension)
        self.identify_errors = dspy.ChainOfThought(IdentifyCriticalErrors)

    def forward(self, prediction: str, ground_truth: str) -> Dict:
        """
        Evaluate a prediction against ground truth.

        Args:
            prediction: Agent's output
            ground_truth: Ground truth reference

        Returns:
            Dict with:
                - scores: Dict mapping dimension name to score
                - explanations: Dict mapping dimension name to explanation
                - critical_errors: List of critical errors
                - overall_score: Weighted overall score
        """
        scores = {}
        explanations = {}

        # Evaluate each dimension
        for dim in self.rubric.dimensions:
            # Format criteria for the dimension
            criteria_text = "\n".join(
                f"Score {score}: {description}"
                for score, description in sorted(dim.criteria.items())
            )

            result = self.evaluate_dimension(
                dimension_name=dim.name,
                dimension_description=dim.description,
                scoring_criteria=criteria_text,
                prediction=prediction,
                ground_truth=ground_truth
            )

            # Convert score to int and ensure it's in valid range
            try:
                score = int(result.score)
            except (ValueError, TypeError):
                score = 3  # Default to middle score if conversion fails
            score = max(1, min(5, score))
            scores[dim.name] = score
            explanations[dim.name] = result.explanation

        # Identify critical errors
        error_result = self.identify_errors(
            prediction=prediction,
            ground_truth=ground_truth,
            rubric=self.rubric.to_text()
        )

        critical_errors = error_result.critical_errors if error_result.critical_errors else []

        # Compute weighted overall score
        overall_score = sum(
            scores[dim.name] * dim.weight
            for dim in self.rubric.dimensions
        )

        return dspy.Prediction(
            scores=scores,
            explanations=explanations,
            critical_errors=critical_errors,
            overall_score=overall_score
        )


def evaluate_output(
    step_name: str,
    prediction: str,
    ground_truth: str
) -> Dict:
    """
    Evaluate an agent output for a specific pipeline step.

    Args:
        step_name: Name of the pipeline step
        prediction: Agent's output
        ground_truth: Ground truth reference

    Returns:
        Evaluation results dict
    """
    rubric = get_rubric(step_name)
    judge = LexicJudge(rubric)
    result = judge(prediction=prediction, ground_truth=ground_truth)

    return {
        "scores": result.scores,
        "explanations": result.explanations,
        "critical_errors": result.critical_errors,
        "overall_score": result.overall_score
    }
