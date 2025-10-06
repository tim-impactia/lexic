"""LLM-as-judge implementation for evaluating agent outputs."""

import dspy
from typing import Dict, List
from evals.judges.rubrics import Rubric, get_rubric
from shared.config import Config


class EvaluateDimension(dspy.Signature):
    """Évaluer une dimension de la sortie de l'agent. IMPORTANT: Répondre en français."""
    dimension_name: str = dspy.InputField(desc="Nom de la dimension évaluée")
    dimension_description: str = dspy.InputField(desc="Ce que cette dimension mesure")
    scoring_criteria: str = dspy.InputField(desc="Critères de notation pour cette dimension (échelle 1-5)")
    prediction: str = dspy.InputField(desc="Sortie de l'agent")
    ground_truth: str = dspy.InputField(desc="Référence vérité terrain")
    score: int = dspy.OutputField(desc="Score de 1 à 5")
    explanation: str = dspy.OutputField(desc="Explication en français pour le score")


class IdentifyCriticalErrors(dspy.Signature):
    """Identifier les erreurs critiques dans la sortie de l'agent. IMPORTANT: Répondre en français."""
    prediction: str = dspy.InputField(desc="Sortie de l'agent")
    ground_truth: str = dspy.InputField(desc="Référence vérité terrain")
    rubric: str = dspy.InputField(desc="Grille d'évaluation")
    critical_errors: List[str] = dspy.OutputField(desc="Liste des erreurs critiques trouvées en français (liste vide si aucune)")


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

            # Ensure score is in valid range
            score = max(1, min(5, result.score))
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
