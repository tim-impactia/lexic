"""Evaluation rubrics for each pipeline step."""

from typing import Dict, List


class EvaluationDimension:
    """Single evaluation dimension with scoring criteria."""

    def __init__(self, name: str, description: str, weight: float, criteria: Dict[int, str]):
        """
        Initialize evaluation dimension.

        Args:
            name: Dimension name
            description: What this dimension measures
            weight: Weight in overall score (0-1)
            criteria: Dict mapping score (1-5) to description
        """
        self.name = name
        self.description = description
        self.weight = weight
        self.criteria = criteria


class Rubric:
    """Evaluation rubric with multiple dimensions."""

    def __init__(self, step_name: str, dimensions: List[EvaluationDimension]):
        """
        Initialize rubric.

        Args:
            step_name: Name of the pipeline step
            dimensions: List of evaluation dimensions
        """
        self.step_name = step_name
        self.dimensions = dimensions

    def to_text(self) -> str:
        """Convert rubric to text format for LLM."""
        lines = [f"# Evaluation Rubric: {self.step_name}\n"]

        for dim in self.dimensions:
            lines.append(f"## {dim.name} (weight: {dim.weight})")
            lines.append(f"{dim.description}\n")
            lines.append("Scoring criteria:")
            for score in sorted(dim.criteria.keys()):
                lines.append(f"- **{score}**: {dim.criteria[score]}")
            lines.append("")

        return "\n".join(lines)


# Qualification rubric
QUALIFICATION_RUBRIC = Rubric(
    step_name="qualification",
    dimensions=[
        EvaluationDimension(
            name="Completeness",
            description="Does the situation report capture all relevant aspects?",
            weight=0.3,
            criteria={
                1: "Missing multiple critical elements (summary, objectives, constraints, or questions)",
                2: "Missing one critical element or several important details",
                3: "All critical elements present but some details lacking",
                4: "Comprehensive with only minor omissions",
                5: "Fully comprehensive, all relevant aspects captured"
            }
        ),
        EvaluationDimension(
            name="Accuracy",
            description="Is the situation report accurate given the client persona and initial facts?",
            weight=0.25,
            criteria={
                1: "Major inaccuracies or misinterpretations",
                2: "Several minor inaccuracies",
                3: "Mostly accurate with few minor errors",
                4: "Accurate with negligible errors",
                5: "Fully accurate, no misinterpretations"
            }
        ),
        EvaluationDimension(
            name="Relevance",
            description="Are irrelevant details excluded?",
            weight=0.2,
            criteria={
                1: "Significant irrelevant content included",
                2: "Some irrelevant content",
                3: "Mostly relevant with minor tangents",
                4: "Highly relevant with minimal extraneous content",
                5: "Perfectly focused, no irrelevant content"
            }
        ),
        EvaluationDimension(
            name="Legal_Insight",
            description="Are the legal questions well-formulated and insightful?",
            weight=0.15,
            criteria={
                1: "Legal questions are vague, poorly formulated, or missing",
                2: "Legal questions are superficial or miss key issues",
                3: "Legal questions are adequate but could be more precise",
                4: "Legal questions are well-formulated and insightful",
                5: "Legal questions are exceptionally precise and demonstrate deep legal insight"
            }
        ),
        EvaluationDimension(
            name="Clarity",
            description="Is the situation report well-structured and clear?",
            weight=0.1,
            criteria={
                1: "Poorly structured, difficult to understand",
                2: "Some structural issues, somewhat unclear",
                3: "Adequately structured and clear",
                4: "Well-structured and clear",
                5: "Exceptionally well-structured and crystal clear"
            }
        )
    ]
)


# Initial analysis rubric
INITIAL_ANALYSIS_RUBRIC = Rubric(
    step_name="initial_analysis",
    dimensions=[
        EvaluationDimension(
            name="Legal_Domain_Accuracy",
            description="Is the legal domain correctly identified?",
            weight=0.25,
            criteria={
                1: "Incorrect legal domain",
                2: "Partially correct but imprecise",
                3: "Correct but could be more specific",
                4: "Accurate and appropriately specific",
                5: "Perfectly accurate and precise"
            }
        ),
        EvaluationDimension(
            name="Legal_Bases_Identification",
            description="Are potential legal bases correctly identified?",
            weight=0.3,
            criteria={
                1: "Missing most relevant legal bases",
                2: "Missing several important legal bases",
                3: "Most relevant legal bases identified",
                4: "All important legal bases identified",
                5: "Comprehensive and precise identification of all relevant legal bases"
            }
        ),
        EvaluationDimension(
            name="Assessment_Quality",
            description="Is the preliminary assessment sound and helpful?",
            weight=0.25,
            criteria={
                1: "Assessment is unsound or unhelpful",
                2: "Assessment has significant weaknesses",
                3: "Assessment is adequate",
                4: "Assessment is sound and helpful",
                5: "Assessment is exceptionally insightful and actionable"
            }
        ),
        EvaluationDimension(
            name="Investigation_Needs",
            description="Are investigation needs correctly identified?",
            weight=0.15,
            criteria={
                1: "Investigation needs are poorly identified",
                2: "Missing important investigation needs",
                3: "Adequate identification of investigation needs",
                4: "Thorough identification of investigation needs",
                5: "Comprehensive and strategic identification of all investigation needs"
            }
        ),
        EvaluationDimension(
            name="Complexity_Assessment",
            description="Is the complexity correctly assessed?",
            weight=0.05,
            criteria={
                1: "Complexity assessment is incorrect",
                2: "Complexity assessment is imprecise",
                3: "Complexity assessment is adequate",
                4: "Complexity assessment is accurate",
                5: "Complexity assessment is precise and well-justified"
            }
        )
    ]
)


# Factual record rubric
FACTUAL_RECORD_RUBRIC = Rubric(
    step_name="factual_record",
    dimensions=[
        EvaluationDimension(
            name="Factual_Completeness",
            description="Are all relevant facts included?",
            weight=0.35,
            criteria={
                1: "Missing many critical facts",
                2: "Missing several important facts",
                3: "Most facts included, some gaps",
                4: "Comprehensive with minor omissions",
                5: "Fully comprehensive, all relevant facts included"
            }
        ),
        EvaluationDimension(
            name="Factual_Accuracy",
            description="Are facts accurately represented?",
            weight=0.3,
            criteria={
                1: "Multiple factual errors or misrepresentations",
                2: "Several factual inaccuracies",
                3: "Mostly accurate with minor errors",
                4: "Accurate with negligible errors",
                5: "Perfectly accurate, no errors"
            }
        ),
        EvaluationDimension(
            name="Chronological_Order",
            description="Is the timeline properly organized chronologically?",
            weight=0.15,
            criteria={
                1: "Timeline is poorly organized or non-chronological",
                2: "Timeline has significant ordering issues",
                3: "Timeline is mostly chronological with minor issues",
                4: "Timeline is well-organized and chronological",
                5: "Timeline is perfectly chronological and clear"
            }
        ),
        EvaluationDimension(
            name="Evidence_Documentation",
            description="Is evidence properly documented?",
            weight=0.15,
            criteria={
                1: "Evidence poorly documented or missing",
                2: "Evidence documentation has gaps",
                3: "Evidence adequately documented",
                4: "Evidence well-documented",
                5: "Evidence comprehensively and precisely documented"
            }
        ),
        EvaluationDimension(
            name="Neutrality",
            description="Are facts presented neutrally without legal conclusions?",
            weight=0.05,
            criteria={
                1: "Significant legal conclusions mixed with facts",
                2: "Some legal conclusions in factual record",
                3: "Mostly neutral with minor interpretations",
                4: "Neutral with negligible interpretations",
                5: "Perfectly neutral, facts only"
            }
        )
    ]
)


# Legal arguments rubric
LEGAL_ARGUMENTS_RUBRIC = Rubric(
    step_name="legal_arguments",
    dimensions=[
        EvaluationDimension(
            name="Legal_Soundness",
            description="Are the legal arguments legally sound?",
            weight=0.35,
            criteria={
                1: "Arguments contain major legal errors",
                2: "Arguments have significant legal weaknesses",
                3: "Arguments are generally sound with minor issues",
                4: "Arguments are sound and well-reasoned",
                5: "Arguments are exceptionally sound and compelling"
            }
        ),
        EvaluationDimension(
            name="Factual_Support",
            description="Are arguments well-supported by facts?",
            weight=0.25,
            criteria={
                1: "Arguments lack factual support",
                2: "Arguments have weak factual support",
                3: "Arguments have adequate factual support",
                4: "Arguments are well-supported by facts",
                5: "Arguments have comprehensive and precise factual support"
            }
        ),
        EvaluationDimension(
            name="Legal_Basis_Application",
            description="Are legal bases correctly applied?",
            weight=0.25,
            criteria={
                1: "Legal bases misapplied or missing",
                2: "Legal bases partially or imprecisely applied",
                3: "Legal bases adequately applied",
                4: "Legal bases well-applied",
                5: "Legal bases expertly and precisely applied"
            }
        ),
        EvaluationDimension(
            name="Reasoning_Quality",
            description="Is the reasoning connecting facts to conclusions clear and logical?",
            weight=0.1,
            criteria={
                1: "Reasoning is unclear or illogical",
                2: "Reasoning has significant gaps",
                3: "Reasoning is adequate",
                4: "Reasoning is clear and logical",
                5: "Reasoning is exceptionally clear, logical, and persuasive"
            }
        ),
        EvaluationDimension(
            name="Completeness",
            description="Are all relevant arguments presented?",
            weight=0.05,
            criteria={
                1: "Missing most relevant arguments",
                2: "Missing several important arguments",
                3: "Most relevant arguments present",
                4: "All important arguments present",
                5: "Comprehensive presentation of all relevant arguments"
            }
        )
    ]
)


# Recommendations rubric
RECOMMENDATIONS_RUBRIC = Rubric(
    step_name="recommendations",
    dimensions=[
        EvaluationDimension(
            name="Actionability",
            description="Are recommendations clear and actionable?",
            weight=0.3,
            criteria={
                1: "Recommendations are vague or not actionable",
                2: "Recommendations are somewhat vague",
                3: "Recommendations are adequately clear and actionable",
                4: "Recommendations are clear and actionable",
                5: "Recommendations are exceptionally clear, specific, and actionable"
            }
        ),
        EvaluationDimension(
            name="Legal_Soundness",
            description="Are recommendations legally sound?",
            weight=0.25,
            criteria={
                1: "Recommendations contain major legal errors",
                2: "Recommendations have legal weaknesses",
                3: "Recommendations are generally sound",
                4: "Recommendations are legally sound",
                5: "Recommendations are exceptionally sound and well-justified"
            }
        ),
        EvaluationDimension(
            name="Risk_Assessment",
            description="Are risks properly identified and assessed?",
            weight=0.2,
            criteria={
                1: "Risks poorly identified or assessed",
                2: "Risk assessment has significant gaps",
                3: "Risks adequately identified and assessed",
                4: "Risks well-identified and assessed",
                5: "Comprehensive and nuanced risk assessment"
            }
        ),
        EvaluationDimension(
            name="Alignment_with_Objectives",
            description="Do recommendations align with client objectives?",
            weight=0.15,
            criteria={
                1: "Recommendations poorly aligned with client objectives",
                2: "Recommendations partially aligned with objectives",
                3: "Recommendations adequately aligned with objectives",
                4: "Recommendations well-aligned with objectives",
                5: "Recommendations perfectly aligned and optimized for client objectives"
            }
        ),
        EvaluationDimension(
            name="Alternatives_and_Next_Steps",
            description="Are alternatives and next steps well-articulated?",
            weight=0.1,
            criteria={
                1: "Alternatives/next steps missing or unclear",
                2: "Alternatives/next steps incomplete",
                3: "Alternatives/next steps adequate",
                4: "Alternatives/next steps well-articulated",
                5: "Comprehensive alternatives with clear, strategic next steps"
            }
        )
    ]
)


# Map step names to rubrics
RUBRICS: Dict[str, Rubric] = {
    "qualification": QUALIFICATION_RUBRIC,
    "initial_analysis": INITIAL_ANALYSIS_RUBRIC,
    "factual_record": FACTUAL_RECORD_RUBRIC,
    "legal_arguments": LEGAL_ARGUMENTS_RUBRIC,
    "recommendations": RECOMMENDATIONS_RUBRIC
}


def get_rubric(step_name: str) -> Rubric:
    """
    Get rubric for a pipeline step.

    Args:
        step_name: Name of the pipeline step

    Returns:
        Rubric for that step

    Raises:
        KeyError: If no rubric exists for that step
    """
    if step_name not in RUBRICS:
        raise KeyError(f"No rubric defined for step: {step_name}")
    return RUBRICS[step_name]
