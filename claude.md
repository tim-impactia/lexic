# Lexic - Legal AI SaaS - Evaluation Framework

## Project Overview

This is the evaluation framework for Lexic, a legal AI SaaS that replicates how a law firm handles clients through multiple AI agents working together. The system is designed for Swiss/French legal practice.

## Core Concept

We're building a multi-agent system that follows the actual legal workflow:

1. **Client Dialogue → Qualification Report** (situation/objectives/constraints)
2. **Qualification Report → Initial Legal Analysis**
3. **Initial Legal Analysis → Investigation Order**
4. **Investigation Order → Client Dialogue → Investigation Report**
5. **Investigation Report → Factual Record**
6. **Factual Record → Update situation/objectives/constraints**
7. **Factual Record → Applicable Legal Bases**
8. **Factual Record + Legal Bases → Legal Arguments**
9. **Arguments → Incomplete facts? → Investigation Order** (loop back if needed)
10. **Arguments + Factual Record → Legal Considerations + Judgment**
11. **Considerations + Judgment → Recommendations**

## Evaluation Strategy

Since we don't have ground truth data for each step, we're creating synthetic data by working backwards from real court decisions:

1. Take a court decision (PDF)
2. Extract structured elements (parties, facts, legal bases, arguments, considerations, judgment)
3. Generate synthetic intermediates working backwards:
   - From the decision, imagine what client persona and initial facts would have led here
   - From the client context, generate what the qualification would have been
   - From the qualification, generate what the initial analysis would have been
   - And so on through the entire pipeline
4. Use these synthetic cases with ground truth at each step to evaluate our agents
5. Use LLM-as-judge to evaluate agent outputs against ground truth

## Technology Stack

- **DSPy**: For building and optimizing LLM-based agents
- **MLFlow**: For experiment tracking and metrics
- **Docling**: For PDF extraction to markdown
- **Claude API**: LLM provider
- **Docker**: For containerization (dev and production)

## File Structure

```
lexic/
├── data/
│   ├── court_decisions/          # Extracted court decisions
│   │   ├── decision_001/
│   │   │   ├── original.pdf
│   │   │   ├── full_text.md
│   │   │   ├── parties.md
│   │   │   ├── facts_timeline.md
│   │   │   ├── legal_bases.md
│   │   │   ├── arguments.md
│   │   │   ├── considerations.md
│   │   │   └── judgment.md
│   │   └── ...
│   │
│   ├── synthetic_cases/          # Generated synthetic cases with ground truth
│   │   ├── case_001/
│   │   │   ├── metadata.md
│   │   │   ├── 01_client_persona.md
│   │   │   ├── 02_initial_facts_known.md
│   │   │   ├── 03_gt_situation.md
│   │   │   ├── 04_gt_initial_analysis.md
│   │   │   ├── 05_gt_investigation_order_1.md
│   │   │   ├── 06_gt_investigation_report_1.md
│   │   │   ├── 09_gt_initial_factual_record.md
│   │   │   ├── 10_gt_final_factual_record.md
│   │   │   ├── 11_gt_applicable_legal_bases.md
│   │   │   ├── 12_gt_legal_arguments.md
│   │   │   ├── 13_gt_considerations.md
│   │   │   ├── 14_gt_judgment.md
│   │   │   └── 15_gt_recommendations.md
│   │   └── ...
│   │
│   └── eval_runs/                # Evaluation run results
│       ├── qualification_20250104_153022/
│       │   ├── case_001_prediction.md
│       │   ├── case_001_evaluation.md
│       │   └── summary.md
│       └── ...
│
├── agents/                       # Production agents (main app)
│   ├── __init__.py
│   ├── qualification.py          # QualificationAgent
│   ├── initial_analysis.py       # InitialAnalysisAgent
│   ├── investigation_order.py    # InvestigationOrderAgent
│   ├── investigation_report.py   # InvestigationReportAgent
│   ├── factual_record.py         # FactualRecordAgent
│   ├── legal_bases.py            # LegalBasisAgent
│   ├── arguments.py              # ArgumentationAgent
│   ├── considerations.py         # ConsiderationAgent
│   ├── recommendations.py        # RecommendationAgent
│   └── pipeline.py               # Full orchestrated pipeline
│
├── shared/
│   ├── __init__.py
│   ├── models.py                 # Dataclasses for all entities
│   ├── config.py                 # Configuration management
│   └── io.py                     # Markdown file I/O utilities
│
├── synthetic_data/               # Synthetic data generation
│   ├── __init__.py
│   ├── extract.py                # Extract from PDFs using DSPy
│   └── generate.py               # Generate synthetic cases using DSPy
│
├── evals/                        # Evaluation framework
│   ├── __init__.py
│   ├── judges/
│   │   ├── __init__.py
│   │   ├── judge.py              # LLM-as-judge implementation
│   │   └── rubrics.py            # Evaluation rubrics for each step
│   │
│   └── orchestrator.py           # Evaluation orchestration with MLFlow
│
├── scripts/
│   ├── 01_extract_decisions.py   # Extract court decisions to markdown
│   ├── 02_generate_synthetic.py  # Generate synthetic cases
│   └── 03_run_eval.py            # Run evaluations
│
├── notebooks/
│   └── explore.ipynb             # Exploration and analysis
│
├── tests/
│   ├── __init__.py
│   ├── test_agents.py
│   ├── test_extraction.py
│   └── test_generation.py
│
├── mlruns/                       # MLFlow artifacts (gitignored)
│
├── Dockerfile                    # For production app
├── docker-compose.yml            # MLFlow + app services
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
└── claude.md                     # This file
```

## DSPy Agent Architecture

Each agent is a DSPy module with:
- Clear input/output signature
- Chain-of-thought reasoning
- Structured output format

Example agent structure:
```python
class QualificationAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.qualify = dspy.ChainOfThought(
            "client_persona, initial_facts -> situation"
        )
    
    def forward(self, client_persona, initial_facts):
        result = self.qualify(
            client_persona=client_persona,
            initial_facts=initial_facts
        )
        return result.situation
```

## Evaluation Rubrics

Each pipeline step has a rubric with dimensions like:
- **Completeness**: Are all relevant aspects captured?
- **Accuracy**: Is the output accurate given inputs?
- **Legal soundness**: Is the legal reasoning correct?
- **Relevance**: Is irrelevant content excluded?
- **Clarity**: Is the output well-structured?

Scores are 1-5 with specific criteria for each score level.

## LLM-as-Judge

The judge is a DSPy module that:
1. Takes prediction, ground truth, and rubric as input
2. Scores each rubric dimension with explanation
3. Identifies critical errors
4. Computes weighted overall score

## MLFlow Integration

We track:
- **Parameters**: Model, step, dataset version, n_cases
- **Metrics**: Per-dimension scores, overall score, error rates, stratified by complexity
- **Artifacts**: Predictions, evaluations, error analysis (as markdown)

## Workflow

### Generate Synthetic Data
```bash
# 1. Place court decision PDFs in data/court_decisions/
# Each should be named decision_XXX.pdf

# 2. Extract decisions to markdown
python scripts/01_extract_decisions.py

# 3. Generate synthetic cases with ground truth
python scripts/02_generate_synthetic.py
```

### Run Evaluations
```bash
# Run evaluation for a specific step
python scripts/03_run_eval.py --step qualification

# View results in MLFlow UI
open http://localhost:5000
```

### Available Pipeline Steps
- `qualification`: Client dialogue → situation report
- `initial_analysis`: Situation → initial legal analysis
- `investigation_order`: Analysis → investigation order
- `investigation_report`: Order → investigation report
- `factual_record`: Facts → structured factual record
- `legal_bases`: Factual record → applicable legal provisions
- `legal_arguments`: Factual record + bases → arguments
- `considerations`: Arguments → legal considerations
- `recommendations`: Considerations → client recommendations

## File Formats

### Markdown Structure

All data files are markdown with YAML frontmatter for metadata:

```markdown
---
id: case_001
source_decision: decision_001
step: qualification
generated_at: 2025-01-04T15:30:22
model: gpt-4o
---

# Ground Truth: Situation

## Summary
[Description of the legal situation]

## Objectives
- Objective 1
- Objective 2

## Constraints
- Time constraints
- Budget constraints
- Risk tolerance

## Legal Questions
- Question 1
- Question 2
```

### Reading/Writing Markdown Files

Use `shared/io.py` utilities:
- `read_markdown(path)`: Returns (metadata_dict, content_string)
- `write_markdown(path, metadata, content)`: Writes markdown with frontmatter
- `list_cases(directory)`: Lists all case directories
- `load_case_step(case_dir, step_name)`: Loads specific step from case

## Docker Setup

### Development
```yaml
# docker-compose.yml includes:
# - MLFlow server for experiment tracking
# - Volume mounts for data/ and code/
```

### Production
```dockerfile
# Dockerfile builds agent API server
# - Exposes REST API for agent pipeline
# - Includes all agents and dependencies
# - Uses production LLM configuration
```

## Testing Strategy

### Unit Tests
- Test individual agents with mock inputs
- Test extraction logic
- Test evaluation rubrics

### Integration Tests
- Test full pipeline on sample cases
- Test judge accuracy against human ratings

### Evaluation Tests
- Continuously run evals on new model versions
- Track metric regressions
- A/B test prompt changes

## Key Design Principles

1. **Simplicity**: Use markdown files and MLFlow, avoid complex databases initially
2. **Transparency**: All data is human-readable markdown
3. **Iterative**: Start with one step, expand gradually
4. **Measurable**: Every step has clear evaluation metrics
5. **Realistic**: Synthetic data mimics real legal workflows

## Next Steps

1. Implement extraction logic for court decisions
2. Implement backward generation for synthetic cases
3. Build first agent (qualification)
4. Create rubric and judge for qualification
5. Run first eval and iterate
6. Expand to remaining steps

## Notes

- All agents use the same LLM configuration from environment
- Swiss/French legal terminology throughout
- Synthetic data validation layer can be added later if needed
- Database (PostgreSQL) can be added when scale requires it
- DVC/Parquet can be added for dataset versioning when needed
