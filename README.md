# Lexic - Legal AI Evaluation Framework

Evaluation framework for a multi-agent legal AI system that replicates Swiss/French legal practice workflows.

## Quick Start

### 1. Setup

```bash
# Clone the repository
cd lexic

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### 2. Prepare Data

Place court decision documents (PDF or DOCX) in `data/court_decisions/` (name them `decision_001.pdf`, `decision_002.docx`, etc.)

### 3. Extract and Generate

```bash
# Extract court decisions to structured markdown
python scripts/01_extract_decisions.py

# Generate synthetic cases with ground truth
python scripts/02_generate_synthetic.py
```

### 4. Run Evaluations

```bash
# Start MLFlow server (in separate terminal)
docker-compose up mlflow

# Or run locally:
mlflow server --host 0.0.0.0 --port 5000

# Run evaluation for a specific step
python scripts/03_run_eval.py --step qualification

# View results in MLFlow UI
open http://localhost:5000
```

## Project Structure

```
lexic/
├── agents/                    # Production agents
│   ├── qualification.py       # Client qualification agent
│   ├── initial_analysis.py    # Initial legal analysis agent
│   ├── factual_record.py      # Factual record agent
│   ├── legal_bases.py         # Legal basis identification agent
│   ├── arguments.py           # Legal argumentation agent
│   ├── considerations.py      # Legal consideration agent
│   └── recommendations.py     # Recommendation agent
│
├── shared/                    # Shared utilities
│   ├── models.py             # Data models
│   ├── config.py             # Configuration
│   └── io.py                 # File I/O utilities
│
├── synthetic_data/           # Data generation
│   ├── extract.py           # PDF extraction
│   └── generate.py          # Synthetic case generation
│
├── evals/                    # Evaluation framework
│   ├── judges/
│   │   ├── rubrics.py       # Evaluation rubrics
│   │   └── judge.py         # LLM-as-judge
│   └── orchestrator.py      # Evaluation orchestration
│
├── scripts/                  # Workflow scripts
│   ├── 01_extract_decisions.py
│   ├── 02_generate_synthetic.py
│   └── 03_run_eval.py
│
└── data/                     # Data directories
    ├── court_decisions/
    ├── synthetic_cases/
    └── eval_runs/
```

## Workflow

### Legal AI Pipeline Steps

1. **Qualification**: Client dialogue → situation report
2. **Initial Analysis**: Situation → legal analysis
3. **Investigation Order**: Analysis → information requests
4. **Investigation Report**: Client response
5. **Factual Record**: Structured facts
6. **Legal Bases**: Applicable laws
7. **Legal Arguments**: Arguments from facts + laws
8. **Considerations**: Legal analysis
9. **Recommendations**: Client advice

### Evaluation Strategy

- **Synthetic Data**: Generate backward from real court decisions
- **Ground Truth**: Each pipeline step has reference outputs
- **LLM-as-Judge**: Evaluate agent outputs using rubrics
- **MLFlow**: Track experiments and metrics

## Available Evaluations

```bash
# Run all pipeline steps
python scripts/03_run_eval.py --step all

# Qualification step
python scripts/03_run_eval.py --step qualification

# Initial analysis step
python scripts/03_run_eval.py --step initial_analysis

# Factual record step
python scripts/03_run_eval.py --step factual_record

# Legal arguments step
python scripts/03_run_eval.py --step legal_arguments

# Recommendations step
python scripts/03_run_eval.py --step recommendations

# Limit number of cases
python scripts/03_run_eval.py --step all --n-cases 5
```

## Docker Usage

```bash
# Start MLFlow only
docker-compose up mlflow

# Start full stack (including API - when implemented)
docker-compose --profile production up

# Build and run
docker-compose build
docker-compose up
```

## Development

```bash
# Run tests
pytest

# Format code
black .
isort .

# Type checking
mypy .
```

## Technology Stack

- **DSPy**: Agent framework with chain-of-thought reasoning
- **Anthropic Claude**: LLM provider
- **MLFlow**: Experiment tracking
- **Docling**: PDF extraction
- **Docker**: Containerization

## Configuration

Edit `.env` to configure:

- `ANTHROPIC_API_KEY`: Your Anthropic API key
- `DEFAULT_MODEL`: Model for agents
- `JUDGE_MODEL`: Model for evaluation
- `MLFLOW_TRACKING_URI`: MLFlow server URL
- `TEMPERATURE`: LLM temperature (default: 0.7)

