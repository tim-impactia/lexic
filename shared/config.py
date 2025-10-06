"""Configuration management for Lexic."""

import os
from pathlib import Path
from typing import Optional


class Config:
    """Configuration for Lexic evaluation framework."""

    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    COURT_DECISIONS_DIR = DATA_DIR / "court_decisions"
    SYNTHETIC_CASES_DIR = DATA_DIR / "synthetic_cases"
    EVAL_RUNS_DIR = DATA_DIR / "eval_runs"
    MLRUNS_DIR = PROJECT_ROOT / "mlruns"

    # LLM Configuration
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "claude-3-5-sonnet-20241022")
    EXTRACTION_MODEL: str = os.getenv("EXTRACTION_MODEL", "claude-3-5-sonnet-20241022")
    GENERATION_MODEL: str = os.getenv("GENERATION_MODEL", "claude-3-5-sonnet-20241022")
    JUDGE_MODEL: str = os.getenv("JUDGE_MODEL", "claude-3-5-sonnet-20241022")

    # DSPy Configuration
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.7"))

    # MLFlow Configuration
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    MLFLOW_EXPERIMENT_NAME: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "lexic-evaluation")

    # Evaluation Configuration
    JUDGE_TEMPERATURE: float = 0.0  # Deterministic for consistency

    @classmethod
    def ensure_dirs(cls):
        """Ensure all required directories exist."""
        for dir_path in [
            cls.COURT_DECISIONS_DIR,
            cls.SYNTHETIC_CASES_DIR,
            cls.EVAL_RUNS_DIR,
            cls.MLRUNS_DIR,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def validate(cls):
        """Validate configuration."""
        if not cls.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")
        cls.ensure_dirs()
