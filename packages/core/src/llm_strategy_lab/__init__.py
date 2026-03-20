"""Core package for llm-strategy-lab."""

from .config import load_experiment_config
from .data_pipeline import prepare_aligned_research_dataset
from .models import (
    BacktestConfig,
    BacktestResult,
    DatasetConfig,
    EnvironmentConfig,
    ExperimentConfig,
    RunRecord,
    RunStatus,
    StrategyArtifact,
    StrategyConfig,
)

__all__ = [
    "__version__",
    "BacktestConfig",
    "BacktestResult",
    "DatasetConfig",
    "EnvironmentConfig",
    "ExperimentConfig",
    "RunRecord",
    "RunStatus",
    "StrategyArtifact",
    "StrategyConfig",
    "load_experiment_config",
    "prepare_aligned_research_dataset",
]

__version__ = "0.1.0"
