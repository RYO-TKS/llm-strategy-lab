"""Core package for llm-strategy-lab."""

from .config import load_experiment_config
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
]

__version__ = "0.1.0"
