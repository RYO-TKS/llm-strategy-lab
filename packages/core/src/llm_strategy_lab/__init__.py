"""Core package for llm-strategy-lab."""

from .backtest import run_daily_backtest
from .config import load_experiment_config
from .data_pipeline import prepare_aligned_research_dataset
from .evaluation import run_backtest_evaluation
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
from .strategies import (
    DoubleSortStrategy,
    MomentumStrategy,
    PlainPCAStrategy,
    SubspacePCAStrategy,
    create_strategy,
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
    "DoubleSortStrategy",
    "MomentumStrategy",
    "PlainPCAStrategy",
    "SubspacePCAStrategy",
    "create_strategy",
    "run_daily_backtest",
    "run_backtest_evaluation",
    "load_experiment_config",
    "prepare_aligned_research_dataset",
]

__version__ = "0.1.0"
