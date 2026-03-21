"""Core package for llm-strategy-lab."""

from .backtest import run_daily_backtest
from .child_runs import create_child_run
from .comparison import compare_runs
from .config import load_experiment_config
from .data_pipeline import prepare_aligned_research_dataset
from .evaluation import run_backtest_evaluation
from .models import (
    BacktestConfig,
    BacktestResult,
    DatasetConfig,
    EnvironmentConfig,
    ExperimentConfig,
    RunComparisonRecord,
    RunRecord,
    RunStatus,
    StrategyArtifact,
    StrategyConfig,
)
from .proposals import (
    ProposalValidationError,
    build_prompt_bundle,
    validate_and_save_proposal,
)
from .runner import run_experiment
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
    "RunComparisonRecord",
    "DatasetConfig",
    "EnvironmentConfig",
    "ExperimentConfig",
    "ProposalValidationError",
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
    "create_child_run",
    "compare_runs",
    "build_prompt_bundle",
    "run_experiment",
    "load_experiment_config",
    "prepare_aligned_research_dataset",
    "validate_and_save_proposal",
]

__version__ = "0.1.0"
