"""Core typed models for experiment configuration and run artifacts."""

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

JsonDict = Dict[str, Any]


def _path_mapping_to_strings(values: Mapping[str, Path]) -> JsonDict:
    return {key: str(value) for key, value in values.items()}


@dataclass(frozen=True)
class StrategyConfig:
    name: str
    params_file: Optional[Path]
    params: JsonDict = field(default_factory=dict)
    raw: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return {
            "name": self.name,
            "params_file": str(self.params_file) if self.params_file else None,
            "params": dict(self.params),
        }


@dataclass(frozen=True)
class DatasetConfig:
    inputs: Mapping[str, Path] = field(default_factory=dict)

    @property
    def us_sectors(self) -> Optional[Path]:
        return self.inputs.get("us_sectors")

    @property
    def jp_sectors(self) -> Optional[Path]:
        return self.inputs.get("jp_sectors")

    @property
    def trading_calendar(self) -> Optional[Path]:
        return self.inputs.get("trading_calendar")

    @property
    def factor_returns(self) -> Optional[Path]:
        return self.inputs.get("factor_returns")

    def to_dict(self) -> JsonDict:
        return _path_mapping_to_strings(self.inputs)


@dataclass(frozen=True)
class BacktestConfig:
    start: Optional[date]
    end: Optional[date]
    rebalance: str
    params: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        payload: JsonDict = {
            "start": self.start.isoformat() if self.start else None,
            "end": self.end.isoformat() if self.end else None,
            "rebalance": self.rebalance,
        }
        payload.update(self.params)
        return payload


@dataclass(frozen=True)
class EnvironmentConfig:
    name: str
    output_root: Path
    log_level: str
    seed: Optional[int]
    raw: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return {
            "name": self.name,
            "output_root": str(self.output_root),
            "log_level": self.log_level,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    environment: EnvironmentConfig
    strategy: StrategyConfig
    dataset: Optional[DatasetConfig]
    backtest: BacktestConfig
    notes: Tuple[str, ...]
    config_path: Path
    project_root: Path
    raw: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return {
            "experiment_id": self.experiment_id,
            "environment": self.environment.to_dict(),
            "strategy": self.strategy.to_dict(),
            "dataset": self.dataset.to_dict() if self.dataset else None,
            "backtest": self.backtest.to_dict(),
            "notes": list(self.notes),
            "config_path": str(self.config_path),
            "project_root": str(self.project_root),
        }


@dataclass(frozen=True)
class StrategyArtifact:
    strategy_name: str
    generated_at_utc: datetime
    signal_columns: Tuple[str, ...]
    parameter_snapshot: JsonDict = field(default_factory=dict)
    artifact_paths: Mapping[str, Path] = field(default_factory=dict)
    explanation: Optional[str] = None
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return {
            "strategy_name": self.strategy_name,
            "generated_at_utc": self.generated_at_utc.isoformat(),
            "signal_columns": list(self.signal_columns),
            "parameter_snapshot": dict(self.parameter_snapshot),
            "artifact_paths": _path_mapping_to_strings(self.artifact_paths),
            "explanation": self.explanation,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class BacktestResult:
    strategy_name: str
    metrics: Mapping[str, float] = field(default_factory=dict)
    series_paths: Mapping[str, Path] = field(default_factory=dict)
    gross_exposure: Optional[float] = None
    net_exposure: Optional[float] = None
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return {
            "strategy_name": self.strategy_name,
            "metrics": dict(self.metrics),
            "series_paths": _path_mapping_to_strings(self.series_paths),
            "gross_exposure": self.gross_exposure,
            "net_exposure": self.net_exposure,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RunComparisonRecord:
    lineage_id: str
    experiment_id: str
    parent_run_id: str
    candidate_run_id: str
    created_at_utc: datetime
    output_dir: Path
    parent_run: JsonDict = field(default_factory=dict)
    candidate_run: JsonDict = field(default_factory=dict)
    metric_comparison: JsonDict = field(default_factory=dict)
    factor_regression_comparison: JsonDict = field(default_factory=dict)
    config_diff: JsonDict = field(default_factory=dict)
    artifact_paths: Mapping[str, Path] = field(default_factory=dict)
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return {
            "lineage_id": self.lineage_id,
            "experiment_id": self.experiment_id,
            "parent_run_id": self.parent_run_id,
            "candidate_run_id": self.candidate_run_id,
            "created_at_utc": self.created_at_utc.isoformat(),
            "output_dir": str(self.output_dir),
            "parent_run": dict(self.parent_run),
            "candidate_run": dict(self.candidate_run),
            "metric_comparison": dict(self.metric_comparison),
            "factor_regression_comparison": dict(self.factor_regression_comparison),
            "config_diff": dict(self.config_diff),
            "artifact_paths": _path_mapping_to_strings(self.artifact_paths),
            "metadata": dict(self.metadata),
        }


class RunStatus(str, Enum):
    SCAFFOLD = "scaffold"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    experiment_id: str
    environment_name: str
    strategy_name: str
    config_path: Path
    output_dir: Path
    status: RunStatus
    started_at_utc: datetime
    finished_at_utc: Optional[datetime] = None
    notes: Tuple[str, ...] = ()
    message: Optional[str] = None
    strategy_artifacts: Tuple[StrategyArtifact, ...] = ()
    backtest_result: Optional[BacktestResult] = None
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        payload: JsonDict = {
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "environment": self.environment_name,
            "strategy": self.strategy_name,
            "config_path": str(self.config_path),
            "output_dir": str(self.output_dir),
            "status": self.status.value,
            "started_at_utc": self.started_at_utc.isoformat(),
            "finished_at_utc": (
                self.finished_at_utc.isoformat() if self.finished_at_utc else None
            ),
            "notes": list(self.notes),
            "message": self.message,
            "strategy_artifacts": [artifact.to_dict() for artifact in self.strategy_artifacts],
            "backtest_result": (
                self.backtest_result.to_dict() if self.backtest_result else None
            ),
            "metadata": dict(self.metadata),
        }
        return payload
