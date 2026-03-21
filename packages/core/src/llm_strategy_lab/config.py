"""Typed YAML configuration loading helpers."""

from datetime import date
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import yaml

from .models import (
    BacktestConfig,
    DatasetConfig,
    EnvironmentConfig,
    ExperimentConfig,
    JsonDict,
    StrategyConfig,
)

KNOWN_PROJECT_ROOT_DIRS = {
    "apps",
    "configs",
    "data",
    "docs",
    "notebooks",
    "packages",
    "runs",
    "scripts",
    "tests",
}
REQUIRED_EXPERIMENT_KEYS = {"experiment_id", "environment", "strategy", "backtest"}


def load_yaml_mapping(path: Path) -> JsonDict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")

    return dict(data)


def find_project_root(start_path: Path) -> Path:
    probe = start_path.resolve()
    if probe.is_file():
        probe = probe.parent

    for candidate in (probe, *probe.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate

    raise FileNotFoundError(f"Could not find project root from: {start_path}")


def resolve_project_path(raw_path: str, *, base_path: Path, project_root: Path) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate.resolve()

    base_dir = base_path.parent
    base_relative = (base_dir / candidate).resolve()
    project_relative = (project_root / candidate).resolve()

    if base_relative.exists():
        return base_relative
    if project_relative.exists():
        return project_relative

    if candidate.parts and candidate.parts[0] in KNOWN_PROJECT_ROOT_DIRS:
        return project_relative
    return base_relative


def _parse_iso_date(field_name: str, raw_value: Optional[str]) -> Optional[date]:
    if raw_value is None:
        return None

    try:
        return date.fromisoformat(raw_value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be YYYY-MM-DD: {raw_value}") from exc


def _ensure_mapping(value: Any, *, field_name: str) -> JsonDict:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping.")
    return dict(value)


def load_environment_config(
    raw_environment: Any,
    *,
    config_path: Path,
    project_root: Path,
) -> EnvironmentConfig:
    if isinstance(raw_environment, str):
        if raw_environment.endswith((".yaml", ".yml")) or "/" in raw_environment:
            environment_path = resolve_project_path(
                raw_environment,
                base_path=config_path,
                project_root=project_root,
            )
        else:
            environment_path = (
                project_root / "configs" / "environments" / raw_environment
            ).with_suffix(
                ".yaml",
            )
        environment_mapping = load_yaml_mapping(environment_path)
    else:
        environment_path = config_path
        environment_mapping = _ensure_mapping(raw_environment, field_name="environment")

    output_root_raw = str(environment_mapping.get("output_root", "runs"))
    output_root = resolve_project_path(
        output_root_raw,
        base_path=environment_path,
        project_root=project_root,
    )
    seed_value = environment_mapping.get("seed")
    seed = int(seed_value) if seed_value is not None else None

    return EnvironmentConfig(
        name=str(environment_mapping.get("name", environment_path.stem)),
        output_root=output_root,
        log_level=str(environment_mapping.get("log_level", "INFO")),
        seed=seed,
        raw=environment_mapping,
    )


def load_strategy_config(
    raw_strategy: Any,
    *,
    config_path: Path,
    project_root: Path,
) -> StrategyConfig:
    strategy_mapping = _ensure_mapping(raw_strategy, field_name="strategy")
    if "name" not in strategy_mapping:
        raise ValueError("strategy.name is required.")

    params_file: Optional[Path] = None
    file_params: JsonDict = {}
    if strategy_mapping.get("params_file"):
        params_file = resolve_project_path(
            str(strategy_mapping["params_file"]),
            base_path=config_path,
            project_root=project_root,
        )
        file_params = load_yaml_mapping(params_file)

    merged_params: JsonDict = {}
    for source in (file_params, strategy_mapping):
        for key, value in source.items():
            if key in {"name", "params_file"}:
                continue
            merged_params[key] = value

    return StrategyConfig(
        name=str(strategy_mapping["name"]),
        params_file=params_file,
        params=merged_params,
        raw=strategy_mapping,
    )


def load_dataset_config(
    raw_dataset: Any,
    *,
    config_path: Path,
    project_root: Path,
) -> DatasetConfig:
    dataset_mapping = _ensure_mapping(raw_dataset, field_name="dataset")
    inputs = {
        key: resolve_project_path(str(value), base_path=config_path, project_root=project_root)
        for key, value in dataset_mapping.items()
    }
    return DatasetConfig(inputs=inputs)


def load_backtest_config(raw_backtest: Any) -> BacktestConfig:
    backtest_mapping = _ensure_mapping(raw_backtest, field_name="backtest")

    params = {
        key: value
        for key, value in backtest_mapping.items()
        if key not in {"start", "end", "rebalance"}
    }

    return BacktestConfig(
        start=_parse_iso_date("backtest.start", backtest_mapping.get("start")),
        end=_parse_iso_date("backtest.end", backtest_mapping.get("end")),
        rebalance=str(backtest_mapping.get("rebalance", "monthly")),
        params=params,
    )


def load_experiment_config(
    config_path: Path,
    *,
    project_root: Optional[Path] = None,
) -> ExperimentConfig:
    resolved_config_path = config_path.resolve()
    resolved_project_root = (
        project_root.resolve()
        if project_root is not None
        else find_project_root(resolved_config_path)
    )
    experiment_mapping = load_yaml_mapping(resolved_config_path)

    missing = sorted(REQUIRED_EXPERIMENT_KEYS - experiment_mapping.keys())
    if missing:
        raise ValueError(f"Config is missing required keys: {', '.join(missing)}")

    notes = experiment_mapping.get("notes")
    if notes is None:
        notes_sequence: Sequence[object] = ()
    elif isinstance(notes, list):
        notes_sequence = notes
    else:
        raise ValueError("notes must be a list when provided.")

    dataset = None
    if "dataset" in experiment_mapping and experiment_mapping["dataset"] is not None:
        dataset = load_dataset_config(
            experiment_mapping["dataset"],
            config_path=resolved_config_path,
            project_root=resolved_project_root,
        )

    return ExperimentConfig(
        experiment_id=str(experiment_mapping["experiment_id"]),
        environment=load_environment_config(
            experiment_mapping["environment"],
            config_path=resolved_config_path,
            project_root=resolved_project_root,
        ),
        strategy=load_strategy_config(
            experiment_mapping["strategy"],
            config_path=resolved_config_path,
            project_root=resolved_project_root,
        ),
        dataset=dataset,
        backtest=load_backtest_config(experiment_mapping["backtest"]),
        notes=tuple(str(note) for note in notes_sequence),
        config_path=resolved_config_path,
        project_root=resolved_project_root,
        raw=experiment_mapping,
    )
