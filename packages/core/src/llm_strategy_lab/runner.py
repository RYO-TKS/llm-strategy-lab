"""Scaffold runner used during repository bootstrap."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .config import load_experiment_config
from .models import ExperimentConfig, RunRecord, RunStatus


def next_run_directory(output_root: Path, experiment_id: str) -> Path:
    experiment_root = output_root / experiment_id
    experiment_root.mkdir(parents=True, exist_ok=True)

    existing_runs = sorted(
        int(path.name)
        for path in experiment_root.iterdir()
        if path.is_dir() and path.name.isdigit()
    )
    next_index = existing_runs[-1] + 1 if existing_runs else 1
    run_dir = experiment_root / f"{next_index:04d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _build_summary(config: ExperimentConfig, run_record: RunRecord) -> str:
    backtest_window = "unspecified"
    if config.backtest.start and config.backtest.end:
        backtest_window = (
            f"{config.backtest.start.isoformat()} -> "
            f"{config.backtest.end.isoformat()}"
        )

    dataset_inputs = "none"
    if config.dataset and config.dataset.inputs:
        dataset_inputs = ", ".join(sorted(config.dataset.inputs.keys()))

    return "\n".join(
        [
            "# Scaffold Run",
            "",
            f"- Run ID: `{run_record.run_id}`",
            f"- Experiment: `{config.experiment_id}`",
            f"- Environment: `{config.environment.name}`",
            f"- Strategy: `{config.strategy.name}`",
            f"- Backtest Window: `{backtest_window}`",
            f"- Dataset Inputs: `{dataset_inputs}`",
            "",
            (
                "This run confirms that typed config loading, run directory allocation, "
                "and artifact output wiring are in place."
            ),
            "It does not execute the real strategy or backtest engine yet.",
            "",
            "## Next Steps",
            "",
            "1. Implement data ingestion and alignment.",
            "2. Add strategy interfaces and concrete strategies.",
            "3. Persist metrics and charts in this run directory.",
            "",
        ]
    )


def create_scaffold_run(config_path: Path, output_root: Path | None = None) -> Path:
    config = load_experiment_config(config_path)
    resolved_output_root = output_root.resolve() if output_root else config.environment.output_root
    run_dir = next_run_directory(
        output_root=resolved_output_root,
        experiment_id=config.experiment_id,
    )
    started_at_utc = datetime.now(timezone.utc)

    run_record = RunRecord(
        run_id=run_dir.name,
        experiment_id=config.experiment_id,
        environment_name=config.environment.name,
        strategy_name=config.strategy.name,
        config_path=config.config_path,
        output_dir=run_dir,
        status=RunStatus.SCAFFOLD,
        started_at_utc=started_at_utc,
        notes=config.notes,
        message=(
            "Repository scaffold is ready. "
            "Implement data, strategy, and backtest logic next."
        ),
        metadata={
            "environment_config": config.environment.to_dict(),
            "strategy_config": config.strategy.to_dict(),
            "dataset_config": config.dataset.to_dict() if config.dataset else None,
            "backtest_config": config.backtest.to_dict(),
        },
    )
    manifest = run_record.to_dict()
    summary = _build_summary(config, run_record)

    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (run_dir / "SUMMARY.md").write_text(summary, encoding="utf-8")
    return run_dir
