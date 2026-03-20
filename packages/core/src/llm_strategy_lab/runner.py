"""Strategy preview runner used during repository bootstrap."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from .config import load_experiment_config
from .data_pipeline import prepare_aligned_research_dataset
from .models import ExperimentConfig, RunRecord, RunStatus
from .strategies import create_strategy


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

    data_alignment = run_record.metadata.get("data_alignment")
    aligned_signal_dates = "n/a"
    quality_event_count = "n/a"
    if isinstance(data_alignment, dict):
        aligned_signal_dates = str(data_alignment.get("aligned_signal_dates", "n/a"))
        quality_event_count = str(data_alignment.get("quality_event_count", "n/a"))

    portfolio_dates_prepared = "n/a"
    portfolio_rows_prepared = "n/a"
    if run_record.strategy_artifacts:
        strategy_metadata = run_record.strategy_artifacts[0].metadata
        portfolio_dates_prepared = str(len(strategy_metadata.get("portfolio_dates", [])))
        portfolio_rows_prepared = str(
            strategy_metadata.get("portfolio_row_count", "n/a")
        )

    return "\n".join(
        [
            "# Strategy Preview Run",
            "",
            f"- Run ID: `{run_record.run_id}`",
            f"- Experiment: `{config.experiment_id}`",
            f"- Environment: `{config.environment.name}`",
            f"- Strategy: `{config.strategy.name}`",
            f"- Backtest Window: `{backtest_window}`",
            f"- Dataset Inputs: `{dataset_inputs}`",
            f"- Aligned Signal Dates: `{aligned_signal_dates}`",
            f"- Data Quality Events: `{quality_event_count}`",
            f"- Portfolio Dates Prepared: `{portfolio_dates_prepared}`",
            f"- Portfolio Rows Prepared: `{portfolio_rows_prepared}`",
            "",
            (
                "This run confirms that typed config loading, market data ingestion, "
                "trading-day alignment, and baseline strategy artifact output are in place."
            ),
            "It does not execute the backtest engine or evaluation metrics yet.",
            "",
            "## Next Steps",
            "",
            "1. Add the remaining PCA-based strategies on the same interface.",
            "2. Connect the standardized portfolio output to the backtest layer.",
            "3. Persist metrics and charts in this run directory.",
            "",
        ]
    )


def create_scaffold_run(config_path: Path, output_root: Path | None = None) -> Path:
    config = load_experiment_config(config_path)
    logging.basicConfig(
        level=getattr(logging, config.environment.log_level.upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)
    resolved_output_root = output_root.resolve() if output_root else config.environment.output_root
    run_dir = next_run_directory(
        output_root=resolved_output_root,
        experiment_id=config.experiment_id,
    )
    started_at_utc = datetime.now(timezone.utc)
    data_alignment_summary = None
    strategy_artifacts = ()
    if config.dataset is not None:
        prepared_dataset = prepare_aligned_research_dataset(
            config.dataset,
            backtest=config.backtest,
            output_dir=run_dir,
            logger=logger,
        )
        data_alignment_summary = prepared_dataset.summary()
        strategy = create_strategy(config.strategy)
        strategy_artifacts = (strategy.run(prepared_dataset, output_dir=run_dir),)

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
            "Aligned data ingestion and baseline strategy artifacts are ready. "
            "Implement the shared backtest engine next."
        ),
        strategy_artifacts=strategy_artifacts,
        metadata={
            "environment_config": config.environment.to_dict(),
            "strategy_config": config.strategy.to_dict(),
            "dataset_config": config.dataset.to_dict() if config.dataset else None,
            "backtest_config": config.backtest.to_dict(),
            "data_alignment": data_alignment_summary,
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
