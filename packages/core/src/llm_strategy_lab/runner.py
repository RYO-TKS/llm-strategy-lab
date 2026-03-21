"""Experiment runner that materializes aligned data, strategy artifacts, and backtests."""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from .backtest import run_daily_backtest
from .config import load_experiment_config, load_strategy_config
from .data_pipeline import prepare_aligned_research_dataset
from .evaluation import run_backtest_evaluation
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


def _apply_strategy_override(
    config: ExperimentConfig,
    *,
    strategy_name: str | None = None,
    strategy_params_file: Path | None = None,
) -> ExperimentConfig:
    if strategy_name is None and strategy_params_file is None:
        return config

    strategy_mapping = dict(config.strategy.raw)
    if strategy_name is not None:
        strategy_mapping["name"] = strategy_name

    effective_name = str(strategy_mapping.get("name", config.strategy.name))
    if strategy_params_file is not None:
        strategy_mapping["params_file"] = str(strategy_params_file)
    elif strategy_name is not None:
        default_params_file = (
            config.project_root / "configs" / "strategies" / f"{effective_name}.default.yaml"
        )
        if default_params_file.exists():
            strategy_mapping["params_file"] = str(default_params_file)
        else:
            strategy_mapping.pop("params_file", None)

    overridden_strategy = load_strategy_config(
        strategy_mapping,
        config_path=config.config_path,
        project_root=config.project_root,
    )
    raw_payload = dict(config.raw)
    raw_payload["strategy"] = dict(strategy_mapping)
    return replace(
        config,
        strategy=overridden_strategy,
        raw=raw_payload,
    )


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

    annual_return = "n/a"
    annual_risk = "n/a"
    return_risk_ratio = "n/a"
    max_drawdown = "n/a"
    average_turnover = "n/a"
    hit_ratio = "n/a"
    ff3_status = "n/a"
    carhart4_status = "n/a"
    chart_count = "0"
    cumulative_rank_ic = "n/a"
    if run_record.backtest_result is not None:
        metrics = run_record.backtest_result.metrics
        annual_return = f"{metrics.get('annual_return', 0.0):.6f}"
        annual_risk = f"{metrics.get('annual_risk', 0.0):.6f}"
        return_risk_ratio = f"{metrics.get('return_risk_ratio', 0.0):.6f}"
        max_drawdown = f"{metrics.get('max_drawdown', 0.0):.6f}"
        average_turnover = f"{metrics.get('average_turnover', 0.0):.6f}"
        hit_ratio = f"{metrics.get('hit_ratio', 0.0):.6f}"
        metadata = run_record.backtest_result.metadata
        factor_statuses = metadata.get("factor_regression_statuses", {})
        if isinstance(factor_statuses, dict):
            ff3_status = str(factor_statuses.get("ff3", "n/a"))
            carhart4_status = str(factor_statuses.get("carhart4", "n/a"))
        chart_paths = metadata.get("chart_paths", {})
        if isinstance(chart_paths, dict):
            chart_count = str(len(chart_paths))
        signal_ic_summary = metadata.get("signal_ic_summary", {})
        if isinstance(signal_ic_summary, dict):
            cumulative_rank_ic = f"{signal_ic_summary.get('final_cumulative_rank_ic', 0.0):.6f}"

    return "\n".join(
        [
            "# Research Run",
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
            f"- Annual Return: `{annual_return}`",
            f"- Annual Risk: `{annual_risk}`",
            f"- Return/Risk: `{return_risk_ratio}`",
            f"- Max Drawdown: `{max_drawdown}`",
            f"- Average Turnover: `{average_turnover}`",
            f"- Hit Ratio: `{hit_ratio}`",
            f"- FF3 Regression: `{ff3_status}`",
            f"- Carhart4 Regression: `{carhart4_status}`",
            f"- Charts Saved: `{chart_count}`",
            f"- Final Cumulative Rank IC: `{cumulative_rank_ic}`",
            "",
            (
                "This run confirms that typed config loading, market data ingestion, "
                "trading-day alignment, strategy artifact output, the shared daily "
                "backtest layer, factor regressions, and chart generation are wired end-to-end."
            ),
            "",
            "## Next Steps",
            "",
            "1. Add transaction-cost and slippage assumptions to the return series.",
            "2. Compare multiple runs side-by-side from the CLI.",
            "3. Feed the standardized metrics and charts into the LLM evaluation loop.",
            "",
        ]
    )


def run_experiment(
    config_path: Path,
    output_root: Path | None = None,
    *,
    strategy_name: str | None = None,
    strategy_params_file: Path | None = None,
) -> Path:
    config = load_experiment_config(config_path)
    config = _apply_strategy_override(
        config,
        strategy_name=strategy_name,
        strategy_params_file=strategy_params_file,
    )
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
    backtest_result = None
    if config.dataset is not None:
        prepared_dataset = prepare_aligned_research_dataset(
            config.dataset,
            backtest=config.backtest,
            output_dir=run_dir,
            logger=logger,
        )
        data_alignment_summary = prepared_dataset.summary()
        strategy = create_strategy(config.strategy)
        prepared_strategy = strategy.generate(prepared_dataset)
        strategy_artifact = strategy.write_artifacts(
            prepared_strategy,
            output_dir=run_dir,
        )
        backtest_result = run_daily_backtest(
            strategy_name=strategy.name,
            dataset=prepared_dataset,
            portfolio=prepared_strategy.portfolio,
            backtest=config.backtest,
            output_dir=run_dir,
        )
        backtest_result = run_backtest_evaluation(
            strategy_name=strategy.name,
            dataset=prepared_dataset,
            signals=prepared_strategy.signals,
            backtest_result=backtest_result,
            factor_path=config.dataset.factor_returns if config.dataset else None,
            output_dir=run_dir,
        )
        strategy_artifacts = (strategy_artifact,)

    run_record = RunRecord(
        run_id=run_dir.name,
        experiment_id=config.experiment_id,
        environment_name=config.environment.name,
        strategy_name=config.strategy.name,
        config_path=config.config_path,
        output_dir=run_dir,
        status=RunStatus.SUCCEEDED,
        started_at_utc=started_at_utc,
        finished_at_utc=datetime.now(timezone.utc),
        notes=config.notes,
        message=(
            "Aligned data ingestion, standardized strategy artifacts, daily "
            "backtest metrics, factor regressions, and chart artifacts completed "
            "successfully."
        ),
        strategy_artifacts=strategy_artifacts,
        backtest_result=backtest_result,
        metadata={
            "environment_config": config.environment.to_dict(),
            "strategy_config": config.strategy.to_dict(),
            "dataset_config": config.dataset.to_dict() if config.dataset else None,
            "backtest_config": config.backtest.to_dict(),
            "cli_overrides": {
                "strategy_name": strategy_name,
                "strategy_params_file": (
                    str(strategy_params_file.resolve()) if strategy_params_file else None
                ),
            },
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


def create_scaffold_run(config_path: Path, output_root: Path | None = None) -> Path:
    return run_experiment(config_path=config_path, output_root=output_root)
