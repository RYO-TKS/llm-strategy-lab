"""Common daily backtest engine and evaluation metrics."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np

from .constants import (
    BACKTEST_SERIES_COLUMNS,
    DATE,
    DRAWDOWN,
    EQUITY_CURVE,
    GROSS_EXPOSURE,
    HIT_RATIO,
    MARKET,
    NET_EXPOSURE,
    RETURN,
    SECTOR,
    SIDE,
    TURNOVER,
    WEIGHT,
)
from .data_pipeline import PreparedResearchDataset
from .models import BacktestConfig, BacktestResult, JsonDict
from .strategies import PortfolioRecord

TRADING_DAYS_PER_YEAR = 252.0
POSITION_EXPORT_COLUMNS = (
    DATE,
    MARKET,
    SECTOR,
    SIDE,
    WEIGHT,
    RETURN,
    "pnl_contribution",
)


@dataclass(frozen=True)
class BacktestSeriesRecord:
    signal_date: date
    daily_return: float
    equity_curve: float
    drawdown: float
    turnover: float
    hit_ratio: float
    gross_exposure: float
    net_exposure: float

    def to_dict(self) -> JsonDict:
        return {
            DATE: self.signal_date.isoformat(),
            RETURN: self.daily_return,
            EQUITY_CURVE: self.equity_curve,
            DRAWDOWN: self.drawdown,
            TURNOVER: self.turnover,
            HIT_RATIO: self.hit_ratio,
            GROSS_EXPOSURE: self.gross_exposure,
            NET_EXPOSURE: self.net_exposure,
        }


@dataclass(frozen=True)
class PositionContributionRecord:
    signal_date: date
    market: str
    sector: str
    side: str
    weight: float
    return_value: float
    pnl_contribution: float

    def to_dict(self) -> JsonDict:
        return {
            DATE: self.signal_date.isoformat(),
            MARKET: self.market,
            SECTOR: self.sector,
            SIDE: self.side,
            WEIGHT: self.weight,
            RETURN: self.return_value,
            "pnl_contribution": self.pnl_contribution,
        }


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[JsonDict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _build_returns_lookup(
    dataset: PreparedResearchDataset,
) -> Dict[date, Dict[str, float]]:
    lookup: Dict[date, Dict[str, float]] = {}
    for row in dataset.jp_open_to_close_returns:
        lookup.setdefault(row.signal_date, {})[row.sector] = row.return_value
    return lookup


def _within_backtest_window(
    signal_date: date,
    backtest: BacktestConfig,
) -> bool:
    if backtest.start and signal_date < backtest.start:
        return False
    if backtest.end and signal_date > backtest.end:
        return False
    return True


def _group_portfolio_by_date(
    portfolio: Sequence[PortfolioRecord],
    *,
    backtest: BacktestConfig,
) -> Dict[date, List[PortfolioRecord]]:
    grouped: Dict[date, List[PortfolioRecord]] = {}
    for row in portfolio:
        if _within_backtest_window(row.signal_date, backtest):
            grouped.setdefault(row.signal_date, []).append(row)
    return grouped


def _compute_turnover(
    current_weights: Mapping[str, float],
    previous_weights: Mapping[str, float],
) -> float:
    sectors = set(current_weights) | set(previous_weights)
    return sum(
        abs(current_weights.get(sector, 0.0) - previous_weights.get(sector, 0.0))
        for sector in sectors
    )


def _annualized_return(equity_curve: float, observation_count: int) -> float:
    if observation_count <= 0:
        return 0.0
    if equity_curve <= 0:
        return -1.0
    return float(equity_curve ** (TRADING_DAYS_PER_YEAR / observation_count) - 1.0)


def _compute_metrics(
    series_rows: Sequence[BacktestSeriesRecord],
    *,
    winning_positions: int,
    observed_positions: int,
) -> Mapping[str, float]:
    if not series_rows:
        return {
            "annual_return": 0.0,
            "annual_risk": 0.0,
            "return_risk_ratio": 0.0,
            "max_drawdown": 0.0,
            "average_turnover": 0.0,
            "hit_ratio": 0.0,
            "average_gross_exposure": 0.0,
            "average_net_exposure": 0.0,
            "cumulative_return": 0.0,
        }

    returns = np.asarray([row.daily_return for row in series_rows], dtype=float)
    turnovers = np.asarray([row.turnover for row in series_rows], dtype=float)
    gross_exposures = np.asarray([row.gross_exposure for row in series_rows], dtype=float)
    net_exposures = np.asarray([row.net_exposure for row in series_rows], dtype=float)
    drawdowns = np.asarray([row.drawdown for row in series_rows], dtype=float)
    final_equity_curve = float(series_rows[-1].equity_curve)

    annual_risk = float(returns.std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR))
    annual_return = _annualized_return(final_equity_curve, len(series_rows))
    return {
        "annual_return": annual_return,
        "annual_risk": annual_risk,
        "return_risk_ratio": (
            annual_return / annual_risk if annual_risk > 0 else 0.0
        ),
        "max_drawdown": abs(float(drawdowns.min())),
        "average_turnover": float(turnovers.mean()),
        "hit_ratio": (
            winning_positions / observed_positions if observed_positions > 0 else 0.0
        ),
        "average_gross_exposure": float(gross_exposures.mean()),
        "average_net_exposure": float(net_exposures.mean()),
        "cumulative_return": final_equity_curve - 1.0,
    }


def run_daily_backtest(
    *,
    strategy_name: str,
    dataset: PreparedResearchDataset,
    portfolio: Sequence[PortfolioRecord],
    backtest: BacktestConfig,
    output_dir: Path,
) -> BacktestResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped_portfolio = _group_portfolio_by_date(portfolio, backtest=backtest)
    returns_lookup = _build_returns_lookup(dataset)

    series_rows: List[BacktestSeriesRecord] = []
    position_rows: List[PositionContributionRecord] = []
    previous_weights: Dict[str, float] = {}
    equity_curve = 1.0
    running_peak = 1.0
    observed_positions = 0
    winning_positions = 0
    missing_return_count = 0

    for signal_date in sorted(grouped_portfolio):
        daily_portfolio = grouped_portfolio[signal_date]
        daily_weights = {
            row.sector: row.weight
            for row in daily_portfolio
        }
        daily_returns = returns_lookup.get(signal_date, {})
        turnover = _compute_turnover(daily_weights, previous_weights)
        gross_exposure = sum(abs(weight) for weight in daily_weights.values())
        net_exposure = sum(daily_weights.values())
        daily_pnl = 0.0
        daily_observed_positions = 0
        daily_winners = 0

        for row in daily_portfolio:
            sector_return = daily_returns.get(row.sector)
            if sector_return is None:
                missing_return_count += 1
                continue

            pnl_contribution = row.weight * sector_return
            daily_pnl += pnl_contribution
            daily_observed_positions += 1
            if pnl_contribution > 0:
                daily_winners += 1
            position_rows.append(
                PositionContributionRecord(
                    signal_date=signal_date,
                    market=row.market,
                    sector=row.sector,
                    side=row.side,
                    weight=row.weight,
                    return_value=sector_return,
                    pnl_contribution=pnl_contribution,
                )
            )

        observed_positions += daily_observed_positions
        winning_positions += daily_winners
        equity_curve *= 1.0 + daily_pnl
        running_peak = max(running_peak, equity_curve)
        drawdown = (equity_curve / running_peak) - 1.0 if running_peak > 0 else 0.0
        series_rows.append(
            BacktestSeriesRecord(
                signal_date=signal_date,
                daily_return=daily_pnl,
                equity_curve=equity_curve,
                drawdown=drawdown,
                turnover=turnover,
                hit_ratio=(
                    daily_winners / daily_observed_positions
                    if daily_observed_positions > 0
                    else 0.0
                ),
                gross_exposure=gross_exposure,
                net_exposure=net_exposure,
            )
        )
        previous_weights = daily_weights

    metrics = _compute_metrics(
        series_rows,
        winning_positions=winning_positions,
        observed_positions=observed_positions,
    )
    daily_path = output_dir / f"{strategy_name}_backtest_daily.csv"
    positions_path = output_dir / f"{strategy_name}_backtest_positions.csv"
    metrics_path = output_dir / f"{strategy_name}_backtest_metrics.json"

    _write_csv(
        daily_path,
        BACKTEST_SERIES_COLUMNS,
        (row.to_dict() for row in series_rows),
    )
    _write_csv(
        positions_path,
        POSITION_EXPORT_COLUMNS,
        (row.to_dict() for row in position_rows),
    )

    metrics_payload = {
        "strategy_name": strategy_name,
        "metrics": dict(metrics),
        "portfolio_day_count": len(series_rows),
        "position_observation_count": observed_positions,
        "winning_position_count": winning_positions,
        "missing_return_count": missing_return_count,
        "turnover_definition": "sum(abs(weight_t - weight_t-1))",
        "annualization_factor": int(TRADING_DAYS_PER_YEAR),
        "series_paths": {
            "daily": str(daily_path),
            "positions": str(positions_path),
        },
    }
    metrics_path.write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return BacktestResult(
        strategy_name=strategy_name,
        metrics=metrics,
        series_paths={
            "daily": daily_path,
            "positions": positions_path,
        },
        gross_exposure=metrics["average_gross_exposure"],
        net_exposure=metrics["average_net_exposure"],
        metadata={
            "metrics_path": str(metrics_path),
            "portfolio_day_count": len(series_rows),
            "position_observation_count": observed_positions,
            "winning_position_count": winning_positions,
            "missing_return_count": missing_return_count,
            "turnover_definition": "sum(abs(weight_t - weight_t-1))",
            "annualization_factor": int(TRADING_DAYS_PER_YEAR),
        },
    )
