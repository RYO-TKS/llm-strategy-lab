"""Strategy interfaces and baseline implementations."""

from __future__ import annotations

import csv
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from .constants import (
    DATE,
    GROSS_EXPOSURE,
    MARKET,
    NET_EXPOSURE,
    RANK,
    SCORE,
    SECTOR,
    SIDE,
    SIGNAL,
    WEIGHT,
)
from .data_pipeline import JP_MARKET, PreparedResearchDataset
from .models import JsonDict, StrategyArtifact, StrategyConfig

SIGNAL_EXPORT_COLUMNS = (
    DATE,
    MARKET,
    SECTOR,
    SIGNAL,
    SCORE,
    RANK,
    "lookback_start",
    "lookback_end",
    "window_size",
)
PORTFOLIO_EXPORT_COLUMNS = (
    DATE,
    MARKET,
    SECTOR,
    SIDE,
    WEIGHT,
    SCORE,
    RANK,
    GROSS_EXPOSURE,
    NET_EXPOSURE,
)


@dataclass(frozen=True)
class SignalRecord:
    signal_date: date
    market: str
    sector: str
    signal: int
    score: float
    rank: int
    lookback_start: date
    lookback_end: date
    window_size: int

    def to_dict(self) -> JsonDict:
        return {
            DATE: self.signal_date.isoformat(),
            MARKET: self.market,
            SECTOR: self.sector,
            SIGNAL: self.signal,
            SCORE: self.score,
            RANK: self.rank,
            "lookback_start": self.lookback_start.isoformat(),
            "lookback_end": self.lookback_end.isoformat(),
            "window_size": self.window_size,
        }


@dataclass(frozen=True)
class PortfolioRecord:
    signal_date: date
    market: str
    sector: str
    side: str
    weight: float
    score: float
    rank: int
    gross_exposure: float
    net_exposure: float

    def to_dict(self) -> JsonDict:
        return {
            DATE: self.signal_date.isoformat(),
            MARKET: self.market,
            SECTOR: self.sector,
            SIDE: self.side,
            WEIGHT: self.weight,
            SCORE: self.score,
            RANK: self.rank,
            GROSS_EXPOSURE: self.gross_exposure,
            NET_EXPOSURE: self.net_exposure,
        }


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[JsonDict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _resolve_quantile(params: Mapping[str, object]) -> float:
    raw_q = params.get("q", params.get("quantile", 0.3))
    try:
        q = float(raw_q)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"q must be numeric, got {raw_q!r}") from exc

    if q <= 0 or q >= 0.5:
        raise ValueError(f"q must satisfy 0 < q < 0.5, got {q}")
    return q


def _resolve_rolling_window(params: Mapping[str, object]) -> int:
    raw_window = params.get("rolling_window", 1)
    try:
        rolling_window = int(raw_window)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"rolling_window must be an integer, got {raw_window!r}"
        ) from exc

    if rolling_window <= 0:
        raise ValueError(
            f"rolling_window must be greater than zero, got {rolling_window}"
        )
    return rolling_window


def _compounded_return(values: Sequence[float]) -> float:
    compounded = 1.0
    for value in values:
        compounded *= 1.0 + value
    return compounded - 1.0


class Strategy(ABC):
    """Common interface for strategy signal generation and portfolio construction."""

    def __init__(self, config: StrategyConfig) -> None:
        self.config = config

    @property
    def name(self) -> str:
        return self.config.name.lower()

    @abstractmethod
    def compute_signal(self, dataset: PreparedResearchDataset) -> Tuple[SignalRecord, ...]:
        """Compute cross-sectional signals aligned to the tradable JP signal dates."""

    @abstractmethod
    def build_portfolio(
        self,
        signals: Sequence[SignalRecord],
    ) -> Tuple[PortfolioRecord, ...]:
        """Convert cross-sectional signals into standardized long/short weights."""

    @abstractmethod
    def explain(
        self,
        *,
        dataset: PreparedResearchDataset,
        signals: Sequence[SignalRecord],
        portfolio: Sequence[PortfolioRecord],
    ) -> JsonDict:
        """Describe signal definition, portfolio rule, and parameterization."""

    def run(
        self,
        dataset: PreparedResearchDataset,
        *,
        output_dir: Path,
    ) -> StrategyArtifact:
        output_dir.mkdir(parents=True, exist_ok=True)
        signals = self.compute_signal(dataset)
        portfolio = self.build_portfolio(signals)
        explanation = self.explain(
            dataset=dataset,
            signals=signals,
            portfolio=portfolio,
        )

        artifact_paths = {
            "signals": output_dir / f"{self.name}_signals.csv",
            "portfolio": output_dir / f"{self.name}_portfolio.csv",
            "explanation": output_dir / f"{self.name}_explanation.json",
        }
        _write_csv(
            artifact_paths["signals"],
            SIGNAL_EXPORT_COLUMNS,
            (row.to_dict() for row in signals),
        )
        _write_csv(
            artifact_paths["portfolio"],
            PORTFOLIO_EXPORT_COLUMNS,
            (row.to_dict() for row in portfolio),
        )
        artifact_paths["explanation"].write_text(
            json.dumps(explanation, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        signal_dates = sorted({row.signal_date for row in signals})
        portfolio_dates = sorted({row.signal_date for row in portfolio})
        metadata = {
            "signal_row_count": len(signals),
            "portfolio_row_count": len(portfolio),
            "signal_dates": [value.isoformat() for value in signal_dates],
            "portfolio_dates": [value.isoformat() for value in portfolio_dates],
            "backtest_ready": True,
            "explanation": explanation,
        }
        return StrategyArtifact(
            strategy_name=self.name,
            generated_at_utc=datetime.now(timezone.utc),
            signal_columns=SIGNAL_EXPORT_COLUMNS,
            parameter_snapshot=dict(self.config.params),
            artifact_paths=artifact_paths,
            explanation=explanation.get("summary"),
            metadata=metadata,
        )


class MomentumStrategy(Strategy):
    """Simple JP-sector momentum baseline using aligned research dates."""

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        self.q = _resolve_quantile(config.params)
        self.rolling_window = _resolve_rolling_window(config.params)

    def compute_signal(self, dataset: PreparedResearchDataset) -> Tuple[SignalRecord, ...]:
        returns_by_date_sector: Dict[Tuple[date, str], float] = {
            (row.signal_date, row.sector): row.return_value
            for row in dataset.jp_open_to_close_returns
        }
        signal_dates = sorted({row.signal_date for row in dataset.jp_open_to_close_returns})
        signal_records: List[SignalRecord] = []

        for current_index in range(self.rolling_window, len(signal_dates)):
            signal_date = signal_dates[current_index]
            lookback_dates = signal_dates[current_index - self.rolling_window : current_index]
            scored_rows: List[Tuple[str, float]] = []

            for sector in dataset.jp_sectors:
                history = [
                    returns_by_date_sector.get((history_date, sector))
                    for history_date in lookback_dates
                ]
                if any(value is None for value in history):
                    continue

                score = _compounded_return(
                    [value for value in history if value is not None]
                )
                scored_rows.append((sector, score))

            if len(scored_rows) < 2:
                continue

            ordered_scores = sorted(
                scored_rows,
                key=lambda item: (-item[1], item[0]),
            )
            bucket_size = min(
                max(1, int(len(ordered_scores) * self.q)),
                len(ordered_scores) // 2,
            )
            if bucket_size == 0:
                continue

            long_sectors = {sector for sector, _score in ordered_scores[:bucket_size]}
            short_sectors = {sector for sector, _score in ordered_scores[-bucket_size:]}

            for rank, (sector, score) in enumerate(ordered_scores, start=1):
                signal_value = 0
                if sector in long_sectors:
                    signal_value = 1
                elif sector in short_sectors:
                    signal_value = -1

                signal_records.append(
                    SignalRecord(
                        signal_date=signal_date,
                        market=JP_MARKET,
                        sector=sector,
                        signal=signal_value,
                        score=score,
                        rank=rank,
                        lookback_start=lookback_dates[0],
                        lookback_end=lookback_dates[-1],
                        window_size=self.rolling_window,
                    )
                )

        return tuple(signal_records)

    def build_portfolio(
        self,
        signals: Sequence[SignalRecord],
    ) -> Tuple[PortfolioRecord, ...]:
        grouped: Dict[date, List[SignalRecord]] = {}
        for signal_row in signals:
            grouped.setdefault(signal_row.signal_date, []).append(signal_row)

        portfolio_rows: List[PortfolioRecord] = []
        for signal_date in sorted(grouped):
            daily_rows = grouped[signal_date]
            long_rows = [row for row in daily_rows if row.signal > 0]
            short_rows = [row for row in daily_rows if row.signal < 0]
            if not long_rows or not short_rows:
                continue

            long_weight = 0.5 / len(long_rows)
            short_weight = -0.5 / len(short_rows)
            gross_exposure = (abs(long_weight) * len(long_rows)) + (
                abs(short_weight) * len(short_rows)
            )
            net_exposure = (long_weight * len(long_rows)) + (
                short_weight * len(short_rows)
            )

            for row in long_rows:
                portfolio_rows.append(
                    PortfolioRecord(
                        signal_date=signal_date,
                        market=row.market,
                        sector=row.sector,
                        side="long",
                        weight=long_weight,
                        score=row.score,
                        rank=row.rank,
                        gross_exposure=gross_exposure,
                        net_exposure=net_exposure,
                    )
                )
            for row in short_rows:
                portfolio_rows.append(
                    PortfolioRecord(
                        signal_date=signal_date,
                        market=row.market,
                        sector=row.sector,
                        side="short",
                        weight=short_weight,
                        score=row.score,
                        rank=row.rank,
                        gross_exposure=gross_exposure,
                        net_exposure=net_exposure,
                    )
                )

        return tuple(portfolio_rows)

    def explain(
        self,
        *,
        dataset: PreparedResearchDataset,
        signals: Sequence[SignalRecord],
        portfolio: Sequence[PortfolioRecord],
    ) -> JsonDict:
        signal_dates = sorted({row.signal_date for row in signals})
        portfolio_dates = sorted({row.signal_date for row in portfolio})
        selected_per_side = min(
            max(1, int(len(dataset.jp_sectors) * self.q)),
            len(dataset.jp_sectors) // 2,
        )
        return {
            "strategy_name": self.name,
            "summary": (
                "Use trailing compounded JP open-to-close returns over the prior "
                f"{self.rolling_window} aligned signal dates, then build an equal-weight "
                "long-short portfolio from the top and bottom quantiles."
            ),
            "signal_definition": (
                "Cross-sectional momentum score is the compounded return of each JP sector "
                "over the trailing aligned signal-date window, excluding the current date."
            ),
            "portfolio_construction": (
                "Select the top and bottom q quantiles by momentum score and assign 50/50 "
                "gross exposure across long and short buckets with equal weights within each side."
            ),
            "parameters": {
                "q": self.q,
                "rolling_window": self.rolling_window,
            },
            "input_market": JP_MARKET,
            "universe_size": len(dataset.jp_sectors),
            "eligible_signal_dates": len(signal_dates),
            "backtest_portfolio_dates": len(portfolio_dates),
            "selected_per_side": selected_per_side,
            "signal_dates": [value.isoformat() for value in signal_dates],
            "portfolio_dates": [value.isoformat() for value in portfolio_dates],
        }


def create_strategy(config: StrategyConfig) -> Strategy:
    strategy_name = config.name.strip().lower()
    if strategy_name == "mom":
        return MomentumStrategy(config)
    raise ValueError(f"Unsupported strategy: {config.name}")
