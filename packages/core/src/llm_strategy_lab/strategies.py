"""Strategy interfaces and baseline implementations."""

from __future__ import annotations

import csv
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

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
from .data_pipeline import JP_MARKET, AlignedMarketReturn, PreparedResearchDataset
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
FeatureSpec = Tuple[str, str]


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


def _resolve_component_count(params: Mapping[str, object]) -> int:
    raw_components = params.get("components", 1)
    try:
        components = int(raw_components)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"components must be an integer, got {raw_components!r}") from exc

    if components <= 0:
        raise ValueError(f"components must be greater than zero, got {components}")
    return components


def _compounded_return(values: Sequence[float]) -> float:
    compounded = 1.0
    for value in values:
        compounded *= 1.0 + value
    return compounded - 1.0


def _selected_per_side(universe_size: int, q: float) -> int:
    return min(
        max(1, int(universe_size * q)),
        universe_size // 2,
    )


def _build_market_lookup(
    rows: Sequence[AlignedMarketReturn],
) -> Dict[date, Dict[str, float]]:
    lookup: Dict[date, Dict[str, float]] = {}
    for row in rows:
        lookup.setdefault(row.signal_date, {})[row.sector] = row.return_value
    return lookup


def _rounded_matrix(values: np.ndarray) -> List[List[float]]:
    return [
        [round(float(item), 6) for item in row]
        for row in values.tolist()
    ]


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


class QuantileLongShortStrategy(Strategy):
    """Shared quantile long-short portfolio construction for ranked JP signals."""

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        self.q = _resolve_quantile(config.params)
        self.rolling_window = _resolve_rolling_window(config.params)

    def _build_ranked_signal_records(
        self,
        *,
        signal_date: date,
        market: str,
        lookback_dates: Sequence[date],
        scored_rows: Sequence[Tuple[str, float]],
    ) -> Tuple[SignalRecord, ...]:
        if not lookback_dates:
            return ()

        ordered_scores = sorted(
            scored_rows,
            key=lambda item: (-item[1], item[0]),
        )
        bucket_size = _selected_per_side(len(ordered_scores), self.q)
        if bucket_size == 0:
            return ()

        long_sectors = {sector for sector, _score in ordered_scores[:bucket_size]}
        short_sectors = {sector for sector, _score in ordered_scores[-bucket_size:]}
        signal_records: List[SignalRecord] = []

        for rank, (sector, score) in enumerate(ordered_scores, start=1):
            signal_value = 0
            if sector in long_sectors:
                signal_value = 1
            elif sector in short_sectors:
                signal_value = -1

            signal_records.append(
                SignalRecord(
                    signal_date=signal_date,
                    market=market,
                    sector=sector,
                    signal=signal_value,
                    score=score,
                    rank=rank,
                    lookback_start=lookback_dates[0],
                    lookback_end=lookback_dates[-1],
                    window_size=len(lookback_dates),
                )
            )

        return tuple(signal_records)

    def compute_signal(self, dataset: PreparedResearchDataset) -> Tuple[SignalRecord, ...]:
        raise NotImplementedError

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


class MomentumStrategy(QuantileLongShortStrategy):
    """Simple JP-sector momentum baseline using aligned research dates."""

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

            signal_records.extend(
                self._build_ranked_signal_records(
                    signal_date=signal_date,
                    market=JP_MARKET,
                    lookback_dates=lookback_dates,
                    scored_rows=scored_rows,
                )
            )

        return tuple(signal_records)

    def explain(
        self,
        *,
        dataset: PreparedResearchDataset,
        signals: Sequence[SignalRecord],
        portfolio: Sequence[PortfolioRecord],
    ) -> JsonDict:
        signal_dates = sorted({row.signal_date for row in signals})
        portfolio_dates = sorted({row.signal_date for row in portfolio})
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
            "selected_per_side": _selected_per_side(len(dataset.jp_sectors), self.q),
            "signal_dates": [value.isoformat() for value in signal_dates],
            "portfolio_dates": [value.isoformat() for value in portfolio_dates],
        }


class PlainPCAStrategy(QuantileLongShortStrategy):
    """Joint JP/US PCA baseline that restores JP scores from current US shocks."""

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        self.components = _resolve_component_count(config.params)
        self._diagnostics_by_date: Dict[date, JsonDict] = {}

    def _build_joint_history(
        self,
        *,
        signal_date: date,
        lookback_dates: Sequence[date],
        us_by_date: Mapping[date, Mapping[str, float]],
        jp_by_date: Mapping[date, Mapping[str, float]],
        us_sectors: Sequence[str],
        jp_sectors: Sequence[str],
    ) -> Optional[JsonDict]:
        feature_specs: Tuple[FeatureSpec, ...] = tuple(
            [("US", sector) for sector in us_sectors]
            + [("JP", sector) for sector in jp_sectors]
        )
        history_rows: List[List[float]] = []
        current_us = us_by_date.get(signal_date)
        if current_us is None:
            return None

        for lookback_date in lookback_dates:
            us_row = us_by_date.get(lookback_date)
            jp_row = jp_by_date.get(lookback_date)
            if us_row is None or jp_row is None:
                return None

            joint_row: List[float] = []
            for market, sector in feature_specs:
                source = us_row if market == "US" else jp_row
                value = source.get(sector)
                if value is None:
                    return None
                joint_row.append(value)
            history_rows.append(joint_row)

        history_matrix = np.asarray(history_rows, dtype=float)
        if history_matrix.shape[0] < 2:
            return None

        means = np.mean(history_matrix, axis=0)
        stds = np.std(history_matrix, axis=0, ddof=1)
        active_mask = stds > 1e-12
        active_specs = [spec for spec, keep in zip(feature_specs, active_mask) if keep]
        if len(active_specs) < 2:
            return None

        active_means = means[active_mask]
        active_stds = stds[active_mask]
        standardized_history = (history_matrix[:, active_mask] - active_means) / active_stds
        correlation_matrix = np.corrcoef(standardized_history, rowvar=False)
        if correlation_matrix.ndim != 2 or not np.all(np.isfinite(correlation_matrix)):
            return None

        eigenvalues, eigenvectors = np.linalg.eigh(correlation_matrix)
        sorted_indices = [
            index
            for index in np.argsort(eigenvalues)[::-1].tolist()
            if float(eigenvalues[index]) > 1e-12
        ]
        top_indices = sorted_indices[: self.components]
        if not top_indices:
            return None

        us_indices = [
            index
            for index, (market, _sector) in enumerate(active_specs)
            if market == "US"
        ]
        jp_indices = [
            index
            for index, (market, _sector) in enumerate(active_specs)
            if market == "JP"
        ]
        if not us_indices or not jp_indices:
            return None

        current_us_values: List[float] = []
        for index in us_indices:
            _market, sector = active_specs[index]
            current_value = current_us.get(sector)
            if current_value is None:
                return None
            current_us_values.append(
                (current_value - float(active_means[index])) / float(active_stds[index])
            )

        current_us_vector = np.asarray(current_us_values, dtype=float)
        jp_scores = {sector: 0.0 for sector in jp_sectors}
        component_details: List[JsonDict] = []

        for component_rank, eigen_index in enumerate(top_indices, start=1):
            eigenvalue = float(eigenvalues[eigen_index])
            component_vector = eigenvectors[:, eigen_index]
            us_loadings = component_vector[us_indices]
            jp_loadings = component_vector[jp_indices]
            us_energy = float(np.dot(us_loadings, us_loadings))
            if us_energy <= 1e-12:
                continue

            factor_score = float(np.dot(current_us_vector, us_loadings) / us_energy)
            for sector_name, loading in zip(
                [active_specs[index][1] for index in jp_indices],
                jp_loadings.tolist(),
            ):
                jp_scores[sector_name] += eigenvalue * factor_score * float(loading)

            component_details.append(
                {
                    "rank": component_rank,
                    "eigenvalue": eigenvalue,
                    "factor_score": factor_score,
                }
            )

        return {
            "signal_date": signal_date.isoformat(),
            "lookback_start": lookback_dates[0].isoformat(),
            "lookback_end": lookback_dates[-1].isoformat(),
            "joint_feature_labels": [
                f"{market}:{sector}"
                for market, sector in active_specs
            ],
            "joint_correlation_matrix": _rounded_matrix(correlation_matrix),
            "top_components": component_details,
            "jp_scores": {
                sector: round(float(score), 10)
                for sector, score in jp_scores.items()
            },
        }

    def compute_signal(self, dataset: PreparedResearchDataset) -> Tuple[SignalRecord, ...]:
        us_by_date = _build_market_lookup(dataset.us_aligned_returns)
        jp_by_date = _build_market_lookup(dataset.jp_open_to_close_returns)
        signal_dates = sorted(set(us_by_date) & set(jp_by_date))
        signal_records: List[SignalRecord] = []
        self._diagnostics_by_date = {}

        for current_index in range(self.rolling_window, len(signal_dates)):
            signal_date = signal_dates[current_index]
            lookback_dates = signal_dates[current_index - self.rolling_window : current_index]
            diagnostic = self._build_joint_history(
                signal_date=signal_date,
                lookback_dates=lookback_dates,
                us_by_date=us_by_date,
                jp_by_date=jp_by_date,
                us_sectors=dataset.us_sectors,
                jp_sectors=dataset.jp_sectors,
            )
            if diagnostic is None:
                continue

            self._diagnostics_by_date[signal_date] = diagnostic
            scored_rows = [
                (sector, float(diagnostic["jp_scores"].get(sector, 0.0)))
                for sector in dataset.jp_sectors
            ]
            signal_records.extend(
                self._build_ranked_signal_records(
                    signal_date=signal_date,
                    market=JP_MARKET,
                    lookback_dates=lookback_dates,
                    scored_rows=scored_rows,
                )
            )

        return tuple(signal_records)

    def explain(
        self,
        *,
        dataset: PreparedResearchDataset,
        signals: Sequence[SignalRecord],
        portfolio: Sequence[PortfolioRecord],
    ) -> JsonDict:
        signal_dates = sorted({row.signal_date for row in signals})
        portfolio_dates = sorted({row.signal_date for row in portfolio})
        latest_date = signal_dates[-1] if signal_dates else None
        latest_diagnostic = (
            self._diagnostics_by_date.get(latest_date) if latest_date else None
        )
        return {
            "strategy_name": self.name,
            "summary": (
                "Build a trailing joint JP/US correlation matrix, estimate plain PCA on that "
                "matrix, project the current US shock onto the retained components, and "
                "restore JP sector scores from the JP loadings."
            ),
            "signal_definition": (
                "The score for each JP sector is the PCA-based reconstruction implied by the "
                "current standardized US aligned return vector and the JP slice of the retained "
                "joint principal components."
            ),
            "portfolio_construction": (
                "Select the top and bottom q quantiles of restored JP scores and assign 50/50 "
                "gross exposure across long and short buckets with equal weights within each side."
            ),
            "parameters": {
                "q": self.q,
                "rolling_window": self.rolling_window,
                "components": self.components,
            },
            "input_markets": ["US", "JP"],
            "us_universe_size": len(dataset.us_sectors),
            "jp_universe_size": len(dataset.jp_sectors),
            "eligible_signal_dates": len(signal_dates),
            "backtest_portfolio_dates": len(portfolio_dates),
            "selected_per_side": _selected_per_side(len(dataset.jp_sectors), self.q),
            "signal_dates": [value.isoformat() for value in signal_dates],
            "portfolio_dates": [value.isoformat() for value in portfolio_dates],
            "latest_diagnostic": latest_diagnostic,
            "comparison_ready_with": "mom",
        }


def create_strategy(config: StrategyConfig) -> Strategy:
    strategy_name = config.name.strip().lower()
    if strategy_name == "mom":
        return MomentumStrategy(config)
    if strategy_name == "pca_plain":
        return PlainPCAStrategy(config)
    raise ValueError(f"Unsupported strategy: {config.name}")
