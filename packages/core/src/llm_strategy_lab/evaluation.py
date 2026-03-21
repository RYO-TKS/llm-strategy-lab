"""Factor regressions, IC analytics, and chart artifact generation."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from .constants import DATE, DRAWDOWN, EQUITY_CURVE, RETURN
from .data_pipeline import PreparedResearchDataset
from .models import BacktestResult, JsonDict
from .strategies import SignalRecord

IC_EXPORT_COLUMNS = (
    DATE,
    "rank_ic",
    "cumulative_rank_ic",
    "observation_count",
)
CARHART_REQUIRED_COLUMNS = ("date", "mkt_rf", "smb", "hml", "umd", "rf")


@dataclass(frozen=True)
class FactorReturnRow:
    signal_date: date
    market_excess_return: float
    smb: float
    hml: float
    rf: float
    umd: Optional[float] = None


@dataclass(frozen=True)
class DailyBacktestRow:
    signal_date: date
    daily_return: float
    equity_curve: float
    drawdown: float


@dataclass(frozen=True)
class SignalICRecord:
    signal_date: date
    rank_ic: float
    cumulative_rank_ic: float
    observation_count: int

    def to_dict(self) -> JsonDict:
        return {
            DATE: self.signal_date.isoformat(),
            "rank_ic": self.rank_ic,
            "cumulative_rank_ic": self.cumulative_rank_ic,
            "observation_count": self.observation_count,
        }


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[JsonDict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _ensure_columns(
    fieldnames: Optional[Sequence[str]],
    required: Sequence[str],
    *,
    path: Path,
) -> None:
    if fieldnames is None:
        raise ValueError(f"CSV header is missing: {path}")
    missing = [column for column in required if column not in fieldnames]
    if missing:
        raise ValueError(f"CSV is missing required columns {missing}: {path}")


def _parse_float(raw_value: str, *, field_name: str) -> float:
    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be numeric: {raw_value}") from exc


def load_factor_returns(path: Path) -> Mapping[date, FactorReturnRow]:
    rows: Dict[date, FactorReturnRow] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _ensure_columns(reader.fieldnames, CARHART_REQUIRED_COLUMNS, path=path)
        for row in reader:
            signal_date = date.fromisoformat(str(row["date"]))
            rows[signal_date] = FactorReturnRow(
                signal_date=signal_date,
                market_excess_return=_parse_float(str(row["mkt_rf"]), field_name="mkt_rf"),
                smb=_parse_float(str(row["smb"]), field_name="smb"),
                hml=_parse_float(str(row["hml"]), field_name="hml"),
                umd=_parse_float(str(row["umd"]), field_name="umd"),
                rf=_parse_float(str(row["rf"]), field_name="rf"),
            )
    return rows


def load_backtest_daily_rows(path: Path) -> List[DailyBacktestRow]:
    rows: List[DailyBacktestRow] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _ensure_columns(reader.fieldnames, (DATE, RETURN, EQUITY_CURVE, DRAWDOWN), path=path)
        for row in reader:
            signal_date = date.fromisoformat(str(row[DATE]))
            rows.append(
                DailyBacktestRow(
                    signal_date=signal_date,
                    daily_return=_parse_float(str(row[RETURN]), field_name=RETURN),
                    equity_curve=_parse_float(str(row[EQUITY_CURVE]), field_name=EQUITY_CURVE),
                    drawdown=_parse_float(str(row[DRAWDOWN]), field_name=DRAWDOWN),
                )
            )
    return rows


def _average_ranks(values: Sequence[float]) -> List[float]:
    indexed_values = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
    ranks = [0.0] * len(values)
    index = 0
    while index < len(indexed_values):
        next_index = index + 1
        while (
            next_index < len(indexed_values)
            and indexed_values[next_index][1] == indexed_values[index][1]
        ):
            next_index += 1
        average_rank = ((index + 1) + next_index) / 2.0
        for item_index in range(index, next_index):
            ranks[indexed_values[item_index][0]] = average_rank
        index = next_index
    return ranks


def _pearson_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    left_values = np.asarray(left, dtype=float)
    right_values = np.asarray(right, dtype=float)
    if left_values.size < 2 or right_values.size < 2:
        return 0.0
    left_std = float(left_values.std(ddof=0))
    right_std = float(right_values.std(ddof=0))
    if left_std == 0.0 or right_std == 0.0:
        return 0.0
    centered_left = left_values - left_values.mean()
    centered_right = right_values - right_values.mean()
    return float(np.mean(centered_left * centered_right) / (left_std * right_std))


def _spearman_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    return _pearson_correlation(_average_ranks(left), _average_ranks(right))


def build_signal_ic_series(
    *,
    signals: Sequence[SignalRecord],
    dataset: PreparedResearchDataset,
) -> List[SignalICRecord]:
    returns_lookup: Dict[date, Dict[str, float]] = {}
    for row in dataset.jp_open_to_close_returns:
        returns_lookup.setdefault(row.signal_date, {})[row.sector] = row.return_value

    grouped_signals: Dict[date, List[SignalRecord]] = {}
    for signal in signals:
        grouped_signals.setdefault(signal.signal_date, []).append(signal)

    cumulative_rank_ic = 0.0
    ic_rows: List[SignalICRecord] = []
    for signal_date in sorted(grouped_signals):
        scores: List[float] = []
        realized_returns: List[float] = []
        for signal in grouped_signals[signal_date]:
            realized_return = returns_lookup.get(signal_date, {}).get(signal.sector)
            if realized_return is None:
                continue
            scores.append(signal.score)
            realized_returns.append(realized_return)
        rank_ic = _spearman_correlation(scores, realized_returns)
        cumulative_rank_ic += rank_ic
        ic_rows.append(
            SignalICRecord(
                signal_date=signal_date,
                rank_ic=rank_ic,
                cumulative_rank_ic=cumulative_rank_ic,
                observation_count=len(scores),
            )
        )
    return ic_rows


def _fit_ols_regression(
    *,
    y_values: np.ndarray,
    x_values: np.ndarray,
    factor_names: Sequence[str],
) -> JsonDict:
    observation_count = int(y_values.shape[0])
    parameter_count = int(x_values.shape[1] + 1)
    if observation_count <= parameter_count:
        return {
            "status": "insufficient_observations",
            "observations": observation_count,
            "required_observations": parameter_count + 1,
            "factor_names": list(factor_names),
        }

    design_matrix = np.column_stack([np.ones(observation_count), x_values])
    coefficients, _residuals, _rank, _singular_values = np.linalg.lstsq(
        design_matrix,
        y_values,
        rcond=None,
    )
    fitted = design_matrix @ coefficients
    residuals = y_values - fitted
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y_values - y_values.mean()) ** 2))
    degrees_of_freedom = observation_count - design_matrix.shape[1]
    sigma_squared = ss_res / degrees_of_freedom if degrees_of_freedom > 0 else 0.0
    xtx_inverse = np.linalg.pinv(design_matrix.T @ design_matrix)
    standard_errors = np.sqrt(np.diag(sigma_squared * xtx_inverse))

    coefficient_names = ("alpha", *factor_names)
    coefficient_map = {
        name: float(value)
        for name, value in zip(coefficient_names, coefficients.tolist())
    }
    t_stats = {}
    for name, value, standard_error in zip(
        coefficient_names,
        coefficients.tolist(),
        standard_errors.tolist(),
    ):
        t_stats[name] = float(value / standard_error) if standard_error > 0 else 0.0

    return {
        "status": "ok",
        "observations": observation_count,
        "coefficients": coefficient_map,
        "t_stats": t_stats,
        "annualized_alpha": float(coefficient_map["alpha"] * 252.0),
        "r_squared": float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0,
        "residual_volatility": float(np.std(residuals, ddof=0)),
        "residual_volatility_annualized": float(np.std(residuals, ddof=0) * np.sqrt(252.0)),
    }


def build_factor_regressions(
    *,
    daily_rows: Sequence[DailyBacktestRow],
    factor_path: Optional[Path],
) -> JsonDict:
    payload: JsonDict = {
        "factor_source_path": str(factor_path) if factor_path is not None else None,
        "matched_dates": [],
        "unmatched_dates": [],
        "models": {},
    }
    if factor_path is None:
        payload["models"] = {
            "ff3": {"status": "missing_factor_data"},
            "carhart4": {"status": "missing_factor_data"},
        }
        return payload

    factor_rows = load_factor_returns(factor_path)
    aligned_dates = [
        row.signal_date
        for row in daily_rows
        if row.signal_date in factor_rows
    ]
    unmatched_dates = [
        row.signal_date
        for row in daily_rows
        if row.signal_date not in factor_rows
    ]
    payload["matched_dates"] = [value.isoformat() for value in aligned_dates]
    payload["unmatched_dates"] = [value.isoformat() for value in unmatched_dates]

    if not aligned_dates:
        payload["models"] = {
            "ff3": {"status": "no_aligned_dates"},
            "carhart4": {"status": "no_aligned_dates"},
        }
        return payload

    daily_returns_by_date = {
        row.signal_date: row.daily_return
        for row in daily_rows
    }
    strategy_returns = np.asarray(
        [
            daily_returns_by_date[signal_date]
            for signal_date in aligned_dates
        ],
        dtype=float,
    )
    risk_free = np.asarray(
        [
            factor_rows[signal_date].rf
            for signal_date in aligned_dates
        ],
        dtype=float,
    )
    excess_returns = strategy_returns - risk_free

    ff3_x = np.asarray(
        [
            [
                factor_rows[signal_date].market_excess_return,
                factor_rows[signal_date].smb,
                factor_rows[signal_date].hml,
            ]
            for signal_date in aligned_dates
        ],
        dtype=float,
    )
    carhart_x = np.asarray(
        [
            [
                factor_rows[signal_date].market_excess_return,
                factor_rows[signal_date].smb,
                factor_rows[signal_date].hml,
                factor_rows[signal_date].umd or 0.0,
            ]
            for signal_date in aligned_dates
        ],
        dtype=float,
    )

    payload["models"] = {
        "ff3": _fit_ols_regression(
            y_values=excess_returns,
            x_values=ff3_x,
            factor_names=("mkt_rf", "smb", "hml"),
        ),
        "carhart4": _fit_ols_regression(
            y_values=excess_returns,
            x_values=carhart_x,
            factor_names=("mkt_rf", "smb", "hml", "umd"),
        ),
    }
    return payload


def _series_to_points(
    values: Sequence[float],
    *,
    width: int,
    height: int,
    left: int,
    top: int,
) -> str:
    if not values:
        return ""
    plot_width = width - left - 40
    plot_height = height - top - 60
    minimum = min(values)
    maximum = max(values)
    if minimum == maximum:
        padding = abs(minimum) * 0.05 if minimum != 0 else 1.0
        minimum -= padding
        maximum += padding

    def x_position(index: int) -> float:
        if len(values) == 1:
            return left + (plot_width / 2.0)
        return left + (plot_width * index / (len(values) - 1))

    def y_position(value: float) -> float:
        normalized = (value - minimum) / (maximum - minimum)
        return top + plot_height - (normalized * plot_height)

    return " ".join(
        f"{x_position(index):.2f},{y_position(value):.2f}"
        for index, value in enumerate(values)
    )


def _write_line_chart_svg(
    *,
    path: Path,
    title: str,
    dates: Sequence[date],
    values: Sequence[float],
    stroke: str,
) -> None:
    width = 840
    height = 420
    left = 72
    top = 56
    plot_width = width - left - 40
    plot_height = height - top - 60
    minimum = min(values) if values else 0.0
    maximum = max(values) if values else 1.0
    if minimum == maximum:
        padding = abs(minimum) * 0.05 if minimum != 0 else 1.0
        minimum -= padding
        maximum += padding

    zero_line = ""
    if minimum <= 0.0 <= maximum:
        zero_y = top + plot_height - ((0.0 - minimum) / (maximum - minimum) * plot_height)
        zero_line = (
            f'<line x1="{left}" y1="{zero_y:.2f}" x2="{left + plot_width}" y2="{zero_y:.2f}" '
            'stroke="#d9dde6" stroke-width="1" stroke-dasharray="4 4" />'
        )

    points = _series_to_points(values, width=width, height=height, left=left, top=top)
    polyline = ""
    markers = ""
    if len(values) >= 2:
        polyline = (
            f'<polyline fill="none" stroke="{stroke}" stroke-width="3" '
            f'points="{points}" />'
        )
    elif len(values) == 1:
        x_coord, y_coord = points.split(",")
        markers = (
            f'<circle cx="{x_coord}" cy="{y_coord}" r="4" fill="{stroke}" />'
        )

    last_value = values[-1] if values else 0.0
    min_label = f"{minimum:.4f}"
    max_label = f"{maximum:.4f}"
    start_label = dates[0].isoformat() if dates else "n/a"
    end_label = dates[-1].isoformat() if dates else "n/a"
    svg_lines = [
        (
            '<svg xmlns="http://www.w3.org/2000/svg" '
            'width="840" height="420" viewBox="0 0 840 420">'
        ),
        '<rect width="100%" height="100%" fill="#ffffff" />',
        (
            f'<text x="{left}" y="30" font-size="22" '
            f'font-family="Menlo, monospace" fill="#1f2937">{title}</text>'
        ),
        (
            f'<text x="{left}" y="48" font-size="12" '
            f'font-family="Menlo, monospace" fill="#6b7280">'
            f"start={start_label} end={end_label} last={last_value:.4f}</text>"
        ),
        (
            f'<line x1="{left}" y1="{top}" x2="{left}" '
            f'y2="{top + plot_height}" stroke="#94a3b8" stroke-width="1" />'
        ),
        (
            f'<line x1="{left}" y1="{top + plot_height}" '
            f'x2="{left + plot_width}" y2="{top + plot_height}" '
            'stroke="#94a3b8" stroke-width="1" />'
        ),
        zero_line,
        polyline,
        markers,
        (
            f'<text x="18" y="{top + 4}" font-size="12" '
            f'font-family="Menlo, monospace" fill="#6b7280">{max_label}</text>'
        ),
        (
            f'<text x="18" y="{top + plot_height}" font-size="12" '
            f'font-family="Menlo, monospace" fill="#6b7280">{min_label}</text>'
        ),
        (
            f'<text x="{left}" y="{top + plot_height + 24}" font-size="12" '
            f'font-family="Menlo, monospace" fill="#6b7280">{start_label}</text>'
        ),
        (
            f'<text x="{left + plot_width - 80}" y="{top + plot_height + 24}" '
            f'font-size="12" font-family="Menlo, monospace" fill="#6b7280">'
            f"{end_label}</text>"
        ),
        "</svg>",
    ]
    path.write_text(
        "\n".join(svg_lines)
        + "\n",
        encoding="utf-8",
    )


def run_backtest_evaluation(
    *,
    strategy_name: str,
    dataset: PreparedResearchDataset,
    signals: Sequence[SignalRecord],
    backtest_result: BacktestResult,
    factor_path: Optional[Path],
    output_dir: Path,
) -> BacktestResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    daily_path = backtest_result.series_paths.get("daily")
    if daily_path is None:
        raise ValueError("BacktestResult.series_paths must include 'daily'")

    daily_rows = load_backtest_daily_rows(daily_path)
    factor_regressions = build_factor_regressions(
        daily_rows=daily_rows,
        factor_path=factor_path,
    )
    factor_regressions["strategy_name"] = strategy_name
    factor_regressions_path = output_dir / f"{strategy_name}_factor_regressions.json"
    factor_regressions_path.write_text(
        json.dumps(factor_regressions, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    ic_rows = build_signal_ic_series(signals=signals, dataset=dataset)
    signal_ic_path = output_dir / f"{strategy_name}_signal_ic.csv"
    _write_csv(signal_ic_path, IC_EXPORT_COLUMNS, (row.to_dict() for row in ic_rows))

    chart_paths = {
        "equity_curve": output_dir / f"{strategy_name}_equity_curve.svg",
        "drawdown": output_dir / f"{strategy_name}_drawdown.svg",
        "cumulative_ic": output_dir / f"{strategy_name}_cumulative_ic.svg",
    }
    chart_dates = [row.signal_date for row in daily_rows]
    _write_line_chart_svg(
        path=chart_paths["equity_curve"],
        title="Equity Curve",
        dates=chart_dates,
        values=[row.equity_curve for row in daily_rows],
        stroke="#0f766e",
    )
    _write_line_chart_svg(
        path=chart_paths["drawdown"],
        title="Drawdown",
        dates=chart_dates,
        values=[row.drawdown for row in daily_rows],
        stroke="#b91c1c",
    )
    _write_line_chart_svg(
        path=chart_paths["cumulative_ic"],
        title="Cumulative Rank IC",
        dates=[row.signal_date for row in ic_rows],
        values=[row.cumulative_rank_ic for row in ic_rows],
        stroke="#1d4ed8",
    )

    signal_ic_summary = {
        "mean_rank_ic": (
            float(np.mean([row.rank_ic for row in ic_rows]))
            if ic_rows
            else 0.0
        ),
        "final_cumulative_rank_ic": (
            float(ic_rows[-1].cumulative_rank_ic)
            if ic_rows
            else 0.0
        ),
        "observation_count": len(ic_rows),
    }

    updated_series_paths = dict(backtest_result.series_paths)
    updated_series_paths["signal_ic"] = signal_ic_path
    updated_metadata = dict(backtest_result.metadata)
    updated_metadata.update(
        {
            "factor_regressions_path": str(factor_regressions_path),
            "factor_regression_statuses": {
                "ff3": factor_regressions["models"]["ff3"]["status"],
                "carhart4": factor_regressions["models"]["carhart4"]["status"],
            },
            "chart_paths": {
                key: str(value)
                for key, value in chart_paths.items()
            },
            "signal_ic_path": str(signal_ic_path),
            "signal_ic_summary": signal_ic_summary,
        }
    )
    return replace(
        backtest_result,
        series_paths=updated_series_paths,
        metadata=updated_metadata,
    )
