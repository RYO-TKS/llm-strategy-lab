"""Market data loading and JP/US trading-day alignment helpers."""

from __future__ import annotations

import csv
import json
import logging
from bisect import bisect_right
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .models import BacktestConfig, DatasetConfig, JsonDict

LOGGER = logging.getLogger(__name__)
US_MARKET = "US"
JP_MARKET = "JP"
EXPECTED_SECTOR_COUNTS = {
    US_MARKET: 11,
    JP_MARKET: 17,
}
PRICE_REQUIRED_FIELDS = ("date", "sector", "open", "close")
CALENDAR_REQUIRED_FIELDS = ("market", "date")


@dataclass(frozen=True)
class MarketPriceBar:
    market: str
    trading_date: date
    sector: str
    open_price: float
    close_price: float


@dataclass(frozen=True)
class MarketReturn:
    market: str
    source_date: date
    sector: str
    return_value: float
    return_type: str

    def to_dict(self, *, signal_date: Optional[date] = None) -> JsonDict:
        return {
            "market": self.market,
            "source_date": self.source_date.isoformat(),
            "signal_date": signal_date.isoformat() if signal_date else None,
            "sector": self.sector,
            "return": self.return_value,
            "return_type": self.return_type,
        }


@dataclass(frozen=True)
class AlignmentPair:
    us_date: date
    jp_signal_date: date

    def to_dict(self) -> JsonDict:
        return {
            "us_date": self.us_date.isoformat(),
            "jp_signal_date": self.jp_signal_date.isoformat(),
        }


@dataclass(frozen=True)
class AlignedMarketReturn:
    market: str
    source_date: date
    signal_date: date
    sector: str
    return_value: float
    return_type: str

    def to_dict(self) -> JsonDict:
        return {
            "market": self.market,
            "source_date": self.source_date.isoformat(),
            "signal_date": self.signal_date.isoformat(),
            "sector": self.sector,
            "return": self.return_value,
            "return_type": self.return_type,
        }


@dataclass(frozen=True)
class DataQualityEvent:
    code: str
    message: str
    market: str
    severity: str = "warning"
    trading_date: Optional[date] = None
    sector: Optional[str] = None

    def to_dict(self) -> JsonDict:
        return {
            "code": self.code,
            "message": self.message,
            "market": self.market,
            "severity": self.severity,
            "trading_date": self.trading_date.isoformat() if self.trading_date else None,
            "sector": self.sector,
        }


@dataclass
class PreparedResearchDataset:
    us_sectors: Tuple[str, ...]
    jp_sectors: Tuple[str, ...]
    alignment_pairs: Tuple[AlignmentPair, ...]
    us_aligned_returns: Tuple[AlignedMarketReturn, ...]
    jp_open_to_close_returns: Tuple[AlignedMarketReturn, ...]
    quality_events: Tuple[DataQualityEvent, ...]
    artifact_paths: Mapping[str, Path] = field(default_factory=dict)

    def summary(self) -> JsonDict:
        signal_dates = sorted({row.signal_date for row in self.jp_open_to_close_returns})
        source_us_dates = sorted({row.source_date for row in self.us_aligned_returns})
        return {
            "us_sector_count": len(self.us_sectors),
            "jp_sector_count": len(self.jp_sectors),
            "aligned_signal_dates": len(signal_dates),
            "aligned_us_dates": len(source_us_dates),
            "alignment_pairs": len(self.alignment_pairs),
            "us_aligned_rows": len(self.us_aligned_returns),
            "jp_open_to_close_rows": len(self.jp_open_to_close_returns),
            "quality_event_count": len(self.quality_events),
            "artifact_paths": {key: str(value) for key, value in self.artifact_paths.items()},
        }


def _ensure_headers(
    fieldnames: Optional[Sequence[str]],
    required: Sequence[str],
    *,
    path: Path,
) -> None:
    if fieldnames is None:
        raise ValueError(f"CSV header is missing: {path}")

    missing = [field for field in required if field not in fieldnames]
    if missing:
        raise ValueError(f"CSV is missing required columns {missing}: {path}")


def _parse_date(raw_value: str, *, field_name: str) -> date:
    try:
        return date.fromisoformat(raw_value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be YYYY-MM-DD: {raw_value}") from exc


def _parse_float(raw_value: str, *, field_name: str) -> float:
    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be numeric: {raw_value}") from exc


def _record_event(
    events: List[DataQualityEvent],
    *,
    code: str,
    message: str,
    market: str,
    logger: logging.Logger,
    trading_date: Optional[date] = None,
    sector: Optional[str] = None,
    severity: str = "warning",
) -> None:
    event = DataQualityEvent(
        code=code,
        message=message,
        market=market,
        severity=severity,
        trading_date=trading_date,
        sector=sector,
    )
    events.append(event)

    log_method = logger.warning if severity == "warning" else logger.info
    log_method(
        "%s [%s market=%s date=%s sector=%s]",
        message,
        code,
        market,
        trading_date.isoformat() if trading_date else "-",
        sector or "-",
    )


def load_market_price_bars(
    path: Path,
    *,
    market: str,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Tuple[MarketPriceBar, ...], Tuple[str, ...], Tuple[DataQualityEvent, ...]]:
    active_logger = logger or LOGGER
    events: List[DataQualityEvent] = []
    rows_by_key: Dict[Tuple[date, str], MarketPriceBar] = {}

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _ensure_headers(reader.fieldnames, PRICE_REQUIRED_FIELDS, path=path)

        for row in reader:
            raw_date = (row.get("date") or "").strip()
            raw_sector = (row.get("sector") or "").strip()
            raw_open = (row.get("open") or "").strip()
            raw_close = (row.get("close") or "").strip()

            if not raw_date or not raw_sector or not raw_open or not raw_close:
                _record_event(
                    events,
                    code="missing_price_field",
                    message=f"Skipped incomplete {market} price row.",
                    market=market,
                    logger=active_logger,
                    trading_date=_parse_date(raw_date, field_name="date") if raw_date else None,
                    sector=raw_sector or None,
                )
                continue

            trading_date = _parse_date(raw_date, field_name="date")
            open_price = _parse_float(raw_open, field_name="open")
            close_price = _parse_float(raw_close, field_name="close")
            key = (trading_date, raw_sector)

            if key in rows_by_key:
                _record_event(
                    events,
                    code="duplicate_sector_row",
                    message=f"Duplicate {market} sector row found; latest row kept.",
                    market=market,
                    logger=active_logger,
                    trading_date=trading_date,
                    sector=raw_sector,
                )

            rows_by_key[key] = MarketPriceBar(
                market=market,
                trading_date=trading_date,
                sector=raw_sector,
                open_price=open_price,
                close_price=close_price,
            )

    rows = tuple(sorted(rows_by_key.values(), key=lambda item: (item.trading_date, item.sector)))
    sector_universe = tuple(sorted({row.sector for row in rows}))
    expected_count = EXPECTED_SECTOR_COUNTS[market]
    if len(sector_universe) != expected_count:
        raise ValueError(
            f"{market} sector universe count mismatch: expected {expected_count}, "
            f"got {len(sector_universe)} from {path}"
        )

    dates = sorted({row.trading_date for row in rows})
    for trading_date in dates:
        sectors_on_date = {
            row.sector
            for row in rows
            if row.trading_date == trading_date
        }
        for missing_sector in sorted(set(sector_universe) - sectors_on_date):
            _record_event(
                events,
                code="missing_sector_row",
                message=f"Sector row missing from {market} prices.",
                market=market,
                logger=active_logger,
                trading_date=trading_date,
                sector=missing_sector,
            )

    active_logger.info(
        "Loaded %s market bars from %s with %s sectors and %s trading dates.",
        market,
        path,
        len(sector_universe),
        len(dates),
    )
    return rows, sector_universe, tuple(events)


def load_trading_calendar(
    path: Path,
    *,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Mapping[str, Tuple[date, ...]], Tuple[DataQualityEvent, ...]]:
    active_logger = logger or LOGGER
    events: List[DataQualityEvent] = []
    open_dates: Dict[str, set[date]] = {
        US_MARKET: set(),
        JP_MARKET: set(),
    }

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _ensure_headers(reader.fieldnames, CALENDAR_REQUIRED_FIELDS, path=path)

        for row in reader:
            market = (row.get("market") or "").strip().upper()
            if market not in open_dates:
                raise ValueError(f"Unsupported market in trading calendar: {market}")

            trading_date = _parse_date((row.get("date") or "").strip(), field_name="date")
            raw_is_open = (row.get("is_open") or "1").strip().lower()
            is_open = raw_is_open in {"1", "true", "yes", "y"}
            if is_open:
                open_dates[market].add(trading_date)

    for market, dates in open_dates.items():
        if not dates:
            raise ValueError(f"Trading calendar has no open dates for {market}.")
        active_logger.info(
            "Loaded %s open dates for %s from %s.",
            len(dates),
            market,
            path,
        )

    calendar = {
        market: tuple(sorted(dates))
        for market, dates in open_dates.items()
    }
    return calendar, tuple(events)


def compute_us_close_to_close_returns(
    bars: Sequence[MarketPriceBar],
) -> Tuple[Tuple[MarketReturn, ...], Tuple[DataQualityEvent, ...]]:
    returns: List[MarketReturn] = []
    events: List[DataQualityEvent] = []
    by_sector: Dict[str, List[MarketPriceBar]] = {}

    for bar in bars:
        by_sector.setdefault(bar.sector, []).append(bar)

    for sector, sector_bars in by_sector.items():
        ordered_bars = sorted(sector_bars, key=lambda item: item.trading_date)
        previous_close: Optional[float] = None

        for current_bar in ordered_bars:
            if previous_close is None:
                previous_close = current_bar.close_price
                continue

            if previous_close == 0:
                events.append(
                    DataQualityEvent(
                        code="zero_previous_close",
                        message="Skipped US return because previous close was zero.",
                        market=US_MARKET,
                        trading_date=current_bar.trading_date,
                        sector=sector,
                    )
                )
                previous_close = current_bar.close_price
                continue

            returns.append(
                MarketReturn(
                    market=US_MARKET,
                    source_date=current_bar.trading_date,
                    sector=sector,
                    return_value=(current_bar.close_price / previous_close) - 1.0,
                    return_type="close_to_close",
                )
            )
            previous_close = current_bar.close_price

    return tuple(sorted(returns, key=lambda item: (item.source_date, item.sector))), tuple(events)


def compute_jp_open_to_close_returns(
    bars: Sequence[MarketPriceBar],
) -> Tuple[Tuple[MarketReturn, ...], Tuple[DataQualityEvent, ...]]:
    returns: List[MarketReturn] = []
    events: List[DataQualityEvent] = []

    for bar in sorted(bars, key=lambda item: (item.trading_date, item.sector)):
        if bar.open_price == 0:
            events.append(
                DataQualityEvent(
                    code="zero_open_price",
                    message="Skipped JP return because open price was zero.",
                    market=JP_MARKET,
                    trading_date=bar.trading_date,
                    sector=bar.sector,
                )
            )
            continue

        returns.append(
            MarketReturn(
                market=JP_MARKET,
                source_date=bar.trading_date,
                sector=bar.sector,
                return_value=(bar.close_price / bar.open_price) - 1.0,
                return_type="open_to_close",
            )
        )

    return tuple(returns), tuple(events)


def _within_backtest_window(trading_date: date, backtest: BacktestConfig) -> bool:
    if backtest.start and trading_date < backtest.start:
        return False
    if backtest.end and trading_date > backtest.end:
        return False
    return True


def align_us_to_jp_next_open(
    us_returns: Sequence[MarketReturn],
    *,
    jp_open_dates: Sequence[date],
    backtest: BacktestConfig,
    logger: Optional[logging.Logger] = None,
) -> Tuple[
    Tuple[AlignmentPair, ...],
    Tuple[AlignedMarketReturn, ...],
    Tuple[DataQualityEvent, ...],
]:
    active_logger = logger or LOGGER
    events: List[DataQualityEvent] = []
    pairs_by_us_date: Dict[date, AlignmentPair] = {}
    aligned_returns: List[AlignedMarketReturn] = []

    ordered_jp_dates = tuple(sorted(jp_open_dates))

    for us_date in sorted({row.source_date for row in us_returns}):
        jp_index = bisect_right(ordered_jp_dates, us_date)
        if jp_index >= len(ordered_jp_dates):
            _record_event(
                events,
                code="missing_next_jp_open",
                message="No next JP open date found for US return date.",
                market=US_MARKET,
                logger=active_logger,
                trading_date=us_date,
            )
            continue

        jp_signal_date = ordered_jp_dates[jp_index]
        if not _within_backtest_window(jp_signal_date, backtest):
            continue

        pairs_by_us_date[us_date] = AlignmentPair(
            us_date=us_date,
            jp_signal_date=jp_signal_date,
        )

    for row in us_returns:
        pair = pairs_by_us_date.get(row.source_date)
        if pair is None:
            continue

        aligned_returns.append(
            AlignedMarketReturn(
                market=row.market,
                source_date=pair.us_date,
                signal_date=pair.jp_signal_date,
                sector=row.sector,
                return_value=row.return_value,
                return_type=row.return_type,
            )
        )

    active_logger.info(
        "Aligned %s US return dates to %s JP signal dates.",
        len(pairs_by_us_date),
        len({pair.jp_signal_date for pair in pairs_by_us_date.values()}),
    )
    return (
        tuple(
            sorted(
                pairs_by_us_date.values(),
                key=lambda item: (item.us_date, item.jp_signal_date),
            )
        ),
        tuple(sorted(aligned_returns, key=lambda item: (item.source_date, item.sector))),
        tuple(events),
    )


def slice_jp_returns_for_signals(
    jp_returns: Sequence[MarketReturn],
    *,
    signal_dates: Iterable[date],
    jp_sectors: Sequence[str],
    logger: Optional[logging.Logger] = None,
) -> Tuple[Tuple[AlignedMarketReturn, ...], Tuple[DataQualityEvent, ...]]:
    active_logger = logger or LOGGER
    events: List[DataQualityEvent] = []
    signal_date_set = set(signal_dates)
    filtered = tuple(
        sorted(
            (
                AlignedMarketReturn(
                    market=row.market,
                    source_date=row.source_date,
                    signal_date=row.source_date,
                    sector=row.sector,
                    return_value=row.return_value,
                    return_type=row.return_type,
                )
                for row in jp_returns
                if row.source_date in signal_date_set
            ),
            key=lambda item: (item.signal_date, item.sector),
        )
    )

    by_date: Dict[date, set[str]] = {}
    for row in filtered:
        by_date.setdefault(row.signal_date, set()).add(row.sector)

    for signal_date in sorted(signal_date_set):
        present_sectors = by_date.get(signal_date, set())
        for missing_sector in sorted(set(jp_sectors) - present_sectors):
            _record_event(
                events,
                code="missing_jp_signal_sector",
                message="JP signal sector is missing for aligned date.",
                market=JP_MARKET,
                logger=active_logger,
                trading_date=signal_date,
                sector=missing_sector,
            )

    return filtered, tuple(events)


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[JsonDict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_prepared_dataset_artifacts(
    dataset: PreparedResearchDataset,
    *,
    output_dir: Path,
) -> Mapping[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths = {
        "alignment_index": output_dir / "alignment_index.csv",
        "us_aligned_returns": output_dir / "us_aligned_close_to_close.csv",
        "jp_open_to_close_returns": output_dir / "jp_open_to_close.csv",
        "data_quality_log": output_dir / "data_quality_log.json",
        "data_alignment_report": output_dir / "data_alignment_report.json",
    }

    _write_csv(
        artifact_paths["alignment_index"],
        ("us_date", "jp_signal_date"),
        (pair.to_dict() for pair in dataset.alignment_pairs),
    )
    _write_csv(
        artifact_paths["us_aligned_returns"],
        ("market", "source_date", "signal_date", "sector", "return", "return_type"),
        (
            row.to_dict()
            for row in dataset.us_aligned_returns
        ),
    )
    _write_csv(
        artifact_paths["jp_open_to_close_returns"],
        ("market", "source_date", "signal_date", "sector", "return", "return_type"),
        (
            row.to_dict()
            for row in dataset.jp_open_to_close_returns
        ),
    )

    quality_log = {
        "events": [event.to_dict() for event in dataset.quality_events],
        "count": len(dataset.quality_events),
    }
    dataset.artifact_paths = artifact_paths
    artifact_paths["data_quality_log"].write_text(
        json.dumps(quality_log, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    artifact_paths["data_alignment_report"].write_text(
        json.dumps(dataset.summary(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return artifact_paths


def prepare_aligned_research_dataset(
    dataset_config: DatasetConfig,
    *,
    backtest: BacktestConfig,
    output_dir: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
) -> PreparedResearchDataset:
    active_logger = logger or LOGGER
    us_bars, us_sectors, us_events = load_market_price_bars(
        dataset_config.us_sectors,
        market=US_MARKET,
        logger=active_logger,
    )
    jp_bars, jp_sectors, jp_events = load_market_price_bars(
        dataset_config.jp_sectors,
        market=JP_MARKET,
        logger=active_logger,
    )
    calendar, calendar_events = load_trading_calendar(
        dataset_config.trading_calendar,
        logger=active_logger,
    )
    us_returns, us_return_events = compute_us_close_to_close_returns(us_bars)
    jp_returns, jp_return_events = compute_jp_open_to_close_returns(jp_bars)
    alignment_pairs, us_aligned_returns, alignment_events = align_us_to_jp_next_open(
        us_returns,
        jp_open_dates=calendar[JP_MARKET],
        backtest=backtest,
        logger=active_logger,
    )
    signal_dates = [pair.jp_signal_date for pair in alignment_pairs]
    jp_signal_returns, jp_signal_events = slice_jp_returns_for_signals(
        jp_returns,
        signal_dates=signal_dates,
        jp_sectors=jp_sectors,
        logger=active_logger,
    )

    dataset = PreparedResearchDataset(
        us_sectors=us_sectors,
        jp_sectors=jp_sectors,
        alignment_pairs=alignment_pairs,
        us_aligned_returns=us_aligned_returns,
        jp_open_to_close_returns=jp_signal_returns,
        quality_events=(
            us_events
            + jp_events
            + calendar_events
            + us_return_events
            + jp_return_events
            + alignment_events
            + jp_signal_events
        ),
    )
    if output_dir is None:
        return dataset

    artifact_paths = write_prepared_dataset_artifacts(dataset, output_dir=output_dir)
    dataset.artifact_paths = artifact_paths
    return dataset
