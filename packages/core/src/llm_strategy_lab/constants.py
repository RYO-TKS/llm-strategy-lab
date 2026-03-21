"""Shared column names and grouped constants used across the project."""


DATE = "date"
TIMESTAMP = "timestamp"
SYMBOL = "symbol"
MARKET = "market"
SECTOR = "sector"
OPEN = "open"
HIGH = "high"
LOW = "low"
CLOSE = "close"
ADJ_CLOSE = "adj_close"
VOLUME = "volume"
RETURN = "return"
FORWARD_RETURN = "forward_return"
SIGNAL = "signal"
SCORE = "score"
RANK = "rank"
SIDE = "side"
WEIGHT = "weight"
TURNOVER = "turnover"
HIT_RATIO = "hit_ratio"
GROSS_EXPOSURE = "gross_exposure"
NET_EXPOSURE = "net_exposure"
EQUITY_CURVE = "equity_curve"
DRAWDOWN = "drawdown"
RUN_ID = "run_id"
EXPERIMENT_ID = "experiment_id"
STRATEGY_NAME = "strategy_name"
STATUS = "status"
CREATED_AT_UTC = "created_at_utc"

PRICE_COLUMNS = (
    DATE,
    SYMBOL,
    MARKET,
    SECTOR,
    OPEN,
    HIGH,
    LOW,
    CLOSE,
    ADJ_CLOSE,
    VOLUME,
)

SIGNAL_COLUMNS = (
    DATE,
    SYMBOL,
    MARKET,
    SECTOR,
    SIGNAL,
    SCORE,
    RANK,
)

PORTFOLIO_COLUMNS = (
    DATE,
    SYMBOL,
    MARKET,
    SECTOR,
    SIDE,
    WEIGHT,
    TURNOVER,
    GROSS_EXPOSURE,
    NET_EXPOSURE,
)

BACKTEST_SERIES_COLUMNS = (
    DATE,
    RETURN,
    EQUITY_CURVE,
    DRAWDOWN,
    TURNOVER,
    HIT_RATIO,
    GROSS_EXPOSURE,
    NET_EXPOSURE,
)

RUN_RECORD_COLUMNS = (
    RUN_ID,
    EXPERIMENT_ID,
    STRATEGY_NAME,
    STATUS,
    CREATED_AT_UTC,
)

COLUMN_GROUPS = {
    "price": PRICE_COLUMNS,
    "signal": SIGNAL_COLUMNS,
    "portfolio": PORTFOLIO_COLUMNS,
    "backtest": BACKTEST_SERIES_COLUMNS,
    "run_record": RUN_RECORD_COLUMNS,
}
