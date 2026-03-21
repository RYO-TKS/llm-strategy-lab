from __future__ import annotations

import csv
import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from llm_strategy_lab.data_pipeline import AlignedMarketReturn, PreparedResearchDataset
from llm_strategy_lab.evaluation import run_backtest_evaluation
from llm_strategy_lab.models import BacktestResult
from llm_strategy_lab.strategies import SignalRecord


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class EvaluationTests(unittest.TestCase):
    def test_run_backtest_evaluation_writes_regressions_ic_and_svg_charts(self) -> None:
        trading_dates = [
            date(2020, 1, 6),
            date(2020, 1, 7),
            date(2020, 1, 8),
            date(2020, 1, 9),
            date(2020, 1, 10),
            date(2020, 1, 13),
            date(2020, 1, 14),
            date(2020, 1, 15),
        ]
        factors = [
            {"mkt_rf": 0.010, "smb": 0.002, "hml": -0.001, "umd": 0.003, "rf": 0.0001},
            {"mkt_rf": -0.004, "smb": 0.001, "hml": 0.002, "umd": 0.001, "rf": 0.0001},
            {"mkt_rf": 0.006, "smb": -0.002, "hml": 0.001, "umd": 0.002, "rf": 0.0001},
            {"mkt_rf": 0.003, "smb": 0.001, "hml": -0.003, "umd": -0.001, "rf": 0.0001},
            {"mkt_rf": -0.002, "smb": -0.001, "hml": 0.002, "umd": 0.004, "rf": 0.0001},
            {"mkt_rf": 0.005, "smb": 0.003, "hml": -0.002, "umd": -0.002, "rf": 0.0001},
            {"mkt_rf": 0.001, "smb": -0.003, "hml": 0.001, "umd": 0.003, "rf": 0.0001},
            {"mkt_rf": -0.003, "smb": 0.002, "hml": -0.001, "umd": 0.002, "rf": 0.0001},
        ]
        alpha = 0.0005
        market_beta = 1.2
        smb_beta = -0.4
        hml_beta = 0.7
        umd_beta = 0.5

        daily_rows = []
        factor_rows = []
        jp_returns = []
        signals = []
        equity_curve = 1.0
        running_peak = 1.0
        sector_scores = {
            "ALPHA": 2.0,
            "BETA": 1.0,
            "GAMMA": -1.0,
            "DELTA": -2.0,
        }
        sector_returns = {
            "ALPHA": 0.03,
            "BETA": 0.01,
            "GAMMA": -0.01,
            "DELTA": -0.02,
        }
        for trading_date, factor in zip(trading_dates, factors):
            strategy_return = (
                factor["rf"]
                + alpha
                + (market_beta * factor["mkt_rf"])
                + (smb_beta * factor["smb"])
                + (hml_beta * factor["hml"])
                + (umd_beta * factor["umd"])
            )
            equity_curve *= 1.0 + strategy_return
            running_peak = max(running_peak, equity_curve)
            drawdown = (equity_curve / running_peak) - 1.0
            daily_rows.append(
                {
                    "date": trading_date.isoformat(),
                    "return": strategy_return,
                    "equity_curve": equity_curve,
                    "drawdown": drawdown,
                    "turnover": 1.0,
                    "hit_ratio": 0.5,
                    "gross_exposure": 1.0,
                    "net_exposure": 0.0,
                }
            )
            factor_rows.append(
                {
                    "date": trading_date.isoformat(),
                    **factor,
                }
            )
            for rank, sector in enumerate(("ALPHA", "BETA", "GAMMA", "DELTA"), start=1):
                jp_returns.append(
                    AlignedMarketReturn(
                        market="JP",
                        source_date=trading_date,
                        signal_date=trading_date,
                        sector=sector,
                        return_value=sector_returns[sector],
                        return_type="open_to_close",
                    )
                )
                signal_value = 1 if sector in {"ALPHA", "BETA"} else -1
                signals.append(
                    SignalRecord(
                        signal_date=trading_date,
                        market="JP",
                        sector=sector,
                        signal=signal_value,
                        score=sector_scores[sector],
                        rank=rank,
                        lookback_start=trading_date,
                        lookback_end=trading_date,
                        window_size=1,
                    )
                )

        dataset = PreparedResearchDataset(
            us_sectors=("US_A", "US_B"),
            jp_sectors=("ALPHA", "BETA", "GAMMA", "DELTA"),
            alignment_pairs=(),
            us_aligned_returns=(),
            jp_open_to_close_returns=tuple(jp_returns),
            quality_events=(),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            daily_path = output_dir / "demo_backtest_daily.csv"
            factor_path = output_dir / "factor_returns.csv"
            _write_csv(
                daily_path,
                [
                    "date",
                    "return",
                    "equity_curve",
                    "drawdown",
                    "turnover",
                    "hit_ratio",
                    "gross_exposure",
                    "net_exposure",
                ],
                daily_rows,
            )
            _write_csv(
                factor_path,
                ["date", "mkt_rf", "smb", "hml", "umd", "rf"],
                factor_rows,
            )

            backtest_result = BacktestResult(
                strategy_name="demo",
                metrics={},
                series_paths={"daily": daily_path},
                metadata={},
            )
            evaluated = run_backtest_evaluation(
                strategy_name="demo",
                dataset=dataset,
                signals=signals,
                backtest_result=backtest_result,
                factor_path=factor_path,
                output_dir=output_dir,
            )

            regression_path = output_dir / "demo_factor_regressions.json"
            ic_path = output_dir / "demo_signal_ic.csv"
            equity_chart = output_dir / "demo_equity_curve.svg"
            drawdown_chart = output_dir / "demo_drawdown.svg"
            cumulative_ic_chart = output_dir / "demo_cumulative_ic.svg"

            payload = json.loads(regression_path.read_text(encoding="utf-8"))
            carhart = payload["models"]["carhart4"]

            self.assertEqual(payload["models"]["ff3"]["status"], "ok")
            self.assertEqual(carhart["status"], "ok")
            self.assertAlmostEqual(carhart["coefficients"]["alpha"], alpha, places=10)
            self.assertAlmostEqual(carhart["coefficients"]["mkt_rf"], market_beta, places=10)
            self.assertAlmostEqual(carhart["coefficients"]["smb"], smb_beta, places=10)
            self.assertAlmostEqual(carhart["coefficients"]["hml"], hml_beta, places=10)
            self.assertAlmostEqual(carhart["coefficients"]["umd"], umd_beta, places=10)
            self.assertEqual(
                evaluated.metadata["factor_regression_statuses"],
                {"ff3": "ok", "carhart4": "ok"},
            )
            self.assertAlmostEqual(
                evaluated.metadata["signal_ic_summary"]["final_cumulative_rank_ic"],
                float(len(trading_dates)),
            )
            self.assertTrue(Path(str(evaluated.series_paths["signal_ic"])).exists())
            self.assertTrue(ic_path.exists())
            self.assertTrue(equity_chart.exists())
            self.assertTrue(drawdown_chart.exists())
            self.assertTrue(cumulative_ic_chart.exists())
            self.assertIn("<svg", equity_chart.read_text(encoding="utf-8"))
            self.assertIn("<svg", drawdown_chart.read_text(encoding="utf-8"))
            self.assertIn("<svg", cumulative_ic_chart.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
