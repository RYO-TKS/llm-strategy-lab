from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from llm_strategy_lab.backtest import run_daily_backtest
from llm_strategy_lab.data_pipeline import AlignedMarketReturn, PreparedResearchDataset
from llm_strategy_lab.models import BacktestConfig, StrategyConfig
from llm_strategy_lab.strategies import PortfolioRecord, create_strategy


def _build_backtest_dataset() -> PreparedResearchDataset:
    signal_dates = [
        date(2020, 1, 6),
        date(2020, 1, 7),
        date(2020, 1, 8),
        date(2020, 1, 9),
        date(2020, 1, 10),
        date(2020, 1, 13),
        date(2020, 1, 14),
        date(2020, 1, 15),
    ]
    global_factor = [-0.04, -0.03, -0.01, 0.0, 0.02, 0.04, 0.05, 0.06]
    country_spread = [0.03, 0.02, 0.01, 0.0, -0.01, -0.02, -0.03, -0.04]
    cyclical_spread = [-0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.04, 0.05]

    us_specs = {
        "ENERGY": (1.0, 1.0, 1.2),
        "CONSUMER_STAPLES": (1.0, 1.0, -1.1),
    }
    jp_specs = {
        "ENERGY_RESOURCES": (1.0, -1.0, 1.4),
        "TRANSPORTATION_LOGISTICS": (1.0, -1.0, 0.6),
        "FOODS": (1.0, -1.0, -0.5),
        "PHARMACEUTICALS": (1.0, -1.0, -1.3),
    }

    us_rows = []
    jp_rows = []
    for index, signal_date in enumerate(signal_dates):
        for sector, (global_loading, country_loading, cyclical_loading) in us_specs.items():
            us_rows.append(
                AlignedMarketReturn(
                    market="US",
                    source_date=signal_date,
                    signal_date=signal_date,
                    sector=sector,
                    return_value=(
                        global_factor[index] * global_loading
                        + country_spread[index] * country_loading
                        + cyclical_spread[index] * cyclical_loading
                    ),
                    return_type="close_to_close",
                )
            )
        for sector, (global_loading, country_loading, cyclical_loading) in jp_specs.items():
            jp_rows.append(
                AlignedMarketReturn(
                    market="JP",
                    source_date=signal_date,
                    signal_date=signal_date,
                    sector=sector,
                    return_value=(
                        global_factor[index] * global_loading
                        + country_spread[index] * country_loading
                        + cyclical_spread[index] * cyclical_loading
                    ),
                    return_type="open_to_close",
                )
            )

    return PreparedResearchDataset(
        us_sectors=tuple(us_specs.keys()),
        jp_sectors=tuple(jp_specs.keys()),
        alignment_pairs=(),
        us_aligned_returns=tuple(us_rows),
        jp_open_to_close_returns=tuple(jp_rows),
        quality_events=(),
    )


class BacktestEngineTests(unittest.TestCase):
    def test_run_daily_backtest_computes_turnover_hit_ratio_and_drawdown(self) -> None:
        dataset = PreparedResearchDataset(
            us_sectors=("US_A",),
            jp_sectors=("JP_A", "JP_B"),
            alignment_pairs=(),
            us_aligned_returns=(),
            jp_open_to_close_returns=(
                AlignedMarketReturn(
                    market="JP",
                    source_date=date(2020, 1, 6),
                    signal_date=date(2020, 1, 6),
                    sector="JP_A",
                    return_value=0.02,
                    return_type="open_to_close",
                ),
                AlignedMarketReturn(
                    market="JP",
                    source_date=date(2020, 1, 6),
                    signal_date=date(2020, 1, 6),
                    sector="JP_B",
                    return_value=-0.01,
                    return_type="open_to_close",
                ),
                AlignedMarketReturn(
                    market="JP",
                    source_date=date(2020, 1, 7),
                    signal_date=date(2020, 1, 7),
                    sector="JP_A",
                    return_value=0.03,
                    return_type="open_to_close",
                ),
                AlignedMarketReturn(
                    market="JP",
                    source_date=date(2020, 1, 7),
                    signal_date=date(2020, 1, 7),
                    sector="JP_B",
                    return_value=-0.01,
                    return_type="open_to_close",
                ),
            ),
            quality_events=(),
        )
        portfolio = (
            PortfolioRecord(
                signal_date=date(2020, 1, 6),
                market="JP",
                sector="JP_A",
                side="long",
                weight=0.5,
                score=1.0,
                rank=1,
                gross_exposure=1.0,
                net_exposure=0.0,
            ),
            PortfolioRecord(
                signal_date=date(2020, 1, 6),
                market="JP",
                sector="JP_B",
                side="short",
                weight=-0.5,
                score=-1.0,
                rank=2,
                gross_exposure=1.0,
                net_exposure=0.0,
            ),
            PortfolioRecord(
                signal_date=date(2020, 1, 7),
                market="JP",
                sector="JP_B",
                side="long",
                weight=0.5,
                score=1.0,
                rank=1,
                gross_exposure=1.0,
                net_exposure=0.0,
            ),
            PortfolioRecord(
                signal_date=date(2020, 1, 7),
                market="JP",
                sector="JP_A",
                side="short",
                weight=-0.5,
                score=-1.0,
                rank=2,
                gross_exposure=1.0,
                net_exposure=0.0,
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_daily_backtest(
                strategy_name="manual",
                dataset=dataset,
                portfolio=portfolio,
                backtest=BacktestConfig(start=None, end=None, rebalance="daily"),
                output_dir=Path(tmpdir),
            )

            self.assertAlmostEqual(result.metrics["average_turnover"], 1.5)
            self.assertAlmostEqual(result.metrics["hit_ratio"], 0.5)
            self.assertAlmostEqual(result.metrics["max_drawdown"], 0.02, places=6)
            self.assertAlmostEqual(result.metrics["average_gross_exposure"], 1.0)
            self.assertAlmostEqual(result.metrics["average_net_exposure"], 0.0)
            self.assertAlmostEqual(result.metrics["cumulative_return"], -0.0053, places=6)
            self.assertTrue(result.series_paths["daily"].exists())
            self.assertTrue(result.series_paths["positions"].exists())

    def test_run_daily_backtest_supports_four_strategies_under_same_contract(self) -> None:
        dataset = _build_backtest_dataset()
        strategy_configs = {
            "mom": StrategyConfig(
                name="mom",
                params_file=None,
                params={"q": 0.25, "rolling_window": 5},
            ),
            "pca_plain": StrategyConfig(
                name="pca_plain",
                params_file=None,
                params={"q": 0.25, "rolling_window": 5, "components": 2},
            ),
            "pca_sub": StrategyConfig(
                name="pca_sub",
                params_file=None,
                params={
                    "q": 0.25,
                    "rolling_window": 5,
                    "components": 3,
                    "regularization": 0.1,
                    "subspace": ["global", "country_spread", "cyclical_vs_defensive"],
                },
            ),
            "double": StrategyConfig(
                name="double",
                params_file=None,
                params={
                    "q": 0.25,
                    "rolling_window": 5,
                    "mom": {"rolling_window": 5},
                    "pca_sub": {
                        "rolling_window": 5,
                        "components": 3,
                        "regularization": 0.1,
                        "subspace": [
                            "global",
                            "country_spread",
                            "cyclical_vs_defensive",
                        ],
                    },
                },
            ),
        }
        metric_key_sets = []

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for name, config in strategy_configs.items():
                strategy = create_strategy(config)
                prepared = strategy.generate(dataset)
                output_dir = root / name
                strategy.write_artifacts(prepared, output_dir=output_dir)
                result = run_daily_backtest(
                    strategy_name=name,
                    dataset=dataset,
                    portfolio=prepared.portfolio,
                    backtest=BacktestConfig(start=None, end=None, rebalance="daily"),
                    output_dir=output_dir,
                )
                metrics_path = Path(str(result.metadata["metrics_path"]))
                payload = json.loads(metrics_path.read_text(encoding="utf-8"))
                expected_days = len({row.signal_date for row in prepared.portfolio})

                self.assertGreaterEqual(len(prepared.portfolio), 2)
                self.assertEqual(payload["strategy_name"], name)
                self.assertEqual(payload["portfolio_day_count"], expected_days)
                self.assertTrue(result.series_paths["daily"].exists())
                self.assertTrue(result.series_paths["positions"].exists())
                self.assertIn("annual_return", payload["metrics"])
                self.assertIn("max_drawdown", payload["metrics"])
                self.assertIn("average_turnover", payload["metrics"])
                self.assertIn("hit_ratio", payload["metrics"])
                metric_key_sets.append(set(payload["metrics"]))

            self.assertEqual(len(metric_key_sets), 4)
            self.assertTrue(all(keys == metric_key_sets[0] for keys in metric_key_sets[1:]))


if __name__ == "__main__":
    unittest.main()
