from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from llm_strategy_lab.data_pipeline import AlignedMarketReturn, PreparedResearchDataset
from llm_strategy_lab.models import StrategyConfig
from llm_strategy_lab.strategies import MomentumStrategy, create_strategy


def _build_dataset() -> PreparedResearchDataset:
    signal_dates = [
        date(2020, 1, 6),
        date(2020, 1, 7),
        date(2020, 1, 8),
        date(2020, 1, 9),
    ]
    sector_returns = {
        "ALPHA": [0.05, 0.03, 0.04, 0.01],
        "BETA": [0.02, 0.01, 0.02, 0.03],
        "GAMMA": [-0.01, -0.02, -0.01, -0.02],
        "DELTA": [-0.03, -0.01, -0.04, -0.05],
    }
    rows = []
    for signal_index, signal_date in enumerate(signal_dates):
        for sector, values in sector_returns.items():
            rows.append(
                AlignedMarketReturn(
                    market="JP",
                    source_date=signal_date,
                    signal_date=signal_date,
                    sector=sector,
                    return_value=values[signal_index],
                    return_type="open_to_close",
                )
            )

    return PreparedResearchDataset(
        us_sectors=("US_A", "US_B"),
        jp_sectors=("ALPHA", "BETA", "GAMMA", "DELTA"),
        alignment_pairs=(),
        us_aligned_returns=(),
        jp_open_to_close_returns=tuple(rows),
        quality_events=(),
    )


class MomentumStrategyTests(unittest.TestCase):
    def test_create_strategy_returns_momentum_strategy(self) -> None:
        strategy = create_strategy(
            StrategyConfig(
                name="mom",
                params_file=None,
                params={"q": 0.25, "rolling_window": 2},
            )
        )
        self.assertIsInstance(strategy, MomentumStrategy)

    def test_momentum_strategy_builds_backtest_ready_portfolio(self) -> None:
        dataset = _build_dataset()
        strategy = MomentumStrategy(
            StrategyConfig(
                name="mom",
                params_file=None,
                params={"q": 0.25, "rolling_window": 2},
            )
        )

        signals = strategy.compute_signal(dataset)
        portfolio = strategy.build_portfolio(signals)
        explanation = strategy.explain(
            dataset=dataset,
            signals=signals,
            portfolio=portfolio,
        )

        self.assertEqual(
            sorted({row.signal_date.isoformat() for row in signals}),
            ["2020-01-08", "2020-01-09"],
        )
        self.assertEqual(len(signals), 8)
        self.assertEqual(len([row for row in signals if row.signal > 0]), 2)
        self.assertEqual(len([row for row in signals if row.signal < 0]), 2)
        self.assertEqual(len(portfolio), 4)
        self.assertEqual({row.side for row in portfolio}, {"long", "short"})
        self.assertTrue(all(abs(row.gross_exposure - 1.0) < 1e-9 for row in portfolio))
        self.assertTrue(all(abs(row.net_exposure) < 1e-9 for row in portfolio))
        self.assertEqual(explanation["strategy_name"], "mom")
        self.assertEqual(explanation["parameters"]["q"], 0.25)
        self.assertEqual(explanation["parameters"]["rolling_window"], 2)
        self.assertEqual(explanation["eligible_signal_dates"], 2)
        self.assertEqual(explanation["selected_per_side"], 1)
        self.assertIn("signal_definition", explanation)
        self.assertIn("portfolio_construction", explanation)

    def test_momentum_strategy_run_writes_signal_and_portfolio_artifacts(self) -> None:
        dataset = _build_dataset()
        strategy = MomentumStrategy(
            StrategyConfig(
                name="mom",
                params_file=None,
                params={"q": 0.25, "rolling_window": 2},
            )
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            artifact = strategy.run(dataset, output_dir=output_dir)

            self.assertEqual(artifact.strategy_name, "mom")
            self.assertTrue(artifact.artifact_paths["signals"].exists())
            self.assertTrue(artifact.artifact_paths["portfolio"].exists())
            self.assertTrue(artifact.artifact_paths["explanation"].exists())
            explanation = json.loads(
                artifact.artifact_paths["explanation"].read_text(encoding="utf-8")
            )
            self.assertTrue(artifact.metadata["backtest_ready"])
            self.assertEqual(explanation["backtest_portfolio_dates"], 2)
            self.assertEqual(artifact.metadata["portfolio_row_count"], 4)


if __name__ == "__main__":
    unittest.main()
