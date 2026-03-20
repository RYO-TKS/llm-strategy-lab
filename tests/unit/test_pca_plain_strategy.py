from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from llm_strategy_lab.data_pipeline import AlignedMarketReturn, PreparedResearchDataset
from llm_strategy_lab.models import StrategyConfig
from llm_strategy_lab.strategies import MomentumStrategy, PlainPCAStrategy, create_strategy


def _build_joint_pca_dataset() -> PreparedResearchDataset:
    signal_dates = [
        date(2020, 1, 6),
        date(2020, 1, 7),
        date(2020, 1, 8),
        date(2020, 1, 9),
        date(2020, 1, 10),
        date(2020, 1, 13),
    ]
    global_factor = [-0.04, -0.02, -0.01, 0.02, 0.04, 0.06]
    spread_factor = [0.03, 0.01, -0.01, -0.02, -0.03, -0.04]

    us_loadings = {
        "US_ALPHA": (0.9, 0.4),
        "US_BETA": (1.1, -0.2),
    }
    jp_loadings = {
        "JP_LEAD": (1.4, 0.05),
        "JP_SUPPORT": (0.55, 0.15),
        "JP_DEFENSIVE": (-0.6, 0.2),
        "JP_LAGGARD": (-1.0, -0.3),
    }

    us_rows = []
    jp_rows = []
    for index, signal_date in enumerate(signal_dates):
        for sector, (global_loading, spread_loading) in us_loadings.items():
            us_rows.append(
                AlignedMarketReturn(
                    market="US",
                    source_date=signal_date,
                    signal_date=signal_date,
                    sector=sector,
                    return_value=(
                        global_factor[index] * global_loading
                        + spread_factor[index] * spread_loading
                    ),
                    return_type="close_to_close",
                )
            )
        for sector, (global_loading, spread_loading) in jp_loadings.items():
            jp_rows.append(
                AlignedMarketReturn(
                    market="JP",
                    source_date=signal_date,
                    signal_date=signal_date,
                    sector=sector,
                    return_value=(
                        global_factor[index] * global_loading
                        + spread_factor[index] * spread_loading
                    ),
                    return_type="open_to_close",
                )
            )

    return PreparedResearchDataset(
        us_sectors=tuple(us_loadings.keys()),
        jp_sectors=tuple(jp_loadings.keys()),
        alignment_pairs=(),
        us_aligned_returns=tuple(us_rows),
        jp_open_to_close_returns=tuple(jp_rows),
        quality_events=(),
    )


class PlainPCAStrategyTests(unittest.TestCase):
    def test_create_strategy_returns_plain_pca_strategy(self) -> None:
        strategy = create_strategy(
            StrategyConfig(
                name="pca_plain",
                params_file=None,
                params={"q": 0.25, "rolling_window": 5, "components": 2},
            )
        )
        self.assertIsInstance(strategy, PlainPCAStrategy)

    def test_plain_pca_reconstructs_expected_jp_signal_from_us_shock(self) -> None:
        dataset = _build_joint_pca_dataset()
        strategy = PlainPCAStrategy(
            StrategyConfig(
                name="pca_plain",
                params_file=None,
                params={"q": 0.25, "rolling_window": 5, "components": 2},
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
            [row.signal_date.isoformat() for row in signals[:1]],
            ["2020-01-13"],
        )
        self.assertEqual(len(signals), 4)
        self.assertEqual(len(portfolio), 2)
        self.assertEqual({row.sector for row in signals if row.signal > 0}, {"JP_SUPPORT"})
        self.assertEqual(
            {row.sector for row in signals if row.signal < 0},
            {"JP_LAGGARD"},
        )
        self.assertEqual(
            {row.sector for row in signals[:2]},
            {"JP_LEAD", "JP_SUPPORT"},
        )
        self.assertEqual({row.side for row in portfolio}, {"long", "short"})
        self.assertEqual(explanation["parameters"]["components"], 2)
        self.assertEqual(explanation["comparison_ready_with"], "mom")
        self.assertEqual(explanation["eligible_signal_dates"], 1)
        self.assertEqual(len(explanation["latest_diagnostic"]["joint_feature_labels"]), 6)
        self.assertEqual(
            len(explanation["latest_diagnostic"]["joint_correlation_matrix"]),
            6,
        )
        self.assertEqual(len(explanation["latest_diagnostic"]["top_components"]), 2)

    def test_plain_pca_and_momentum_share_artifact_contract(self) -> None:
        dataset = _build_joint_pca_dataset()
        pca_strategy = PlainPCAStrategy(
            StrategyConfig(
                name="pca_plain",
                params_file=None,
                params={"q": 0.25, "rolling_window": 5, "components": 2},
            )
        )
        mom_strategy = MomentumStrategy(
            StrategyConfig(
                name="mom",
                params_file=None,
                params={"q": 0.25, "rolling_window": 5},
            )
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            pca_artifact = pca_strategy.run(dataset, output_dir=output_dir / "pca")
            mom_artifact = mom_strategy.run(dataset, output_dir=output_dir / "mom")

            self.assertEqual(pca_artifact.signal_columns, mom_artifact.signal_columns)
            self.assertTrue(pca_artifact.metadata["backtest_ready"])
            self.assertTrue(mom_artifact.metadata["backtest_ready"])
            self.assertTrue(pca_artifact.artifact_paths["signals"].exists())
            self.assertTrue(mom_artifact.artifact_paths["signals"].exists())
            pca_explanation = json.loads(
                pca_artifact.artifact_paths["explanation"].read_text(encoding="utf-8")
            )
            self.assertEqual(pca_explanation["latest_diagnostic"]["signal_date"], "2020-01-13")


if __name__ == "__main__":
    unittest.main()
