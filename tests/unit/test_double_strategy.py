from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from llm_strategy_lab.data_pipeline import AlignedMarketReturn, PreparedResearchDataset
from llm_strategy_lab.models import StrategyConfig
from llm_strategy_lab.strategies import (
    DoubleSortStrategy,
    MomentumStrategy,
    SubspacePCAStrategy,
    create_strategy,
)


def _build_double_sort_dataset() -> PreparedResearchDataset:
    signal_dates = [
        date(2020, 1, 6),
        date(2020, 1, 7),
        date(2020, 1, 8),
        date(2020, 1, 9),
        date(2020, 1, 10),
        date(2020, 1, 13),
    ]
    global_factor = [-0.03, -0.02, 0.0, 0.01, 0.03, 0.05]
    country_spread = [0.02, 0.01, 0.0, -0.01, -0.02, -0.03]
    cyclical_spread = [-0.02, -0.01, 0.0, 0.01, 0.03, 0.04]

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


def _double_params() -> dict[str, object]:
    return {
        "q": 0.25,
        "rolling_window": 5,
        "mom": {
            "rolling_window": 5,
        },
        "pca_sub": {
            "rolling_window": 5,
            "components": 3,
            "regularization": 0.1,
            "subspace": ["global", "country_spread", "cyclical_vs_defensive"],
        },
    }


class DoubleSortStrategyTests(unittest.TestCase):
    def test_create_strategy_returns_double_sort_strategy(self) -> None:
        strategy = create_strategy(
            StrategyConfig(
                name="double",
                params_file=None,
                params=_double_params(),
            )
        )
        self.assertIsInstance(strategy, DoubleSortStrategy)

    def test_double_sort_intersects_mom_and_pca_sub_buckets(self) -> None:
        dataset = _build_double_sort_dataset()
        strategy = DoubleSortStrategy(
            StrategyConfig(
                name="double",
                params_file=None,
                params=_double_params(),
            )
        )

        signals = strategy.compute_signal(dataset)
        portfolio = strategy.build_portfolio(signals)
        explanation = strategy.explain(
            dataset=dataset,
            signals=signals,
            portfolio=portfolio,
        )

        self.assertEqual(len(signals), 4)
        self.assertEqual(len(portfolio), 2)
        self.assertEqual({row.sector for row in signals if row.signal > 0}, {"ENERGY_RESOURCES"})
        self.assertEqual({row.sector for row in signals if row.signal < 0}, {"PHARMACEUTICALS"})
        self.assertEqual({row.side for row in portfolio}, {"long", "short"})
        self.assertTrue(all(abs(row.gross_exposure - 1.0) < 1e-9 for row in portfolio))
        self.assertTrue(all(abs(row.net_exposure) < 1e-9 for row in portfolio))
        latest = explanation["latest_diagnostic"]
        self.assertEqual(
            latest["quadrant_members"]["mom_high_pca_high"],
            ["ENERGY_RESOURCES"],
        )
        self.assertEqual(
            latest["quadrant_members"]["mom_low_pca_low"],
            ["PHARMACEUTICALS"],
        )
        self.assertEqual(latest["quadrant_members"]["mom_high_pca_low"], [])
        self.assertEqual(latest["quadrant_members"]["mom_low_pca_high"], [])
        self.assertEqual(
            set(latest["quadrant_members"]["other"]),
            {"TRANSPORTATION_LOGISTICS", "FOODS"},
        )
        self.assertEqual(explanation["comparison_ready_with"], ["mom", "pca_plain", "pca_sub"])

    def test_double_sort_shares_portfolio_contract_with_mom_and_pca_sub(self) -> None:
        dataset = _build_double_sort_dataset()
        double_strategy = DoubleSortStrategy(
            StrategyConfig(
                name="double",
                params_file=None,
                params=_double_params(),
            )
        )
        mom_strategy = MomentumStrategy(
            StrategyConfig(
                name="mom",
                params_file=None,
                params={"q": 0.25, "rolling_window": 5},
            )
        )
        pca_sub_strategy = SubspacePCAStrategy(
            StrategyConfig(
                name="pca_sub",
                params_file=None,
                params={
                    "q": 0.25,
                    "rolling_window": 5,
                    "components": 3,
                    "regularization": 0.1,
                    "subspace": ["global", "country_spread", "cyclical_vs_defensive"],
                },
            )
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            double_artifact = double_strategy.run(dataset, output_dir=output_dir / "double")
            mom_artifact = mom_strategy.run(dataset, output_dir=output_dir / "mom")
            pca_sub_artifact = pca_sub_strategy.run(dataset, output_dir=output_dir / "pca_sub")

            self.assertEqual(double_artifact.signal_columns, mom_artifact.signal_columns)
            self.assertEqual(double_artifact.signal_columns, pca_sub_artifact.signal_columns)
            self.assertTrue(double_artifact.metadata["backtest_ready"])
            self.assertTrue(double_artifact.artifact_paths["portfolio"].exists())
            explanation = json.loads(
                double_artifact.artifact_paths["explanation"].read_text(encoding="utf-8")
            )
            self.assertEqual(explanation["latest_diagnostic"]["long_bucket_size"], 1)
            self.assertEqual(explanation["latest_diagnostic"]["short_bucket_size"], 1)


if __name__ == "__main__":
    unittest.main()
