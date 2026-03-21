from __future__ import annotations

import unittest
from datetime import date
from pathlib import Path

from llm_strategy_lab.data_pipeline import AlignedMarketReturn, PreparedResearchDataset
from llm_strategy_lab.models import StrategyConfig
from llm_strategy_lab.strategies import PlainPCAStrategy, SubspacePCAStrategy, create_strategy


def _build_subspace_dataset() -> PreparedResearchDataset:
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
        "ENERGY": (1.0, 1.0, 1.0),
        "CONSUMER_STAPLES": (1.0, 1.0, -1.0),
    }
    jp_specs = {
        "ENERGY_RESOURCES": (1.0, -1.0, 1.0),
        "FOODS": (1.0, -1.0, -1.0),
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


class SubspacePCAStrategyTests(unittest.TestCase):
    def test_create_strategy_returns_subspace_pca_strategy(self) -> None:
        strategy = create_strategy(
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
        self.assertIsInstance(strategy, SubspacePCAStrategy)

    def test_subspace_pca_uses_predefined_basis_and_recovers_expected_direction(self) -> None:
        dataset = _build_subspace_dataset()
        strategy = SubspacePCAStrategy(
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

        signals = strategy.compute_signal(dataset)
        portfolio = strategy.build_portfolio(signals)
        explanation = strategy.explain(
            dataset=dataset,
            signals=signals,
            portfolio=portfolio,
        )

        self.assertEqual(len(signals), 2)
        self.assertEqual(len(portfolio), 2)
        self.assertEqual({row.sector for row in signals if row.signal > 0}, {"ENERGY_RESOURCES"})
        self.assertEqual({row.sector for row in signals if row.signal < 0}, {"FOODS"})
        self.assertEqual(explanation["parameters"]["regularization"], 0.1)
        self.assertEqual(
            explanation["parameters"]["subspace"],
            ["global", "country_spread", "cyclical_vs_defensive"],
        )
        self.assertEqual(explanation["comparison_ready_with"], ["pca_plain", "mom"])
        latest = explanation["latest_diagnostic"]
        self.assertEqual(
            latest["subspace_names"],
            ["global", "country_spread", "cyclical_vs_defensive"],
        )
        self.assertEqual(len(latest["projected_subspace_matrix"]), 3)
        self.assertEqual(len(latest["regularized_subspace_matrix"]), 3)
        self.assertEqual(len(latest["orthonormal_basis"][0]), 3)
        self.assertGreaterEqual(len(latest["top_components"]), 2)

    def test_subspace_and_plain_pca_are_comparable_on_same_input(self) -> None:
        dataset = _build_subspace_dataset()
        pca_sub = SubspacePCAStrategy(
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
        pca_plain = PlainPCAStrategy(
            StrategyConfig(
                name="pca_plain",
                params_file=None,
                params={"q": 0.25, "rolling_window": 5, "components": 2},
            )
        )

        sub_signals = pca_sub.compute_signal(dataset)
        plain_signals = pca_plain.compute_signal(dataset)

        self.assertEqual(len(sub_signals), len(plain_signals))
        self.assertEqual(
            {row.signal_date.isoformat() for row in sub_signals},
            {row.signal_date.isoformat() for row in plain_signals},
        )

    def test_pca_sub_math_note_exists(self) -> None:
        note_path = Path("docs/specs/pca-subspace-notes.md")
        note_text = note_path.read_text(encoding="utf-8")

        self.assertIn("S_t^(reg)", note_text)
        self.assertIn("country_spread", note_text)
        self.assertIn("cyclical_vs_defensive", note_text)


if __name__ == "__main__":
    unittest.main()
