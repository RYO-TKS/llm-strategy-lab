from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from llm_strategy_lab.config import load_experiment_config
from llm_strategy_lab.constants import COLUMN_GROUPS, PRICE_COLUMNS, SIGNAL
from llm_strategy_lab.data_pipeline import prepare_aligned_research_dataset
from llm_strategy_lab.runner import create_scaffold_run


def _write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class ConfigAndRunnerTests(unittest.TestCase):
    def test_load_experiment_config_resolves_paths_and_typed_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_file(root / "pyproject.toml", "[project]\nname = 'llm-strategy-lab'\n")
            _write_file(
                root / "configs" / "environments" / "dev.yaml",
                "name: dev\noutput_root: runs\nlog_level: INFO\nseed: 7\n",
            )
            _write_file(
                root / "configs" / "strategies" / "mom.yaml",
                "name: mom\nq: 0.3\nrolling_window: 1\n",
            )
            _write_file(
                root / "configs" / "experiments" / "sample.yaml",
                "\n".join(
                    [
                        "experiment_id: sample_research",
                        "environment: dev",
                        "strategy:",
                        "  name: mom",
                        "  params_file: configs/strategies/mom.yaml",
                        "dataset:",
                        "  us_sectors: data/sample/us.csv",
                        "  jp_sectors: data/sample/jp.csv",
                        "  trading_calendar: data/sample/trading_calendar.csv",
                        "  factor_returns: data/sample/factor_returns.csv",
                        "backtest:",
                        '  start: "2020-01-01"',
                        '  end: "2020-12-31"',
                        "  rebalance: monthly",
                        "notes:",
                        "  - first note",
                        "  - second note",
                    ]
                )
                + "\n",
            )

            config = load_experiment_config(root / "configs" / "experiments" / "sample.yaml")

            self.assertEqual(config.experiment_id, "sample_research")
            self.assertEqual(config.environment.name, "dev")
            self.assertEqual(config.environment.output_root, (root / "runs").resolve())
            self.assertEqual(
                config.strategy.params_file,
                (root / "configs" / "strategies" / "mom.yaml").resolve(),
            )
            self.assertEqual(config.strategy.params["rolling_window"], 1)
            self.assertEqual(
                config.dataset.us_sectors,
                (root / "data" / "sample" / "us.csv").resolve(),
            )
            self.assertEqual(
                config.dataset.factor_returns,
                (root / "data" / "sample" / "factor_returns.csv").resolve(),
            )
            self.assertEqual(config.backtest.start, date(2020, 1, 1))
            self.assertEqual(config.backtest.end, date(2020, 12, 31))
            self.assertEqual(config.notes, ("first note", "second note"))

    def test_load_experiment_config_merges_inline_strategy_params(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_file(root / "pyproject.toml", "[project]\nname = 'llm-strategy-lab'\n")
            _write_file(
                root / "configs" / "environments" / "dev.yaml",
                "name: dev\noutput_root: runs\n",
            )
            _write_file(
                root / "configs" / "strategies" / "mom.yaml",
                "name: mom\nquantile: 0.3\nrolling_window: 120\n",
            )
            _write_file(
                root / "configs" / "experiments" / "sample.yaml",
                "\n".join(
                    [
                        "experiment_id: sample_research",
                        "environment: dev",
                        "strategy:",
                        "  name: mom",
                        "  params_file: configs/strategies/mom.yaml",
                        "  quantile: 0.2",
                        "backtest:",
                        "  rebalance: monthly",
                    ]
                )
                + "\n",
            )

            config = load_experiment_config(root / "configs" / "experiments" / "sample.yaml")

            self.assertEqual(config.strategy.params["quantile"], 0.2)
            self.assertEqual(config.strategy.params["rolling_window"], 120)

    def test_create_scaffold_run_uses_environment_output_root_and_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_file(root / "pyproject.toml", "[project]\nname = 'llm-strategy-lab'\n")
            _write_file(
                root / "configs" / "environments" / "dev.yaml",
                "name: dev\noutput_root: runs\nlog_level: INFO\nseed: 42\n",
            )
            _write_file(
                root / "configs" / "strategies" / "mom.yaml",
                "name: mom\nq: 0.3\nrolling_window: 1\n",
            )
            _write_file(
                root / "data" / "sample" / "us.csv",
                "\n".join(
                    [
                        "date,sector,open,close",
                        "2020-01-06,COMMUNICATION_SERVICES,99.4,100.0",
                        "2020-01-07,COMMUNICATION_SERVICES,100.4,101.8",
                        "2020-01-08,COMMUNICATION_SERVICES,101.2,101.1",
                        "2020-01-06,CONSUMER_DISCRETIONARY,106.4,107.0",
                        "2020-01-07,CONSUMER_DISCRETIONARY,107.4,108.8",
                        "2020-01-08,CONSUMER_DISCRETIONARY,108.2,108.1",
                        "2020-01-06,CONSUMER_STAPLES,113.4,114.0",
                        "2020-01-07,CONSUMER_STAPLES,114.4,115.8",
                        "2020-01-08,CONSUMER_STAPLES,115.2,115.1",
                        "2020-01-06,ENERGY,120.4,121.0",
                        "2020-01-07,ENERGY,121.4,122.8",
                        "2020-01-08,ENERGY,122.2,122.1",
                        "2020-01-06,FINANCIALS,127.4,128.0",
                        "2020-01-07,FINANCIALS,128.4,129.8",
                        "2020-01-08,FINANCIALS,129.2,129.1",
                        "2020-01-06,HEALTH_CARE,134.4,135.0",
                        "2020-01-07,HEALTH_CARE,135.4,136.8",
                        "2020-01-08,HEALTH_CARE,136.2,136.1",
                        "2020-01-06,INDUSTRIALS,141.4,142.0",
                        "2020-01-07,INDUSTRIALS,142.4,143.8",
                        "2020-01-08,INDUSTRIALS,143.2,143.1",
                        "2020-01-06,INFORMATION_TECHNOLOGY,148.4,149.0",
                        "2020-01-07,INFORMATION_TECHNOLOGY,149.4,150.8",
                        "2020-01-08,INFORMATION_TECHNOLOGY,150.2,150.1",
                        "2020-01-06,MATERIALS,155.4,156.0",
                        "2020-01-07,MATERIALS,156.4,157.8",
                        "2020-01-08,MATERIALS,157.2,157.1",
                        "2020-01-06,REAL_ESTATE,162.4,163.0",
                        "2020-01-07,REAL_ESTATE,163.4,164.8",
                        "2020-01-08,REAL_ESTATE,164.2,164.1",
                        "2020-01-06,UTILITIES,169.4,170.0",
                        "2020-01-07,UTILITIES,170.4,171.8",
                        "2020-01-08,UTILITIES,171.2,171.1",
                    ]
                )
                + "\n",
            )
            _write_file(
                root / "data" / "sample" / "jp.csv",
                "\n".join(
                    [
                        "date,sector,open,close",
                        "2020-01-07,FOODS,49.7,50.7",
                        "2020-01-08,FOODS,50.2,51.0",
                        "2020-01-10,FOODS,50.1,49.6",
                        "2020-01-07,ENERGY_RESOURCES,52.7,53.7",
                        "2020-01-08,ENERGY_RESOURCES,53.2,54.0",
                        "2020-01-10,ENERGY_RESOURCES,53.1,52.6",
                        "2020-01-07,CONSTRUCTION_MATERIALS,55.7,56.7",
                        "2020-01-08,CONSTRUCTION_MATERIALS,56.2,57.0",
                        "2020-01-10,CONSTRUCTION_MATERIALS,56.1,55.6",
                        "2020-01-07,RAW_MATERIALS_CHEMICALS,58.7,59.7",
                        "2020-01-08,RAW_MATERIALS_CHEMICALS,59.2,60.0",
                        "2020-01-10,RAW_MATERIALS_CHEMICALS,59.1,58.6",
                        "2020-01-07,PHARMACEUTICALS,61.7,62.7",
                        "2020-01-08,PHARMACEUTICALS,62.2,63.0",
                        "2020-01-10,PHARMACEUTICALS,62.1,61.6",
                        "2020-01-07,AUTOMOBILES_TRANSPORTATION,64.7,65.7",
                        "2020-01-08,AUTOMOBILES_TRANSPORTATION,65.2,66.0",
                        "2020-01-10,AUTOMOBILES_TRANSPORTATION,65.1,64.6",
                        "2020-01-07,STEEL_NONFERROUS,67.7,68.7",
                        "2020-01-08,STEEL_NONFERROUS,68.2,69.0",
                        "2020-01-10,STEEL_NONFERROUS,68.1,67.6",
                        "2020-01-07,MACHINERY,70.7,71.7",
                        "2020-01-08,MACHINERY,71.2,72.0",
                        "2020-01-10,MACHINERY,71.1,70.6",
                        "2020-01-07,ELECTRIC_APPLIANCES_PRECISION,73.7,74.7",
                        "2020-01-08,ELECTRIC_APPLIANCES_PRECISION,74.2,75.0",
                        "2020-01-10,ELECTRIC_APPLIANCES_PRECISION,74.1,73.6",
                        "2020-01-07,IT_SERVICES_OTHERS,76.7,77.7",
                        "2020-01-08,IT_SERVICES_OTHERS,77.2,78.0",
                        "2020-01-10,IT_SERVICES_OTHERS,77.1,76.6",
                        "2020-01-07,ELECTRIC_POWER_GAS,79.7,80.7",
                        "2020-01-08,ELECTRIC_POWER_GAS,80.2,81.0",
                        "2020-01-10,ELECTRIC_POWER_GAS,80.1,79.6",
                        "2020-01-07,TRANSPORTATION_LOGISTICS,82.7,83.7",
                        "2020-01-08,TRANSPORTATION_LOGISTICS,83.2,84.0",
                        "2020-01-10,TRANSPORTATION_LOGISTICS,83.1,82.6",
                        "2020-01-07,COMMERCIAL_WHOLESALE_TRADE,85.7,86.7",
                        "2020-01-08,COMMERCIAL_WHOLESALE_TRADE,86.2,87.0",
                        "2020-01-10,COMMERCIAL_WHOLESALE_TRADE,86.1,85.6",
                        "2020-01-07,RETAIL_TRADE,88.7,89.7",
                        "2020-01-08,RETAIL_TRADE,89.2,90.0",
                        "2020-01-10,RETAIL_TRADE,89.1,88.6",
                        "2020-01-07,BANKS,91.7,92.7",
                        "2020-01-08,BANKS,92.2,93.0",
                        "2020-01-10,BANKS,92.1,91.6",
                        "2020-01-07,FINANCIALS_EX_BANKS,94.7,95.7",
                        "2020-01-08,FINANCIALS_EX_BANKS,95.2,96.0",
                        "2020-01-10,FINANCIALS_EX_BANKS,95.1,94.6",
                        "2020-01-07,REAL_ESTATE,97.7,98.7",
                        "2020-01-08,REAL_ESTATE,98.2,99.0",
                        "2020-01-10,REAL_ESTATE,98.1,97.6",
                    ]
                )
                + "\n",
            )
            _write_file(
                root / "data" / "sample" / "trading_calendar.csv",
                "\n".join(
                    [
                        "market,date,is_open",
                        "US,2020-01-06,1",
                        "US,2020-01-07,1",
                        "US,2020-01-08,1",
                        "JP,2020-01-06,0",
                        "JP,2020-01-07,1",
                        "JP,2020-01-08,1",
                        "JP,2020-01-09,0",
                        "JP,2020-01-10,1",
                    ]
                )
                + "\n",
            )
            _write_file(
                root / "data" / "sample" / "factor_returns.csv",
                "\n".join(
                    [
                        "date,mkt_rf,smb,hml,umd,rf",
                        "2020-01-07,0.0040,0.0010,-0.0005,0.0008,0.0001",
                        "2020-01-08,0.0035,0.0008,-0.0003,0.0006,0.0001",
                        "2020-01-10,-0.0025,-0.0004,0.0007,-0.0006,0.0001",
                    ]
                )
                + "\n",
            )
            config_path = root / "configs" / "experiments" / "sample.yaml"
            _write_file(
                config_path,
                "\n".join(
                    [
                        "experiment_id: sample_research",
                        "environment: dev",
                        "strategy:",
                        "  name: mom",
                        "  params_file: configs/strategies/mom.yaml",
                        "dataset:",
                        "  us_sectors: data/sample/us.csv",
                        "  jp_sectors: data/sample/jp.csv",
                        "  trading_calendar: data/sample/trading_calendar.csv",
                        "  factor_returns: data/sample/factor_returns.csv",
                        "backtest:",
                        '  start: "2020-01-01"',
                        '  end: "2020-12-31"',
                        "  rebalance: monthly",
                        "notes:",
                        "  - typed config ready",
                    ]
                )
                + "\n",
            )

            first_run = create_scaffold_run(config_path=config_path)
            second_run = create_scaffold_run(config_path=config_path)

            self.assertEqual(first_run.name, "0001")
            self.assertEqual(second_run.name, "0002")
            self.assertEqual(first_run.parent, (root / "runs" / "sample_research").resolve())

            manifest = json.loads((first_run / "manifest.json").read_text(encoding="utf-8"))
            summary = (first_run / "SUMMARY.md").read_text(encoding="utf-8")

            self.assertEqual(manifest["status"], "succeeded")
            self.assertEqual(manifest["environment"], "dev")
            self.assertEqual(manifest["strategy"], "mom")
            self.assertEqual(manifest["notes"], ["typed config ready"])
            self.assertEqual(manifest["metadata"]["environment_config"]["seed"], 42)
            self.assertEqual(manifest["metadata"]["data_alignment"]["aligned_signal_dates"], 2)
            self.assertEqual(len(manifest["strategy_artifacts"]), 1)
            self.assertEqual(manifest["strategy_artifacts"][0]["strategy_name"], "mom")
            self.assertTrue(manifest["strategy_artifacts"][0]["metadata"]["backtest_ready"])
            self.assertIsNotNone(manifest["backtest_result"])
            self.assertIn("annual_return", manifest["backtest_result"]["metrics"])
            self.assertTrue((first_run / "alignment_index.csv").exists())
            self.assertTrue((first_run / "us_aligned_close_to_close.csv").exists())
            self.assertTrue((first_run / "jp_open_to_close.csv").exists())
            self.assertTrue((first_run / "data_quality_log.json").exists())
            self.assertTrue((first_run / "mom_signals.csv").exists())
            self.assertTrue((first_run / "mom_portfolio.csv").exists())
            self.assertTrue((first_run / "mom_explanation.json").exists())
            self.assertTrue((first_run / "mom_backtest_daily.csv").exists())
            self.assertTrue((first_run / "mom_backtest_positions.csv").exists())
            self.assertTrue((first_run / "mom_backtest_metrics.json").exists())
            self.assertTrue((first_run / "mom_factor_regressions.json").exists())
            self.assertTrue((first_run / "mom_signal_ic.csv").exists())
            self.assertTrue((first_run / "mom_equity_curve.svg").exists())
            self.assertTrue((first_run / "mom_drawdown.svg").exists())
            self.assertTrue((first_run / "mom_cumulative_ic.svg").exists())
            self.assertEqual(
                manifest["backtest_result"]["metadata"]["factor_regression_statuses"]["ff3"],
                "insufficient_observations",
            )
            self.assertIn("Backtest Window", summary)
            self.assertIn("Dataset Inputs", summary)
            self.assertIn("Aligned Signal Dates", summary)
            self.assertIn("Portfolio Dates Prepared", summary)
            self.assertIn("Annual Return", summary)
            self.assertIn("Max Drawdown", summary)
            self.assertIn("FF3 Regression", summary)
            self.assertIn("Charts Saved", summary)

    def test_constants_expose_expected_groups(self) -> None:
        self.assertIn("price", COLUMN_GROUPS)
        self.assertIn("signal", COLUMN_GROUPS)
        self.assertIn(SIGNAL, COLUMN_GROUPS["signal"])
        self.assertIn("date", PRICE_COLUMNS)
        self.assertIn("market", COLUMN_GROUPS["portfolio"])
        self.assertIn("sector", COLUMN_GROUPS["portfolio"])

    def test_prepare_aligned_research_dataset_skips_to_next_jp_open_and_logs_missing_sector(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_file(root / "pyproject.toml", "[project]\nname = 'llm-strategy-lab'\n")
            _write_file(
                root / "configs" / "environments" / "dev.yaml",
                "name: dev\noutput_root: runs\n",
            )
            us_rows = ["date,sector,open,close"]
            jp_rows = ["date,sector,open,close"]
            us_sectors = [
                "COMMUNICATION_SERVICES",
                "CONSUMER_DISCRETIONARY",
                "CONSUMER_STAPLES",
                "ENERGY",
                "FINANCIALS",
                "HEALTH_CARE",
                "INDUSTRIALS",
                "INFORMATION_TECHNOLOGY",
                "MATERIALS",
                "REAL_ESTATE",
                "UTILITIES",
            ]
            jp_sectors = [
                "FOODS",
                "ENERGY_RESOURCES",
                "CONSTRUCTION_MATERIALS",
                "RAW_MATERIALS_CHEMICALS",
                "PHARMACEUTICALS",
                "AUTOMOBILES_TRANSPORTATION",
                "STEEL_NONFERROUS",
                "MACHINERY",
                "ELECTRIC_APPLIANCES_PRECISION",
                "IT_SERVICES_OTHERS",
                "ELECTRIC_POWER_GAS",
                "TRANSPORTATION_LOGISTICS",
                "COMMERCIAL_WHOLESALE_TRADE",
                "RETAIL_TRADE",
                "BANKS",
                "FINANCIALS_EX_BANKS",
                "REAL_ESTATE",
            ]
            for idx, sector in enumerate(us_sectors):
                base = 100 + idx * 5
                us_rows.extend(
                    [
                        f"2020-01-06,{sector},{base - 0.5:.1f},{base:.1f}",
                        f"2020-01-07,{sector},{base + 0.2:.1f},{base + 1.5:.1f}",
                        f"2020-01-08,{sector},{base + 0.4:.1f},{base + 1.0:.1f}",
                    ]
                )
            for idx, sector in enumerate(jp_sectors):
                base = 50 + idx * 2
                jp_rows.extend(
                    [
                        f"2020-01-07,{sector},{base - 0.3:.1f},{base + 0.5:.1f}",
                        f"2020-01-08,{sector},{base + 0.1:.1f},{base + 0.8:.1f}",
                    ]
                )
                if sector != "REAL_ESTATE":
                    jp_rows.append(
                        f"2020-01-10,{sector},{base + 0.2:.1f},{base - 0.4:.1f}"
                    )
            _write_file(root / "data" / "sample" / "us.csv", "\n".join(us_rows) + "\n")
            _write_file(root / "data" / "sample" / "jp.csv", "\n".join(jp_rows) + "\n")
            _write_file(
                root / "data" / "sample" / "calendar.csv",
                "\n".join(
                    [
                        "market,date,is_open",
                        "US,2020-01-06,1",
                        "US,2020-01-07,1",
                        "US,2020-01-08,1",
                        "JP,2020-01-07,1",
                        "JP,2020-01-08,1",
                        "JP,2020-01-09,0",
                        "JP,2020-01-10,1",
                    ]
                )
                + "\n",
            )
            config = load_experiment_config(
                _write_and_return_path(
                    root / "configs" / "experiments" / "sample.yaml",
                    "\n".join(
                        [
                            "experiment_id: sample_research",
                            "environment:",
                            "  name: dev",
                            "  output_root: runs",
                            "strategy:",
                            "  name: pca_sub",
                            "backtest:",
                            '  start: "2020-01-01"',
                            '  end: "2020-12-31"',
                            "  rebalance: monthly",
                            "dataset:",
                            "  us_sectors: data/sample/us.csv",
                            "  jp_sectors: data/sample/jp.csv",
                            "  trading_calendar: data/sample/calendar.csv",
                        ]
                    )
                    + "\n",
                )
            )

            prepared = prepare_aligned_research_dataset(
                config.dataset,
                backtest=config.backtest,
            )

            self.assertEqual(len(prepared.alignment_pairs), 2)
            self.assertEqual(
                [pair.jp_signal_date.isoformat() for pair in prepared.alignment_pairs],
                ["2020-01-08", "2020-01-10"],
            )
            self.assertTrue(
                all(
                    pair.jp_signal_date > pair.us_date
                    for pair in prepared.alignment_pairs
                )
            )
            self.assertEqual(len(prepared.us_aligned_returns), 22)
            self.assertEqual(len(prepared.jp_open_to_close_returns), 33)
            self.assertTrue(
                any(event.code == "missing_jp_signal_sector" for event in prepared.quality_events)
            )


def _write_and_return_path(path: Path, content: str) -> Path:
    _write_file(path, content)
    return path


if __name__ == "__main__":
    unittest.main()
