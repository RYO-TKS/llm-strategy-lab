from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from llm_strategy_lab.config import load_experiment_config
from llm_strategy_lab.constants import COLUMN_GROUPS, PRICE_COLUMNS, SIGNAL
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
                root / "configs" / "strategies" / "pca_sub.default.yaml",
                "name: pca_sub\nlookback_window: 60\nquantile: 0.3\n",
            )
            _write_file(
                root / "configs" / "experiments" / "sample.yaml",
                "\n".join(
                    [
                        "experiment_id: sample_research",
                        "environment: dev",
                        "strategy:",
                        "  name: pca_sub",
                        "  params_file: configs/strategies/pca_sub.default.yaml",
                        "dataset:",
                        "  us_sectors: data/sample/us.csv",
                        "  jp_sectors: data/sample/jp.csv",
                        "  trading_calendar: data/sample/trading_calendar.csv",
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
                (root / "configs" / "strategies" / "pca_sub.default.yaml").resolve(),
            )
            self.assertEqual(config.strategy.params["lookback_window"], 60)
            self.assertEqual(
                config.dataset.us_sectors,
                (root / "data" / "sample" / "us.csv").resolve(),
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
                root / "configs" / "strategies" / "pca_sub.default.yaml",
                "name: pca_sub\nlookback_window: 60\nquantile: 0.3\n",
            )
            config_path = root / "configs" / "experiments" / "sample.yaml"
            _write_file(
                config_path,
                "\n".join(
                    [
                        "experiment_id: sample_research",
                        "environment: dev",
                        "strategy:",
                        "  name: pca_sub",
                        "  params_file: configs/strategies/pca_sub.default.yaml",
                        "dataset:",
                        "  us_sectors: data/sample/us.csv",
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

            self.assertEqual(manifest["status"], "scaffold")
            self.assertEqual(manifest["environment"], "dev")
            self.assertEqual(manifest["strategy"], "pca_sub")
            self.assertEqual(manifest["notes"], ["typed config ready"])
            self.assertEqual(manifest["metadata"]["environment_config"]["seed"], 42)
            self.assertIn("Backtest Window", summary)
            self.assertIn("Dataset Inputs", summary)

    def test_constants_expose_expected_groups(self) -> None:
        self.assertIn("price", COLUMN_GROUPS)
        self.assertIn("signal", COLUMN_GROUPS)
        self.assertIn(SIGNAL, COLUMN_GROUPS["signal"])
        self.assertIn("date", PRICE_COLUMNS)


if __name__ == "__main__":
    unittest.main()
