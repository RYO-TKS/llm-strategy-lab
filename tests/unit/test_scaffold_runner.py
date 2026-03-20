from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from llm_strategy_lab.runner import (
    create_scaffold_run,
    load_config,
)

SAMPLE_CONFIG = """\
experiment_id: sample_research
environment: dev
strategy:
  name: pca_sub
backtest:
  start: "2020-01-01"
  end: "2020-12-31"
"""


class ScaffoldRunnerTests(unittest.TestCase):
    def test_load_config_returns_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(SAMPLE_CONFIG, encoding="utf-8")

            loaded = load_config(config_path)

            self.assertEqual(loaded["experiment_id"], "sample_research")
            self.assertEqual(loaded["strategy"]["name"], "pca_sub")

    def test_create_scaffold_run_writes_artifacts_and_increments_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "config.yaml"
            config_path.write_text(SAMPLE_CONFIG, encoding="utf-8")

            first_run = create_scaffold_run(config_path=config_path, output_root=root / "runs")
            second_run = create_scaffold_run(config_path=config_path, output_root=root / "runs")

            self.assertEqual(first_run.name, "0001")
            self.assertEqual(second_run.name, "0002")

            manifest = json.loads((first_run / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "scaffold")
            self.assertTrue((first_run / "SUMMARY.md").exists())


if __name__ == "__main__":
    unittest.main()
