from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from llm_strategy_lab.comparison import compare_runs
from llm_strategy_lab.runner import run_experiment


class RunComparisonTests(unittest.TestCase):
    def test_compare_runs_writes_lineage_manifest_and_summary(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        config_path = project_root / "configs" / "experiments" / "sample_research.yaml"

        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "runs"
            parent_run = run_experiment(config_path=config_path, output_root=output_root)
            candidate_run = run_experiment(
                config_path=config_path,
                output_root=output_root,
                strategy_name="pca_plain",
            )

            comparison_dir = compare_runs(parent_run=parent_run, candidate_run=candidate_run)

            self.assertEqual(
                comparison_dir,
                (output_root / "comparisons" / "sample_research-0001-to-0002").resolve(),
            )
            manifest_path = comparison_dir / "comparison_manifest.json"
            summary_path = comparison_dir / "SUMMARY.md"
            self.assertTrue(manifest_path.exists())
            self.assertTrue(summary_path.exists())

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            summary = summary_path.read_text(encoding="utf-8")

            self.assertEqual(manifest["lineage_id"], "sample_research-0001-to-0002")
            self.assertEqual(manifest["parent_run_id"], "0001")
            self.assertEqual(manifest["candidate_run_id"], "0002")
            self.assertEqual(manifest["parent_run"]["strategy"], "mom")
            self.assertEqual(manifest["candidate_run"]["strategy"], "pca_plain")
            self.assertIn("annual_return", manifest["metric_comparison"]["delta"])
            self.assertEqual(
                manifest["factor_regression_comparison"]["models"]["ff3"]["parent_status"],
                "ok",
            )
            self.assertEqual(
                manifest["factor_regression_comparison"]["models"]["ff3"]["candidate_status"],
                "ok",
            )
            self.assertEqual(
                manifest["config_diff"]["changed"]["strategy_config.name"]["candidate"],
                "pca_plain",
            )
            self.assertEqual(
                manifest["config_diff"]["changed"]["cli_overrides.strategy_name"]["candidate"],
                "pca_plain",
            )
            self.assertIn("Lineage ID is `sample_research-0001-to-0002`", summary)
            self.assertIn("`strategy_config.name` changed from `mom` to `pca_plain`.", summary)


if __name__ == "__main__":
    unittest.main()
