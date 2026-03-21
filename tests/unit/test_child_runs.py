from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml
from llm_strategy_lab.child_runs import create_child_run
from llm_strategy_lab.comparison import compare_runs
from llm_strategy_lab.runner import run_experiment


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


class ChildRunTests(unittest.TestCase):
    def test_create_child_run_generates_snapshot_and_lineage_metadata(self) -> None:
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
            proposal_file = _write_json(
                Path(tmpdir) / "child_proposal.json",
                {
                    "proposal_id": "proposal-child-0001",
                    "lineage_id": "sample_research-0001-to-0002",
                    "parent_run_id": "0001",
                    "candidate_run_id": "0002",
                    "hypothesis": (
                        "Keeping the pca_plain family while broadening the component count may "
                        "retain fit quality and recover part of the return loss."
                    ),
                    "rationale": (
                        "The current candidate improved model fit but still trails the parent "
                        "on annual return, so the next step should preserve the same family "
                        "and test a narrow parameter delta."
                    ),
                    "strategy_delta": {
                        "summary": (
                            "Switch to pca_plain, raise components to 3, "
                            "and rebalance weekly."
                        ),
                        "strategy_name": "pca_plain",
                        "parameter_changes": [
                            {
                                "path": "strategy_config.params.components",
                                "operation": "set",
                                "previous_value": 2,
                                "value": 3,
                                "reason": "Expand the PCA basis slightly.",
                            },
                            {
                                "path": "backtest_config.rebalance",
                                "operation": "set",
                                "previous_value": "monthly",
                                "value": "weekly",
                                "reason": "Shorten the rebalance interval for the child run.",
                            },
                        ],
                    },
                    "expected_impact": {
                        "summary": "Aim for better annual return without discarding the fit gain.",
                        "metric_expectations": [
                            {
                                "metric": "annual_return",
                                "direction": "increase",
                                "reason": "The parent still leads on annual return.",
                            }
                        ],
                    },
                },
            )

            child_run_dir = create_child_run(comparison_dir, proposal_file)

            self.assertEqual(child_run_dir, (output_root / "sample_research" / "0003").resolve())
            snapshot_path = child_run_dir / "child_config_snapshot.yaml"
            proposal_snapshot_path = child_run_dir / "applied_proposal_artifact.json"
            manifest_path = child_run_dir / "manifest.json"
            self.assertTrue(snapshot_path.exists())
            self.assertTrue(proposal_snapshot_path.exists())
            self.assertTrue(manifest_path.exists())

            snapshot = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(snapshot["strategy"]["name"], "pca_plain")
            self.assertEqual(snapshot["strategy"]["components"], 3)
            self.assertEqual(snapshot["backtest"]["rebalance"], "weekly")
            self.assertEqual(manifest["strategy"], "pca_plain")
            self.assertEqual(manifest["config_path"], str(snapshot_path.resolve()))
            self.assertEqual(manifest["metadata"]["strategy_config"]["params"]["components"], 3)
            self.assertEqual(manifest["metadata"]["backtest_config"]["rebalance"], "weekly")
            self.assertEqual(manifest["metadata"]["lineage"]["parent_run_id"], "0001")
            self.assertEqual(manifest["metadata"]["lineage"]["candidate_run_id"], "0002")
            self.assertEqual(
                manifest["metadata"]["lineage"]["proposal_id"],
                "proposal-child-0001",
            )
            self.assertEqual(
                manifest["metadata"]["lineage"]["child_config_snapshot_path"],
                str(snapshot_path.resolve()),
            )
            self.assertEqual(
                manifest["metadata"]["lineage"]["proposal_snapshot_path"],
                str(proposal_snapshot_path.resolve()),
            )
            self.assertEqual(
                manifest["metadata"]["proposal_summary"]["applied_changes"][0]["path"],
                "strategy_config.params.components",
            )
            self.assertEqual(
                manifest["metadata"]["proposal_summary"]["applied_changes"][1]["path"],
                "backtest_config.rebalance",
            )


if __name__ == "__main__":
    unittest.main()
