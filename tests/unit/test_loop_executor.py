from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from llm_strategy_lab.comparison import compare_runs
from llm_strategy_lab.loop_executor import (
    _build_quality_gate,
    generate_auto_proposal,
    run_improvement_loop,
)
from llm_strategy_lab.runner import run_experiment


class LoopExecutorTests(unittest.TestCase):
    def test_generate_auto_proposal_writes_schema_compatible_payload(self) -> None:
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
            proposal_output_dir = Path(tmpdir) / "proposal"

            proposal_path = generate_auto_proposal(
                comparison_dir,
                output_root=proposal_output_dir,
                iteration_index=1,
            )

            proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
            self.assertEqual(
                proposal["proposal_id"],
                "auto-sample_research-0001-to-0002-iter-01",
            )
            self.assertEqual(proposal["lineage_id"], "sample_research-0001-to-0002")
            self.assertEqual(proposal["strategy_delta"]["strategy_name"], "pca_plain")
            self.assertGreaterEqual(len(proposal["strategy_delta"]["parameter_changes"]), 1)

    def test_build_quality_gate_accepts_improving_candidate(self) -> None:
        quality_gate = _build_quality_gate(
            {
                "metric_comparison": {
                    "parent": {
                        "annual_return": 1.0,
                        "return_risk_ratio": 2.0,
                        "max_drawdown": 0.05,
                        "average_turnover": 0.2,
                    },
                    "candidate": {
                        "annual_return": 1.2,
                        "return_risk_ratio": 2.3,
                        "max_drawdown": 0.05,
                        "average_turnover": 0.19,
                    },
                    "delta": {
                        "annual_return": 0.2,
                        "return_risk_ratio": 0.3,
                        "max_drawdown": 0.0,
                        "average_turnover": -0.01,
                    },
                }
            },
            min_annual_return_delta=0.0,
            min_return_risk_ratio_delta=0.0,
            max_drawdown_increase=0.0,
            max_turnover_increase=0.0,
        )

        self.assertTrue(quality_gate["passed"])
        self.assertTrue(all(check["passed"] for check in quality_gate["checks"]))

    def test_run_improvement_loop_writes_reject_artifact_for_sample_data(self) -> None:
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

            loop_dir = run_improvement_loop(
                parent_run=parent_run,
                candidate_run=candidate_run,
                output_root=output_root,
                max_iterations=1,
                no_improvement_limit=1,
            )

            self.assertEqual(
                loop_dir,
                (output_root / "loops" / "sample_research" / "0001").resolve(),
            )
            manifest_path = loop_dir / "loop_manifest.json"
            summary_path = loop_dir / "SUMMARY.md"
            self.assertTrue(manifest_path.exists())
            self.assertTrue(summary_path.exists())

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["stop_reason"], "quality_gate_failed")
            self.assertEqual(manifest["final_accepted_run_id"], "0001")
            self.assertEqual(len(manifest["iterations"]), 1)
            first_iteration = manifest["iterations"][0]
            self.assertEqual(first_iteration["decision"], "reject")
            self.assertEqual(first_iteration["decision_reason"], "quality_gate_failed")
            self.assertTrue(
                Path(first_iteration["artifact_paths"]["prompt_bundle"]).exists()
            )
            self.assertTrue(
                Path(first_iteration["artifact_paths"]["generated_proposal"]).exists()
            )
            self.assertTrue(
                Path(first_iteration["artifact_paths"]["proposal_artifact"]).exists()
            )
            self.assertTrue(Path(first_iteration["artifact_paths"]["child_run_dir"]).exists())
            self.assertTrue(
                Path(first_iteration["artifact_paths"]["candidate_comparison"]).exists()
            )


if __name__ == "__main__":
    unittest.main()
