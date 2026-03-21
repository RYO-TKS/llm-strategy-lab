from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from llm_strategy_lab.comparison import compare_runs
from llm_strategy_lab.proposals import (
    ProposalValidationError,
    build_prompt_bundle,
    validate_and_save_proposal,
)
from llm_strategy_lab.runner import run_experiment


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


class ProposalWorkflowTests(unittest.TestCase):
    def test_build_prompt_bundle_writes_bundle_and_schema(self) -> None:
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

            prompt_bundle_path = build_prompt_bundle(comparison_dir)

            schema_path = comparison_dir / "proposal_schema.json"
            self.assertEqual(prompt_bundle_path, comparison_dir / "prompt_bundle.json")
            self.assertTrue(prompt_bundle_path.exists())
            self.assertTrue(schema_path.exists())

            prompt_bundle = json.loads(prompt_bundle_path.read_text(encoding="utf-8"))
            schema = json.loads(schema_path.read_text(encoding="utf-8"))

            self.assertEqual(prompt_bundle["lineage_id"], "sample_research-0001-to-0002")
            self.assertEqual(prompt_bundle["parent_run_id"], "0001")
            self.assertEqual(prompt_bundle["candidate_run_id"], "0002")
            self.assertEqual(
                prompt_bundle["comparison_summary"]["candidate_strategy"],
                "pca_plain",
            )
            self.assertTrue(prompt_bundle["comparison_summary"]["metric_highlights"])
            self.assertEqual(
                schema["properties"]["lineage_id"]["const"],
                "sample_research-0001-to-0002",
            )
            self.assertIn("Respond with JSON only", prompt_bundle["messages"][0]["content"])
            self.assertIn("Top metric deltas:", prompt_bundle["messages"][1]["content"])

    def test_validate_and_save_proposal_rejects_invalid_and_saves_valid_artifact(self) -> None:
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

            invalid_proposal_path = _write_json(
                Path(tmpdir) / "invalid_proposal.json",
                {
                    "proposal_id": "proposal-0001",
                    "lineage_id": "sample_research-0001-to-0002",
                    "parent_run_id": "0001",
                    "candidate_run_id": "0002",
                    "hypothesis": "PCA plain should continue to beat the baseline.",
                    "rationale": (
                        "The comparison suggests different factor exposure and "
                        "turnover behavior."
                    ),
                    "strategy_delta": {
                        "summary": "Keep PCA and extend one parameter.",
                        "strategy_name": "pca_plain",
                        "parameter_changes": [
                            {
                                "path": "strategy_config.params.components",
                                "operation": "set",
                                "value": 3,
                                "reason": "Expand the PCA basis slightly.",
                            }
                        ],
                    },
                },
            )

            with self.assertRaises(ProposalValidationError) as error_context:
                validate_and_save_proposal(comparison_dir, invalid_proposal_path)

            self.assertIn("expected_impact", str(error_context.exception))
            validation_report = json.loads(
                (comparison_dir / "proposal_validation.json").read_text(encoding="utf-8")
            )
            self.assertFalse(validation_report["valid"])
            self.assertFalse((comparison_dir / "proposal_artifact.json").exists())

            valid_proposal = {
                "proposal_id": "proposal-0002",
                "lineage_id": "sample_research-0001-to-0002",
                "parent_run_id": "0001",
                "candidate_run_id": "0002",
                "hypothesis": (
                    "Increasing PCA components while keeping the same framework could retain "
                    "signal breadth without reverting to the baseline factor mix."
                ),
                "rationale": (
                    "The candidate improves model fit materially, but return and turnover moved "
                    "in the wrong direction, so the next change should stay inside the same "
                    "strategy family and test a narrower structural delta."
                ),
                "strategy_delta": {
                    "summary": "Keep pca_plain and raise components from 2 to 3.",
                    "strategy_name": "pca_plain",
                    "parameter_changes": [
                        {
                            "path": "strategy_config.params.components",
                            "operation": "set",
                            "previous_value": 2,
                            "value": 3,
                            "reason": (
                                "Test whether a slightly broader factor basis "
                                "improves return capture."
                            ),
                        }
                    ],
                },
                "expected_impact": {
                    "summary": (
                        "Target better return quality while keeping risk close "
                        "to the current candidate."
                    ),
                    "metric_expectations": [
                        {
                            "metric": "annual_return",
                            "direction": "increase",
                            "reason": "The current candidate trails the parent on annual return.",
                        },
                        {
                            "metric": "average_turnover",
                            "direction": "decrease",
                            "reason": (
                                "Turnover rose sharply versus the parent and "
                                "should be controlled."
                            ),
                        },
                    ],
                },
            }
            valid_proposal_path = _write_json(Path(tmpdir) / "valid_proposal.json", valid_proposal)

            proposal_artifact_path = validate_and_save_proposal(comparison_dir, valid_proposal_path)

            self.assertEqual(proposal_artifact_path, comparison_dir / "proposal_artifact.json")
            saved_proposal = json.loads(proposal_artifact_path.read_text(encoding="utf-8"))
            saved_validation_report = json.loads(
                (comparison_dir / "proposal_validation.json").read_text(encoding="utf-8")
            )

            self.assertEqual(saved_proposal["proposal_id"], "proposal-0002")
            self.assertTrue(saved_validation_report["valid"])
            self.assertEqual(saved_validation_report["errors"], [])


if __name__ == "__main__":
    unittest.main()
