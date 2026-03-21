from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from llm_strategy_lab.cli import main


class CliTests(unittest.TestCase):
    def test_main_runs_explicit_run_command_with_strategy_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "runs"
            params_file = Path(tmpdir) / "pca_sub.yaml"
            with patch(
                "llm_strategy_lab.cli.run_experiment",
                return_value=output_root / "sample_research" / "0001",
            ) as mock_run:
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    result = main(
                        [
                            "run",
                            "--config",
                            "configs/experiments/sample_research.yaml",
                            "--output-root",
                            str(output_root),
                            "--strategy",
                            "pca_sub",
                            "--strategy-params-file",
                            str(params_file),
                        ]
                    )

        self.assertEqual(result, 0)
        self.assertIn("Research run created at", stdout.getvalue())
        self.assertEqual(
            mock_run.call_args.kwargs,
            {
                "config_path": Path("configs/experiments/sample_research.yaml"),
                "output_root": output_root,
                "strategy_name": "pca_sub",
                "strategy_params_file": params_file,
            },
        )

    def test_main_runs_sample_subcommand(self) -> None:
        sample_config = Path("/tmp/sample_research.yaml")
        run_dir = Path("/tmp/runs/sample_research/0001")
        with patch(
            "llm_strategy_lab.cli.resolve_sample_config_path",
            return_value=sample_config,
        ), patch(
            "llm_strategy_lab.cli.run_experiment",
            return_value=run_dir,
        ) as mock_run:
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                result = main(["sample", "--strategy", "double"])

        self.assertEqual(result, 0)
        self.assertIn(str(run_dir), stdout.getvalue())
        self.assertEqual(mock_run.call_args.kwargs["config_path"], sample_config)
        self.assertEqual(mock_run.call_args.kwargs["strategy_name"], "double")
        self.assertIsNone(mock_run.call_args.kwargs["output_root"])

    def test_main_runs_compare_subcommand(self) -> None:
        comparison_dir = Path("/tmp/runs/comparisons/sample_research-0001-to-0002")
        with patch(
            "llm_strategy_lab.cli.compare_runs",
            return_value=comparison_dir,
        ) as mock_compare:
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                result = main(
                    [
                        "compare",
                        "--parent-run",
                        "runs/sample_research/0001",
                        "--candidate-run",
                        "runs/sample_research/0002",
                    ]
                )

        self.assertEqual(result, 0)
        self.assertIn(str(comparison_dir), stdout.getvalue())
        self.assertEqual(
            mock_compare.call_args.kwargs,
            {
                "parent_run": Path("runs/sample_research/0001"),
                "candidate_run": Path("runs/sample_research/0002"),
                "output_root": None,
            },
        )

    def test_main_runs_prompt_bundle_subcommand(self) -> None:
        bundle_path = Path("/tmp/runs/comparisons/sample_research-0001-to-0002/prompt_bundle.json")
        with patch(
            "llm_strategy_lab.cli.build_prompt_bundle",
            return_value=bundle_path,
        ) as mock_bundle:
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                result = main(
                    [
                        "prompt-bundle",
                        "--comparison",
                        "runs/comparisons/sample_research-0001-to-0002",
                    ]
                )

        self.assertEqual(result, 0)
        self.assertIn(str(bundle_path), stdout.getvalue())
        self.assertEqual(
            mock_bundle.call_args.kwargs,
            {
                "comparison_reference": Path("runs/comparisons/sample_research-0001-to-0002"),
                "output_root": None,
            },
        )

    def test_main_runs_validate_proposal_subcommand(self) -> None:
        proposal_artifact_path = Path(
            "/tmp/runs/comparisons/sample_research-0001-to-0002/proposal_artifact.json"
        )
        with patch(
            "llm_strategy_lab.cli.validate_and_save_proposal",
            return_value=proposal_artifact_path,
        ) as mock_validate:
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                result = main(
                    [
                        "validate-proposal",
                        "--comparison",
                        "runs/comparisons/sample_research-0001-to-0002",
                        "--proposal-file",
                        "tmp/proposal.json",
                    ]
                )

        self.assertEqual(result, 0)
        self.assertIn(str(proposal_artifact_path), stdout.getvalue())
        self.assertEqual(
            mock_validate.call_args.kwargs,
            {
                "comparison_reference": Path("runs/comparisons/sample_research-0001-to-0002"),
                "proposal_file": Path("tmp/proposal.json"),
                "output_root": None,
            },
        )

    def test_main_runs_child_run_subcommand(self) -> None:
        child_run_dir = Path("/tmp/runs/sample_research/0003")
        with patch(
            "llm_strategy_lab.cli.create_child_run",
            return_value=child_run_dir,
        ) as mock_child_run:
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                result = main(
                    [
                        "child-run",
                        "--comparison",
                        "runs/comparisons/sample_research-0001-to-0002",
                        "--proposal-file",
                        "tmp/proposal.json",
                    ]
                )

        self.assertEqual(result, 0)
        self.assertIn(str(child_run_dir), stdout.getvalue())
        self.assertEqual(
            mock_child_run.call_args.kwargs,
            {
                "comparison_reference": Path("runs/comparisons/sample_research-0001-to-0002"),
                "proposal_file": Path("tmp/proposal.json"),
                "output_root": None,
            },
        )

    def test_main_runs_loop_subcommand(self) -> None:
        loop_dir = Path("/tmp/runs/loops/sample_research/0001")
        with patch(
            "llm_strategy_lab.cli.run_improvement_loop",
            return_value=loop_dir,
        ) as mock_loop:
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                result = main(
                    [
                        "loop",
                        "--comparison",
                        "runs/comparisons/sample_research-0001-to-0002",
                        "--max-iterations",
                        "2",
                        "--no-improvement-limit",
                        "1",
                    ]
                )

        self.assertEqual(result, 0)
        self.assertIn(str(loop_dir), stdout.getvalue())
        self.assertEqual(
            mock_loop.call_args.kwargs,
            {
                "comparison_reference": Path("runs/comparisons/sample_research-0001-to-0002"),
                "parent_run": None,
                "candidate_run": None,
                "output_root": None,
                "max_iterations": 2,
                "no_improvement_limit": 1,
                "min_annual_return_delta": 0.0,
                "min_return_risk_ratio_delta": 0.0,
                "max_drawdown_increase": 0.0,
                "max_turnover_increase": 0.0,
            },
        )

    def test_main_supports_legacy_config_arguments(self) -> None:
        run_dir = Path("/tmp/runs/sample_research/0002")
        with patch(
            "llm_strategy_lab.cli.run_experiment",
            return_value=run_dir,
        ) as mock_run:
            result = main(
                [
                    "--config",
                    "configs/experiments/sample_research.yaml",
                    "--strategy",
                    "mom",
                ]
            )

        self.assertEqual(result, 0)
        self.assertEqual(
            mock_run.call_args.kwargs["config_path"],
            Path("configs/experiments/sample_research.yaml"),
        )
        self.assertEqual(mock_run.call_args.kwargs["strategy_name"], "mom")


if __name__ == "__main__":
    unittest.main()
