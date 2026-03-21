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
