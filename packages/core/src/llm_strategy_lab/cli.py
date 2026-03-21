"""Command-line entry points for llm-strategy-lab."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .comparison import compare_runs
from .config import find_project_root
from .runner import run_experiment

STRATEGY_CHOICES = ("mom", "pca_plain", "pca_sub", "double")


def _add_run_arguments(parser: argparse.ArgumentParser, *, include_config: bool) -> None:
    if include_config:
        parser.add_argument(
            "--config",
            required=True,
            help="Path to the experiment config file.",
        )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional override for the run artifact root directory.",
    )
    parser.add_argument(
        "--strategy",
        choices=STRATEGY_CHOICES,
        default=None,
        help="Optional strategy override for the selected experiment config.",
    )
    parser.add_argument(
        "--strategy-params-file",
        default=None,
        help="Optional path to a strategy params YAML used with --strategy.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-strategy-lab",
        description="Run configured research experiments and sample baselines.",
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser(
        "run",
        help="Run an experiment from a config file.",
    )
    _add_run_arguments(run_parser, include_config=True)

    sample_parser = subparsers.add_parser(
        "sample",
        help="Run the bundled sample experiment config.",
    )
    _add_run_arguments(sample_parser, include_config=False)

    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare two completed runs and write lineage artifacts.",
    )
    compare_parser.add_argument(
        "--parent-run",
        required=True,
        help="Path to the parent/baseline run directory or manifest.json.",
    )
    compare_parser.add_argument(
        "--candidate-run",
        required=True,
        help="Path to the candidate run directory or manifest.json.",
    )
    compare_parser.add_argument(
        "--output-root",
        default=None,
        help="Optional override for the comparison artifact root directory.",
    )
    return parser


def build_legacy_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-strategy-lab",
        description="Legacy single-command entry point; prefer `run` or `sample`.",
    )
    _add_run_arguments(parser, include_config=True)
    return parser


def resolve_sample_config_path() -> Path:
    candidate_roots = []
    for probe in (Path.cwd(), Path(__file__)):
        try:
            candidate_roots.append(find_project_root(probe))
        except FileNotFoundError:
            continue

    for project_root in candidate_roots:
        config_path = (project_root / "configs" / "experiments" / "sample_research.yaml").resolve()
        if config_path.exists():
            return config_path

    raise FileNotFoundError("Could not resolve configs/experiments/sample_research.yaml")


def _run_from_namespace(args: argparse.Namespace, *, config_path: Path) -> int:
    run_dir = run_experiment(
        config_path=config_path,
        output_root=Path(args.output_root) if args.output_root else None,
        strategy_name=args.strategy,
        strategy_params_file=(
            Path(args.strategy_params_file)
            if args.strategy_params_file
            else None
        ),
    )
    print(f"Research run created at: {run_dir}")
    return 0


def _compare_from_namespace(args: argparse.Namespace) -> int:
    comparison_dir = compare_runs(
        parent_run=Path(args.parent_run),
        candidate_run=Path(args.candidate_run),
        output_root=Path(args.output_root) if args.output_root else None,
    )
    print(f"Run comparison created at: {comparison_dir}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args_list = list(argv) if argv is not None else sys.argv[1:]
    if not args_list:
        build_parser().print_help()
        return 2

    if args_list[0] in {"run", "sample", "compare"}:
        parser = build_parser()
        args = parser.parse_args(args_list)
        if args.command == "run":
            return _run_from_namespace(args, config_path=Path(args.config))
        if args.command == "sample":
            return _run_from_namespace(args, config_path=resolve_sample_config_path())
        if args.command == "compare":
            return _compare_from_namespace(args)
        parser.print_help()
        return 2

    legacy_parser = build_legacy_parser()
    legacy_args = legacy_parser.parse_args(args_list)
    return _run_from_namespace(legacy_args, config_path=Path(legacy_args.config))


if __name__ == "__main__":
    raise SystemExit(main())
