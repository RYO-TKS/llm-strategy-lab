"""Command-line entry points for llm-strategy-lab."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .child_runs import create_child_run
from .comparison import compare_runs
from .config import find_project_root
from .loop_executor import run_improvement_loop
from .proposals import build_prompt_bundle, validate_and_save_proposal
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

    prompt_bundle_parser = subparsers.add_parser(
        "prompt-bundle",
        help="Generate an LLM prompt bundle and proposal schema from a comparison.",
    )
    prompt_bundle_parser.add_argument(
        "--comparison",
        required=True,
        help="Path to the comparison directory or comparison_manifest.json.",
    )
    prompt_bundle_parser.add_argument(
        "--output-root",
        default=None,
        help="Optional override for the prompt bundle output directory.",
    )

    validate_proposal_parser = subparsers.add_parser(
        "validate-proposal",
        help="Validate a proposal JSON against the generated proposal schema.",
    )
    validate_proposal_parser.add_argument(
        "--comparison",
        required=True,
        help="Path to the comparison directory or comparison_manifest.json.",
    )
    validate_proposal_parser.add_argument(
        "--proposal-file",
        required=True,
        help="Path to the proposal JSON file to validate.",
    )
    validate_proposal_parser.add_argument(
        "--output-root",
        default=None,
        help="Optional override for the validated proposal output directory.",
    )

    child_run_parser = subparsers.add_parser(
        "child-run",
        help="Validate a proposal and materialize the next child run from it.",
    )
    child_run_parser.add_argument(
        "--comparison",
        required=True,
        help="Path to the comparison directory or comparison_manifest.json.",
    )
    child_run_parser.add_argument(
        "--proposal-file",
        required=True,
        help="Path to the proposal JSON file to validate and apply.",
    )
    child_run_parser.add_argument(
        "--output-root",
        default=None,
        help="Optional override for the child run output root directory.",
    )

    loop_parser = subparsers.add_parser(
        "loop",
        help="Run the iterative proposal -> child run -> comparison loop.",
    )
    loop_parser.add_argument(
        "--comparison",
        default=None,
        help="Optional path to the comparison directory or comparison_manifest.json.",
    )
    loop_parser.add_argument(
        "--parent-run",
        default=None,
        help="Optional path to the parent/baseline run directory or manifest.json.",
    )
    loop_parser.add_argument(
        "--candidate-run",
        default=None,
        help="Optional path to the candidate run directory or manifest.json.",
    )
    loop_parser.add_argument(
        "--output-root",
        default=None,
        help="Optional override for the loop and child run output root directory.",
    )
    loop_parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum number of loop iterations to execute.",
    )
    loop_parser.add_argument(
        "--no-improvement-limit",
        type=int,
        default=1,
        help="Stop after this many consecutive rejected iterations.",
    )
    loop_parser.add_argument(
        "--min-annual-return-delta",
        type=float,
        default=0.0,
        help="Minimum annual_return delta required for acceptance.",
    )
    loop_parser.add_argument(
        "--min-return-risk-ratio-delta",
        type=float,
        default=0.0,
        help="Minimum return_risk_ratio delta required for acceptance.",
    )
    loop_parser.add_argument(
        "--max-drawdown-increase",
        type=float,
        default=0.0,
        help="Maximum allowed max_drawdown increase for acceptance.",
    )
    loop_parser.add_argument(
        "--max-turnover-increase",
        type=float,
        default=0.0,
        help="Maximum allowed average_turnover increase for acceptance.",
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


def _build_prompt_bundle_from_namespace(args: argparse.Namespace) -> int:
    bundle_path = build_prompt_bundle(
        comparison_reference=Path(args.comparison),
        output_root=Path(args.output_root) if args.output_root else None,
    )
    print(f"Prompt bundle created at: {bundle_path}")
    return 0


def _validate_proposal_from_namespace(args: argparse.Namespace) -> int:
    proposal_artifact_path = validate_and_save_proposal(
        comparison_reference=Path(args.comparison),
        proposal_file=Path(args.proposal_file),
        output_root=Path(args.output_root) if args.output_root else None,
    )
    print(f"Validated proposal saved at: {proposal_artifact_path}")
    return 0


def _create_child_run_from_namespace(args: argparse.Namespace) -> int:
    child_run_dir = create_child_run(
        comparison_reference=Path(args.comparison),
        proposal_file=Path(args.proposal_file),
        output_root=Path(args.output_root) if args.output_root else None,
    )
    print(f"Child run created at: {child_run_dir}")
    return 0


def _run_loop_from_namespace(args: argparse.Namespace) -> int:
    loop_dir = run_improvement_loop(
        comparison_reference=Path(args.comparison) if args.comparison else None,
        parent_run=Path(args.parent_run) if args.parent_run else None,
        candidate_run=Path(args.candidate_run) if args.candidate_run else None,
        output_root=Path(args.output_root) if args.output_root else None,
        max_iterations=args.max_iterations,
        no_improvement_limit=args.no_improvement_limit,
        min_annual_return_delta=args.min_annual_return_delta,
        min_return_risk_ratio_delta=args.min_return_risk_ratio_delta,
        max_drawdown_increase=args.max_drawdown_increase,
        max_turnover_increase=args.max_turnover_increase,
    )
    print(f"Improvement loop created at: {loop_dir}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args_list = list(argv) if argv is not None else sys.argv[1:]
    if not args_list:
        build_parser().print_help()
        return 2

    if args_list[0] in {
        "run",
        "sample",
        "compare",
        "prompt-bundle",
        "validate-proposal",
        "child-run",
        "loop",
    }:
        parser = build_parser()
        args = parser.parse_args(args_list)
        if args.command == "run":
            return _run_from_namespace(args, config_path=Path(args.config))
        if args.command == "sample":
            return _run_from_namespace(args, config_path=resolve_sample_config_path())
        if args.command == "compare":
            return _compare_from_namespace(args)
        if args.command == "prompt-bundle":
            return _build_prompt_bundle_from_namespace(args)
        if args.command == "validate-proposal":
            return _validate_proposal_from_namespace(args)
        if args.command == "child-run":
            return _create_child_run_from_namespace(args)
        if args.command == "loop":
            return _run_loop_from_namespace(args)
        parser.print_help()
        return 2

    legacy_parser = build_legacy_parser()
    legacy_args = legacy_parser.parse_args(args_list)
    return _run_from_namespace(legacy_args, config_path=Path(legacy_args.config))


if __name__ == "__main__":
    raise SystemExit(main())
