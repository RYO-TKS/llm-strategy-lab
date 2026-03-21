"""Command-line entry points for llm-strategy-lab."""

from __future__ import annotations

import argparse
from pathlib import Path

from .runner import create_scaffold_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-strategy-lab",
        description="Run the aligned-data, strategy, and backtest experiment runner.",
    )
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
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    run_dir = create_scaffold_run(
        config_path=Path(args.config),
        output_root=Path(args.output_root) if args.output_root else None,
    )
    print(f"Research run created at: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
