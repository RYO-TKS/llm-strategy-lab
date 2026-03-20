"""Scaffold runner used during repository bootstrap."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only before dependency install
    yaml = None


REQUIRED_KEYS = {"experiment_id", "environment", "strategy", "backtest"}


def load_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    text = config_path.read_text(encoding="utf-8")
    if yaml is None:
        raise RuntimeError("PyYAML is required. Run 'make setup' before loading YAML configs.")

    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping.")

    missing = sorted(REQUIRED_KEYS - data.keys())
    if missing:
        raise ValueError(f"Config is missing required keys: {', '.join(missing)}")

    return data


def next_run_directory(output_root: Path, experiment_id: str) -> Path:
    experiment_root = output_root / experiment_id
    experiment_root.mkdir(parents=True, exist_ok=True)

    existing_runs = sorted(
        int(path.name)
        for path in experiment_root.iterdir()
        if path.is_dir() and path.name.isdigit()
    )
    next_index = existing_runs[-1] + 1 if existing_runs else 1
    run_dir = experiment_root / f"{next_index:04d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def create_scaffold_run(config_path: Path, output_root: Path) -> Path:
    config = load_config(config_path)
    run_dir = next_run_directory(
        output_root=output_root,
        experiment_id=str(config["experiment_id"]),
    )

    strategy = config.get("strategy", {})
    manifest = {
        "experiment_id": config["experiment_id"],
        "environment": config["environment"],
        "strategy": strategy.get("name", "unknown"),
        "status": "scaffold",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "message": (
            "Repository scaffold is ready. "
            "Implement data, strategy, and backtest logic next."
        ),
    }

    summary = "\n".join(
        [
            "# Scaffold Run",
            "",
            f"- Experiment: `{manifest['experiment_id']}`",
            f"- Environment: `{manifest['environment']}`",
            f"- Strategy: `{manifest['strategy']}`",
            "",
            (
                "This run confirms that the repository bootstrap, config loading, "
                "and artifact output path are wired."
            ),
            "It does not execute the real strategy or backtest engine yet.",
            "",
            "## Next Steps",
            "",
            "1. Implement data ingestion and alignment.",
            "2. Add strategy interfaces and concrete strategies.",
            "3. Persist metrics and charts in this run directory.",
            "",
        ]
    )

    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (run_dir / "SUMMARY.md").write_text(summary, encoding="utf-8")
    return run_dir
