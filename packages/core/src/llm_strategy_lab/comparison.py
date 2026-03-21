"""Run comparison helpers for lineage tracking and side-by-side evaluation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .models import RunComparisonRecord

COMPARABLE_CONFIG_KEYS = (
    "environment_config",
    "strategy_config",
    "dataset_config",
    "backtest_config",
    "cli_overrides",
)


def _normalize_run_directory(run_reference: Path) -> Path:
    resolved = run_reference.resolve()
    if resolved.is_file():
        if resolved.name != "manifest.json":
            raise ValueError(f"Expected a run directory or manifest.json, got: {resolved}")
        resolved = resolved.parent
    manifest_path = resolved / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Run manifest not found: {manifest_path}")
    return resolved


def _load_manifest(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "succeeded":
        raise ValueError(f"Comparison requires succeeded runs: {manifest_path}")
    if not isinstance(manifest.get("backtest_result"), dict):
        raise ValueError(f"Comparison requires backtest artifacts: {manifest_path}")
    return manifest


def _extract_run_reference(manifest: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
    return {
        "run_id": manifest["run_id"],
        "experiment_id": manifest["experiment_id"],
        "strategy": manifest["strategy"],
        "status": manifest["status"],
        "output_dir": str(run_dir),
        "manifest_path": str((run_dir / "manifest.json").resolve()),
        "summary_path": str((run_dir / "SUMMARY.md").resolve()),
    }


def _derive_lineage_id(
    parent_manifest: Mapping[str, Any],
    candidate_manifest: Mapping[str, Any],
) -> str:
    parent_experiment = str(parent_manifest["experiment_id"])
    candidate_experiment = str(candidate_manifest["experiment_id"])
    if parent_experiment == candidate_experiment:
        return (
            f"{parent_experiment}-"
            f"{parent_manifest['run_id']}-to-{candidate_manifest['run_id']}"
        )
    return (
        f"{parent_experiment}-{parent_manifest['run_id']}"
        f"-to-{candidate_experiment}-{candidate_manifest['run_id']}"
    )


def _resolve_output_root(
    parent_run_dir: Path,
    candidate_run_dir: Path,
    output_root: Path | None,
) -> Path:
    if output_root is not None:
        resolved = output_root.resolve()
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved

    parent_output_root = parent_run_dir.parents[1]
    candidate_output_root = candidate_run_dir.parents[1]
    if parent_output_root != candidate_output_root:
        raise ValueError(
            "Parent and candidate run directories are under different output roots; "
            "pass --output-root explicitly."
        )
    return parent_output_root


def _numeric_delta(parent: Any, candidate: Any) -> Any:
    if isinstance(parent, Mapping) and isinstance(candidate, Mapping):
        nested: dict[str, Any] = {}
        for key in sorted(set(parent) | set(candidate)):
            if key in parent and key in candidate:
                delta = _numeric_delta(parent[key], candidate[key])
                if delta not in ({}, None):
                    nested[str(key)] = delta
        return nested
    if isinstance(parent, (int, float)) and isinstance(candidate, (int, float)):
        return float(candidate) - float(parent)
    return None


def _build_metric_comparison(
    parent_manifest: Mapping[str, Any],
    candidate_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    parent_metrics = dict(parent_manifest["backtest_result"]["metrics"])
    candidate_metrics = dict(candidate_manifest["backtest_result"]["metrics"])
    delta = {
        key: float(candidate_metrics[key]) - float(parent_metrics[key])
        for key in sorted(set(parent_metrics) & set(candidate_metrics))
    }
    return {
        "parent": parent_metrics,
        "candidate": candidate_metrics,
        "delta": delta,
    }


def _load_factor_regression_models(
    manifest: Mapping[str, Any],
) -> tuple[str | None, dict[str, Any]]:
    metadata = manifest["backtest_result"].get("metadata", {})
    factor_path = metadata.get("factor_regressions_path")
    if not factor_path:
        return None, {}
    factor_payload = json.loads(Path(factor_path).read_text(encoding="utf-8"))
    models = factor_payload.get("models", {})
    return str(factor_path), dict(models) if isinstance(models, Mapping) else {}


def _build_factor_regression_comparison(
    parent_manifest: Mapping[str, Any],
    candidate_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    parent_path, parent_models = _load_factor_regression_models(parent_manifest)
    candidate_path, candidate_models = _load_factor_regression_models(candidate_manifest)
    model_comparison: dict[str, Any] = {}
    for model_name in sorted(set(parent_models) | set(candidate_models)):
        parent_model = parent_models.get(model_name)
        candidate_model = candidate_models.get(model_name)
        if not isinstance(parent_model, Mapping) or not isinstance(candidate_model, Mapping):
            model_comparison[model_name] = {
                "parent": parent_model,
                "candidate": candidate_model,
                "delta": {},
            }
            continue
        model_comparison[model_name] = {
            "parent_status": parent_model.get("status"),
            "candidate_status": candidate_model.get("status"),
            "parent": dict(parent_model),
            "candidate": dict(candidate_model),
            "delta": _numeric_delta(parent_model, candidate_model),
        }
    return {
        "parent_path": parent_path,
        "candidate_path": candidate_path,
        "models": model_comparison,
    }


def _flatten_mapping(
    mapping: Mapping[str, Any],
    *,
    prefix: str = "",
) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in mapping.items():
        dotted_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.update(_flatten_mapping(value, prefix=dotted_key))
            continue
        flattened[dotted_key] = value
    return flattened


def _build_config_snapshot(manifest: Mapping[str, Any]) -> dict[str, Any]:
    metadata = manifest.get("metadata", {})
    snapshot = {
        key: metadata.get(key)
        for key in COMPARABLE_CONFIG_KEYS
        if key in metadata
    }
    return snapshot


def _build_config_diff(
    parent_manifest: Mapping[str, Any],
    candidate_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    parent_snapshot = _build_config_snapshot(parent_manifest)
    candidate_snapshot = _build_config_snapshot(candidate_manifest)
    parent_flat = _flatten_mapping(parent_snapshot)
    candidate_flat = _flatten_mapping(candidate_snapshot)
    added = {
        key: candidate_flat[key]
        for key in sorted(set(candidate_flat) - set(parent_flat))
    }
    removed = {
        key: parent_flat[key]
        for key in sorted(set(parent_flat) - set(candidate_flat))
    }
    changed = {
        key: {
            "parent": parent_flat[key],
            "candidate": candidate_flat[key],
        }
        for key in sorted(set(parent_flat) & set(candidate_flat))
        if parent_flat[key] != candidate_flat[key]
    }
    return {
        "parent_snapshot": parent_snapshot,
        "candidate_snapshot": candidate_snapshot,
        "added": added,
        "removed": removed,
        "changed": changed,
    }


def _build_markdown_summary(record: RunComparisonRecord) -> str:
    lines = [
        "# Run Comparison",
        "",
        (
            f"Parent run `{record.parent_run_id}` (`{record.parent_run['strategy']}`) "
            f"was compared against candidate run `{record.candidate_run_id}` "
            f"(`{record.candidate_run['strategy']}`)."
        ),
        (
            f"Lineage ID is `{record.lineage_id}` and the comparison was created at "
            f"`{record.created_at_utc.isoformat()}`."
        ),
        "",
        "## Metrics",
        "",
    ]

    metric_delta = record.metric_comparison.get("delta", {})
    parent_metrics = record.metric_comparison.get("parent", {})
    candidate_metrics = record.metric_comparison.get("candidate", {})
    for metric_name in sorted(metric_delta):
        lines.append(
            (
                f"- `{metric_name}` changed from `{parent_metrics[metric_name]:.6f}` "
                f"to `{candidate_metrics[metric_name]:.6f}` "
                f"(`delta={metric_delta[metric_name]:.6f}`)."
            )
        )

    lines.extend(["", "## Factor Regressions", ""])
    models = record.factor_regression_comparison.get("models", {})
    for model_name in sorted(models):
        model_payload = models[model_name]
        lines.append(
            (
                f"- `{model_name}` status moved from "
                f"`{model_payload.get('parent_status', 'n/a')}` to "
                f"`{model_payload.get('candidate_status', 'n/a')}`."
            )
        )
        delta = model_payload.get("delta", {})
        if isinstance(delta, Mapping):
            annualized_alpha_delta = delta.get("annualized_alpha")
            r_squared_delta = delta.get("r_squared")
            if annualized_alpha_delta is not None:
                lines.append(
                    f"  annualized_alpha delta: `{float(annualized_alpha_delta):.6f}`."
                )
            if r_squared_delta is not None:
                lines.append(f"  r_squared delta: `{float(r_squared_delta):.6f}`.")

    changed = record.config_diff.get("changed", {})
    added = record.config_diff.get("added", {})
    removed = record.config_diff.get("removed", {})
    lines.extend(["", "## Config Diff", ""])
    if not changed and not added and not removed:
        lines.append("No config differences were detected across the comparable snapshots.")
    else:
        for key in sorted(changed):
            diff = changed[key]
            lines.append(
                f"- `{key}` changed from `{diff['parent']}` to `{diff['candidate']}`."
            )
        for key in sorted(added):
            lines.append(f"- `{key}` was added with value `{added[key]}`.")
        for key in sorted(removed):
            lines.append(f"- `{key}` was removed from the candidate snapshot.")

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            (
                f"- JSON manifest: "
                f"`{record.artifact_paths['comparison_manifest']}`"
            ),
            (
                f"- Parent manifest: "
                f"`{record.parent_run['manifest_path']}`"
            ),
            (
                f"- Candidate manifest: "
                f"`{record.candidate_run['manifest_path']}`"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def compare_runs(
    parent_run: Path,
    candidate_run: Path,
    *,
    output_root: Path | None = None,
) -> Path:
    parent_run_dir = _normalize_run_directory(parent_run)
    candidate_run_dir = _normalize_run_directory(candidate_run)
    parent_manifest = _load_manifest(parent_run_dir)
    candidate_manifest = _load_manifest(candidate_run_dir)

    lineage_id = _derive_lineage_id(parent_manifest, candidate_manifest)
    resolved_output_root = _resolve_output_root(
        parent_run_dir,
        candidate_run_dir,
        output_root,
    )
    comparison_dir = (resolved_output_root / "comparisons" / lineage_id).resolve()
    comparison_dir.mkdir(parents=True, exist_ok=True)

    comparison_manifest_path = comparison_dir / "comparison_manifest.json"
    summary_path = comparison_dir / "SUMMARY.md"
    created_at_utc = datetime.now(timezone.utc)
    record = RunComparisonRecord(
        lineage_id=lineage_id,
        experiment_id=str(candidate_manifest["experiment_id"]),
        parent_run_id=str(parent_manifest["run_id"]),
        candidate_run_id=str(candidate_manifest["run_id"]),
        created_at_utc=created_at_utc,
        output_dir=comparison_dir,
        parent_run=_extract_run_reference(parent_manifest, parent_run_dir),
        candidate_run=_extract_run_reference(candidate_manifest, candidate_run_dir),
        metric_comparison=_build_metric_comparison(parent_manifest, candidate_manifest),
        factor_regression_comparison=_build_factor_regression_comparison(
            parent_manifest,
            candidate_manifest,
        ),
        config_diff=_build_config_diff(parent_manifest, candidate_manifest),
        artifact_paths={
            "comparison_manifest": comparison_manifest_path,
            "summary": summary_path,
        },
        metadata={
            "comparison_kind": "parent_child",
            "parent_strategy": parent_manifest["strategy"],
            "candidate_strategy": candidate_manifest["strategy"],
        },
    )

    comparison_manifest_path.write_text(
        json.dumps(record.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(_build_markdown_summary(record), encoding="utf-8")
    return comparison_dir
