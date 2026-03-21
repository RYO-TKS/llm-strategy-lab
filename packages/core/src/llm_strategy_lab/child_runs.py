"""Child run generation from validated proposals."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from .comparison import _normalize_run_directory
from .config import find_project_root, load_experiment_config
from .proposals import _normalize_comparison_directory, validate_and_save_proposal
from .runner import _execute_loaded_experiment, next_run_directory

JsonDict = dict[str, Any]


def _load_json(path: Path) -> JsonDict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object at: {path}")
    return payload


def _resolve_default_params_file(project_root: Path, strategy_name: str) -> str | None:
    default_params_file = project_root / "configs" / "strategies" / f"{strategy_name}.default.yaml"
    if default_params_file.exists():
        return str(default_params_file.resolve())
    return None


def _base_strategy_mapping(
    parent_metadata: Mapping[str, Any],
    *,
    project_root: Path,
    target_strategy_name: str,
) -> JsonDict:
    parent_strategy_config = parent_metadata["strategy_config"]
    parent_strategy_name = str(parent_strategy_config["name"])

    if target_strategy_name == parent_strategy_name:
        strategy_mapping: JsonDict = {"name": parent_strategy_name}
        params_file = parent_strategy_config.get("params_file")
        if params_file is not None:
            strategy_mapping["params_file"] = params_file
        params = parent_strategy_config.get("params", {})
        if isinstance(params, Mapping):
            strategy_mapping.update(dict(params))
        return strategy_mapping

    strategy_mapping = {"name": target_strategy_name}
    default_params_file = _resolve_default_params_file(project_root, target_strategy_name)
    if default_params_file is not None:
        strategy_mapping["params_file"] = default_params_file
    return strategy_mapping


def _translate_change_path(change_path: str) -> tuple[str, ...]:
    parts = tuple(change_path.split("."))
    if parts == ("strategy_config", "name"):
        return ("strategy", "name")
    if parts == ("strategy_config", "params_file"):
        return ("strategy", "params_file")
    if len(parts) >= 3 and parts[:2] == ("strategy_config", "params"):
        return ("strategy", *parts[2:])
    if len(parts) >= 2 and parts[0] == "backtest_config":
        return ("backtest", *parts[1:])
    raise ValueError(f"Unsupported proposal change path: {change_path}")


def _apply_nested_change(
    payload: JsonDict,
    *,
    translated_path: tuple[str, ...],
    operation: str,
    value: Any,
) -> None:
    cursor: JsonDict = payload
    for key in translated_path[:-1]:
        next_value = cursor.get(key)
        if not isinstance(next_value, dict):
            next_value = {}
            cursor[key] = next_value
        cursor = next_value

    leaf_key = translated_path[-1]
    if operation in {"set", "add"}:
        cursor[leaf_key] = value
        return
    if operation == "remove":
        cursor.pop(leaf_key, None)
        return
    raise ValueError(f"Unsupported proposal operation: {operation}")


def _build_child_config_snapshot(
    parent_manifest: Mapping[str, Any],
    proposal_payload: Mapping[str, Any],
    *,
    output_root: Path,
    project_root: Path,
) -> tuple[JsonDict, list[JsonDict]]:
    parent_metadata = parent_manifest["metadata"]
    strategy_delta = proposal_payload.get("strategy_delta", {})
    if not isinstance(strategy_delta, Mapping):
        raise ValueError("proposal.strategy_delta must be a mapping")

    target_strategy_name = str(
        strategy_delta.get(
            "strategy_name",
            parent_metadata["strategy_config"]["name"],
        )
    )
    snapshot = {
        "experiment_id": parent_manifest["experiment_id"],
        "environment": {
            **dict(parent_metadata["environment_config"]),
            "output_root": str(output_root.resolve()),
        },
        "strategy": _base_strategy_mapping(
            parent_metadata,
            project_root=project_root,
            target_strategy_name=target_strategy_name,
        ),
        "dataset": dict(parent_metadata["dataset_config"])
        if parent_metadata.get("dataset_config") is not None
        else None,
        "backtest": dict(parent_metadata["backtest_config"]),
        "notes": [
            *list(parent_manifest.get("notes", [])),
            (
                f"child run generated from proposal {proposal_payload['proposal_id']} "
                f"(lineage {proposal_payload['lineage_id']})"
            ),
        ],
    }

    applied_changes: list[JsonDict] = []
    parameter_changes = strategy_delta.get("parameter_changes", [])
    if not isinstance(parameter_changes, list):
        raise ValueError("proposal.strategy_delta.parameter_changes must be a list")

    for change in parameter_changes:
        if not isinstance(change, Mapping):
            raise ValueError("proposal parameter change entries must be mappings")
        translated_path = _translate_change_path(str(change["path"]))
        _apply_nested_change(
            snapshot,
            translated_path=translated_path,
            operation=str(change["operation"]),
            value=change.get("value"),
        )
        applied_changes.append(
            {
                "path": str(change["path"]),
                "translated_path": ".".join(translated_path),
                "operation": str(change["operation"]),
                "value": change.get("value"),
                "previous_value": change.get("previous_value"),
                "reason": change.get("reason"),
            }
        )

    return snapshot, applied_changes


def create_child_run(
    comparison_reference: Path,
    proposal_file: Path,
    *,
    output_root: Path | None = None,
) -> Path:
    comparison_dir = _normalize_comparison_directory(comparison_reference)
    comparison_manifest = _load_json(comparison_dir / "comparison_manifest.json")
    proposal_artifact_path = validate_and_save_proposal(
        comparison_dir,
        proposal_file.resolve(),
        output_root=comparison_dir,
    )
    proposal_payload = _load_json(proposal_artifact_path)

    parent_run_dir = _normalize_run_directory(Path(comparison_manifest["parent_run"]["output_dir"]))
    parent_manifest = _load_json(parent_run_dir / "manifest.json")
    project_root = find_project_root(Path(parent_manifest["config_path"]))

    resolved_output_root = (
        output_root.resolve()
        if output_root is not None
        else Path(parent_manifest["metadata"]["environment_config"]["output_root"]).resolve()
    )
    child_run_dir = next_run_directory(resolved_output_root, str(parent_manifest["experiment_id"]))
    snapshot_payload, applied_changes = _build_child_config_snapshot(
        parent_manifest,
        proposal_payload,
        output_root=resolved_output_root,
        project_root=project_root,
    )
    snapshot_path = child_run_dir / "child_config_snapshot.yaml"
    proposal_snapshot_path = child_run_dir / "applied_proposal_artifact.json"
    snapshot_path.write_text(
        yaml.safe_dump(snapshot_payload, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    proposal_snapshot_path.write_text(
        json.dumps(proposal_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    child_config = load_experiment_config(snapshot_path, project_root=project_root)
    return _execute_loaded_experiment(
        child_config,
        run_dir=child_run_dir,
        cli_overrides={
            "strategy_name": None,
            "strategy_params_file": None,
        },
        resolved_output_root=resolved_output_root,
        extra_metadata={
            "lineage": {
                "lineage_id": proposal_payload["lineage_id"],
                "parent_run_id": proposal_payload["parent_run_id"],
                "candidate_run_id": proposal_payload["candidate_run_id"],
                "proposal_id": proposal_payload["proposal_id"],
                "proposal_artifact_path": str(proposal_artifact_path),
                "proposal_snapshot_path": str(proposal_snapshot_path.resolve()),
                "comparison_manifest_path": str(
                    (comparison_dir / "comparison_manifest.json").resolve()
                ),
                "child_config_snapshot_path": str(snapshot_path.resolve()),
            },
            "proposal_summary": {
                "hypothesis": proposal_payload["hypothesis"],
                "rationale": proposal_payload["rationale"],
                "expected_impact": proposal_payload["expected_impact"],
                "applied_changes": applied_changes,
            },
        },
    )
