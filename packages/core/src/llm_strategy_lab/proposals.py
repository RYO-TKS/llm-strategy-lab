"""Prompt bundle generation and proposal validation for the LLM loop."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator

JsonDict = dict[str, Any]


@dataclass(frozen=True)
class ProposalValidationError(ValueError):
    errors: tuple[str, ...]

    def __str__(self) -> str:
        return "; ".join(self.errors)


def _normalize_comparison_directory(comparison_reference: Path) -> Path:
    resolved = comparison_reference.resolve()
    if resolved.is_file():
        if resolved.name != "comparison_manifest.json":
            raise ValueError(
                "Expected a comparison directory or comparison_manifest.json, "
                f"got: {resolved}"
            )
        resolved = resolved.parent
    manifest_path = resolved / "comparison_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Comparison manifest not found: {manifest_path}")
    return resolved


def _load_json(path: Path) -> JsonDict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object at: {path}")
    return payload


def _load_comparison_manifest(comparison_dir: Path) -> JsonDict:
    return _load_json(comparison_dir / "comparison_manifest.json")


def _read_text(path_value: str | None) -> str | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def _metric_highlights(metric_comparison: Mapping[str, Any]) -> list[JsonDict]:
    parent_metrics = metric_comparison.get("parent", {})
    candidate_metrics = metric_comparison.get("candidate", {})
    delta = metric_comparison.get("delta", {})
    if not isinstance(parent_metrics, Mapping):
        return []
    if not isinstance(candidate_metrics, Mapping):
        return []
    if not isinstance(delta, Mapping):
        return []

    highlights: list[JsonDict] = []
    for metric_name, metric_delta in sorted(
        delta.items(),
        key=lambda item: abs(float(item[1])),
        reverse=True,
    ):
        highlights.append(
            {
                "metric": str(metric_name),
                "parent": parent_metrics.get(metric_name),
                "candidate": candidate_metrics.get(metric_name),
                "delta": float(metric_delta),
            }
        )
    return highlights


def _factor_highlights(factor_comparison: Mapping[str, Any]) -> list[JsonDict]:
    models = factor_comparison.get("models", {})
    if not isinstance(models, Mapping):
        return []
    highlights: list[JsonDict] = []
    for model_name in sorted(models):
        model_payload = models[model_name]
        if not isinstance(model_payload, Mapping):
            continue
        delta = model_payload.get("delta", {})
        annualized_alpha_delta = None
        r_squared_delta = None
        observations_delta = None
        if isinstance(delta, Mapping):
            annualized_alpha_delta = delta.get("annualized_alpha")
            r_squared_delta = delta.get("r_squared")
            observations_delta = delta.get("observations")
        highlights.append(
            {
                "model": str(model_name),
                "parent_status": model_payload.get("parent_status"),
                "candidate_status": model_payload.get("candidate_status"),
                "annualized_alpha_delta": annualized_alpha_delta,
                "r_squared_delta": r_squared_delta,
                "observations_delta": observations_delta,
            }
        )
    return highlights


def _config_highlights(config_diff: Mapping[str, Any]) -> JsonDict:
    changed = config_diff.get("changed", {})
    added = config_diff.get("added", {})
    removed = config_diff.get("removed", {})
    return {
        "changed": dict(changed) if isinstance(changed, Mapping) else {},
        "added": dict(added) if isinstance(added, Mapping) else {},
        "removed": dict(removed) if isinstance(removed, Mapping) else {},
    }


def _proposal_schema(comparison_manifest: Mapping[str, Any]) -> JsonDict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Strategy Improvement Proposal",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "proposal_id",
            "lineage_id",
            "parent_run_id",
            "candidate_run_id",
            "hypothesis",
            "rationale",
            "strategy_delta",
            "expected_impact",
        ],
        "properties": {
            "proposal_id": {
                "type": "string",
                "minLength": 1,
            },
            "lineage_id": {
                "type": "string",
                "const": comparison_manifest["lineage_id"],
            },
            "parent_run_id": {
                "type": "string",
                "const": comparison_manifest["parent_run_id"],
            },
            "candidate_run_id": {
                "type": "string",
                "const": comparison_manifest["candidate_run_id"],
            },
            "hypothesis": {
                "type": "string",
                "minLength": 20,
            },
            "rationale": {
                "type": "string",
                "minLength": 20,
            },
            "strategy_delta": {
                "type": "object",
                "additionalProperties": False,
                "required": ["summary", "parameter_changes"],
                "properties": {
                    "summary": {
                        "type": "string",
                        "minLength": 10,
                    },
                    "strategy_name": {
                        "type": "string",
                        "minLength": 1,
                    },
                    "parameter_changes": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["path", "operation", "value"],
                            "properties": {
                                "path": {
                                    "type": "string",
                                    "minLength": 1,
                                },
                                "operation": {
                                    "type": "string",
                                    "enum": ["set", "add", "remove"],
                                },
                                "value": {},
                                "previous_value": {},
                                "reason": {
                                    "type": "string",
                                    "minLength": 5,
                                },
                            },
                        },
                    },
                },
            },
            "expected_impact": {
                "type": "object",
                "additionalProperties": False,
                "required": ["summary", "metric_expectations"],
                "properties": {
                    "summary": {
                        "type": "string",
                        "minLength": 10,
                    },
                    "metric_expectations": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["metric", "direction", "reason"],
                            "properties": {
                                "metric": {
                                    "type": "string",
                                    "minLength": 1,
                                },
                                "direction": {
                                    "type": "string",
                                    "enum": ["increase", "decrease", "stable"],
                                },
                                "reason": {
                                    "type": "string",
                                    "minLength": 5,
                                },
                            },
                        },
                    },
                },
            },
        },
    }


def _system_prompt() -> str:
    return (
        "You are proposing the next strategy experiment. "
        "Respond with JSON only. Follow the proposal schema exactly, avoid extra "
        "keys, and ground every claim in the supplied run artifacts."
    )


def _user_prompt(bundle: Mapping[str, Any]) -> str:
    comparison_summary = bundle["comparison_summary"]
    config_changes = comparison_summary["config_highlights"]["changed"]
    config_change_lines = [
        f"- {path}: parent={values['parent']}, candidate={values['candidate']}"
        for path, values in sorted(config_changes.items())
    ]
    metric_lines = [
        (
            f"- {item['metric']}: parent={item['parent']}, "
            f"candidate={item['candidate']}, delta={item['delta']}"
        )
        for item in comparison_summary["metric_highlights"][:5]
    ]
    factor_lines = [
        (
            f"- {item['model']}: parent_status={item['parent_status']}, "
            f"candidate_status={item['candidate_status']}, "
            f"annualized_alpha_delta={item['annualized_alpha_delta']}, "
            f"r_squared_delta={item['r_squared_delta']}"
        )
        for item in comparison_summary["factor_highlights"]
    ]
    return "\n".join(
        [
            "Use the comparison context below to propose the next experiment delta.",
            f"Lineage ID: {bundle['lineage_id']}",
            f"Parent Run: {bundle['parent_run_id']}",
            f"Candidate Run: {bundle['candidate_run_id']}",
            "Top metric deltas:",
            *metric_lines,
            "Factor regression summary:",
            *factor_lines,
            "Changed config fields:",
            *(config_change_lines or ["- none"]),
            "Return exactly one JSON object that validates against the supplied schema.",
        ]
    )


def build_prompt_bundle(
    comparison_reference: Path,
    *,
    output_root: Path | None = None,
) -> Path:
    comparison_dir = _normalize_comparison_directory(comparison_reference)
    output_dir = output_root.resolve() if output_root else comparison_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison_manifest = _load_comparison_manifest(comparison_dir)
    schema_path = output_dir / "proposal_schema.json"
    prompt_bundle_path = output_dir / "prompt_bundle.json"
    comparison_summary_path = Path(comparison_manifest["artifact_paths"]["summary"])

    schema_payload = _proposal_schema(comparison_manifest)
    schema_path.write_text(
        json.dumps(schema_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    prompt_bundle = {
        "lineage_id": comparison_manifest["lineage_id"],
        "experiment_id": comparison_manifest["experiment_id"],
        "parent_run_id": comparison_manifest["parent_run_id"],
        "candidate_run_id": comparison_manifest["candidate_run_id"],
        "proposal_schema_path": str(schema_path),
        "artifacts": {
            "comparison_manifest": str((comparison_dir / "comparison_manifest.json").resolve()),
            "comparison_summary": str(comparison_summary_path),
            "parent_summary": comparison_manifest["parent_run"]["summary_path"],
            "candidate_summary": comparison_manifest["candidate_run"]["summary_path"],
        },
        "comparison_summary": {
            "parent_strategy": comparison_manifest["parent_run"]["strategy"],
            "candidate_strategy": comparison_manifest["candidate_run"]["strategy"],
            "metric_highlights": _metric_highlights(
                comparison_manifest["metric_comparison"]
            ),
            "factor_highlights": _factor_highlights(
                comparison_manifest["factor_regression_comparison"]
            ),
            "config_highlights": _config_highlights(comparison_manifest["config_diff"]),
        },
        "source_summaries": {
            "comparison_summary": _read_text(str(comparison_summary_path)),
            "parent_summary": _read_text(comparison_manifest["parent_run"]["summary_path"]),
            "candidate_summary": _read_text(
                comparison_manifest["candidate_run"]["summary_path"]
            ),
        },
        "messages": [
            {"role": "system", "content": _system_prompt()},
            {"role": "user", "content": ""},
        ],
    }
    prompt_bundle["messages"][1]["content"] = _user_prompt(prompt_bundle)
    prompt_bundle_path.write_text(
        json.dumps(prompt_bundle, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return prompt_bundle_path


def _validation_error_messages(errors: Sequence[Any]) -> tuple[str, ...]:
    messages: list[str] = []
    for error in errors:
        path = ".".join(str(part) for part in error.absolute_path)
        location = path if path else "<root>"
        messages.append(f"{location}: {error.message}")
    return tuple(messages)


def validate_and_save_proposal(
    comparison_reference: Path,
    proposal_file: Path,
    *,
    output_root: Path | None = None,
) -> Path:
    comparison_dir = _normalize_comparison_directory(comparison_reference)
    output_dir = output_root.resolve() if output_root else comparison_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_bundle_path = build_prompt_bundle(comparison_dir, output_root=output_dir)
    schema_path = output_dir / "proposal_schema.json"
    schema_payload = _load_json(schema_path)
    proposal_payload = _load_json(proposal_file.resolve())

    validator = Draft202012Validator(schema_payload)
    errors = sorted(validator.iter_errors(proposal_payload), key=lambda error: list(error.path))
    validation_report_path = output_dir / "proposal_validation.json"
    validation_payload = {
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "proposal_source": str(proposal_file.resolve()),
        "proposal_schema_path": str(schema_path),
        "prompt_bundle_path": str(prompt_bundle_path),
        "valid": not errors,
        "errors": _validation_error_messages(errors),
    }
    validation_report_path.write_text(
        json.dumps(validation_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if errors:
        raise ProposalValidationError(validation_payload["errors"])

    proposal_artifact_path = output_dir / "proposal_artifact.json"
    proposal_artifact_path.write_text(
        json.dumps(proposal_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return proposal_artifact_path
