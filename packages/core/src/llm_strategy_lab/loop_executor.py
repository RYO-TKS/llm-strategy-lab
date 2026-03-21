"""Iterative improvement loop executor for proposal generation and child runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from .child_runs import create_child_run
from .comparison import compare_runs
from .proposals import build_prompt_bundle, validate_and_save_proposal

JsonDict = dict[str, Any]


def _load_json(path: Path) -> JsonDict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object at: {path}")
    return payload


def _normalize_comparison_reference(comparison_reference: Path) -> Path:
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


def _next_loop_directory(output_root: Path, experiment_id: str) -> Path:
    loop_root = output_root / "loops" / experiment_id
    loop_root.mkdir(parents=True, exist_ok=True)
    existing_loops = sorted(
        int(path.name)
        for path in loop_root.iterdir()
        if path.is_dir() and path.name.isdigit()
    )
    next_index = existing_loops[-1] + 1 if existing_loops else 1
    loop_dir = loop_root / f"{next_index:04d}"
    loop_dir.mkdir(parents=True, exist_ok=False)
    return loop_dir


def _resolve_initial_comparison(
    *,
    comparison_reference: Optional[Path],
    parent_run: Optional[Path],
    candidate_run: Optional[Path],
    output_root: Optional[Path],
) -> Path:
    if comparison_reference is not None:
        return _normalize_comparison_reference(comparison_reference)
    if parent_run is None or candidate_run is None:
        raise ValueError(
            "Provide either --comparison or both --parent-run and --candidate-run."
        )
    return compare_runs(parent_run=parent_run, candidate_run=candidate_run, output_root=output_root)


def _comparison_output_root(comparison_manifest: Mapping[str, Any]) -> Path:
    parent_output_dir = Path(comparison_manifest["parent_run"]["output_dir"]).resolve()
    return parent_output_dir.parents[1]


def _numeric_value(mapping: Mapping[str, Any], key: str) -> float:
    value = mapping.get(key, 0.0)
    return float(value) if value is not None else 0.0


def _round_float(value: float) -> float:
    return round(float(value), 6)


def _build_quality_gate(
    comparison_manifest: Mapping[str, Any],
    *,
    min_annual_return_delta: float,
    min_return_risk_ratio_delta: float,
    max_drawdown_increase: float,
    max_turnover_increase: float,
) -> JsonDict:
    parent_metrics = comparison_manifest["metric_comparison"]["parent"]
    candidate_metrics = comparison_manifest["metric_comparison"]["candidate"]
    delta = comparison_manifest["metric_comparison"]["delta"]

    checks = [
        {
            "metric": "annual_return",
            "rule": f"delta >= {min_annual_return_delta}",
            "baseline": parent_metrics.get("annual_return"),
            "candidate": candidate_metrics.get("annual_return"),
            "delta": delta.get("annual_return"),
            "passed": _numeric_value(delta, "annual_return") >= min_annual_return_delta,
        },
        {
            "metric": "return_risk_ratio",
            "rule": f"delta >= {min_return_risk_ratio_delta}",
            "baseline": parent_metrics.get("return_risk_ratio"),
            "candidate": candidate_metrics.get("return_risk_ratio"),
            "delta": delta.get("return_risk_ratio"),
            "passed": (
                _numeric_value(delta, "return_risk_ratio")
                >= min_return_risk_ratio_delta
            ),
        },
        {
            "metric": "max_drawdown",
            "rule": f"delta <= {max_drawdown_increase}",
            "baseline": parent_metrics.get("max_drawdown"),
            "candidate": candidate_metrics.get("max_drawdown"),
            "delta": delta.get("max_drawdown"),
            "passed": _numeric_value(delta, "max_drawdown") <= max_drawdown_increase,
        },
        {
            "metric": "average_turnover",
            "rule": f"delta <= {max_turnover_increase}",
            "baseline": parent_metrics.get("average_turnover"),
            "candidate": candidate_metrics.get("average_turnover"),
            "delta": delta.get("average_turnover"),
            "passed": _numeric_value(delta, "average_turnover") <= max_turnover_increase,
        },
    ]
    return {
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
    }


def _candidate_snapshot(comparison_manifest: Mapping[str, Any]) -> JsonDict:
    config_diff = comparison_manifest.get("config_diff", {})
    snapshot = config_diff.get("candidate_snapshot", {})
    if not isinstance(snapshot, Mapping):
        raise ValueError("comparison candidate_snapshot must be a mapping")
    return dict(snapshot)


def _auto_parameter_changes(
    comparison_manifest: Mapping[str, Any],
) -> list[JsonDict]:
    candidate_snapshot = _candidate_snapshot(comparison_manifest)
    strategy_config = candidate_snapshot.get("strategy_config", {})
    if not isinstance(strategy_config, Mapping):
        raise ValueError("candidate strategy snapshot is missing")
    params = strategy_config.get("params", {})
    if not isinstance(params, Mapping):
        raise ValueError("candidate strategy params must be a mapping")

    delta = comparison_manifest["metric_comparison"]["delta"]
    parameter_changes: list[JsonDict] = []

    q_key = "q" if "q" in params else "quantile" if "quantile" in params else None
    if q_key is not None:
        current_q = float(params[q_key])
        if _numeric_value(delta, "average_turnover") > 0:
            next_q = max(0.1, round(current_q - 0.05, 2))
            reason = "Turnover increased, so narrow the selected quantile."
        elif _numeric_value(delta, "annual_return") < 0:
            next_q = min(0.45, round(current_q + 0.05, 2))
            reason = "Annual return fell, so widen the selected quantile modestly."
        else:
            next_q = current_q
            reason = ""
        if next_q != current_q:
            parameter_changes.append(
                {
                    "path": f"strategy_config.params.{q_key}",
                    "operation": "set",
                    "previous_value": current_q,
                    "value": next_q,
                    "reason": reason,
                }
            )

    if "rolling_window" in params:
        current_window = int(params["rolling_window"])
        if (
            _numeric_value(delta, "hit_ratio") < 0
            or _numeric_value(delta, "max_drawdown") > 0
        ):
            next_window = current_window + 1
            parameter_changes.append(
                {
                    "path": "strategy_config.params.rolling_window",
                    "operation": "set",
                    "previous_value": current_window,
                    "value": next_window,
                    "reason": "Increase the lookback window to smooth unstable signals.",
                }
            )

    if "components" in params and _numeric_value(delta, "annual_return") < 0:
        current_components = int(params["components"])
        parameter_changes.append(
            {
                "path": "strategy_config.params.components",
                "operation": "set",
                "previous_value": current_components,
                "value": current_components + 1,
                "reason": "Expand the retained component count to test broader signal capture.",
            }
        )

    if "regularization" in params and _numeric_value(delta, "max_drawdown") > 0:
        current_regularization = float(params["regularization"])
        next_regularization = min(round(current_regularization + 0.05, 2), 0.5)
        if next_regularization != current_regularization:
            parameter_changes.append(
                {
                    "path": "strategy_config.params.regularization",
                    "operation": "set",
                    "previous_value": current_regularization,
                    "value": next_regularization,
                    "reason": "Increase shrinkage because drawdown deteriorated.",
                }
            )

    if not parameter_changes:
        raise ValueError("No heuristic proposal changes could be generated from the comparison.")
    return parameter_changes


def generate_auto_proposal(
    comparison_reference: Path,
    *,
    output_root: Path,
    iteration_index: int,
) -> Path:
    comparison_dir = _normalize_comparison_reference(comparison_reference)
    comparison_manifest = _load_json(comparison_dir / "comparison_manifest.json")
    output_root.mkdir(parents=True, exist_ok=True)

    candidate_snapshot = _candidate_snapshot(comparison_manifest)
    strategy_config = candidate_snapshot["strategy_config"]
    metric_delta = comparison_manifest["metric_comparison"]["delta"]
    factor_models = comparison_manifest["factor_regression_comparison"]["models"]
    factor_model_names = ", ".join(sorted(factor_models))
    parameter_changes = _auto_parameter_changes(comparison_manifest)

    proposal_payload = {
        "proposal_id": (
            f"auto-{comparison_manifest['lineage_id']}-iter-{iteration_index:02d}"
        ),
        "lineage_id": comparison_manifest["lineage_id"],
        "parent_run_id": comparison_manifest["parent_run_id"],
        "candidate_run_id": comparison_manifest["candidate_run_id"],
        "hypothesis": (
            f"Adjusting {strategy_config['name']} using the latest comparison deltas may "
            "improve return quality while keeping the next run inside the same lineage."
        ),
        "rationale": (
            "The current comparison shows annual_return delta "
            f"{_round_float(_numeric_value(metric_delta, 'annual_return'))}, "
            "return_risk_ratio delta "
            f"{_round_float(_numeric_value(metric_delta, 'return_risk_ratio'))}, "
            "average_turnover delta "
            f"{_round_float(_numeric_value(metric_delta, 'average_turnover'))}, "
            f"and factor models [{factor_model_names}]. "
            "The heuristic proposer responds by applying a narrow strategy parameter delta."
        ),
        "strategy_delta": {
            "summary": (
                f"Auto-generated proposal for {strategy_config['name']} from "
                f"comparison {comparison_manifest['lineage_id']}."
            ),
            "strategy_name": strategy_config["name"],
            "parameter_changes": parameter_changes,
        },
        "expected_impact": {
            "summary": (
                "Aim to improve annual return and return/risk while preventing further "
                "drawdown or turnover deterioration."
            ),
            "metric_expectations": [
                {
                    "metric": "annual_return",
                    "direction": "increase",
                    "reason": "The loop treats annual return as the primary acceptance metric.",
                },
                {
                    "metric": "return_risk_ratio",
                    "direction": "increase",
                    "reason": "The loop requires return/risk to stay flat or improve.",
                },
                {
                    "metric": "average_turnover",
                    "direction": "decrease",
                    "reason": "High turnover is explicitly constrained by the quality gate.",
                },
            ],
        },
    }
    generated_proposal_path = output_root / "generated_proposal.json"
    generated_proposal_path.write_text(
        json.dumps(proposal_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return generated_proposal_path


def _build_loop_summary(loop_manifest: Mapping[str, Any]) -> str:
    lines = [
        "# Improvement Loop",
        "",
        f"- Loop ID: `{loop_manifest['loop_id']}`",
        f"- Experiment: `{loop_manifest['experiment_id']}`",
        f"- Status: `{loop_manifest['status']}`",
        f"- Stop Reason: `{loop_manifest['stop_reason']}`",
        f"- Final Accepted Run: `{loop_manifest['final_accepted_run_id']}`",
        "",
        "## Iterations",
        "",
    ]
    for iteration in loop_manifest["iterations"]:
        lines.append(
            (
                f"- Iteration `{iteration['iteration']}`: baseline "
                f"`{iteration['baseline_run_id']}` -> child `{iteration['child_run_id']}`, "
                f"decision `{iteration['decision']}`, reason `{iteration['decision_reason']}`."
            )
        )
    lines.append("")
    return "\n".join(lines)


def run_improvement_loop(
    *,
    comparison_reference: Optional[Path] = None,
    parent_run: Optional[Path] = None,
    candidate_run: Optional[Path] = None,
    output_root: Optional[Path] = None,
    max_iterations: int = 3,
    no_improvement_limit: int = 1,
    min_annual_return_delta: float = 0.0,
    min_return_risk_ratio_delta: float = 0.0,
    max_drawdown_increase: float = 0.0,
    max_turnover_increase: float = 0.0,
    proposal_generator: Callable[..., Path] = generate_auto_proposal,
    child_run_creator: Callable[..., Path] = create_child_run,
    comparison_creator: Callable[..., Path] = compare_runs,
) -> Path:
    if max_iterations <= 0:
        raise ValueError("max_iterations must be greater than zero")
    if no_improvement_limit <= 0:
        raise ValueError("no_improvement_limit must be greater than zero")

    initial_comparison_dir = _resolve_initial_comparison(
        comparison_reference=comparison_reference,
        parent_run=parent_run,
        candidate_run=candidate_run,
        output_root=output_root,
    )
    initial_comparison_manifest = _load_json(initial_comparison_dir / "comparison_manifest.json")
    resolved_output_root = (
        output_root.resolve()
        if output_root is not None
        else _comparison_output_root(initial_comparison_manifest)
    )
    loop_dir = _next_loop_directory(
        resolved_output_root,
        initial_comparison_manifest["experiment_id"],
    )
    iterations_dir = loop_dir / "iterations"
    iterations_dir.mkdir(parents=True, exist_ok=True)

    current_comparison_dir = initial_comparison_dir
    current_comparison_manifest = initial_comparison_manifest
    current_baseline_run_dir = Path(
        current_comparison_manifest["parent_run"]["output_dir"]
    ).resolve()
    final_accepted_run_id = str(current_comparison_manifest["parent_run_id"])
    consecutive_no_improvement = 0
    stop_reason = "max_iterations_reached"
    status = "succeeded"
    iteration_records: list[JsonDict] = []
    started_at_utc = datetime.now(timezone.utc)

    for iteration_index in range(1, max_iterations + 1):
        iteration_dir = iterations_dir / f"{iteration_index:02d}"
        iteration_dir.mkdir(parents=True, exist_ok=True)

        prompt_bundle_path = build_prompt_bundle(
            current_comparison_dir,
            output_root=iteration_dir,
        )
        generated_proposal_path = proposal_generator(
            current_comparison_dir,
            output_root=iteration_dir,
            iteration_index=iteration_index,
        )
        proposal_artifact_path = validate_and_save_proposal(
            current_comparison_dir,
            generated_proposal_path,
            output_root=iteration_dir,
        )
        child_run_dir = child_run_creator(
            current_comparison_dir,
            generated_proposal_path,
            output_root=resolved_output_root,
        )
        candidate_comparison_dir = comparison_creator(
            parent_run=current_baseline_run_dir,
            candidate_run=child_run_dir,
            output_root=resolved_output_root,
        )
        candidate_comparison_manifest = _load_json(
            candidate_comparison_dir / "comparison_manifest.json"
        )
        quality_gate = _build_quality_gate(
            candidate_comparison_manifest,
            min_annual_return_delta=min_annual_return_delta,
            min_return_risk_ratio_delta=min_return_risk_ratio_delta,
            max_drawdown_increase=max_drawdown_increase,
            max_turnover_increase=max_turnover_increase,
        )
        decision = "accept" if quality_gate["passed"] else "reject"
        decision_reason = (
            "quality_gate_passed" if quality_gate["passed"] else "quality_gate_failed"
        )

        if decision == "accept":
            current_baseline_run_dir = child_run_dir.resolve()
            final_accepted_run_id = str(candidate_comparison_manifest["candidate_run_id"])
            consecutive_no_improvement = 0
        else:
            consecutive_no_improvement += 1

        iteration_records.append(
            {
                "iteration": iteration_index,
                "baseline_run_id": current_comparison_manifest["parent_run_id"],
                "input_candidate_run_id": current_comparison_manifest["candidate_run_id"],
                "child_run_id": candidate_comparison_manifest["candidate_run_id"],
                "decision": decision,
                "decision_reason": decision_reason,
                "quality_gate": quality_gate,
                "consecutive_no_improvement": consecutive_no_improvement,
                "artifact_paths": {
                    "input_comparison": str(
                        (current_comparison_dir / "comparison_manifest.json").resolve()
                    ),
                    "prompt_bundle": str(prompt_bundle_path.resolve()),
                    "generated_proposal": str(generated_proposal_path.resolve()),
                    "proposal_artifact": str(proposal_artifact_path.resolve()),
                    "child_run_dir": str(child_run_dir.resolve()),
                    "candidate_comparison": str(
                        (candidate_comparison_dir / "comparison_manifest.json").resolve()
                    ),
                },
            }
        )

        current_comparison_dir = candidate_comparison_dir
        current_comparison_manifest = candidate_comparison_manifest

        if decision == "reject" and consecutive_no_improvement >= no_improvement_limit:
            stop_reason = (
                "quality_gate_failed"
                if no_improvement_limit == 1
                else "no_improvement_limit_reached"
            )
            break

        if iteration_index == max_iterations:
            stop_reason = "max_iterations_reached"
            break

    loop_manifest: JsonDict = {
        "loop_id": loop_dir.name,
        "experiment_id": initial_comparison_manifest["experiment_id"],
        "status": status,
        "started_at_utc": started_at_utc.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "stop_reason": stop_reason,
        "settings": {
            "max_iterations": max_iterations,
            "no_improvement_limit": no_improvement_limit,
            "min_annual_return_delta": min_annual_return_delta,
            "min_return_risk_ratio_delta": min_return_risk_ratio_delta,
            "max_drawdown_increase": max_drawdown_increase,
            "max_turnover_increase": max_turnover_increase,
        },
        "initial_context": {
            "comparison_manifest_path": str(
                (initial_comparison_dir / "comparison_manifest.json").resolve()
            ),
            "parent_run_id": initial_comparison_manifest["parent_run_id"],
            "candidate_run_id": initial_comparison_manifest["candidate_run_id"],
        },
        "final_accepted_run_id": final_accepted_run_id,
        "iterations": iteration_records,
    }
    loop_manifest_path = loop_dir / "loop_manifest.json"
    summary_path = loop_dir / "SUMMARY.md"
    loop_manifest_path.write_text(
        json.dumps(loop_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(_build_loop_summary(loop_manifest), encoding="utf-8")
    return loop_dir
