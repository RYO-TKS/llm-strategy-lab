# Implementation Status

この文書は、2026-03-21 時点の実装到達点と検証導線を GPT / reviewer が即確認できるようにまとめたものです。

## 完了範囲

M1 は完了しています。対象は Issue `#2` から `#10` で、typed config、日米営業日整合データ取り込み、4 戦略、共通 backtest、因子回帰、チャート生成、CLI まで実装済みです。

M2 も完了しています。対象は Issue `#21` から `#24` で、run comparison、lineage manifest、prompt bundle、proposal schema validation、child run generation、改善 loop executor まで実装済みです。

## 主要導線

単発実験は `llm-strategy-lab sample` と `llm-strategy-lab run --config ...` で確認できます。

比較導線は `llm-strategy-lab compare --parent-run ... --candidate-run ...` で確認できます。

LLM 準備導線は `llm-strategy-lab prompt-bundle --comparison ...` と `llm-strategy-lab validate-proposal --comparison ... --proposal-file ...` で確認できます。

child run 導線は `llm-strategy-lab child-run --comparison ... --proposal-file ...` で確認できます。

改善 loop は `llm-strategy-lab loop --parent-run ... --candidate-run ... --max-iterations 1` で確認できます。

## 確認コマンド

```bash
make lint
make test
make run-sample
llm-strategy-lab run --config configs/experiments/sample_research.yaml --strategy pca_plain
llm-strategy-lab compare --parent-run runs/sample_research/0016 --candidate-run runs/sample_research/0017
llm-strategy-lab prompt-bundle --comparison runs/comparisons/sample_research-0016-to-0017
llm-strategy-lab validate-proposal --comparison runs/comparisons/sample_research-0016-to-0017 --proposal-file /tmp/proposal.json
llm-strategy-lab child-run --comparison runs/comparisons/sample_research-0016-to-0017 --proposal-file /tmp/proposal.json
llm-strategy-lab loop --parent-run runs/sample_research/0016 --candidate-run runs/sample_research/0017 --max-iterations 1
```

## 代表 artifact

baseline sample run は `runs/sample_research/0016/` にあります。

comparison artifact の例は `runs/comparisons/sample_research-0016-to-0017/` にあります。ここには `comparison_manifest.json`, `prompt_bundle.json`, `proposal_schema.json`, `proposal_artifact.json`, `proposal_validation.json` が揃っています。

child run の例は `runs/sample_research/0025/` にあります。ここには `child_config_snapshot.yaml`, `applied_proposal_artifact.json`, `manifest.json` があり、manifest の `metadata.lineage` から parent / candidate / proposal を追えます。

改善 loop の例は `runs/loops/sample_research/0001/` にあります。ここには `loop_manifest.json` と `iterations/01/` の proposal 系 artifact があり、`stop_reason=quality_gate_failed` と `decision=reject` を確認できます。

## 実装ファイル

run comparison は `packages/core/src/llm_strategy_lab/comparison.py` です。

proposal generation / validation は `packages/core/src/llm_strategy_lab/proposals.py` です。

child run generation は `packages/core/src/llm_strategy_lab/child_runs.py` です。

改善 loop executor は `packages/core/src/llm_strategy_lab/loop_executor.py` です。

CLI 導線は `packages/core/src/llm_strategy_lab/cli.py` に集約しています。

## 現在の見方

M1 / M2 の必須実装は揃っています。次に残るのは M3 の実用モードと M4 の UI です。
