# LLM Strategy Lab

LLM を用いた投資戦略の生成、改善、検証を行う研究用基盤です。
最初の目標は、日米業種リードラグ戦略を再現しつつ、Issue 駆動で段階的に拡張できる最小構成を固めることです。

## スコープ

- 戦略テンプレートのバックテスト
- LLM による改善ループ
- 日米業種リードラグ戦略の再現
- 実運用の自動発注は対象外

## ディレクトリ概要

- `packages/`: コアロジック
- `apps/`: UI / API
- `configs/`: 実験設定
- `docs/`: 仕様、設計、運用メモ
- `runs/`: 実験成果物

## 開発

```bash
make setup
make lint
make test
make run-sample
```

CLI からは次で再現できます。

```bash
llm-strategy-lab sample
llm-strategy-lab run --config configs/experiments/sample_research.yaml
llm-strategy-lab run --config configs/experiments/sample_research.yaml --strategy pca_sub
llm-strategy-lab compare --parent-run runs/sample_research/0016 --candidate-run runs/sample_research/0017
llm-strategy-lab prompt-bundle --comparison runs/comparisons/sample_research-0016-to-0017
llm-strategy-lab validate-proposal --comparison runs/comparisons/sample_research-0016-to-0017 --proposal-file tmp/proposal.json
llm-strategy-lab child-run --comparison runs/comparisons/sample_research-0016-to-0017 --proposal-file tmp/proposal.json
llm-strategy-lab loop --parent-run runs/sample_research/0016 --candidate-run runs/sample_research/0017 --max-iterations 1
```

`make run-sample` は `llm-strategy-lab sample` のショートカットで、現時点で MOM ベースライン戦略の signal / portfolio artifact に加えて、
日次 backtest、metrics JSON、FF3 / Carhart4 回帰 JSON、equity curve / drawdown / cumulative IC の SVG を `runs/` に生成します。
transaction cost を加味した評価は M1 の後続 Issue で進めます。
実装済みの戦略インターフェースは `mom`, `pca_plain`, `pca_sub`, `double` で、sample 設定は現時点では `mom` を参照しています。
sample データは CLI override で `pca_plain` と `pca_sub` を回しても signal / backtest が空にならない長さにしてあります。
`llm-strategy-lab compare` は 2 つの completed run を比較し、`runs/comparisons/{lineage_id}/` に comparison manifest と Markdown summary を保存します。
`llm-strategy-lab prompt-bundle` は comparison artifact から `prompt_bundle.json` と `proposal_schema.json` を生成します。
`llm-strategy-lab validate-proposal` は proposal JSON を schema で検証し、valid な場合だけ `proposal_artifact.json` を保存します。
`llm-strategy-lab child-run` は validated proposal を parent run に適用し、child 側の run directory に `child_config_snapshot.yaml` を残したうえで実行し、manifest に parent / proposal / lineage 参照を保存します。
`llm-strategy-lab loop` は comparison から prompt bundle と auto proposal を生成し、child run 実行と baseline 比較、quality gate 判定、accept / reject / stop_reason の保存までを 1 コマンドで実行します。

## GitHub 運用

Issue template、PR template、CODEOWNERS、CI ワークフローはこのリポジトリ内に用意しています。
一方で、labels、milestones、Project、branch protection は GitHub 側の設定なので、リモートリポジトリ作成後に適用してください。

ラベル作成用の補助スクリプトは `scripts/setup/github_labels.sh` に置いてあります。

## マイルストーン

- M1: Phase 1 最小動作品
- M2: LLM 改善ループ
- M3: 実用モード
- M4: UI
