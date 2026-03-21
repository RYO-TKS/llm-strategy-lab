# Local Development

## 前提

推奨 Python は 3.11 系です。CI も 3.11 を前提にしています。
このマシンでは Python 3.9 でも最低限の構文確認はできますが、ローカル運用は 3.11 に寄せる方が安全です。

## 初回セットアップ

```bash
make setup
```

仮想環境 `.venv/` を作成し、開発依存をインストールします。

## よく使うコマンド

```bash
make help
make lint
make test
make run-sample
llm-strategy-lab sample
llm-strategy-lab run --config configs/experiments/sample_research.yaml --strategy pca_plain
```

`make run-sample` は `llm-strategy-lab sample` を呼び、`runs/` に `runs/{experiment_id}/{iteration}` 構造で成果物を保存します。
`llm-strategy-lab run --config ... --strategy ...` を使うと、同じ experiment config を読みつつ CLI 側で戦略だけ差し替えられます。
sample 実行では MOM ベースライン戦略の signal / portfolio 成果物に加えて、backtest daily series、position contribution series、metrics JSON、FF3 / Carhart4 回帰 JSON、equity curve / drawdown / cumulative IC の SVG を生成します。
`pca_plain` と `pca_sub` と `double` も同じ strategy interface で追加済みですが、sample 設定は現時点では `mom` を使います。

## GitHub 側の作業

labels、milestones、Project、branch protection はローカルでは反映できません。
ラベルは `scripts/setup/github_labels.sh OWNER/REPO` で作成できます。
