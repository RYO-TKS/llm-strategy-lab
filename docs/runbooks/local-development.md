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
llm-strategy-lab compare --parent-run runs/sample_research/0016 --candidate-run runs/sample_research/0017
llm-strategy-lab prompt-bundle --comparison runs/comparisons/sample_research-0016-to-0017
llm-strategy-lab validate-proposal --comparison runs/comparisons/sample_research-0016-to-0017 --proposal-file tmp/proposal.json
llm-strategy-lab child-run --comparison runs/comparisons/sample_research-0016-to-0017 --proposal-file tmp/proposal.json
```

`make run-sample` は `llm-strategy-lab sample` を呼び、`runs/` に `runs/{experiment_id}/{iteration}` 構造で成果物を保存します。
`llm-strategy-lab run --config ... --strategy ...` を使うと、同じ experiment config を読みつつ CLI 側で戦略だけ差し替えられます。
sample 実行では MOM ベースライン戦略の signal / portfolio 成果物に加えて、backtest daily series、position contribution series、metrics JSON、FF3 / Carhart4 回帰 JSON、equity curve / drawdown / cumulative IC の SVG を生成します。
`pca_plain` と `pca_sub` と `double` も同じ strategy interface で追加済みで、sample データは CLI override で PCA 系を回しても非空 artifact が出る日数を含みます。
`llm-strategy-lab compare` は 2 つの succeeded run を比較し、metrics 差分、因子回帰差分、config diff、`parent_run_id` / `candidate_run_id` / `lineage_id` を含む comparison manifest を `runs/comparisons/` に保存します。
`llm-strategy-lab prompt-bundle` は comparison から LLM 入力用の `prompt_bundle.json` と `proposal_schema.json` を生成します。
`llm-strategy-lab validate-proposal` は proposal JSON を schema で検証し、invalid proposal は弾いて `proposal_validation.json` に理由を残し、valid proposal だけ `proposal_artifact.json` に保存します。
`llm-strategy-lab child-run` は proposal を再検証して parent run ベースの child config snapshot を生成し、`child_config_snapshot.yaml` を child run directory に保存したうえで実験を実行し、manifest に lineage 参照を残します。

## GitHub 側の作業

labels、milestones、Project、branch protection はローカルでは反映できません。
ラベルは `scripts/setup/github_labels.sh OWNER/REPO` で作成できます。
