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
```

`make run-sample` は `configs/experiments/sample_research.yaml` を読み込み、`runs/` に MOM ベースライン戦略の signal / portfolio 成果物を生成します。

## GitHub 側の作業

labels、milestones、Project、branch protection はローカルでは反映できません。
ラベルは `scripts/setup/github_labels.sh OWNER/REPO` で作成できます。
