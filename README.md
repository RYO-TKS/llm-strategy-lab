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

`make run-sample` は現時点で MOM ベースライン戦略の signal / portfolio artifact に加えて、
日次 backtest、metrics JSON、daily/position series を `runs/` に生成します。
チャート生成と transaction cost を加味した評価は M1 の後続 Issue で進めます。
実装済みの戦略インターフェースは `mom`, `pca_plain`, `pca_sub`, `double` で、sample 設定は現時点では `mom` を参照しています。

## GitHub 運用

Issue template、PR template、CODEOWNERS、CI ワークフローはこのリポジトリ内に用意しています。
一方で、labels、milestones、Project、branch protection は GitHub 側の設定なので、リモートリポジトリ作成後に適用してください。

ラベル作成用の補助スクリプトは `scripts/setup/github_labels.sh` に置いてあります。

## マイルストーン

- M1: Phase 1 最小動作品
- M2: LLM 改善ループ
- M3: 実用モード
- M4: UI
