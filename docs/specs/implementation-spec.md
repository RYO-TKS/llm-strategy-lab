# Implementation Spec

Phase 1 では、次の 4 要素を最小動作品として揃えます。

1. データ取込基盤
2. 戦略インターフェースと主要戦略
3. バックテストと評価
4. 実験ランナー CLI

## 現時点の前提

このリポジトリは、Issue 駆動で最小構成を拡張していく研究基盤です。
`make run-sample` は現時点で aligned dataset、MOM ベースライン戦略の signal / portfolio artifact、日次バックテスト series、metrics JSON、FF3 / Carhart4 回帰 JSON、主要 SVG チャートを出力します。
CLI は `llm-strategy-lab run --config ...` と `llm-strategy-lab sample` を持ち、`--strategy` による戦略差し替えをサポートします。
transaction cost を加味した評価はまだ未実装です。
PCA 系の実装メモは [pca-subspace-notes.md](pca-subspace-notes.md) に残します。

## Phase 1 で満たすべきこと

設定ファイルから実験を選べること。

戦略ごとの差し替えが共通インターフェースで行えること。

バックテスト結果と評価指標を成果物として保存できること。

少なくとも 4 戦略を同一条件で比較できること。

## 除外事項

自動発注、ブローカー接続、秘匿データの自動取得は初期スコープから外します。
