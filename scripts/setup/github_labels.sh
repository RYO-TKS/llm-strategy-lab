#!/usr/bin/env bash
set -euo pipefail

if ! command -v gh >/dev/null 2>&1; then
  echo "GitHub CLI 'gh' is required."
  exit 1
fi

REPO="${1:-${GH_REPO:-}}"
if [ -z "$REPO" ]; then
  echo "Usage: $0 owner/repo"
  exit 1
fi

create_label() {
  local name="$1"
  local color="$2"
  local description="$3"
  gh label create "$name" --repo "$REPO" --color "$color" --description "$description" --force
}

create_label "type:feature" "1D76DB" "新機能追加"
create_label "type:bug" "D73A4A" "不具合修正"
create_label "type:research" "5319E7" "検証・比較実験・論文反映"
create_label "type:task" "0E8A16" "雑務・整備・運用タスク"

create_label "area:common" "0E4F8A" "共通定義・型・設定"
create_label "area:data" "0052CC" "データ取得・整形・営業日整合"
create_label "area:strategies" "006B75" "戦略ロジック"
create_label "area:backtest" "0366D6" "バックテスト本体"
create_label "area:eval" "1B5E20" "評価指標・因子回帰・チャート"
create_label "area:llm" "6F42C1" "LLM連携・プロンプト・応答処理"
create_label "area:sandbox" "8A63D2" "生成コード隔離実行"
create_label "area:ui" "FBCA04" "フロントエンドUI"
create_label "area:api" "D4C5F9" "API層"
create_label "area:core" "1D3557" "実験ランナー・共通実行制御"
create_label "area:docs" "C2E0C6" "仕様書・README・運用文書"
create_label "area:infra" "BFDADC" "CI・開発環境・共通設定"

create_label "prio:P0" "B60205" "最優先。他を止めて対応"
create_label "prio:P1" "D93F0B" "高優先"
create_label "prio:P2" "FBCA04" "通常優先"

create_label "needs-spec" "F9D0C4" "仕様の明文化が必要"
create_label "needs-review" "FEF2C0" "レビュー待ち"
create_label "blocked" "000000" "他作業待ちで停止中"
create_label "good-first-task" "7057FF" "初手で取り組みやすい"
create_label "wont-do" "FFFFFF" "今回は実施しない"

echo "Labels applied to $REPO"
