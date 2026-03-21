# PCA_SUB Notes

`pca_sub` は `pca_plain` と同じ日米整合済みデータを入力に使いますが、固有ベクトルをフル空間で直接推定せず、事前定義した部分空間に制約してから推定します。

## 入力行列

シグナル日 `t` に対し、直近 `W` 本の lookback window から標準化済みの結合リターン行列 `Z_t` を作ります。列は `US 11 業種 + JP 17 業種` の結合ベクトルです。

## 事前部分空間

初期 basis は次の 3 本です。

1. `global`
   全特徴量に同符号を与える全体因子です。
2. `country_spread`
   US を `+1`、JP を `-1` とする国別スプレッド因子です。
3. `cyclical_vs_defensive`
   景気敏感セクターを `+1`、ディフェンシブセクターを `-1`、その他を `0` とする因子です。

raw basis 行列を `B` とし、QR 分解で直交化した `Q` を推定に使います。

## 正則化付き固有空間推定

標準化後の結合相関行列を `C_t = Corr(Z_t)` とすると、まず部分空間上へ射影して

`S_t = Q^T C_t Q`

を作ります。その後、対角 shrinkage で

`S_t^(reg) = (1 - lambda) S_t + lambda diag(diag(S_t))`

を作り、`S_t^(reg)` の固有分解を取ります。実装では正の固有値だけを残し、上位 `components` 本を使います。

## 米国ショックの射影と日本側復元

部分空間で得た固有ベクトル `u_k` はフル空間に

`v_k = Q u_k`

として持ち上げます。シグナル日当日の標準化済み US ベクトルを `x_t^(US)` とすると、各成分のスコアは

`f_k = (x_t^(US) . v_k^(US)) / (||v_k^(US)||^2 + lambda)`

で求めます。JP 側の復元スコアは

`score_j = sum_k lambda_k f_k v_{k,j}^{(JP)}`

です。

## ポートフォリオ化

復元された JP score を cross-sectional に順位付けし、`q` 分位の上位を long、下位を short に採用します。ウェイトは `mom` と `pca_plain` と同じ 50/50 gross の等ウェイト long-short です。
