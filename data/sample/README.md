# Sample Data

小さな再現用データをここに置きます。
ライセンス不明な外部市場データや秘匿データはコミットしません。

このリポジトリの sample CSV は、Issue `#3` の日米営業日整合を再現するための最小データです。

- `us_sectors.csv`: `date,sector,open,close`
- `jp_sectors.csv`: `date,sector,open,close`
- `trading_calendar.csv`: `market,date,is_open`
- `factor_returns.csv`: `date,mkt_rf,smb,hml,umd,rf`

US は 11 業種、JP は 17 業種を含みます。
US は `close-to-close`、JP は `open-to-close` を計算し、US 日付の次の JP 営業日に整合させます。
factor 系 CSV は FF3 / Carhart4 の回帰疎通確認用の最小データです。
