# Help / Documentation (Externalized)

このディレクトリは、Streamlitアプリの **Helpタブの文章を外部ファイル化** するための素案です。

- 長文化しやすい説明を `iwa-collections-analysis.py` から分離することで、
  - コードの見通し向上
  - ドキュメントの改訂がしやすい
  - 将来的に英語版/社内規約版など派生を作りやすい

を狙います。

## 想定する読み込み方法（例）

- `help_docs/` 配下の Markdown を `st.markdown()` で読み込み
- もしくは `help_docs/*.md` を結合して1画面に表示

## 重要：Guardrails 表示（必須要件）

アプリ内で表示されるガードレールの色は以下を意味します。

- ✅ **緑：OK**（通常の解釈が可能）
- ⚠️ **黄：注意**（解釈に注意。**Help → Guardrails** を参照）
- ⛔ **赤：危険域**（この項目は **計算スキップ/参考値扱い**。**Help → Guardrails** を参照）

ドキュメント側でも、黄色/赤に該当する時の典型原因と対応を明記します。

---

## ファイル構成

- `overview.md`：アプリ全体の目的と使い方
- `guardrails.md`：ガードレール判定の考え方・典型例・対応
- `km_logrank.md`：KM曲線とlog-rank（2群/4群）
- `cox_models.md`：Cox（単変量/共変量調整/4群）
- `interaction.md`：連続値 + 相互作用モデルの読み方
- `cut_smoothness.md`：cutoff scan（安定性評価）の意味
- `faq.md`：よくある質問（計算が落ちる/真っ白/列不足 など）

