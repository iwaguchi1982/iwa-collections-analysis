# Continuous + Interaction（連続値 + 相互作用）

このセクションは、2遺伝子を **二値化せず** 連続値で使い、
さらに **相互作用（gene1 × gene2）** を入れたCoxモデルです。

## モデルのイメージ

- gene1(z)
- gene2(z)
- gene1(z) × gene2(z)
- （必要なら共変量）

ここで z は z-score（平均0, 分散1）です。

## 何が分かる？

- gene1 が上がるほどリスクが増/減するか
- gene2 が上がるほどリスクが増/減するか
- **gene1の効果がgene2の値によって変わるか**（interaction）

## 注意点

- interactionが非有意でも「2遺伝子が無意味」とは限りません
- 多重検定・探索的解析ではp値が過小評価され得ます
- ⚠️/⛔ が出たら Guardrails を優先

## 表示ラベル

このセクションでは、
- `expression_1(z)` を **Gene1名（z）**
- `expression_2(z)` を **Gene2名（z）**
- `expression_1×expression_2` を **Gene1×Gene2（interaction）**

のように、実際の遺伝子名で表示するのが推奨です。

