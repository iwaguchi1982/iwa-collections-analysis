# Overview

このアプリは、遺伝子発現（1遺伝子または2遺伝子）と生存データ（例：OS_MONTHS / is_dead）を結合して、

- **Kaplan–Meier（KM）曲線とLog-rank検定**（2群 / 4群）
- **Cox比例ハザードモデル**（共変量調整あり/なし、時間依存性モデル対応）
- **2遺伝子の連続値 + 相互作用（interaction）モデル**
- **Cutoffの頑健性評価**（Cut smoothness scan / Bootstrap Re-sampling）
- **機序・パスウェイ解析**（GSEA / ORA によるメカニズム探索）
- **マルチコホート検証**（外部コホートのImport、Validation、Meta-Analysis）
- **交絡因子の厳密な補正**（Propensity Score Matching: PSM）
- **標的の優先順位付けとスコアリング**（Target Triage & Prioritization dashboard）

を統合的かつ直感的に実行・確認できるようにするための、包括的なバイオマーカー探索・創薬標的評価プラットフォームです。

> **🔰 はじめての方へ:**
> 初めてご利用になる際は、先に Help メニュー内の **「🔰 Beginner's Guide (解釈の注意点)」** を必ずご一読ください。p値の解釈や多重検定の罠など、生存時間解析において非常に陥りやすいミスを防ぐための重要事項がまとまっています。
内容はアプリ製作者の主張が織り交ぜられていますので、取捨選択を行ってください。

## 推奨ワークフロー

1. **Auto Discovery**
   - 候補ペア（Gene1×Gene2）を探索
   - 「Apply to Analysis Settings」で **Analysis側の設定に反映**
2. **Analysis**
   - 「Run Detailed Analysis」を実行
   - KM / Cox / Interaction を確認
   - 追加で **Cut smoothness scan** を実行し、cutoff依存性を確認
3. **解釈はGuardrailsとセットで**
   - ⚠️/⛔ が出たときは **Help → Guardrails** を参照

---

## 用語

- **OS_MONTHS**：追跡期間（例：月）
- **is_dead**：イベント（死亡）フラグ（1=イベント発生、0=検閲）
- **cutoff**：対象データ(発現量など)をHigh/Lowに分ける分割点（パーセンタイル）
- **HiHi / HiLo / LoHi / LoLo**：2遺伝子をそれぞれHigh/Lowに二値化した4群
- **Signature Score**: 複数遺伝子の発現量を統合（平均Z-score等）した疑似的な発現スコア
- **Mutant / WT**: 遺伝子変異の有無（Mutant=変異あり、WT=野生型）
- **Amplification / Deletion / Diploid**: Copic Number Alteration(コピー数変異)のステータス

