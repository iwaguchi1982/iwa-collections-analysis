# Iwa Collections Analysis: サービス概要およびデプロイガイド

本ドキュメントは、本アプリケーション（Iwa Collections Analysis）をホスティングおよび運用・提供する事業者（企業・システム管理部門など）向けに、アプリケーションが解決する課題（ビジネスバリュー）と、具体的な動作環境およびデプロイ手順をまとめたものです。

> 本プロジェクトは探索研究向けの開発中ソフトウェアです。
> 臨床判断、規制対応用途、診断用途を意図したものではありません。

---

## 1. アプリケーションの概要 (Executive Summary)

### アプリケーション名
**Iwa Collections Analysis**

### 開発の背景と目的
がんに特化した世界最大級のオープンデータベースである「cBioPortal」等の公開データを活用し、次世代のがん治療薬の標的（ターゲット）となる遺伝子の探索、および既存薬の転用（ドラッグリパーパシング）を支援するための、インタラクティブなデータサイエンス・プラットフォームです。

従来のバイオインフォマティクス解析は、コマンドラインやR/Pythonのプログラミングを含む高度な専門知識を要求し、時間がかかるものでした。本システムは、洗練されたGUI（グラフィカルユーザーインターフェース）を通じて、**「非プログラマーの医学系研究者や製薬企業の探索担当者が、数クリックで高度な統計解析と仮説検証のサイクルを回せる」** 状態を実現します。

### 主要機能と提供価値 (Value Proposition)
1. **目的別ナビゲーション (Mission-Based Routing)**: 
   - 「予後マーカー探索」「新規治療標的探索」「既存薬リパーパス」の3つの研究目的に沿って、最適な解析ツール（タブ）を適切な順番で自動提示します。
2. **高度な統計補正による信頼性の担保**: 
   - カプランマイヤー曲線やCox比例ハザードモデルといった基本的な生存分析にとどまらず、**傾向スコアマッチング (PSM)** による年齢やステージ等の交絡因子の徹底排除をGUIから実行可能です。
3. **探索から検証・実用化までのシームレスな体験**:
   - `Auto Discovery`: 全遺伝子ペアから予後予測に優れた組み合わせを自動探索（多重検定FDRの自動信号機判定付き）。
   - `Cross-Cohort Validation` & `Meta-Analysis`: 別のデータセット（外部コホート）を用いて結果を検証し、過学習を防ぎます。
   - `Companion Diagnostics`: この薬はどの集団（Stage IIIなど）に最も効果的か？という「患者層別化」の提案を自動出力します。
4. **創薬適性・安全性（Druggability & Safety）の統合**:
   - 探索した標的遺伝子が、細胞のどこにあるのか（細胞表面か核内か）、全身毒性リスク（CRISPR必須性）はないか、既存の承認薬や治験薬はないかという「創薬実現性」のデータをOpenTargetsから（またはセキュアなMockデータから）取得して統合表示する機能を備えています。
5. **堅牢なガードレール（フェイルセーフUI）**:
   - イベント（死亡等）の数が少なすぎるデータセットや、欠損値が多いデータをユーザーが誤って解析しようとした場合、アプリ側が検知して**自動で計算をブロック・警告**し、誤った統計解釈による開発の無駄を防ぎます（Event DANGER, Missing CAUTION）。

---

## 2. システム・アーキテクチャ

本アプリケーションは、Python製データアプリフレームワークである **Streamlit** を用いて構築された**ステートレスなフロントエンド＋バックエンド統合アプリケーション**です。

- **言語**: Python 3.12+ (または 3.11+)
- **フレームワーク**: Streamlit
- **主要ライブラリ**: 
  - データ処理: `pandas`, `numpy`
  - 生存時間解析・統計: `lifelines`, `statsmodels`, `scipy`
  - 可視化: `matplotlib`, `seaborn`
  - 外部API通信: `requests`
- **データストレージ (バックエンド)**: 
  - データベースサーバー（SQL等）は不要です。初期提供のCSV/TSVファイル（またはユーザーがアップロードするファイル）をメモリ上に展開・キャッシュ（`@st.cache_data`）して高速計算します。
- **外部通信**:
  - `api.opentargets.io` (GraphQL) に対するアウトバウンド通信を行う機能（Live APIモード）を有します。※セキュリティ・プライバシーの観点で、通信を遮断する手動トグル（Mockモード）も標準実装されています。

---

## 3. デプロイおよび運用ガイド (Deployment Guide)

### 前提条件 (システム要件)
- **OS**: Linux (Ubuntu 22.04 推奨), macOS, または Windows WSL2
- **CPU**: 4 vCPU 以上推奨（`Auto Discovery` 等の大規模行列計算・Cox回帰ループにCPUパワーを消費します）
- **RAM**: 8GB 以上必須（16GB以上を推奨。cBioPortal等の巨大な発現量マトリクスをPandasでメモリ上に保持するため）

### A. Pixi (推奨パッケージマネージャ) を用いたローカル/開発環境デプロイ
プロジェクトのルートディレクトリにある `pixi.toml` が環境を定義しています。

1. **Pixiのインストール**
   ```bash
   curl -fsSL https://pixi.sh/install.sh | bash
   ```
2. **依存関係のインストール**
   ```bash
   cd /path/to/iwa-collections-analysis
   pixi install
   ```
3. **Pixi 環境内でアプリケーションを起動**
   ```bash
   python -m streamlit run iwa_collections_analysis.py --server.port 8501 --server.address 0.0.0.0
   ```
   **または、Pixi シェルに入らずに直接起動する場合**
   ```bash
   pixi run python -m streamlit run iwa_collections_analysis.py --server.port 8501 --server.address 0.0.0.0
   ```

### B. Docker を用いた本番 / クラウド環境へのデプロイ（参考例）
コンテナベース（Docker）でのホスティングが最も管理が容易です。以下のような標準的な `Dockerfile` で動作します。

```dockerfile
# Dockerfile 例
FROM python:3.12-slim

WORKDIR /app

# 必要なシステムライブラリ（一部C拡張のビルド用）
RUN apt-get update && apt-get install -y \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# requirements.txt の生成 (pixiやpoetryからエクスポートしたもの)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# コードとデータのコピー
COPY . .

# Streamlitのデフォルトポート
EXPOSE 8501

# ヘルスチェックの設定 (任意)
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# アプリの起動
CMD ["python", "-m", "streamlit", "run", "iwa_collections_analysis.py", "--server.port", "8501", "--server.address", "0.0.0.0", "--server.enableCORS", "false"]
```

### ネットワーク・セキュリティ設定の要件
1. **リバースプロキシ・HTTPS化**:
   Streamlit自体はHTTPSの終端機能を持たないため、本番運用においてはNginx, AWS ALB, GCP Cloud Load Balancing などを前段に配置し、SSL証明書の設定とリバースプロキシを行ってください。
2. **認証システム**:
   本アプリ単体にはログインシステムは組み込まれていません。クラウドインフラのアカウント認証（例: AWS Cognito, IAP）や、SAML/OIDC対応のリバースプロキシ（例: OAuth2 Proxy）を通じたアクセス制限を施すことを強く推奨します（特にLive APIモードではなくMockを使用し、ユーザーの検索履歴を社内秘にする場合）。

### パフォーマンスチューニング
大規模コホート（例：5,000人以上×2万遺伝子）のデータを扱う場合、Streamlitのデータキャッシュサイズ上限によりメモリ不足やOOM (Out Of Memory) キルが発生する可能性があります。
環境変数や `~/.streamlit/config.toml` にて、必要に応じてキャッシュメモリの上限を調整できます。

---

## 4. 管理者向けFAQ

**Q. 「Data scope limited」または「メモリ確保エラー」が出ると報告があった**
A. ホストマシンのRAM（メモリ）が不足しています。インスタンスのスケールアップを検討するか、より小さなデータセットをAnalysisへ投入するようにユーザーへ案内してください。

**Q. 外部通信 (Live APIモード) でデータが取得できない**
A. 企業のファイアウォール・プロキシが `https://api.opentargets.io` との通信を遮断しているか、OpenTargetsのサーバーがメンテナンス中の可能性があります。ユーザーへ提供アプリの画面上から「Mock Database」モードへの切り替えを案内してください。

**Q. UIに⛔や⚠️のアラートが出て計算がストップするという問い合わせ**
A. アプリケーションの正常な「ガードレール機能」の動作です。サンプルの欠損が多すぎるか、イベント（死亡数）が圧倒的に足りないため、誤った結果の出力を防止しています。イベントの多い別のガン種コホートを選択するよう、ユーザーに案内してください。
