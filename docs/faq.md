# データセットの追加と高速な読み込みについて (Parquet化)

本アプリケーションは、大規模ながん種（PanCancerなど）のデータを扱うことを想定し、データのロードを超高速化するために **Parquet (パーケイ) 形式** をサポートしています。

新たにデータセットを追加してアプリに読み込ませる際は、以下の手順を実施してください。

### 1. データの配置
cBioPortal 等からダウンロードしたデータセットのフォルダを、`iwa-collections-analysis/data/` の直下に配置します。
(例: `iwa-collections-analysis/data/brca_tcga_pan_can_atlas_2018/`)

フォルダ内に最低限、以下に該当する臨床データ(`data_clinical*.txt`)と発現系データ(`data_mrna_seq*.txt`等)のテキストファイルが含まれている必要があります。
※ 1-Geneの「Mutation」「CNA」解析オプションを利用する場合は、追加で `data_mutations.txt` や `data_cna.txt` (または `data_linear_CNA.txt`) も同じフォルダに配置してください。

### 2. データ変換スクリプトの実行（必須）
そのままアプリを起動してもテキストデータを読み込むため動作はしますが、ロードに非常に長い時間がかかります。
これを一瞬で読み込めるようなバイナリ形式 (`.parquet`) に変換するため、ターミナルで以下のコマンドを1度だけ実行してください。

```bash
pixi run python scripts/convert_to_parquet.py
```

このスクリプトは `data/` 配下をスキャンし、まだ変換されていないデータセットを見つけると自動的に読み込み元のテキストをパースし、各データフォルダ内に `clinical.parquet` と `expression.parquet` を生成します。

### 3. アプリの起動
変換が完了したら、通常通りアプリを起動してください。

```bash
pixi run streamlit run iwa-collections-analysis.py
```

アプリ側は `.parquet` ファイルの存在を検知すると、テキストファイルよりも最優先でそれを読み直すため、データロードの待ち時間が劇的に短縮されます。
