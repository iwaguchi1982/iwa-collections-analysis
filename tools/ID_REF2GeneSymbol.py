import pandas as pd
import GEOparse

# 1) 発現行列を読む
expr = pd.read_csv("./GSE68465_parsed/expression.tsv", sep="\t")

# 先頭列 ID_REF が probe ID
# 例: 1007_s_at, 1053_at ...

# 2) GPL96 の注釈を取得
gpl = GEOparse.get_GEO(geo="GPL96", destdir=".")

# GPLの注釈表を確認
gpl_table = gpl.table.copy()
print(gpl_table.columns.tolist())

# 3) probe ID と gene symbol の対応表を作る
# たいてい GPL 側に ID と Gene Symbol 系の列があります
# 列名は環境で微妙に違うことがあるので、まず確認が大事
probe_col = "ID"
symbol_col = "Gene Symbol"   # もし無ければ "Gene.Symbol" などを確認

anno = gpl_table[[probe_col, symbol_col]].copy()
anno.columns = ["ID_REF", "gene_symbol"]

# 4) 発現行列とマージ
expr2 = expr.merge(anno, on="ID_REF", how="left")

# 5) gene symbol が空のものを除く
expr2["gene_symbol"] = expr2["gene_symbol"].astype(str).str.strip()
expr2 = expr2[(expr2["gene_symbol"] != "") & (expr2["gene_symbol"] != "nan")]

# 6) 数値列だけ取り出す
sample_cols = [c for c in expr2.columns if c not in ["ID_REF", "gene_symbol"]]

# 7) 同じ gene symbol に複数 probe があるので平均でまとめる
expr_gene = expr2.groupby("gene_symbol", as_index=False)[sample_cols].mean()

# 8) 保存
expr_gene.to_csv("expression_gene_symbol.tsv", sep="\t", index=False)

print(expr_gene.head())
print(expr_gene.shape)