"""Details of the top block-association candidates and known-gene blocks."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"

pheno = pd.read_csv(PROCESSED / "dormancy_phenotype.csv")
y = pheno["dsds50"].to_numpy(float)
n = len(y)
y_std = (y - y.mean()) / y.std()
pcs = pd.read_csv(TABLES / "accession_pcs.csv")
X = np.column_stack([np.ones(n), pcs[[f"PC{i}" for i in range(1, 6)]].to_numpy()])
inv = np.linalg.inv(X.T @ X)
r = y_std - X @ (inv @ (X.T @ y_std))

codes = np.load(PROCESSED / "genotypes_83.npy")
snp = pd.read_csv(PROCESSED / "snp_metadata.csv")
snp_ann = pd.read_csv(PROCESSED / "snp_annotation.csv")
blocks = pd.read_csv(PROCESSED / "haplotype_blocks.csv")
genes = pd.read_csv(ROOT / "data" / "annotations" / "genes_tair10.csv")
scan = pd.read_csv(TABLES / "block_association_pc5.csv").set_index("block_id")
n_blocks = len(blocks)

focus = list(scan.nsmallest(20, "p_raw").index)
# add blocks of the strongest known genes
for gene in ["AT2G01570", "AT5G45830", "AT1G15550", "AT3G47620", "AT5G66880", "AT1G28300", "AT2G29090", "AT5G27320"]:
    row = genes[genes["gene"] == gene].iloc[0]
    sub = blocks[(blocks["chrom"] == row["chr"]) & (blocks["start"] <= row["end"]) & (blocks["end"] >= row["start"])]
    if len(sub):
        focus.append(int(sub["block_id"].iloc[0]))
focus = sorted(set(focus))

rows = []
for b in focus:
    blk = blocks[blocks["block_id"] == b].iloc[0]
    s, e = int(blk["snp_start_row"]), int(blk["snp_end_row"])
    dos = (codes[s:e, :] == snp["minor_code"].to_numpy()[s:e, None]).astype(np.float64).T  # n x k
    dos = dos - dos.mean(0)
    dr = dos - X @ (inv @ (X.T @ dos))
    sxx = np.einsum("ij,ij->j", dr, dr)
    beta = (dr.T @ r) / sxx
    df = n - X.shape[1]
    sigma2 = float(r @ r / df)
    tstat = beta / np.sqrt(sigma2 / sxx)
    p = 2 * stats.t.sf(np.abs(tstat), df)
    j = int(np.argmin(p))
    row_snp = snp.iloc[s + j]
    ann = snp_ann.iloc[s + j]
    d = (codes[s + j, :] == row_snp["minor_code"]).astype(int)
    cnt = [int((d == 0).sum()), int((d == 1).sum())]
    med = [float(np.median(y[d == g])) for g in (0, 1)]
    carriers = d > 0
    genes_in = genes[
        (genes["chr"] == blk["chrom"]) & (genes["start"] <= blk["end"]) & (genes["end"] >= blk["start"])
    ]
    gene_list = "; ".join(
        f"{r.gene}({r['name']})" if str(r["name"]) != r.gene else r.gene for _, r in genes_in.head(12).iterrows()
    )
    rows.append(
        {
            "block_id": b,
            "chrom": int(blk["chrom"]),
            "start": int(blk["start"]),
            "end": int(blk["end"]),
            "n_snps": int(e - s),
            "p_raw": float(scan.loc[b, "p_raw"]),
            "p_graph": float(scan.loc[b, "p_graph"]),
            "p_embedding": float(scan.loc[b, "p_embedding"]),
            "pct_raw": float(100 * scan["p_raw"].rank(method="min").loc[b] / n_blocks),
            "pct_embedding": float(100 * scan["p_embedding"].rank(method="min").loc[b] / n_blocks),
            "top_snp": f"{int(row_snp['chrom'])}:{int(row_snp['pos'])}",
            "top_snp_p": float(p[j]),
            "top_snp_maf": float(row_snp["maf_83"]),
            "top_snp_minor": str(row_snp["minor_base"]),
            "top_snp_gene": str(ann["nearest_gene"]),
            "top_snp_gene_name": str(ann["nearest_gene_name"]),
            "top_snp_inside_gene": bool(ann["inside_gene"]),
            "n_major_minor": "/".join(str(c) for c in cnt),
            "median_major_minor": "/".join(f"{m:.1f}" for m in med),
            "median_major": float(np.median(y[~carriers])) if (~carriers).any() else np.nan,
            "median_minor": float(np.median(y[carriers])) if carriers.any() else np.nan,
            "genes_in_block": gene_list,
        }
    )

out = pd.DataFrame(rows).sort_values("p_raw")
out.to_csv(TABLES / "candidate_regions.csv", index=False)
print(out.head(20).to_string(index=False))
