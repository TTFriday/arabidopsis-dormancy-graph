"""Sensitivity of the block scan to phenotype transform and residual kinship."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"

pheno = pd.read_csv(PROCESSED / "dormancy_phenotype.csv")
y_raw = pheno["dsds50"].to_numpy(float)
n = len(y_raw)
y_std = (y_raw - y_raw.mean()) / y_raw.std()
y_rank = stats.norm.ppf((stats.rankdata(y_raw) - 0.5) / n)
y_rank = (y_rank - y_rank.mean()) / y_rank.std()

pcs = pd.read_csv(TABLES / "accession_pcs.csv")
X = np.column_stack([np.ones(n), pcs[[f"PC{i}" for i in range(1, 6)]].to_numpy()])
p = X.shape[1]

codes = np.load(PROCESSED / "genotypes_83.npy")
snp = pd.read_csv(PROCESSED / "snp_metadata.csv")
maf = snp["maf_83"].to_numpy()
keep = np.flatnonzero((maf > 0.05) & (snp["minor_code"].to_numpy() < 4))
minor = snp["minor_code"].to_numpy()[keep]
G = (codes == snp["minor_code"].to_numpy()[:, None]).astype(np.float64)
blocks = pd.read_csv(PROCESSED / "haplotype_blocks.csv")
n_blocks = len(blocks)

edges = np.load(PROCESSED / "block_edges.npz")
src, dst = edges["edge_index"]
neighbours: list[set[int]] = [set() for _ in range(n_blocks)]
for i, j in zip(src, dst):
    neighbours[int(i)].add(int(j))
    neighbours[int(j)].add(int(i))
groups = [sorted({b, *neighbours[b]}) for b in range(n_blocks)]
emb = np.load(PROCESSED / "graph_embeddings_83.npy")

af = G[keep].mean(1) / 2
Zg = G[keep] - 2 * af[:, None]
K = Zg.T @ Zg / (2 * np.sum(af * (1 - af)))
ev, evec = np.linalg.eigh(K)
ev = np.clip(ev, 0, None)

cfg = {
    "rank_pc5": (y_rank, np.eye(n)),
    "raw_pc5_k0.1": (y_std, (evec / np.sqrt(1 + 0.1 * ev)).T),
    "rank_pc5_k0.1": (y_rank, (evec / np.sqrt(1 + 0.1 * ev)).T),
}


def satter(q, lam):
    lam = lam[lam > 1e-10]
    c1 = float(lam.sum())
    c2 = float((lam**2).sum())
    return float(stats.chi2.sf(q / (c2 / c1), c1**2 / c2))


def run(y, W, tag):
    yt, Xt = W @ y, W @ X
    inv = np.linalg.inv(Xt.T @ Xt)
    rt = yt - Xt @ (inv @ (Xt.T @ yt))
    Zr_raw, Zr_emb = {}, {}
    for b in range(n_blocks):
        s = int(blocks.loc[b, "snp_start_row"])
        e = int(blocks.loc[b, "snp_end_row"])
        Z = G[s:e, :].T
        Z = Z - Z.mean(0)
        Zt = W @ Z
        Zr_raw[b] = Zt - Xt @ (inv @ (Xt.T @ Zt))
        Ze = emb[:, b, :]
        Ze = Ze - Ze.mean(0)
        sd = Ze.std(0)
        ok = sd > 1e-8
        Ze = Ze[:, ok] / sd[ok]
        Zt = W @ Ze
        Zr_emb[b] = Zt - Xt @ (inv @ (Xt.T @ Zt))
    out = []
    for b in range(n_blocks):
        rec = {"block_id": b}
        for kind in ["raw", "graph", "embedding"]:
            if kind == "raw":
                Z = Zr_raw[b]
            elif kind == "embedding":
                Z = Zr_emb[b]
            else:
                Z = np.concatenate([Zr_raw[j] for j in groups[b]], axis=1)
            A = Z.T @ Z
            q = float(np.sum((Z.T @ rt) ** 2))
            rec[f"p_{kind}"] = satter(q, np.linalg.eigvalsh(A))
            rec[f"Q_{kind}"] = q
        out.append(rec)
    res = pd.DataFrame(out).merge(
        blocks[["chrom", "start", "end", "nearest_gene", "nearest_dist"]], left_on="block_id", right_index=True
    )
    summ = {}
    dog = res[(res["chrom"] == 5) & (res["start"] <= 18_591_506) & (res["end"] >= 18_589_418)]
    for kind in ["raw", "graph", "embedding"]:
        pv = res[f"p_{kind}"].to_numpy()
        qv = multipletests(pv, method="fdr_bh")[1]
        res[f"q_{kind}"] = qv
        top = res.sort_values(f"p_{kind}").head(20)
        top[["block_id", "chrom", "start", "end", f"p_{kind}", f"q_{kind}", "nearest_gene"]].to_csv(
            TABLES / f"sens_{tag}_top20_{kind}.csv", index=False
        )
        ranks = stats.rankdata(pv)
        summ[kind] = {
            "min_p": float(pv.min()),
            "n_p_0.01": int((pv < 0.01).sum()),
            "n_fdr_0.05": int((qv < 0.05).sum()),
            "top_block": int(top.iloc[0]["block_id"]),
            "top_gene": top.iloc[0]["nearest_gene"],
            "dog1_p": float(dog[f"p_{kind}"].iloc[0]),
            "dog1_percentile": float(100 * ranks[dog.index[0]] / n_blocks),
        }
    res.to_csv(TABLES / f"sens_{tag}_scan.csv", index=False)
    return summ


summary = {}
for tag, (y, W) in cfg.items():
    print("running", tag, flush=True)
    summary[tag] = run(y, W, tag)
    print(json.dumps(summary[tag], indent=2), flush=True)

(TABLES / "block_scan_sensitivity.json").write_text(json.dumps(summary, indent=2))
