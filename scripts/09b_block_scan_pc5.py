"""Block-level association scan with a calibrated fixed-effects null (PC1-5)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"
rng = np.random.default_rng(20261026)

pheno = pd.read_csv(PROCESSED / "dormancy_phenotype.csv")
y = pheno["dsds50"].to_numpy(float)
y = (y - y.mean()) / y.std()
n = len(y)

pcs = pd.read_csv(TABLES / "accession_pcs.csv")
X = np.column_stack([np.ones(n), pcs[[f"PC{i}" for i in range(1, 6)]].to_numpy()])
p = X.shape[1]

codes = np.load(PROCESSED / "genotypes_83.npy")
snp = pd.read_csv(PROCESSED / "snp_metadata.csv")
maf = snp["maf_83"].to_numpy()
keep = np.flatnonzero((maf > 0.05) & (snp["minor_code"].to_numpy() < 4))
minor = snp["minor_code"].to_numpy()[keep]
G = (codes == snp["minor_code"].to_numpy()[:, None]).astype(np.float32)
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

# residualise y and X once
XtX_inv = np.linalg.inv(X.T @ X)
r = y - X @ (XtX_inv @ (X.T @ y))
rss0 = float(r @ r)
m = n - p
print(f"residual df {m}, rss0 {rss0:.3f}", flush=True)

# raw block bases, residualised
print("residualising blocks", flush=True)
Zr_raw = np.zeros((n_blocks, n, 25), dtype=np.float32)
Zr_emb = np.zeros((n_blocks, n, 16), dtype=np.float32)
n_snp = np.zeros(n_blocks, dtype=np.int32)
for b in range(n_blocks):
    s = int(blocks.loc[b, "snp_start_row"])
    e = int(blocks.loc[b, "snp_end_row"])
    Z = G[s:e, :].T.astype(np.float64)
    Z = Z - Z.mean(0)
    Zr = Z - X @ (XtX_inv @ (X.T @ Z))
    k = Zr.shape[1]
    Zr_raw[b, :, :k] = Zr
    n_snp[b] = k
    Ze = emb[:, b, :].astype(np.float64)
    Ze = Ze - Ze.mean(0)
    sd = Ze.std(0)
    ok = sd > 1e-8
    Ze = Ze[:, ok] / sd[ok]
    Ze = Ze - X @ (XtX_inv @ (X.T @ Ze))
    Zr_emb[b, :, : Ze.shape[1]] = Ze
    if (b + 1) % 2000 == 0:
        print(f"  {b+1}/{n_blocks}", flush=True)

# group member matrix for graph kernel
from scipy import sparse
rows, cols = [], []
for b, g in enumerate(groups):
    rows.extend([b] * len(g))
    cols.extend(g)
S = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_blocks, n_blocks))


def satterthwaite(q, lam):
    lam = lam[lam > 1e-10]
    c1 = float(lam.sum())
    c2 = float((lam**2).sum())
    scale = c2 / c1
    df = c1**2 / c2
    return float(stats.chi2.sf(q / scale, df)), df, scale


def scan(kind: str):
    q = np.zeros(n_blocks)
    df = np.zeros(n_blocks)
    sc = np.zeros(n_blocks)
    pv = np.zeros(n_blocks)
    for b in range(n_blocks):
        if kind == "raw":
            Z = Zr_raw[b, :, : n_snp[b]].astype(np.float64)
        elif kind == "embedding":
            k = int(np.sum(np.abs(Zr_emb[b]).sum(0) > 1e-12))
            Z = Zr_emb[b, :, :k].astype(np.float64)
        else:
            members = groups[b]
            Z = np.concatenate(
                [Zr_raw[j, :, : n_snp[j]].astype(np.float64) for j in members], axis=1
            )
        A = Z.T @ Z
        qq = float(np.sum((Z.T @ r) ** 2))
        lam = np.linalg.eigvalsh(A)
        pval, d, sca = satterthwaite(qq, lam)
        q[b], pv[b], df[b], sc[b] = qq, pval, d, sca
    return q, pv, df, sc


res = pd.DataFrame(
    {
        "block_id": np.arange(n_blocks),
        "chrom": blocks["chrom"].to_numpy(),
        "start": blocks["start"].to_numpy(),
        "end": blocks["end"].to_numpy(),
        "n_snps": n_snp,
        "group_size": [len(g) for g in groups],
        "nearest_gene": blocks["nearest_gene"].to_numpy(),
        "nearest_dist": blocks["nearest_dist"].to_numpy(),
        "gene_count": blocks["gene_count"].to_numpy(),
    }
)
for kind in ["raw", "graph", "embedding"]:
    print("scan", kind, flush=True)
    q, pv, df, sca = scan(kind)
    res[f"Q_{kind}"] = q
    res[f"p_{kind}"] = pv
    res[f"df_{kind}"] = df
    if kind == "raw":
        scale_raw, df_raw = sca, df
    res[f"q_{kind}"] = multipletests(pv, method="fdr_bh")[1]

# rotation-based FWER for the raw kernel (exact residual-space rotation)
print("rotation null", flush=True)
Ux, _, _ = np.linalg.svd(X, full_matrices=True)
B = Ux[:, p:]  # n x m orthonormal basis of the residual space
Zc = np.zeros((n_blocks, m, 25), dtype=np.float32)
for b in range(n_blocks):
    Zc[b, :, : n_snp[b]] = B.T @ Zr_raw[b, :, : n_snp[b]].astype(np.float64)
focus = sorted(
    set(res.nsmallest(20, "p_raw")["block_id"])
    | set(res.nsmallest(20, "p_graph")["block_id"])
    | set(res.nsmallest(20, "p_embedding")["block_id"])
    | {7960}
)
fidx = {b: i for i, b in enumerate(focus)}
focus_p = np.zeros((1000, len(focus)))
min_p_null = np.zeros(1000)
max_null = np.zeros(1000)
for t in range(1000):
    cs = rng.standard_normal(m)
    proj = np.einsum("bij,i->bj", Zc, cs)
    qv = np.sum(proj**2, axis=1)
    pv_star = stats.chi2.sf(qv / scale_raw, df_raw)
    min_p_null[t] = pv_star.min()
    focus_p[t] = pv_star[focus]
    max_null[t] = qv.max()
min_p_obs = float(res["p_raw"].min())
max_obs = float(res["Q_raw"].max())
fwer = float((1 + np.sum(min_p_null <= min_p_obs)) / 1001)
fwer_maxq = float((1 + np.sum(max_null >= max_obs)) / 1001)

pobs = res.set_index("block_id").loc[focus, "p_raw"].to_numpy()
p_rot = (1 + (focus_p <= pobs).sum(axis=0)) / 1001
rot = pd.DataFrame(
    {
        "block_id": focus,
        "Q_raw": res.set_index("block_id").loc[focus, "Q_raw"].to_numpy(),
        "p_raw": pobs,
        "p_rotation_raw": p_rot,
    }
).merge(blocks[["chrom", "start", "end", "nearest_gene"]], left_on="block_id", right_index=True)
rot.to_csv(TABLES / "block_pc5_rotation_focus.csv", index=False)

summary = {
    "model": "OLS with intercept + PC1-5; standardised DSDS50",
    "n": int(n),
    "n_blocks": int(n_blocks),
    "fwer_raw_minp": fwer,
    "fwer_raw_maxq": fwer_maxq,
    "rotation_minp_median": float(np.median(min_p_null)),
    "rotation_maxq_median": float(np.median(max_null)),
}
for kind in ["raw", "graph", "embedding"]:
    ok = res[f"p_{kind}"].notna()
    summary[kind] = {
        "min_p": float(res.loc[ok, f"p_{kind}"].min()),
        "n_p_0.01": int((res.loc[ok, f"p_{kind}"] < 0.01).sum()),
        "n_fdr_0.05": int((res.loc[ok, f"q_{kind}"] < 0.05).sum()),
        "n_fdr_0.10": int((res.loc[ok, f"q_{kind}"] < 0.10).sum()),
        "top_block": int(res.loc[res.loc[ok, f"p_{kind}"].idxmin(), "block_id"]),
    }
    top = res[ok].sort_values(f"p_{kind}").head(20)
    top[
        ["block_id", "chrom", "start", "end", "p_" + kind, "q_" + kind, "nearest_gene", "nearest_dist"]
    ].to_csv(TABLES / f"block_pc5_top20_{kind}.csv", index=False)

dog = res[(res["chrom"] == 5) & (res["start"] <= 18_591_506) & (res["end"] >= 18_589_418)]
summary["dog1_region"] = {}
for kind in ["raw", "graph", "embedding"]:
    summary["dog1_region"][kind] = {
        "block_ids": dog["block_id"].tolist(),
        "p": dog[f"p_{kind}"].round(6).tolist(),
        "percentile": [
            float(100.0 * stats.rankdata(res[f"p_{kind}"])[i] / n_blocks) for i in dog.index
        ],
    }
    if kind == "raw":
        summary["dog1_region"]["p_rotation_raw"] = [
            float(rot.loc[rot["block_id"] == b, "p_rotation_raw"].iloc[0]) for b in dog["block_id"]
        ]
summary["rotation_focus_top"] = (
    rot.sort_values("p_rotation_raw")
    .head(10)[["block_id", "chrom", "start", "nearest_gene", "p_raw", "p_rotation_raw"]]
    .round(6)
    .to_dict(orient="records")
)

res.to_csv(TABLES / "block_association_pc5.csv", index=False)
(TABLES / "block_association_pc5_summary.json").write_text(json.dumps(summary, indent=2))

# ---- figure: QQ and Miami plot ----
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), dpi=170)
ax = axes[0]
colors = {"raw": "#4c4c4c", "graph": "#3b6ea8", "embedding": "#b4483f"}
for kind in ["raw", "graph", "embedding"]:
    pv = np.sort(res[f"p_{kind}"].dropna().to_numpy())
    nn = len(pv)
    k = np.arange(1, nn + 1)
    exp = -np.log10((k - 0.5) / nn)
    lo = -np.log10(stats.beta.ppf(0.975, k, nn - k + 1))
    hi = -np.log10(stats.beta.ppf(0.025, k, nn - k + 1))
    ax.fill_between(exp, lo, hi, color="#d9d9d9", alpha=0.55, linewidth=0)
    ax.plot(exp, -np.log10(pv), color=colors[kind], linewidth=1.15, label=kind)
ax.plot([0, 6], [0, 6], "--", color="#888888", linewidth=0.8)
ax.set_xlim(0, 5.2)
ax.set_ylim(0, 6.5)
ax.set_xlabel("Expected $-\\log_{10}P$")
ax.set_ylabel("Observed $-\\log_{10}P$")
ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)

ax = axes[1]
offset = 0.0
ticks, labels = [], []
chrom_offset = {}
chrom_colors = {1: "#4c72b0", 2: "#dd8452", 3: "#55a868", 4: "#c44e52", 5: "#8172b3"}
for c in range(1, 6):
    sub = res[res["chrom"] == c]
    span = sub["end"].max() / 1e6
    chrom_offset[c] = offset
    x = offset + sub["start"] / 1e6
    ax.scatter(x, -np.log10(sub["p_raw"]), s=5, color=chrom_colors[c], alpha=0.65, linewidths=0)
    ticks.append(offset + span / 2)
    labels.append(str(c))
    offset += span + 4
ax.axhline(-np.log10(0.01), color="#999999", linestyle=":", linewidth=0.8)
top = res.nsmallest(1, "p_raw").iloc[0]
xtop = chrom_offset[int(top["chrom"])] + top["start"] / 1e6
ax.annotate(
    f"block {int(top['block_id'])} ({top['nearest_gene']})",
    xy=(xtop, -np.log10(top["p_raw"])),
    xytext=(6, 8),
    textcoords="offset points",
    fontsize=7,
    arrowprops=dict(arrowstyle="-", color="#666666", linewidth=0.6),
)
xdog = chrom_offset[5] + dog["start"].iloc[0] / 1e6
ax.annotate(
    "DOG1",
    xy=(xdog, -np.log10(dog["p_raw"].iloc[0])),
    xytext=(-28, 8),
    textcoords="offset points",
    fontsize=7,
    color="#8172b3",
    arrowprops=dict(arrowstyle="-", color="#8172b3", linewidth=0.6),
)
ax.set_xticks(ticks)
ax.set_xticklabels(labels)
ax.set_xlabel("Chromosome")
ax.set_ylabel("$-\\log_{10}P$, raw block kernel")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(ROOT / "results" / "figures" / "fig6_block_association.png")
plt.close(fig)

print(json.dumps(summary, indent=2))
print("top raw:")
print(pd.read_csv(TABLES / "block_pc5_top20_raw.csv").head(10).to_string())
