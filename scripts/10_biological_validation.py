"""Known-dormancy-gene ranking and enrichment for the block association scan."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
RNG = np.random.default_rng(20261027)

scan = pd.read_csv(TABLES / "block_association_pc5.csv")
blocks = pd.read_csv(PROCESSED / "haplotype_blocks.csv")
genes = pd.read_csv(ROOT / "data" / "annotations" / "genes_tair10.csv")
n_blocks = len(blocks)

KNOWN = {
    "AT5G45830": "DOG1",
    "AT3G24650": "ABI3",
    "AT2G40220": "ABI4",
    "AT2G36270": "ABI5",
    "AT1G21970": "LEC1",
    "AT1G28300": "LEC2",
    "AT3G26790": "FUS3",
    "AT1G18100": "MFT",
    "AT2G20180": "PIF1/PIL5",
    "AT2G18790": "PHYB",
    "AT4G36930": "SPT",
    "AT3G05120": "GID1A",
    "AT3G63010": "GID1B",
    "AT5G27320": "GID1C",
    "AT2G01570": "RGA",
    "AT1G14920": "GAI",
    "AT1G66350": "RGL1",
    "AT3G03450": "RGL2",
    "AT5G17490": "RGL3",
    "AT4G24210": "SLY1",
    "AT3G24220": "NCED6",
    "AT1G78390": "NCED9",
    "AT4G19230": "CYP707A1",
    "AT2G29090": "CYP707A2",
    "AT5G51760": "AHG1",
    "AT3G11410": "AHG3/PP2CA",
    "AT1G72770": "HAB1",
    "AT4G26080": "ABI1",
    "AT5G57050": "ABI2",
    "AT5G08590": "SNRK2.1",
    "AT3G50500": "SNRK2.2",
    "AT5G66880": "SNRK2.3",
    "AT4G25420": "GA20OX1",
    "AT5G51810": "GA20OX2",
    "AT1G15550": "GA3OX1",
    "AT1G80340": "GA3OX2",
    "AT5G67030": "ABA1",
    "AT1G52340": "ABA2",
    "AT1G16540": "ABA3",
    "AT2G27150": "AAO3",
    "AT4G17870": "PYR1",
    "AT5G46790": "PYL1",
    "AT2G26040": "PYL2",
    "AT2G38310": "PYL4",
    "AT5G05440": "PYL5",
    "AT2G40330": "PYL6",
    "AT5G53160": "PYL8/RCAR3",
    "AT4G37470": "KAI2",
    "AT2G42620": "MAX2",
    "AT5G57710": "SMAX1",
    "AT5G35550": "TT2",
    "AT4G09820": "TT8",
    "AT5G24520": "TTG1",
    "AT5G48100": "TT10",
    "AT1G61720": "BAN",
    "AT1G17260": "AHA10",
    "AT3G47620": "TCP14",
    "AT1G69690": "TCP15",
}

# ---- map every gene to its best-overlap block ----
blocks_by_chr = {c: blocks[blocks["chrom"] == c].reset_index(drop=True) for c in range(1, 6)}
block_ids = []
for _, row in genes.iterrows():
    sub = blocks_by_chr[int(row["chr"])]
    starts = sub["start"].to_numpy()
    ends = sub["end"].to_numpy()
    gs, ge = int(row["start"]), int(row["end"])
    j = int(np.searchsorted(starts, gs, side="right")) - 1
    cand = [k for k in range(max(0, j - 1), min(len(sub), j + 3))]
    ov = [max(0, min(ends[k], ge) - max(starts[k], gs)) for k in cand]
    if max(ov) > 0:
        best = cand[int(np.argmax(ov))]
    else:
        dist = [0 if (starts[k] <= ge and ends[k] >= gs) else min(abs(starts[k] - ge), abs(ends[k] - gs)) for k in cand]
        best = cand[int(np.argmin(dist))]
    block_ids.append(int(sub.loc[best, "block_id"]))
genes["block_id"] = block_ids
genes[["gene", "chr", "start", "end", "block_id"]].to_csv(
    TABLES / "gene_block_map.csv", index=False
)

# percentiles from the primary scan (1 = most significant)
scan_idx = scan.set_index("block_id")
for kind in ["raw", "graph", "embedding"]:
    rank_map = scan.set_index("block_id")[f"p_{kind}"].rank(method="min")
    genes[f"pct_{kind}"] = 100.0 * genes["block_id"].map(rank_map) / n_blocks

known = genes[genes["gene"].isin(KNOWN)].copy()
known["short"] = known["gene"].map(KNOWN)
known = known[["gene", "short", "chr", "start", "end", "block_id", "pct_raw", "pct_graph", "pct_embedding"]]
known = known.merge(
    scan[["block_id", "p_raw", "p_graph", "p_embedding", "nearest_gene"]], on="block_id", how="left"
)
known.sort_values("pct_raw").to_csv(TABLES / "known_dormancy_gene_ranks.csv", index=False)

# ---- enrichment tests ----
summary = {"n_known": int(len(known)), "n_blocks": int(n_blocks)}
for kind in ["raw", "graph", "embedding"]:
    pct = known[f"pct_{kind}"].to_numpy()
    all_pct = genes[f"pct_{kind}"].to_numpy()
    obs_mean = float(pct.mean())
    null_mean = np.array(
        [RNG.choice(all_pct, size=len(pct), replace=False).mean() for _ in range(20000)]
    )
    p_mean = float((1 + np.sum(null_mean <= obs_mean)) / 20001)
    row = {
        "mean_percentile": obs_mean,
        "median_percentile": float(np.median(pct)),
        "min_percentile": float(pct.min()),
        "p_mean_permutation": p_mean,
    }
    for frac in [0.01, 0.05, 0.10]:
        top_blocks = set(scan.nsmallest(int(round(frac * n_blocks)), f"p_{kind}")["block_id"])
        obs = int(known["block_id"].isin(top_blocks).sum())
        null = np.array(
            [
                int(known["block_id"].isin(RNG.choice(scan["block_id"], size=len(top_blocks), replace=False)).sum())
                for _ in range(20000)
            ]
        )
        row[f"top{int(frac*100)}_count"] = obs
        row[f"top{int(frac*100)}_expected"] = float(len(known) * frac)
        row[f"top{int(frac*100)}_p"] = float((1 + np.sum(null >= obs)) / 20001)
    summary[kind] = row

summary["dog1"] = known[known["gene"] == "AT5G45830"][
    ["block_id", "pct_raw", "pct_graph", "pct_embedding", "p_raw", "p_graph", "p_embedding"]
].to_dict(orient="records")[0]
(TABLES / "known_gene_enrichment.json").write_text(json.dumps(summary, indent=2))

# ---- figure ----
fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), dpi=170, gridspec_kw={"width_ratios": [1.35, 1]})
ax = axes[0]
top = known.sort_values("pct_raw").head(15).iloc[::-1]
ypos = np.arange(len(top))
for off, kind, color, label in [
    (0.18, "raw", "#4c4c4c", "raw block"),
    (0.0, "graph", "#3b6ea8", "graph-smoothed"),
    (-0.18, "embedding", "#b4483f", "SSL embedding"),
]:
    ax.scatter(top[f"pct_{kind}"], ypos + off, s=26, color=color, label=label, zorder=3, linewidths=0)
ax.axvline(5, color="#c44e52", linestyle="--", linewidth=0.9)
ax.set_xscale("log")
ax.set_xlim(0.3, 100)
ax.set_yticks(ypos)
ax.set_yticklabels(top["short"], fontsize=8)
ax.set_xlabel("Block percentile (log scale, lower = stronger)")
ax.legend(frameon=False, fontsize=7, loc="lower left")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", color="#e6e6e6", linewidth=0.5, zorder=0)

ax = axes[1]
all_pct = genes["pct_raw"].to_numpy()
obs_mean = float(known["pct_raw"].mean())
null_mean = np.array([RNG.choice(all_pct, size=len(known), replace=False).mean() for _ in range(20000)])
ax.hist(null_mean, bins=40, color="#bdbdbd", edgecolor="white", linewidth=0.3)
ax.axvline(obs_mean, color="#c44e52", linewidth=1.5)
ax.set_xlabel("Mean percentile of 58 random genes")
ax.set_ylabel("Count")
ax.set_title(f"Observed {obs_mean:.1f} (P={summary['raw']['p_mean_permutation']:.3f})", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIGURES / "fig7_known_gene_enrichment.png")
plt.close(fig)

print(json.dumps(summary, indent=2))
print(known.sort_values("pct_raw").head(15).to_string(index=False))
