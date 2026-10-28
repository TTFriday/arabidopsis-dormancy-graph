"""Assemble the main figures from saved result tables.

The individual analysis scripts also write diagnostic figures. This script
only redraws the six main figures, so that panel layout, labels
and colour schemes are consistent.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures" / "main"
FIGURES.mkdir(parents=True, exist_ok=True)

BLUE = "#3b6ea8"
RED = "#b4483f"
GREY = "#4c4c4c"
GREEN = "#4f8a5b"
ORANGE = "#c47f3a"


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)


def workflow_figure():
    fig, ax = plt.subplots(figsize=(11.5, 4.6), dpi=200)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    boxes = [
        (0.015, 0.18, 0.20, 0.60),
        (0.265, 0.18, 0.20, 0.60),
        (0.515, 0.18, 0.20, 0.60),
        (0.765, 0.18, 0.22, 0.60),
    ]
    for x, y, w, h in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h,
                boxstyle="round,pad=0.012,rounding_size=0.018",
                facecolor="#f5f7fa", edgecolor="#9aa7b4", linewidth=0.9,
            )
        )

    ax.text(0.115, 0.865, "Data", ha="center", fontsize=11, weight="bold")
    ax.text(0.365, 0.865, "Haplotype block graph", ha="center", fontsize=11, weight="bold")
    ax.text(0.615, 0.865, "Masked pretraining", ha="center", fontsize=11, weight="bold")
    ax.text(0.875, 0.865, "Dual-kernel models", ha="center", fontsize=11, weight="bold")

    ax.text(0.115, 0.745, "1,307 accessions", ha="center", fontsize=8.5)
    ax.text(0.115, 0.695, "214,051 SNPs", ha="center", fontsize=8.5)
    ax.text(0.115, 0.60, "83 accessions", ha="center", fontsize=8.5)
    ax.text(0.115, 0.55, "DSDS50 phenotype", ha="center", fontsize=8.5)
    ax.text(0.115, 0.44, "TAIR10 / Araport11", ha="center", fontsize=8)
    ax.text(0.115, 0.39, "27,445 genes", ha="center", fontsize=8)
    ax.plot([0.055, 0.175], [0.49, 0.49], color="#c9d1d9", linewidth=0.8)

    # small genotype matrix
    rng = np.random.default_rng(7)
    for i in range(6):
        for j in range(6):
            ax.add_patch(
                Rectangle(
                    (0.075 + j * 0.012, 0.235 + i * 0.018),
                    0.010, 0.015,
                    facecolor=BLUE if rng.random() > 0.45 else "#dce6f1",
                    edgecolor="white", linewidth=0.2,
                )
            )

    ax.text(0.365, 0.745, "8,589 blocks", ha="center", fontsize=8.5)
    ax.text(0.365, 0.70, "25 SNPs per block", ha="center", fontsize=8.5)
    ax.text(0.365, 0.60, "node: mean dosage + 3 PCs", ha="center", fontsize=8)
    ax.text(0.365, 0.55, "edges: adjacent + LD (|r| > 0.5)", ha="center", fontsize=8)

    # graph sketch
    xs = np.array([0.295, 0.335, 0.375, 0.415, 0.445, 0.315, 0.355, 0.395, 0.430])
    ys = np.array([0.30, 0.34, 0.31, 0.35, 0.30, 0.40, 0.43, 0.40, 0.43])
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            if np.hypot(xs[i] - xs[j], ys[i] - ys[j]) < 0.055:
                ax.plot([xs[i], xs[j]], [ys[i], ys[j]], color="#b8c4d0", linewidth=0.6, zorder=1)
    ax.scatter(xs, ys, s=24, color=BLUE, edgecolor="white", linewidth=0.4, zorder=2)

    ax.text(0.615, 0.745, "15% blocks masked", ha="center", fontsize=8.5)
    ax.text(0.615, 0.70, "reconstruct 4 features", ha="center", fontsize=8.5)
    ax.text(0.615, 0.60, "2-layer graph Transformer", ha="center", fontsize=8)
    ax.text(0.615, 0.55, "d = 16, 4 heads", ha="center", fontsize=8)
    ax.text(0.615, 0.44, "1,176 train / 131 val", ha="center", fontsize=8)
    ax.text(0.615, 0.39, "no phenotype used", ha="center", fontsize=8)

    # mask illustration
    for k in range(5):
        x = 0.555 + k * 0.026
        color = "#dce6f1" if k == 2 else BLUE
        ax.add_patch(Rectangle((x, 0.25), 0.018, 0.05, facecolor=color, edgecolor="white", linewidth=0.4))
    ax.text(0.610, 0.225, "masked", ha="center", fontsize=7, color=RED)

    ax.text(0.875, 0.745, r"$K = 0.5K_{\mathrm{SNP}} + 0.5K_{\mathrm{block}}$", ha="center", fontsize=9.5)
    ax.text(0.875, 0.66, "GBLUP prediction", ha="center", fontsize=8.5)
    ax.text(0.875, 0.61, "nested 5-fold CV × 30", ha="center", fontsize=8.5)
    ax.text(0.875, 0.52, "block-level score test", ha="center", fontsize=8.5)
    ax.text(0.875, 0.47, "rotation null + BH-FDR", ha="center", fontsize=8.5)
    ax.text(0.875, 0.38, "known-gene enrichment", ha="center", fontsize=8.5)
    ax.text(0.875, 0.33, "candidate region ranking", ha="center", fontsize=8.5)

    for x0, x1 in [(0.217, 0.263), (0.467, 0.513), (0.717, 0.763)]:
        ax.add_patch(
            FancyArrowPatch(
                (x0, 0.48), (x1, 0.48),
                arrowstyle="-|>", mutation_scale=11,
                linewidth=1.1, color="#5b6b7b",
            )
        )

    fig.tight_layout()
    fig.savefig(FIGURES / "fig1_workflow.png")
    plt.close(fig)


def qc_figure():
    pheno = pd.read_csv(PROCESSED / "dormancy_phenotype.csv")
    y = pheno["dsds50"].to_numpy(float)
    pcs = pd.read_csv(TABLES / "accession_pcs.csv")
    ld = pd.read_csv(TABLES / "ld_decay_pairs.csv")
    blocks = pd.read_csv(PROCESSED / "haplotype_blocks.csv")

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.2), dpi=200)
    ax = axes[0, 0]
    ax.hist(y, bins=20, color=BLUE, edgecolor="white", linewidth=0.6)
    ax.set_xlabel("DSDS50 (days)")
    ax.set_ylabel("Accessions")
    ax.set_title("(a) Phenotype distribution (n = 83)", loc="left", fontsize=10)
    style(ax)

    ax = axes[0, 1]
    sc = ax.scatter(
        pcs["PC1"], pcs["PC2"], c=y, cmap="viridis", s=30,
        edgecolor="white", linewidth=0.4,
    )
    ax.set_xlabel("PC1 (4.1%)")
    ax.set_ylabel("PC2 (3.6%)")
    ax.set_title("(b) Population structure", loc="left", fontsize=10)
    cb = fig.colorbar(sc, ax=ax, shrink=0.82, pad=0.02)
    cb.set_label("DSDS50 (days)", fontsize=8)
    style(ax)

    ax = axes[1, 0]
    bins = np.arange(0, 500_001, 10_000)
    ld["bin"] = pd.cut(ld["distance_bp"], bins)
    ld_mean = ld.groupby("bin", observed=True)["r2"].mean()
    centers = np.array([b.mid for b in ld_mean.index]) / 1000
    ax.plot(centers, ld_mean.values, marker="o", ms=3, color=BLUE, linewidth=1.1)
    ax.set_xlabel("Distance (kb)")
    ax.set_ylabel("Mean $r^2$")
    ax.set_title("(c) LD decay (9,245 pruned SNPs)", loc="left", fontsize=10)
    style(ax)

    ax = axes[1, 1]
    span_kb = (blocks["end"] - blocks["start"]).to_numpy() / 1000
    ax.hist(span_kb, bins=np.linspace(0, 50, 26), color=GREEN, edgecolor="white", linewidth=0.5)
    med = np.median(span_kb)
    p90 = np.quantile(span_kb, 0.9)
    ax.axvline(med, color=RED, linestyle="--", linewidth=1.0)
    ax.text(med + 1.0, ax.get_ylim()[1] * 0.88, f"median {med:.1f} kb", color=RED, fontsize=8)
    ax.text(22, ax.get_ylim()[1] * 0.78, f"8,589 blocks\np90 = {p90:.1f} kb", fontsize=8)
    ax.set_xlabel("Block span (kb)")
    ax.set_ylabel("Number of blocks")
    ax.set_title("(d) Haplotype block span", loc="left", fontsize=10)
    style(ax)

    fig.tight_layout()
    fig.savefig(FIGURES / "fig2_data_qc.png")
    plt.close(fig)


def prediction_figure():
    y = pd.read_csv(PROCESSED / "dormancy_phenotype.csv")["dsds50"].to_numpy(float)
    preds = np.load(TABLES / "prediction_cv_predictions.npz")
    metrics = json.loads((TABLES / "prediction_metrics.json").read_text())
    control = json.loads((TABLES / "ssl_control.json").read_text())
    ext = np.load(TABLES / "external_validation_predictions.npz")
    ext_metrics = json.loads((TABLES / "external_validation.json").read_text())["models"]

    fig = plt.figure(figsize=(12.2, 4.0), dpi=200)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.35, 1.0], wspace=0.32)

    ax = fig.add_subplot(gs[0, 0])
    pred = preds["dual_SNP_block"].mean(axis=0)
    ax.scatter(y, pred, s=22, alpha=0.78, color=BLUE, edgecolor="white", linewidth=0.4)
    lo, hi = min(y.min(), pred.min()) - 5, max(y.max(), pred.max()) + 5
    ax.plot([lo, hi], [lo, hi], "--", color="#888888", linewidth=0.9)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Observed DSDS50 (days)")
    ax.set_ylabel("Predicted DSDS50 (days)")
    ax.set_title(
        f"(a) Dual kernel, r = {metrics['models']['dual-SNP-block']['pearson_mean']:.3f}",
        loc="left", fontsize=10,
    )
    style(ax)

    ax = fig.add_subplot(gs[0, 1])
    order = [
        ("SNP-GBLUP", "SNP-GBLUP"),
        ("block-kernel", "Block kernel"),
        ("graph-kernel", "Graph kernel (SSL)"),
        ("dual-SNP-block", "Dual SNP + block"),
        ("triple-kernel", "Triple kernel"),
        ("graph-embedding-ridge", "Embedding ridge"),
    ]
    means = [metrics["models"][k]["pearson_mean"] for k, _ in order]
    sds = [metrics["models"][k]["pearson_sd"] for k, _ in order]
    labels = [lab for _, lab in order]
    means.append(control["pearson_mean"])
    sds.append(control["pearson_sd"])
    labels.append("Graph kernel (random init)")
    colors = ["#9aa7b4", "#9aa7b4", "#9aa7b4", BLUE, "#6f94b8", "#c9a0a0", "#d9d9d9"]
    ypos = np.arange(len(means))
    ax.barh(ypos, means, xerr=sds, color=colors, edgecolor="none", height=0.66, error_kw={"lw": 0.7})
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.8)
    ax.set_xlabel("Pearson r (5-fold CV)")
    ax.set_title("(b) Model comparison", loc="left", fontsize=10)
    ax.axvline(metrics["models"]["SNP-GBLUP"]["pearson_mean"], color="#777777", linestyle=":", linewidth=0.8)
    style(ax)

    ax = fig.add_subplot(gs[0, 2])
    obs = ext["observed"]
    pred_ext = ext["dual_SNP_block"]
    obs_z = (obs - obs.mean()) / obs.std()
    pred_z = (pred_ext - pred_ext.mean()) / pred_ext.std()
    ax.scatter(obs_z, pred_z, s=25, color=GREEN, alpha=0.82, edgecolor="white", linewidth=0.4)
    lo, hi = -2.2, 3.2
    ax.plot([lo, hi], [lo, hi], "--", color="#888888", linewidth=0.9)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Observed 279_DSDS50 (z)")
    ax.set_ylabel("Predicted DSDS50 (z)")
    ax.set_title(
        f" (c) External validation (n = 29), $\\rho$ = {ext_metrics['dual-SNP-block']['spearman']:.3f}",
        loc="left", fontsize=9,
    )
    style(ax)

    fig.savefig(FIGURES / "fig3_prediction.png", bbox_inches="tight")
    plt.close(fig)


def association_figure():
    res = pd.read_csv(TABLES / "block_association_pc5.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), dpi=200)
    ax = axes[0]
    colors = {"raw": GREY, "graph": BLUE, "embedding": RED}
    labels = {"raw": "Raw block", "graph": "Graph-smoothed", "embedding": "SSL embedding"}
    for kind in ["raw", "graph", "embedding"]:
        pv = np.sort(res[f"p_{kind}"].dropna().to_numpy())
        nn = len(pv)
        k = np.arange(1, nn + 1)
        exp = -np.log10((k - 0.5) / nn)
        lo = -np.log10(stats.beta.ppf(0.975, k, nn - k + 1))
        hi = -np.log10(stats.beta.ppf(0.025, k, nn - k + 1))
        ax.fill_between(exp, lo, hi, color="#d9d9d9", alpha=0.5, linewidth=0)
        ax.plot(exp, -np.log10(pv), color=colors[kind], linewidth=1.1, label=labels[kind])
    ax.plot([0, 6], [0, 6], "--", color="#888888", linewidth=0.8)
    ax.set_xlim(0, 5.2)
    ax.set_ylim(0, 6.5)
    ax.set_xlabel("Expected $-\\log_{10}P$")
    ax.set_ylabel("Observed $-\\log_{10}P$")
    ax.set_title("(a) Block-level QQ", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    style(ax)

    ax = axes[1]
    offset = 0.0
    ticks, tick_labels = [], []
    chrom_colors = {1: "#4c72b0", 2: "#dd8452", 3: "#55a868", 4: "#c44e52", 5: "#8172b3"}
    offsets = {}
    for c in range(1, 6):
        sub = res[res["chrom"] == c]
        span = sub["end"].max() / 1e6
        offsets[c] = offset
        x = offset + sub["start"] / 1e6
        ax.scatter(x, -np.log10(sub["p_raw"]), s=5, color=chrom_colors[c], alpha=0.65, linewidths=0)
        ticks.append(offset + span / 2)
        tick_labels.append(str(c))
        offset += span + 4
    ax.axhline(-np.log10(0.01), color="#999999", linestyle=":", linewidth=0.8)
    top = res.nsmallest(1, "p_raw").iloc[0]
    xtop = offsets[int(top["chrom"])] + top["start"] / 1e6
    ax.annotate(
        f"block {int(top['block_id'])}\n({top['nearest_gene']})",
        xy=(xtop, -np.log10(top["p_raw"])),
        xytext=(4, 8), textcoords="offset points", fontsize=7,
        arrowprops=dict(arrowstyle="-", color="#666666", linewidth=0.6),
    )
    dog = res[(res["chrom"] == 5) & (res["start"] <= 18_591_506) & (res["end"] >= 18_589_418)]
    xdog = offsets[5] + dog["start"].iloc[0] / 1e6
    ax.annotate(
        "DOG1",
        xy=(xdog, -np.log10(dog["p_raw"].iloc[0])),
        xytext=(-24, 8), textcoords="offset points", fontsize=7, color="#8172b3",
        arrowprops=dict(arrowstyle="-", color="#8172b3", linewidth=0.6),
    )
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("Chromosome")
    ax.set_ylabel("$-\\log_{10}P$, raw block kernel")
    ax.set_title("(b) Block-level association", loc="left", fontsize=10, pad=10)
    ax.set_ylim(-0.3, 7.0)
    style(ax)

    fig.tight_layout()
    fig.savefig(FIGURES / "fig4_block_association.png")
    plt.close(fig)


def enrichment_figure():
    known = pd.read_csv(TABLES / "known_dormancy_gene_ranks.csv")
    summary = json.loads((TABLES / "known_gene_enrichment.json").read_text())
    rng = np.random.default_rng(20261027)
    gene_map = pd.read_csv(TABLES / "gene_block_map.csv")

    fig, axes = plt.subplots(
        1, 2, figsize=(10.2, 4.3), dpi=200,
        gridspec_kw={"width_ratios": [1.35, 1]},
    )
    ax = axes[0]
    top = known.sort_values("pct_raw").head(15).iloc[::-1]
    ypos = np.arange(len(top))
    for off, kind, color, label in [
        (0.18, "raw", GREY, "Raw block"),
        (0.0, "graph", BLUE, "Graph-smoothed"),
        (-0.18, "embedding", RED, "SSL embedding"),
    ]:
        ax.scatter(top[f"pct_{kind}"], ypos + off, s=28, color=color, label=label, zorder=3, linewidths=0)
    ax.axvline(5, color=RED, linestyle="--", linewidth=0.9)
    ax.text(5.6, 2.0, "top 5%", color=RED, fontsize=6.5, rotation=90, va="bottom")
    ax.set_xscale("log")
    ax.set_xlim(0.3, 100)
    ax.set_yticks(ypos)
    ax.set_yticklabels(top["short"], fontsize=8)
    ax.set_xlabel("Block percentile (log scale, lower = stronger)")
    ax.set_title("(a) Known dormancy genes", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    ax.grid(axis="x", color="#e6e6e6", linewidth=0.5, zorder=0)
    style(ax)

    # null distribution for the graph-smoothed ranking
    scan = pd.read_csv(TABLES / "block_association_pc5.csv")
    rank_map = scan.set_index("block_id")["p_graph"].rank(method="min")
    gene_map["pct_graph"] = 100.0 * gene_map["block_id"].map(rank_map) / len(scan)
    all_pct = gene_map["pct_graph"].dropna().to_numpy()
    obs = float(known["pct_graph"].mean())
    null = np.array([rng.choice(all_pct, size=len(known), replace=False).mean() for _ in range(20000)])
    ax = axes[1]
    ax.hist(null, bins=40, color="#bdbdbd", edgecolor="white", linewidth=0.3)
    ax.axvline(obs, color=BLUE, linewidth=1.6)
    ax.set_xlabel("Mean percentile of 58 random genes")
    ax.set_ylabel("Count")
    ax.set_title(
        f"(b) Graph-smoothed null: {obs:.1f} (P = {summary['graph']['p_mean_permutation']:.3f})",
        loc="left", fontsize=9,
    )
    style(ax)

    fig.tight_layout()
    fig.savefig(FIGURES / "fig5_known_gene_enrichment.png")
    plt.close(fig)


def candidate_figure():
    blocks = pd.read_csv(PROCESSED / "haplotype_blocks.csv")
    scan = pd.read_csv(TABLES / "block_association_pc5.csv")
    genes = pd.read_csv(ROOT / "data" / "annotations" / "genes_tair10.csv")
    snp_p = np.load(TABLES / "snp_scan_pc5_pvalues.npz")
    snp = pd.DataFrame(
        {"chrom": snp_p["chrom"], "pos": snp_p["pos"], "p": snp_p["p"]}
    )

    regions = [
        (1, 28_850_000, 29_150_000, 1995, "AT1G77350"),
        (3, 6_300_000, 6_500_000, 3681, "AT3G18580"),
        (5, 18_540_000, 18_680_000, 7960, "DOG1"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6), dpi=200)
    for ax, (chrom, lo, hi, bid, label) in zip(axes, regions):
        sp = snp[(snp["chrom"] == chrom) & (snp["pos"] >= lo) & (snp["pos"] <= hi)]
        ax.scatter(
            sp["pos"] / 1e6, -np.log10(sp["p"]), s=9, color="#b9c2cc",
            alpha=0.75, linewidths=0, label="SNP",
        )
        bl = scan[(scan["chrom"] == chrom) & (scan["start"] >= lo) & (scan["start"] <= hi)]
        ax.scatter(
            bl["start"] / 1e6, -np.log10(bl["p_raw"]), s=30, color=BLUE,
            alpha=0.9, edgecolor="white", linewidth=0.4, label="Block", zorder=3,
        )
        row = scan[scan["block_id"] == bid].iloc[0]
        ax.scatter(
            row["start"] / 1e6, -np.log10(row["p_raw"]), s=85,
            facecolor="none", edgecolor=RED, linewidth=1.3, zorder=4,
        )
        ax.annotate(
            label,
            xy=(row["start"] / 1e6, -np.log10(row["p_raw"])),
            xytext=(6, 10), textcoords="offset points", fontsize=8,
            arrowprops=dict(arrowstyle="-", color=RED, linewidth=0.6),
        )
        # gene models are omitted here; candidate blocks are listed in the candidate table
        ax.axhline(-np.log10(0.01), color="#999999", linestyle=":", linewidth=0.7)
        ax.set_xlim(lo / 1e6, hi / 1e6)
        ax.set_ylim(0, max(6.2, -np.log10(sp["p"].min()) + 0.8))
        ax.set_xlabel(f"Chromosome {chrom} (Mb)")
        ax.set_ylabel("$-\\log_{10}P$")
        style(ax)
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig6_candidate_regions.png")
    plt.close(fig)


def main():
    workflow_figure()
    qc_figure()
    prediction_figure()
    association_figure()
    enrichment_figure()
    candidate_figure()
    print(f"figures written to {FIGURES}")


if __name__ == "__main__":
    main()
