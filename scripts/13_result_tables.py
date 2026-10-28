"""Build the numbered result tables."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"


def table1() -> None:
    pheno = pd.read_csv(PROCESSED / "dormancy_phenotype.csv")
    y = pheno["dsds50"]
    qc = json.loads((TABLES / "qc_summary.json").read_text())
    block = json.loads((PROCESSED / "block_summary.json").read_text())
    ssl = json.loads((PROCESSED / "ssl_summary.json").read_text())
    rows = [
        ("Accessions with genotypes", "1,307"),
        ("Accessions with DSDS50 phenotype", f"{len(y)}"),
        ("SNP markers", f"{block['snps']:,}"),
        ("SNPs with MAF > 0.05 in 83 accessions", f"{qc['common_snps_maf05']:,}"),
        ("LD-pruned SNPs for PCA/kinship", f"{qc['ld_pruned_snps']:,}"),
        ("PC1-PC5 variance explained", " / ".join(f"{v*100:.2f}%" for v in qc["pc_variance_explained_top5"])),
        ("Haplotype blocks", f"{block['n_blocks']:,}"),
        ("Median / 90th percentile block span", f"{block['span_bp_median']/1000:.1f} / {block['span_bp_p90']/1000:.1f} kb"),
        ("Graph edges (adjacency / LD)", f"{block['edges']:,} ({block['adjacency_edges']:,} / {block['ld_edges']:,})"),
        ("Genes annotated (TAIR10/Araport11)", "27,445"),
        ("DSDS50 mean ± SD (range)", f"{y.mean():.1f} ± {y.std():.1f} ({y.min():.0f}-{y.max():.0f}) days"),
        ("DSDS50 median (IQR)", f"{y.median():.1f} ({y.quantile(.25):.1f}-{y.quantile(.75):.1f}) days"),
        ("Pretraining accessions (train / validation)", f"{ssl['train_accessions']:,} / {ssl['val_accessions']:,}"),
        ("Graph encoder embedding dimension", f"{ssl['d']} (2 layers, 4 heads)"),
    ]
    pd.DataFrame(rows, columns=["Item", "Value"]).to_csv(
        TABLES / "table1_data_summary.csv", index=False
    )


def table2() -> None:
    metrics = json.loads((TABLES / "prediction_metrics.json").read_text())
    ext = json.loads((TABLES / "external_validation.json").read_text())["models"]
    control = json.loads((TABLES / "ssl_control.json").read_text())
    rows = []
    labels = {
        "SNP-GBLUP": "SNP-GBLUP",
        "block-kernel": "Haplotype block kernel",
        "graph-kernel": "Graph kernel (SSL)",
        "dual-SNP-block": "Dual SNP + block kernel",
        "triple-kernel": "Triple kernel",
        "graph-embedding-ridge": "Sample embedding + ridge",
    }
    for key, label in labels.items():
        m = metrics["models"][key]
        e = ext.get(key)
        rows.append(
            {
                "Model": label,
                "CV Pearson r": f"{m['pearson_mean']:.3f} ± {m['pearson_sd']:.3f}",
                "CV Spearman": f"{m['spearman_mean']:.3f}",
                "CV MAE (days)": f"{m['mae_mean']:.1f}",
                "External Pearson": "—" if e is None else f"{e['pearson']:.3f}",
                "External Spearman": "—" if e is None else f"{e['spearman']:.3f}",
            }
        )
    rows.append(
        {
            "Model": "Graph kernel (random init)",
            "CV Pearson r": f"{control['pearson_mean']:.3f} ± {control['pearson_sd']:.3f}",
            "CV Spearman": f"{control['spearman_mean']:.3f}",
            "CV MAE (days)": f"{control['mae_mean']:.1f}",
            "External Pearson": "—",
            "External Spearman": "—",
        }
    )
    pd.DataFrame(rows).to_csv(TABLES / "table2_prediction.csv", index=False)


def table3() -> None:
    known = pd.read_csv(TABLES / "known_dormancy_gene_ranks.csv")
    out = known.sort_values("pct_raw").head(15)[
        [
            "short", "gene", "block_id", "pct_raw", "pct_graph", "pct_embedding",
            "p_raw", "p_graph", "p_embedding",
        ]
    ].copy()
    out.columns = [
        "Gene", "Locus", "Block", "Percentile raw", "Percentile graph",
        "Percentile embedding", "P raw", "P graph", "P embedding",
    ]
    for col in ["Percentile raw", "Percentile graph", "Percentile embedding"]:
        out[col] = out[col].map(lambda v: f"{v:.2f}")
    for col in ["P raw", "P graph", "P embedding"]:
        out[col] = out[col].map(lambda v: f"{v:.3g}")
    out.to_csv(TABLES / "table3_known_genes.csv", index=False)


def table4() -> None:
    cand = pd.read_csv(TABLES / "candidate_regions.csv")
    scan = pd.read_csv(TABLES / "block_association_pc5.csv").set_index("block_id")
    cand = cand.merge(
        scan[["q_raw", "q_graph", "q_embedding"]], left_on="block_id", right_index=True
    )
    blocks = [1995, 1626, 7053, 5941, 4866, 1987, 3454, 3681, 3839, 6363, 7960, 2106]
    cand = cand.set_index("block_id").loc[blocks].reset_index()
    cand["Region (Mb)"] = cand.apply(
        lambda r: f"{int(r.chrom)}:{r.start/1e6:.3f}-{r.end/1e6:.3f}", axis=1
    )
    out = cand[
        [
            "block_id", "Region (Mb)", "p_raw", "q_raw", "p_graph", "q_graph",
            "top_snp", "top_snp_p", "top_snp_maf", "n_major_minor",
            "median_major_minor", "top_snp_gene_name", "genes_in_block",
        ]
    ].copy()
    out.columns = [
        "Block", "Region (Mb)", "P raw", "q raw", "P graph", "q graph",
        "Top SNP", "Top SNP P", "Top SNP MAF", "Major/minor",
        "Median major/minor (d)", "Top SNP gene", "Genes in block",
    ]
    out["P raw"] = out["P raw"].map(lambda v: f"{v:.2e}")
    out["P graph"] = out["P graph"].map(lambda v: f"{v:.2e}")
    out["q raw"] = out["q raw"].map(lambda v: f"{v:.3f}")
    out["q graph"] = out["q graph"].map(lambda v: f"{v:.3f}")
    out["Top SNP P"] = out["Top SNP P"].map(lambda v: f"{v:.2e}")
    out["Top SNP MAF"] = out["Top SNP MAF"].map(lambda v: f"{v:.3f}")
    out.to_csv(TABLES / "table4_candidates.csv", index=False)


def table_s1() -> None:
    sens = json.loads((TABLES / "block_scan_sensitivity.json").read_text())
    rows = []
    for config, models in sens.items():
        for kind, v in models.items():
            rows.append(
                {
                    "Configuration": config,
                    "Kernel": kind,
                    "Min P": v["min_p"],
                    "Top block": v["top_block"],
                    "Top gene": v["top_gene"],
                    "DOG1 percentile": v["dog1_percentile"],
                    "DOG1 P": v["dog1_p"],
                }
            )
    pd.DataFrame(rows).to_csv(TABLES / "tableS1_sensitivity.csv", index=False)


def table_s2() -> None:
    grouped = json.loads((TABLES / "grouped_cv.json").read_text())
    rows = []
    for k, models in grouped["models"].items():
        for name, v in models.items():
            rows.append(
                {
                    "Clusters": k,
                    "Model": name,
                    "Pearson": v["pearson"],
                    "Spearman": v["spearman"],
                    "MAE": v["mae"],
                }
            )
    pd.DataFrame(rows).to_csv(TABLES / "tableS2_grouped_cv.csv", index=False)


def main() -> None:
    table1()
    table2()
    table3()
    table4()
    table_s1()
    table_s2()
    print("tables written")


if __name__ == "__main__":
    main()
