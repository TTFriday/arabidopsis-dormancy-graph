"""SNP-level PC1-5 linear scan used as a benchmark for the block scan.

The model is the same as in 09b: DSDS50 standardised, intercept plus PC1-5.
Residualising both phenotype and genotypes on the covariates makes the scan
fast and gives a direct SNP-level comparison for block prioritisation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"


def main() -> None:
    pheno = pd.read_csv(PROCESSED / "dormancy_phenotype.csv")
    y = pheno["dsds50"].to_numpy(float)
    y = StandardScaler().fit_transform(y[:, None]).ravel()

    pcs = pd.read_csv(TABLES / "accession_pcs.csv")
    X = np.column_stack(
        [np.ones(len(y)), pcs[[f"PC{i}" for i in range(1, 6)]].to_numpy(float)]
    )

    codes = np.load(PROCESSED / "genotypes_83.npy")
    snp = pd.read_csv(PROCESSED / "snp_metadata.csv")
    maf = snp["maf_83"].to_numpy()
    keep = np.flatnonzero((maf > 0.05) & (snp["minor_code"].to_numpy() < 4))
    minor = snp["minor_code"].to_numpy()[keep]
    D = (codes[keep, :] == minor[:, None]).astype(np.float64).T

    inv = np.linalg.inv(X.T @ X)
    H = X @ inv @ X.T
    Ry = y - H @ y
    RD = D - H @ D

    sxx = np.einsum("ij,ij->j", RD, RD)
    beta = (Ry @ RD) / sxx
    df = len(y) - X.shape[1]
    sigma2 = float(Ry @ Ry / df)
    tstat = beta / np.sqrt(sigma2 / sxx)
    pval = 2 * stats.t.sf(np.abs(tstat), df)
    lam = float(np.median(stats.chi2.isf(pval, 1)) / 0.4549)

    meta = snp.iloc[keep].reset_index(drop=True)
    scan = pd.DataFrame(
        {
            "chrom": meta["chrom"],
            "pos": meta["pos"],
            "maf": meta["maf_83"],
            "beta": beta,
            "t": tstat,
            "p": pval,
        }
    )
    scan["q"] = multipletests(scan["p"], method="fdr_bh")[1]
    scan = scan.sort_values("p").reset_index(drop=True)
    scan.head(1000).to_csv(TABLES / "snp_scan_pc5_top1000.csv", index=False)
    np.savez_compressed(
        TABLES / "snp_scan_pc5_pvalues.npz",
        p=pval.astype(np.float32),
        beta=beta.astype(np.float32),
        chrom=meta["chrom"].to_numpy(np.int8),
        pos=meta["pos"].to_numpy(np.int64),
    )

    summary = {
        "model": "OLS with intercept + PC1-5; standardised DSDS50",
        "n": int(len(y)),
        "n_snps": int(len(keep)),
        "df": int(df),
        "lambda_gc": lam,
        "min_p": float(scan["p"].min()),
        "n_p_1e-5": int((scan["p"] < 1e-5).sum()),
        "n_fdr_0.05": int((scan["q"] < 0.05).sum()),
        "top": scan.head(10)[["chrom", "pos", "maf", "beta", "p", "q"]].to_dict(
            orient="records"
        ),
    }
    (TABLES / "snp_scan_pc5_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
