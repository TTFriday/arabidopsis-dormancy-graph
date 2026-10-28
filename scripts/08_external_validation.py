"""Exploratory external validation on 29 accessions with 279_DSDS50.

The 29 accessions have no 8_Seed Dormancy value and are not used for model
fitting. Predictions are trained on the 83 accessions and evaluated by rank
correlation because the two dormancy scores come from different experiments.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
RNG = np.random.default_rng(20261024)

LAMBDAS = np.array([1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0])


def choose_lambda(K: np.ndarray, y: np.ndarray, seed: int = 0) -> float:
    best = (np.inf, LAMBDAS[-1])
    for lam in LAMBDAS:
        pred = np.zeros(len(y))
        for tr, te in KFold(5, shuffle=True, random_state=seed).split(K):
            alpha = np.linalg.solve(
                K[np.ix_(tr, tr)] + lam * np.eye(len(tr)),
                y[tr] - y[tr].mean(),
            )
            pred[te] = K[np.ix_(te, tr)] @ alpha + y[tr].mean()
        mse = float(np.mean((pred - y) ** 2))
        if mse < best[0]:
            best = (mse, float(lam))
    return best[1]


def fit_predict(Ktr: np.ndarray, Kte: np.ndarray, y: np.ndarray, seed: int = 0):
    lam = choose_lambda(Ktr, y, seed)
    alpha = np.linalg.solve(Ktr + lam * np.eye(len(y)), y - y.mean())
    return Kte @ alpha + y.mean(), lam


def kernel_metrics(pred: np.ndarray, obs: np.ndarray) -> dict:
    return {
        "pearson": float(stats.pearsonr(pred, obs)[0]),
        "spearman": float(stats.spearmanr(pred, obs).statistic),
        "mae": float(np.mean(np.abs(pred - obs))),
        "n": int(len(obs)),
    }


def bootstrap_spearman(pred: np.ndarray, obs: np.ndarray, n_boot: int = 2000) -> list[float]:
    vals = []
    for _ in range(n_boot):
        idx = RNG.integers(0, len(obs), len(obs))
        if np.std(pred[idx]) == 0 or np.std(obs[idx]) == 0:
            continue
        vals.append(float(stats.spearmanr(pred[idx], obs[idx]).statistic))
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]


def main() -> None:
    pheno = pd.read_csv(PROCESSED / "dormancy_phenotype.csv")
    y = pheno["dsds50"].to_numpy(float)

    raw = pd.read_csv(
        ROOT / "data" / "raw" / "atpolydb" / "phenotype_published_raw.tsv",
        sep="\t",
        encoding="latin1",
    )
    ext = raw[(raw["279_DSDS50"].notna()) & (~raw["8_Seed Dormancy"].notna())].copy()
    y_ext = ext["279_DSDS50"].to_numpy(float)
    ext_ids = ext["ecotype id"].to_numpy(int)

    ext_codes_path = PROCESSED / "genotypes_external29.npy"
    train_subset = PROCESSED / "block_features_83.npy"
    test_subset = PROCESSED / "block_features_external29.npy"

    codes = np.load(PROCESSED / "genotypes_83.npy")
    snp = pd.read_csv(PROCESSED / "snp_metadata.csv")
    maf = snp["maf_83"].to_numpy()
    keep = np.flatnonzero((maf > 0.05) & (snp["minor_code"].to_numpy() < 4))
    minor = snp["minor_code"].to_numpy()[keep]
    Xtr = (codes[keep, :] == minor[:, None]).astype(np.float32).T
    if ext_codes_path.exists():
        Xte = (np.load(ext_codes_path)[keep, :] == minor[:, None]).astype(np.float32).T
    else:
        with open(
            ROOT / "data" / "raw" / "atpolydb" / "call_method_75" / "call_method_75_TAIR9.csv",
            "rb",
        ) as handle:
            handle.readline()
            header = handle.readline().decode().rstrip("\n").split(",")
        header_ids = np.array([int(x) for x in header[2:]])
        id_to_col = {v: i for i, v in enumerate(header_ids)}
        test_cols = np.array([id_to_col[i] for i in ext_ids], dtype=int)
        all_codes = np.load(PROCESSED / "genotypes_all_1307.npy", mmap_mode="r")
        Xte = (
            np.asarray(all_codes[keep, :])[:, test_cols] == minor[:, None]
        ).astype(np.float32).T
    Ktr_snp = Xtr @ Xtr.T / Xtr.shape[1]
    Kte_snp = Xte @ Xtr.T / Xtr.shape[1]

    if train_subset.exists() and test_subset.exists():
        Btr = np.asarray(
            np.load(train_subset, mmap_mode="r")[:, :, 0], dtype=np.float64
        ).T
        Bte = np.asarray(
            np.load(test_subset, mmap_mode="r")[:, :, 0], dtype=np.float64
        ).T
    else:
        train_cols = np.load(PROCESSED / "dormancy_cols_in_1307.npy")
        block_feats = np.load(PROCESSED / "block_features.npy", mmap_mode="r")[:, :, 0]
        Btr = np.asarray(block_feats[train_cols], dtype=np.float64).T
        Bte = np.asarray(block_feats[test_cols], dtype=np.float64).T
    Ktr_block = np.zeros((len(y), len(y)), dtype=np.float64)
    Kte_block = np.zeros((len(y_ext), len(y)), dtype=np.float64)
    used = 0
    for b in range(Btr.shape[0]):
        mu = Btr[b].mean()
        sd = Btr[b].std()
        if sd < 1e-8:
            continue
        ztr = (Btr[b] - mu) / sd
        zte = (Bte[b] - mu) / sd
        Ktr_block += np.outer(ztr, ztr)
        Kte_block += np.outer(zte, ztr)
        used += 1
    Ktr_block /= used
    Kte_block /= used

    models = {
        "SNP-GBLUP": (Ktr_snp, Kte_snp),
        "block-kernel": (Ktr_block, Kte_block),
        "dual-SNP-block": (
            0.5 * Ktr_snp + 0.5 * Ktr_block,
            0.5 * Kte_snp + 0.5 * Kte_block,
        ),
    }
    results = {}
    preds = {}
    for name, (Ktr, Kte) in models.items():
        pred, lam = fit_predict(Ktr, Kte, y)
        preds[name] = pred
        results[name] = kernel_metrics(pred, y_ext)
        results[name]["lambda"] = lam
        results[name]["spearman_boot_ci"] = bootstrap_spearman(pred, y_ext)
        print(f"[08] {name}: {results[name]}", flush=True)

    np.savez_compressed(
        TABLES / "external_validation_predictions.npz",
        **{k.replace("-", "_"): v for k, v in preds.items()},
        observed=y_ext,
        ecotype_id=ext_ids,
    )
    (TABLES / "external_validation.json").write_text(
        json.dumps(
            {
                "n_train": int(len(y)),
                "n_external": int(len(y_ext)),
                "external_trait": "279_DSDS50",
                "note": "Exploratory; phenotypes come from a different experiment.",
                "models": results,
            },
            indent=2,
        )
    )

    fig, ax = plt.subplots(figsize=(4.6, 3.8), dpi=160)
    pred = preds["dual-SNP-block"]
    ax.scatter(y_ext, pred, s=24, color="#3b6ea8", alpha=0.8, edgecolor="white", linewidth=0.4)
    ax.set_xlabel("279_DSDS50 (external, days)")
    ax.set_ylabel("Predicted dormancy score")
    rho = results["dual-SNP-block"]["spearman"]
    ax.set_title(f"External set (n=29, Spearman = {rho:.3f})", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig5_external_validation.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
