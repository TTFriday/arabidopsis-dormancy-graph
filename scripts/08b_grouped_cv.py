"""Grouped cross-validation across genetic subpopulations.

Random 5-fold CV measures interpolation within the sampled panel. To probe
extrapolation, accessions are clustered on PC1-5 (k = 6, 8, 10) and entire
clusters are held out. The same kernel models and inner lambda selection are
used as in the main prediction analysis. Negative correlations here are an
expected consequence of predicting a held-out subpopulation from the others,
not a defect of the kernel itself.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold, KFold

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"
SEED = 20261024


def load_prediction_module():
    path = ROOT / "scripts" / "07_prediction_models.py"
    spec = importlib.util.spec_from_file_location("pred07", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select_lambda(K: np.ndarray, y: np.ndarray, lambdas: np.ndarray, seed: int = 0) -> float:
    best_mse, best_lam = np.inf, float(lambdas[-1])
    for lam in lambdas:
        pred = np.zeros(len(y))
        for tr, te in KFold(5, shuffle=True, random_state=seed).split(K):
            alpha = np.linalg.solve(
                K[np.ix_(tr, tr)] + lam * np.eye(len(tr)), y[tr] - y[tr].mean()
            )
            pred[te] = K[np.ix_(te, tr)] @ alpha + y[tr].mean()
        mse = float(np.mean((pred - y) ** 2))
        if mse < best_mse:
            best_mse, best_lam = mse, float(lam)
    return best_lam


def grouped_predictions(
    K: np.ndarray, y: np.ndarray, groups: np.ndarray, lambdas: np.ndarray
) -> dict:
    pred = np.zeros(len(y))
    chosen = []
    for tr, te in GroupKFold(n_splits=len(np.unique(groups))).split(K, y, groups):
        lam = select_lambda(K[np.ix_(tr, tr)], y[tr], lambdas)
        alpha = np.linalg.solve(
            K[np.ix_(tr, tr)] + lam * np.eye(len(tr)), y[tr] - y[tr].mean()
        )
        pred[te] = K[np.ix_(te, tr)] @ alpha + y[tr].mean()
        chosen.append(float(lam))
    return {
        "pearson": float(stats.pearsonr(pred, y)[0]),
        "spearman": float(stats.spearmanr(pred, y).statistic),
        "mae": float(np.mean(np.abs(pred - y))),
        "lambdas": chosen,
        "prediction": pred,
    }


def main() -> None:
    mod = load_prediction_module()
    y, K_snp, K_block, K_graph, _, _ = mod.load_inputs()
    K_dual = 0.5 * K_snp + 0.5 * K_block

    pcs = pd.read_csv(TABLES / "accession_pcs.csv")
    Z = pcs[[f"PC{i}" for i in range(1, 6)]].to_numpy(float)
    Z = (Z - Z.mean(axis=0)) / Z.std(axis=0)

    models = {"SNP-GBLUP": K_snp, "dual-SNP-block": K_dual}
    results = {}
    for k in (6, 8, 10):
        groups = KMeans(n_clusters=k, n_init=50, random_state=SEED).fit_predict(Z)
        results[f"k={k}"] = {}
        for name, K in models.items():
            res = grouped_predictions(K, y, groups, mod.LAMBDAS)
            res.pop("prediction")
            res["cluster_sizes"] = np.bincount(groups).tolist()
            results[f"k={k}"][name] = res
            print(
                f"[08b] k={k} {name}: r={res['pearson']:.3f}, "
                f"rho={res['spearman']:.3f}",
                flush=True,
            )

    (TABLES / "grouped_cv.json").write_text(
        json.dumps(
            {
                "note": (
                    "Clusters are defined on PC1-5 without phenotypes; whole "
                    "clusters are held out. Exploratory stress test of "
                    "cross-subpopulation extrapolation."
                ),
                "k_values": [6, 8, 10],
                "models": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
