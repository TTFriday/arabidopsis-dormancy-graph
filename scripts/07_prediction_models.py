"""Nested-CV prediction with SNP, haplotype-block and graph kernels.

The primary model is a fixed equal-weight dual kernel:
    K = 0.5 * K_SNP + 0.5 * K_block
where K_block is the average linear kernel of block mean dosages. K_SNP is the
linear genomic relationship matrix used by the GBLUP baseline. A graph-kernel
variant replaces K_block with kernels derived from the self-supervised block
embeddings. All kernels use genotype information only; phenotypes enter only
inside training folds.

Outputs
-------
results/tables/prediction_metrics.json
results/tables/prediction_model_comparison.csv
results/tables/prediction_cv_predictions.npz
results/figures/fig4_prediction.png
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
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
RNG = np.random.default_rng(20261028)

N_REPEATS = 30
N_PERMUTATIONS = 30
PERMUTATION_REPEATS = 10
LAMBDAS = np.array([1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0])


def load_inputs():
    pheno = pd.read_csv(PROCESSED / "dormancy_phenotype.csv")
    y = pheno["dsds50"].to_numpy(float)

    codes = np.load(PROCESSED / "genotypes_83.npy")
    snp = pd.read_csv(PROCESSED / "snp_metadata.csv")
    maf = snp["maf_83"].to_numpy()
    keep = np.flatnonzero((maf > 0.05) & (snp["minor_code"].to_numpy() < 4))
    minor = snp["minor_code"].to_numpy()[keep]
    Xsnp = (codes[keep, :] == minor[:, None]).astype(np.float32).T
    K_snp = Xsnp @ Xsnp.T / Xsnp.shape[1]

    cols = np.load(PROCESSED / "dormancy_cols_in_1307.npy")
    subset_path = PROCESSED / "block_features_83.npy"
    if subset_path.exists():
        block_mean = np.asarray(
            np.load(subset_path, mmap_mode="r")[:, :, 0], dtype=np.float64
        )
    else:
        block_feats = np.load(PROCESSED / "block_features.npy", mmap_mode="r")
        block_mean = np.asarray(block_feats[cols, :, 0], dtype=np.float64)
    K_block = block_mean_kernel(block_mean)

    graph_path = PROCESSED / "graph_embeddings_83.npy"
    if graph_path.exists():
        graph_nodes = np.load(graph_path)
        K_graph = graph_kernel(graph_nodes)
    else:
        K_graph = None

    sample_path = PROCESSED / "sample_embeddings_1307.npy"
    sample_emb = np.load(sample_path)[cols] if sample_path.exists() else None
    return y, K_snp, K_block, K_graph, sample_emb, int(Xsnp.shape[1])


def block_mean_kernel(mean_dosage: np.ndarray) -> np.ndarray:
    z = mean_dosage - mean_dosage.mean(axis=0)
    sd = z.std(axis=0)
    ok = sd > 1e-8
    z = z[:, ok] / sd[ok]
    return z @ z.T / z.shape[1]


def graph_kernel(nodes: np.ndarray) -> np.ndarray:
    """Average linear kernel of standardised block embeddings."""
    n, n_blocks, d = nodes.shape
    K = np.zeros((n, n), dtype=np.float64)
    used = 0
    for b in range(n_blocks):
        z = nodes[:, b, :].astype(np.float64)
        z = z - z.mean(axis=0)
        sd = z.std(axis=0)
        ok = sd > 1e-8
        if not ok.any():
            continue
        z = z[:, ok] / sd[ok]
        K += z @ z.T / z.shape[1]
        used += 1
    return K / max(used, 1)


def kernel_cv(K: np.ndarray, y: np.ndarray, repeats: int = N_REPEATS) -> dict:
    n = len(y)
    preds = np.zeros((repeats, n))
    chosen = []
    for rep in range(repeats):
        for tr, te in KFold(5, shuffle=True, random_state=rep).split(K):
            Ktr = K[np.ix_(tr, tr)]
            Kte = K[np.ix_(te, tr)]
            ytr = y[tr]
            best_mse = np.inf
            best_lam = LAMBDAS[-1]
            for lam in LAMBDAS:
                inner_pred = np.zeros(len(tr))
                for itr, ite in KFold(5, shuffle=True, random_state=rep + 100).split(tr):
                    alpha = np.linalg.solve(
                        Ktr[np.ix_(itr, itr)] + lam * np.eye(len(itr)),
                        ytr[itr] - ytr[itr].mean(),
                    )
                    inner_pred[ite] = Ktr[np.ix_(ite, itr)] @ alpha + ytr[itr].mean()
                mse = float(np.mean((inner_pred - ytr) ** 2))
                if mse < best_mse:
                    best_mse = mse
                    best_lam = float(lam)
            alpha = np.linalg.solve(Ktr + best_lam * np.eye(len(tr)), ytr - ytr.mean())
            preds[rep, te] = Kte @ alpha + ytr.mean()
            chosen.append(best_lam)
    r = np.array([stats.pearsonr(preds[i], y)[0] for i in range(repeats)])
    rho = np.array([stats.spearmanr(preds[i], y).statistic for i in range(repeats)])
    mae = np.array([np.mean(np.abs(preds[i] - y)) for i in range(repeats)])
    rmse = np.array([np.sqrt(np.mean((preds[i] - y) ** 2)) for i in range(repeats)])
    return {
        "pred_repeats": preds,
        "pearson_mean": float(r.mean()),
        "pearson_sd": float(r.std()),
        "spearman_mean": float(rho.mean()),
        "mae_mean": float(mae.mean()),
        "rmse_mean": float(rmse.mean()),
        "lambda_mode": float(pd.Series(chosen).mode()[0]),
        "repeat_r": r.tolist(),
    }


def embedding_cv(X: np.ndarray, y: np.ndarray, repeats: int = N_REPEATS) -> dict:
    n = len(y)
    preds = np.zeros((repeats, n))
    for rep in range(repeats):
        for tr, te in KFold(5, shuffle=True, random_state=rep).split(X):
            model = make_pipeline(
                StandardScaler(),
                RidgeCV(alphas=np.logspace(-4, 2, 19)),
            )
            model.fit(X[tr], y[tr])
            preds[rep, te] = model.predict(X[te])
    r = np.array([stats.pearsonr(preds[i], y)[0] for i in range(repeats)])
    rho = np.array([stats.spearmanr(preds[i], y).statistic for i in range(repeats)])
    mae = np.array([np.mean(np.abs(preds[i] - y)) for i in range(repeats)])
    return {
        "pred_repeats": preds,
        "pearson_mean": float(r.mean()),
        "pearson_sd": float(r.std()),
        "spearman_mean": float(rho.mean()),
        "mae_mean": float(mae.mean()),
        "rmse_mean": float(
            np.mean([np.sqrt(np.mean((preds[i] - y) ** 2)) for i in range(preds.shape[0])])
        ),
        "repeat_r": r.tolist(),
    }


def summarize_predictions(preds: np.ndarray, y: np.ndarray) -> dict:
    mean_pred = preds.mean(axis=0)
    r = float(stats.pearsonr(mean_pred, y)[0])
    rho = float(stats.spearmanr(mean_pred, y).statistic)
    mae = float(np.mean(np.abs(mean_pred - y)))
    return {"pooled_pearson": r, "pooled_spearman": rho, "pooled_mae": mae}


def main() -> None:
    y, K_snp, K_block, K_graph, sample_emb, n_snps = load_inputs()
    print(f"[07] y n={len(y)}, K_snp={K_snp.shape}", flush=True)
    kernels = {
        "SNP-GBLUP": K_snp,
        "block-kernel": K_block,
        "dual-SNP-block": 0.5 * K_snp + 0.5 * K_block,
    }
    if K_graph is not None:
        kernels["graph-kernel"] = K_graph
        kernels["dual-SNP-graph"] = 0.5 * K_snp + 0.5 * K_graph
        kernels["triple-kernel"] = (K_snp + K_block + K_graph) / 3.0

    results = {}
    preds_all = {}
    for name, K in kernels.items():
        print(f"[07] evaluating {name}", flush=True)
        res = kernel_cv(K, y)
        results[name] = {k: v for k, v in res.items() if k != "pred_repeats"}
        results[name].update(summarize_predictions(res["pred_repeats"], y))
        preds_all[name] = res["pred_repeats"]

    if sample_emb is not None:
        print("[07] evaluating graph sample embedding + ridge", flush=True)
        res = embedding_cv(sample_emb, y)
        results["graph-embedding-ridge"] = {
            k: v for k, v in res.items() if k != "pred_repeats"
        }
        results["graph-embedding-ridge"].update(summarize_predictions(res["pred_repeats"], y))
        preds_all["graph-embedding-ridge"] = res["pred_repeats"]

    # paired comparison against SNP-GBLUP and dual-SNP-block
    primary = "dual-SNP-block"
    comparisons = []
    for name, res in results.items():
        if name == "SNP-GBLUP":
            continue
        diffs = np.array(res["repeat_r"]) - np.array(results["SNP-GBLUP"]["repeat_r"])
        p = stats.wilcoxon(diffs, alternative="greater").pvalue if np.any(diffs != 0) else 1.0
        comparisons.append(
            {
                "model": name,
                "delta_r_vs_snp": float(diffs.mean()),
                "wilcoxon_p_vs_snp": float(p),
            }
        )
    for c in comparisons:
        if c["model"] != primary:
            diffs = np.array(results[c["model"]]["repeat_r"]) - np.array(results[primary]["repeat_r"])
            c["delta_r_vs_primary"] = float(diffs.mean())
            c["wilcoxon_p_vs_primary"] = float(
                stats.wilcoxon(diffs).pvalue if np.any(diffs != 0) else 1.0
            )
        else:
            c["delta_r_vs_primary"] = 0.0
            c["wilcoxon_p_vs_primary"] = 1.0

    comp_df = pd.DataFrame(comparisons).sort_values("wilcoxon_p_vs_snp")
    comp_df.to_csv(TABLES / "prediction_model_comparison.csv", index=False)

    np.savez_compressed(
        TABLES / "prediction_cv_predictions.npz",
        **{name.replace("-", "_"): pred for name, pred in preds_all.items()},
    )

    # permutation null for the primary model
    print("[07] permutation null for dual-SNP-block", flush=True)
    K_primary = 0.5 * K_snp + 0.5 * K_block
    observed = float(results[primary]["pearson_mean"])
    perm_r = []
    for s in range(N_PERMUTATIONS):
        yp = RNG.permutation(y)
        r = kernel_cv(K_primary, yp, repeats=PERMUTATION_REPEATS)["pearson_mean"]
        perm_r.append(float(r))
        print(f"[07] permutation {s + 1}/{N_PERMUTATIONS}: r={r:.3f}", flush=True)

    perm_p = float(
        (1 + np.sum(np.array(perm_r) >= observed)) / (len(perm_r) + 1)
    )
    permutation = {
        "observed_dual_r": observed,
        "n_permutations": len(perm_r),
        "repeats_per_permutation": PERMUTATION_REPEATS,
        "perm_r": perm_r,
        "perm_max": float(np.max(perm_r)),
        "perm_mean": float(np.mean(perm_r)),
        "perm_sd": float(np.std(perm_r)),
        "p_permutation": perm_p,
    }
    (TABLES / "prediction_permutation.json").write_text(json.dumps(permutation, indent=2))

    summary = {
        "n": int(len(y)),
        "n_snps": n_snps,
        "n_repeats": N_REPEATS,
        "primary_model": primary,
        "models": results,
        "permutation_primary_r": perm_r,
        "permutation_p": perm_p,
    }
    (TABLES / "prediction_metrics.json").write_text(json.dumps(summary, indent=2))

    # figure
    order = [
        "SNP-GBLUP",
        "block-kernel",
        "dual-SNP-block",
        "graph-kernel",
        "graph-embedding-ridge",
        "triple-kernel",
    ]
    order = [m for m in order if m in results]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), dpi=160)
    ax = axes[0]
    pred = preds_all[primary].mean(axis=0)
    ax.scatter(y, pred, s=18, alpha=0.75, color="#3b6ea8", edgecolor="white", linewidth=0.4)
    lo = min(y.min(), pred.min()) - 5
    hi = max(y.max(), pred.max()) + 5
    ax.plot([lo, hi], [lo, hi], "--", color="#888888", linewidth=0.9)
    ax.set_xlabel("Observed DSDS50 (days)")
    ax.set_ylabel("Predicted DSDS50 (days)")
    ax.set_title(f"Dual kernel (r = {results[primary]['pearson_mean']:.3f})", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    means = [results[m]["pearson_mean"] for m in order]
    sds = [results[m]["pearson_sd"] for m in order]
    x = np.arange(len(order))
    ax.barh(x, means, xerr=sds, color="#7fa6c9", edgecolor="none", height=0.65)
    ax.set_yticks(x)
    ax.set_yticklabels(order, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Pearson r (5-fold CV)")
    ax.set_xlim(0, 0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig4_prediction.png")
    plt.close(fig)

    print(
        comp_df[
            ["model", "delta_r_vs_snp", "wilcoxon_p_vs_snp", "delta_r_vs_primary"]
        ].to_string(index=False)
    )
    print(json.dumps({k: summary["models"][k]["pearson_mean"] for k in results}, indent=2))


if __name__ == "__main__":
    main()
