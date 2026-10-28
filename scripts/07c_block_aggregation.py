"""Sensitivity of the block kernel to block granularity.

Adjacent blocks on the same chromosome are merged in groups of two and four,
and the resulting mean-dosage kernel is evaluated with the same nested CV.
This checks whether the predictive contribution of the block representation
depends on the 25-SNP block definition.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"


def load_module():
    path = ROOT / "scripts" / "07_prediction_models.py"
    spec = importlib.util.spec_from_file_location("pred07", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def merge_blocks(mean_dosage: np.ndarray, chrom: np.ndarray, size: int) -> np.ndarray:
    merged = []
    for c in np.unique(chrom):
        idx = np.flatnonzero(chrom == c)
        for start in range(0, len(idx), size):
            merged.append(mean_dosage[:, idx[start : start + size]].mean(axis=1))
    return np.column_stack(merged)


def main() -> None:
    pred = load_module()
    y, _, K_block, _, _, _ = pred.load_inputs()

    cols = np.load(PROCESSED / "dormancy_cols_in_1307.npy")
    subset_path = PROCESSED / "block_features_83.npy"
    if subset_path.exists():
        mean_dosage = np.asarray(
            np.load(subset_path, mmap_mode="r")[:, :, 0], dtype=np.float64
        )
    else:
        block_feats = np.load(PROCESSED / "block_features.npy", mmap_mode="r")
        mean_dosage = np.asarray(block_feats[cols, :, 0], dtype=np.float64)
    blocks = pd.read_csv(PROCESSED / "haplotype_blocks.csv")
    chrom = blocks["chrom"].to_numpy()

    results = {"1x (25-SNP blocks)": pred.kernel_cv(K_block, y)["pearson_mean"]}
    for size in (2, 4):
        merged = merge_blocks(mean_dosage, chrom, size)
        K = pred.block_mean_kernel(merged)
        res = pred.kernel_cv(K, y)
        results[f"{size}x merge ({merged.shape[1]} windows)"] = res["pearson_mean"]
        print(f"[07c] {size}x: {res['pearson_mean']:.4f}", flush=True)

    (TABLES / "block_aggregation_sensitivity.json").write_text(
        json.dumps(results, indent=2)
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
