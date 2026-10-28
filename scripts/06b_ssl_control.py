"""No-pretraining control for the graph encoder.

The same graph Transformer is instantiated with the training seed but no
gradient updates, and its block embeddings are passed through the identical
kernel and cross-validation protocol. The comparison isolates the contribution
of masked graph pretraining from the architecture and the input features.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"
SEED = 20261024


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    torch.set_num_threads(1)
    ssl = load_module("ssl06", ROOT / "scripts" / "06_self_supervised_pretrain.py")
    pred = load_module("pred07", ROOT / "scripts" / "07_prediction_models.py")

    feats, src, dst, efeat, _, _ = ssl.build_graph_tensors()
    cols = np.load(PROCESSED / "dormancy_cols_in_1307.npy")

    torch.manual_seed(SEED)
    model = ssl.MaskedGraphEncoder(
        in_dim=feats.shape[2],
        d=16,
        heads=4,
        n_layers=2,
        edge_dim=efeat.shape[1],
        dropout=0.10,
    )
    model.eval()
    nodes, _ = ssl.encode_all(
        model,
        torch.from_numpy(feats[cols]),
        src,
        dst,
        efeat,
        batch_size=16,
    )

    y, K_snp, K_block, K_trained, _, _ = pred.load_inputs()
    K_random = pred.graph_kernel(nodes.astype(np.float64))
    res = pred.kernel_cv(K_random, y, repeats=30)

    tri = np.triu_indices(len(y), k=1)

    def similarity(A: np.ndarray, B: np.ndarray) -> float:
        return float(np.corrcoef(A[tri], B[tri])[0, 1])

    out = {
        "control": "random-initialised encoder, no pretraining",
        "seed": SEED,
        "pearson_mean": res["pearson_mean"],
        "pearson_sd": res["pearson_sd"],
        "spearman_mean": res["spearman_mean"],
        "mae_mean": res["mae_mean"],
        "repeat_r": res["repeat_r"],
        "kernel_similarity": {
            "snp_vs_block": similarity(K_snp, K_block),
            "block_vs_graph_trained": similarity(K_block, K_trained),
            "block_vs_graph_random": similarity(K_block, K_random),
            "graph_trained_vs_random": similarity(K_trained, K_random),
        },
    }
    (TABLES / "ssl_control.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "repeat_r"}, indent=2))


if __name__ == "__main__":
    main()
