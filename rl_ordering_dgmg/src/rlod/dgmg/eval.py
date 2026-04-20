from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import numpy as np

from rlod.graphs.validators import is_connected_from_adj, is_tree_from_adj, is_binary_tree_undirected


@dataclass
class EvalConfig:
    n_min: int = 5
    n_max: int = 20


def _gini(x: np.ndarray) -> float:
    # gini para distribuição de graus (tipo dissertação)
    x = x.astype(float)
    if x.size == 0:
        return 0.0
    if np.all(x == 0):
        return 0.0
    x = np.sort(x)
    n = x.size
    cumx = np.cumsum(x)
    g = (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n
    return float(g)


def evaluate_generated(adjs: List[np.ndarray], cfg: EvalConfig) -> Dict[str, float]:
    n = len(adjs)
    if n == 0:
        return {}

    valid = 0
    valid_size = 0
    connected = 0
    tree = 0
    binary = 0

    nodes_list = []
    edges_list = []
    gini_list = []

    for adj in adjs:
        nn = int(adj.shape[0])
        nodes_list.append(nn)
        m = int(adj.sum() // 2)
        edges_list.append(m)

        deg = adj.sum(axis=1).astype(int)
        gini_list.append(_gini(deg))

        in_size = (cfg.n_min <= nn <= cfg.n_max)
        if in_size:
            valid_size += 1

        conn = is_connected_from_adj(adj)
        if conn:
            connected += 1

        is_tree = is_tree_from_adj(adj)
        if is_tree:
            tree += 1

        is_bin = is_binary_tree_undirected(adj, root=0)
        if is_bin:
            binary += 1

        # "valid" = árvore binária (estrutural)
        if is_bin:
            valid += 1

    nodes_arr = np.array(nodes_list, dtype=float)
    edges_arr = np.array(edges_list, dtype=float)
    gini_arr = np.array(gini_list, dtype=float)

    return {
        "num_samples": float(n),
        "valid_ratio": valid / n,
        "valid_size_ratio": valid_size / n,
        "connected_ratio": connected / n,
        "tree_ratio": tree / n,
        "binary_ratio": binary / n,
        "avg_nodes": float(nodes_arr.mean()),
        "std_nodes": float(nodes_arr.std()),
        "avg_edges": float(edges_arr.mean()),
        "std_edges": float(edges_arr.std()),
        "avg_gini_degree": float(gini_arr.mean()),
        "std_gini_degree": float(gini_arr.std()),
    }
