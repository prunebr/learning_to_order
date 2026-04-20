# src/rlod/eval/ba_metrics.py
from __future__ import annotations
from typing import Dict, List

import numpy as np
import networkx as nx


def gini_degree_from_adj(A: np.ndarray) -> float:
    n = int(A.shape[0])
    if n == 0:
        return 0.0
    d = A.sum(axis=0).astype(np.float64)
    s = float(d.sum())
    if s <= 0:
        return 0.0
    d_sorted = np.sort(d)
    i = np.arange(1, n + 1, dtype=np.float64)
    num = float(((2 * i - n - 1) * d_sorted).sum())
    den = float(n * s)
    return num / den


def hubiness_from_adj(A: np.ndarray) -> float:
    n = int(A.shape[0])
    if n < 2:
        return 0.0
    dmax = float(A.sum(axis=0).max())
    return dmax / float(n - 1)


def connected_from_adj(A: np.ndarray) -> bool:
    n = int(A.shape[0])
    if n == 0:
        return False
    G = nx.from_numpy_array(A)
    return nx.is_connected(G)


def evaluate_generated_ba(adjs: List[np.ndarray], n_min: int, n_max: int) -> Dict[str, float]:
    S = len(adjs)
    if S == 0:
        return {
            "average_size": 0.0,
            "std_size": 0.0,
            "valid_size_ratio": 0.0,
            "connected_ratio": 0.0,
            "gini_degree": 0.0,
            "avg_gini_degree": 0.0,
            "hubiness": 0.0,
            "avg_edges": 0.0,
            "avg_nodes": 0.0,
            "std_nodes": 0.0,
        }

    sizes = np.array([a.shape[0] for a in adjs], dtype=np.float64)
    valid_size_ratio = float(((sizes >= n_min) & (sizes <= n_max)).mean())
    connected_ratio = float(np.mean([connected_from_adj(a) for a in adjs]))
    gini_degree = float(np.mean([gini_degree_from_adj(a) for a in adjs]))
    hubiness = float(np.mean([hubiness_from_adj(a) for a in adjs]))
    avg_edges = float(np.mean([a.sum() / 2.0 for a in adjs]))

    return {
        "average_size": float(sizes.mean()),
        "std_size": float(sizes.std()),
        "valid_size_ratio": valid_size_ratio,
        "connected_ratio": connected_ratio,
        "gini_degree": gini_degree,
        "avg_gini_degree": gini_degree,
        "hubiness": hubiness,
        "avg_edges": avg_edges,
        "avg_nodes": float(sizes.mean()),
        "std_nodes": float(sizes.std()),
    }