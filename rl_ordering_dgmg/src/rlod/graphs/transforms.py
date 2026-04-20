from __future__ import annotations

from typing import Iterable, List, Tuple
import numpy as np


def edges_to_adj(n_nodes: int, edges: np.ndarray) -> np.ndarray:
    """
    Converte edges (m,2) com u<v para adj (n,n) uint8 simétrica.
    """
    adj = np.zeros((n_nodes, n_nodes), dtype=np.uint8)
    if edges.size == 0:
        return adj

    u = edges[:, 0]
    v = edges[:, 1]
    adj[u, v] = 1
    adj[v, u] = 1
    np.fill_diagonal(adj, 0)
    return adj


def relabel_edges(edges: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """
    Aplica uma permutação de rótulos: novo_id = perm[old_id]
    edges: (m,2) com u<v no espaço antigo -> retorna (m,2) com u'<v' no novo espaço
    """
    u = perm[edges[:, 0]]
    v = perm[edges[:, 1]]
    uu = np.minimum(u, v)
    vv = np.maximum(u, v)
    out = np.stack([uu, vv], axis=1).astype(np.int64)
    # Remover possíveis duplicatas (em tese não deveria criar)
    out = np.unique(out, axis=0)
    return out
