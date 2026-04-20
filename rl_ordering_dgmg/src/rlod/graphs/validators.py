from __future__ import annotations

import numpy as np
from typing import Tuple


def is_connected_from_adj(adj: np.ndarray) -> bool:
    n = adj.shape[0]
    if n == 0:
        return False
    if n == 1:
        return True

    seen = np.zeros(n, dtype=bool)
    stack = [0]
    seen[0] = True

    while stack:
        v = stack.pop()
        neigh = np.where(adj[v] > 0)[0]
        for u in neigh:
            if not seen[u]:
                seen[u] = True
                stack.append(u)

    return bool(seen.all())


def is_tree_from_adj(adj: np.ndarray) -> bool:
    n = adj.shape[0]
    # árvore simples: conectada e m = n-1
    m = int(adj.sum() // 2)
    return (m == n - 1) and is_connected_from_adj(adj)


def is_binary_tree_undirected(adj: np.ndarray, root: int | None = None) -> bool:
    """
    Critério estrutural (não-orientado):
      - deve ser árvore
      - grau máximo <= 3
      - se root for dado: grau(root) <= 2
    """
    if not is_tree_from_adj(adj):
        return False

    deg = adj.sum(axis=1)
    if int(deg.max()) > 3:
        return False

    if root is not None:
        if int(deg[int(root)]) > 2:
            return False

    return True
