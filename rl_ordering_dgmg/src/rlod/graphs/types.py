from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple
import numpy as np


@dataclass(frozen=True)
class GraphSample:
    """
    Formato padrão do projeto para um grafo "raw" (antes de ordenação/seq DGMG).

    - n_nodes: número de nós
    - edges: lista de arestas não-direcionadas únicas (u, v) com u < v
    - adj: matriz de adjacência (uint8 0/1), simétrica, sem self-loop
    - root: nó raiz (se fizer sentido na família; para árvores binárias, faz)
    - meta: dicionário de metadados (família, parâmetros do gerador, etc.)
    """
    n_nodes: int
    edges: np.ndarray                 # shape (m, 2), dtype=int64, u < v
    adj: np.ndarray                   # shape (n, n), dtype=uint8
    root: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        # Para serialização estável (pickle/json-friendly)
        return {
            "n_nodes": int(self.n_nodes),
            "edges": np.asarray(self.edges, dtype=np.int64),
            "adj": np.asarray(self.adj, dtype=np.uint8),
            "root": int(self.root),
            "meta": dict(self.meta),
        }

    @staticmethod
    def from_dict(d: Any) -> "GraphSample":
        # Accept either:
        #  (1) the project dict format: {"n_nodes","edges","adj",...}
        #  (2) a raw adjacency matrix (np.ndarray), used by some dataset scripts (e.g., BA)

        # Case (2): raw adjacency matrix
        if isinstance(d, np.ndarray):
            A = np.asarray(d, dtype=np.uint8)
            if A.ndim != 2 or A.shape[0] != A.shape[1]:
                raise ValueError(f"Adjacency must be square (n,n), got shape={A.shape}")

            # force 0/1, symmetry, no self-loops (defensive)
            np.fill_diagonal(A, 0)
            A = ((A + A.T) > 0).astype(np.uint8)

            # edges: all (u,v) with u<v and A[u,v]=1
            # np.triu(A,1).nonzero() returns (rows, cols) of upper-triangular ones
            u, v = np.triu(A, k=1).nonzero()
            edges = np.stack([u, v], axis=1).astype(np.int64, copy=False)

            return GraphSample(
                n_nodes=int(A.shape[0]),
                edges=edges,
                adj=A,
                root=0,
                meta={},
            )

        # Case (1): dict format (existing behavior)
        return GraphSample(
            n_nodes=int(d["n_nodes"]),
            edges=np.asarray(d["edges"], dtype=np.int64),
            adj=np.asarray(d["adj"], dtype=np.uint8),
            root=int(d.get("root", 0)),
            meta=dict(d.get("meta", {})),
        )

class GraphSource(Protocol):
    """
    Interface (protocolo) de uma fonte de grafos.
    Implementações devem ser determinísticas dado um RNG/seed.
    """
    def sample(self) -> GraphSample:
        ...

    def sample_batch(self, k: int) -> List[GraphSample]:
        ...
