from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from .types import GraphSample
from .transforms import edges_to_adj, relabel_edges
from .validators import is_binary_tree_undirected


@dataclass
class BinaryTreeSourceConfig:
    n_min: int = 5
    n_max: int = 20
    mode: str = "any"   # "any" | "full" | "perfect"
    relabel_nodes: bool = True
    max_tries: int = 100


class BinaryTreeSource:
    """
    Gera árvores binárias enraizadas e exporta como grafo não-direcionado.

    mode:
      - "any": cada nó tem no máximo 2 filhos (grau<=3), root grau<=2
      - "full": todo nó interno tem exatamente 2 filhos (n deve ser ímpar)
      - "perfect": árvore perfeita (n = 2^{d+1}-1)
    """
    def __init__(self, cfg: BinaryTreeSourceConfig, seed: int = 0):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

        if self.cfg.n_min < 1 or self.cfg.n_max < self.cfg.n_min:
            raise ValueError("Intervalo inválido n_min/n_max.")

        if self.cfg.mode not in {"any", "full", "perfect"}:
            raise ValueError("mode deve ser: any | full | perfect")

    def sample(self) -> GraphSample:
        for _ in range(self.cfg.max_tries):
            n = int(self.rng.integers(self.cfg.n_min, self.cfg.n_max + 1))

            if self.cfg.mode == "perfect":
                sample = self._sample_perfect_in_range()
            elif self.cfg.mode == "full":
                if n % 2 == 0:
                    continue
                sample = self._sample_full(n)
            else:
                sample = self._sample_any(n)

            if self.cfg.relabel_nodes:
                sample = self._relabel_sample(sample)

            # sanity: garantir que o que sai é realmente binário
            if is_binary_tree_undirected(sample.adj, root=sample.root):
                return sample

        raise RuntimeError("Falhou em gerar uma árvore binária válida dentro de max_tries.")

    def sample_batch(self, k: int) -> List[GraphSample]:
        return [self.sample() for _ in range(k)]

    # -------------------- modos de geração --------------------

    def _sample_any(self, n: int) -> GraphSample:
        # Root fixo 0
        edges: List[Tuple[int, int]] = []
        root = 0

        # slots = lista de nós com slots disponíveis (cada nó começa com 2 slots)
        # representamos slots como contagem de slots livres por nó
        free_slots = {0: 2}

        for new_node in range(1, n):
            # escolher um pai proporcional ao número de slots livres
            parents = []
            weights = []
            for p, slots in free_slots.items():
                if slots > 0:
                    parents.append(p)
                    weights.append(slots)

            p = int(self.rng.choice(parents, p=np.array(weights) / np.sum(weights)))
            edges.append((min(p, new_node), max(p, new_node)))

            # consumir um slot do pai
            free_slots[p] -= 1

            # novo nó nasce com 2 slots (pode ter até 2 filhos)
            free_slots[new_node] = 2

        edges_arr = np.array(edges, dtype=np.int64)
        adj = edges_to_adj(n, edges_arr)
        return GraphSample(
            n_nodes=n,
            edges=np.unique(edges_arr, axis=0),
            adj=adj,
            root=root,
            meta={"family": "binary_tree", "mode": "any"},
        )

    def _sample_full(self, n: int) -> GraphSample:
        """
        Full binary tree: cada nó interno tem 2 filhos.
        Construção: começa com root folha; expande folhas adicionando 2 filhos até atingir n.
        Requer n ímpar.
        """
        if n % 2 == 0:
            raise ValueError("Full binary tree requer n ímpar.")

        edges: List[Tuple[int, int]] = []
        root = 0

        next_node = 1
        leaves = [0]  # nós que ainda não foram expandidos (0 filhos)

        while next_node + 1 < n:
            leaf_idx = int(self.rng.integers(0, len(leaves)))
            parent = leaves.pop(leaf_idx)

            left = next_node
            right = next_node + 1
            next_node += 2

            edges.append((min(parent, left), max(parent, left)))
            edges.append((min(parent, right), max(parent, right)))

            # esses novos nós são folhas por enquanto
            leaves.append(left)
            leaves.append(right)

        if next_node != n:
            # Por construção, isso não deveria acontecer se n é ímpar
            raise RuntimeError("Falha ao completar full tree: contagem de nós inesperada.")

        edges_arr = np.array(edges, dtype=np.int64)
        adj = edges_to_adj(n, edges_arr)
        return GraphSample(
            n_nodes=n,
            edges=np.unique(edges_arr, axis=0),
            adj=adj,
            root=root,
            meta={"family": "binary_tree", "mode": "full"},
        )

    def _sample_perfect_in_range(self) -> GraphSample:
        """
        Escolhe uma profundidade d tal que n=2^{d+1}-1 esteja dentro de [n_min,n_max],
        e gera a árvore perfeita.
        """
        possible_n = []
        d = 0
        while True:
            n = (2 ** (d + 1)) - 1
            if n > self.cfg.n_max:
                break
            if n >= self.cfg.n_min:
                possible_n.append((d, n))
            d += 1

        if not possible_n:
            raise RuntimeError("Não há árvores perfeitas dentro do intervalo n_min/n_max.")

        d, n = possible_n[int(self.rng.integers(0, len(possible_n)))]
        return self._sample_perfect(depth=d)

    def _sample_perfect(self, depth: int) -> GraphSample:
        """
        Árvore perfeita: todos níveis completos até depth.
        depth=0 -> 1 nó
        """
        root = 0
        edges: List[Tuple[int, int]] = []

        # indexação em heap: pai i, filhos 2i+1 e 2i+2
        n = (2 ** (depth + 1)) - 1
        for i in range((2 ** depth) - 1):  # nós internos
            l = 2 * i + 1
            r = 2 * i + 2
            edges.append((min(i, l), max(i, l)))
            edges.append((min(i, r), max(i, r)))

        edges_arr = np.array(edges, dtype=np.int64)
        adj = edges_to_adj(n, edges_arr)
        return GraphSample(
            n_nodes=n,
            edges=np.unique(edges_arr, axis=0),
            adj=adj,
            root=root,
            meta={"family": "binary_tree", "mode": "perfect", "depth": depth},
        )

    # -------------------- helpers --------------------

    def _relabel_sample(self, s: GraphSample) -> GraphSample:
        """
        Relabela nós aleatoriamente para evitar viés de indexação.
        Mantém o root coerente após a permutação.
        """
        perm = self.rng.permutation(s.n_nodes).astype(np.int64)
        inv_perm = np.empty_like(perm)
        inv_perm[perm] = np.arange(s.n_nodes, dtype=np.int64)

        new_edges = relabel_edges(s.edges, perm)
        new_adj = edges_to_adj(s.n_nodes, new_edges)
        new_root = int(perm[s.root])

        return GraphSample(
            n_nodes=s.n_nodes,
            edges=new_edges,
            adj=new_adj,
            root=new_root,
            meta={**s.meta, "relabel_nodes": True},
        )
