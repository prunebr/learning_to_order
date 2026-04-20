from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
import numpy as np

from rlod.graphs.types import GraphSample
from .actions import ActionType


def validate_order(order: List[int], n_nodes: int) -> None:
    if len(order) != n_nodes:
        raise ValueError(f"order len={len(order)} != n_nodes={n_nodes}")
    s = set(order)
    if len(s) != n_nodes:
        raise ValueError("order tem repetição.")
    if min(s) != 0 or max(s) != n_nodes - 1:
        raise ValueError("order deve conter exatamente 0..n-1.")


@dataclass(frozen=True)
class DecisionSequence:
    n_nodes: int
    actions: np.ndarray               # (L,) int16
    args: np.ndarray                  # (L,) int16  (-1 quando não tem argumento)
    order: List[int]                  # ordenação original (ids do grafo)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_nodes": int(self.n_nodes),
            "actions": np.asarray(self.actions, dtype=np.int16),
            "args": np.asarray(self.args, dtype=np.int16),
            "order": list(map(int, self.order)),
            "meta": dict(self.meta),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "DecisionSequence":
        return DecisionSequence(
            n_nodes=int(d["n_nodes"]),
            actions=np.asarray(d["actions"], dtype=np.int16),
            args=np.asarray(d["args"], dtype=np.int16),
            order=list(map(int, d["order"])),
            meta=dict(d.get("meta", {})),
        )


def build_decision_sequence(sample: GraphSample, order: List[int]) -> DecisionSequence:
    """
    Converte (grafo + ordenação) -> sequência de ações DGMG.

    Importante: CHOOSE_DEST usa o índice no "espaço de construção" (0..t-1),
    NÃO o id original do grafo.
    """
    n = sample.n_nodes
    validate_order(order, n)
    adj = sample.adj

    actions: List[int] = []
    args: List[int] = []

    orig_to_new: Dict[int, int] = {}
    added: set[int] = set()

    for t, v in enumerate(order):
        # vizinhos já adicionados (antes de adicionar v)
        prev_added = added
        neigh = np.flatnonzero(adj[v]).astype(int).tolist()
        prev_neighbors = [u for u in neigh if u in prev_added]

        # ADD_NODE
        actions.append(int(ActionType.ADD_NODE))
        args.append(-1)

        # agora marca como adicionado
        orig_to_new[v] = t
        added.add(v)

        # ordenar os destinos por índice de construção (determinístico)
        prev_neighbors.sort(key=lambda u: orig_to_new[u])

        # para cada aresta (v -> u_prev)
        for u in prev_neighbors:
            actions.append(int(ActionType.ADD_EDGE))
            args.append(-1)

            actions.append(int(ActionType.CHOOSE_DEST))
            args.append(int(orig_to_new[u]))

        # STOP_EDGE para esse nó
        actions.append(int(ActionType.STOP_EDGE))
        args.append(-1)

    # STOP_NODE final
    actions.append(int(ActionType.STOP_NODE))
    args.append(-1)

    actions_arr = np.asarray(actions, dtype=np.int16)
    args_arr = np.asarray(args, dtype=np.int16)

    meta = {
        "family": sample.meta.get("family", "unknown"),
        "generator_mode": sample.meta.get("mode", "unknown"),
        "n_edges": int(sample.edges.shape[0]),
        "seq_len": int(actions_arr.shape[0]),
    }

    return DecisionSequence(
        n_nodes=n,
        actions=actions_arr,
        args=args_arr,
        order=order,
        meta=meta,
    )


def reconstruct_adj_from_sequence(seq: DecisionSequence) -> np.ndarray:
    """
    Reconstroi adj (n,n) a partir da sequência, para sanity-check.
    """
    n = seq.n_nodes
    adj = np.zeros((n, n), dtype=np.uint8)

    cur_node = -1
    pending_edge = False

    for a, arg in zip(seq.actions.tolist(), seq.args.tolist()):
        at = ActionType(int(a))

        if at == ActionType.ADD_NODE:
            cur_node += 1
            pending_edge = False

        elif at == ActionType.ADD_EDGE:
            if cur_node < 0:
                raise ValueError("ADD_EDGE antes de ADD_NODE.")
            pending_edge = True

        elif at == ActionType.CHOOSE_DEST:
            if not pending_edge:
                raise ValueError("CHOOSE_DEST sem ADD_EDGE anterior.")
            dest = int(arg)
            if dest < 0 or dest >= cur_node:
                raise ValueError(f"dest inválido {dest} (cur_node={cur_node})")
            adj[cur_node, dest] = 1
            adj[dest, cur_node] = 1
            pending_edge = False

        elif at == ActionType.STOP_EDGE:
            pending_edge = False

        elif at == ActionType.STOP_NODE:
            break

        else:
            raise ValueError(f"Ação desconhecida: {a}")

    np.fill_diagonal(adj, 0)
    return adj


def reconstruct_adj_original_from_sequence(seq: DecisionSequence) -> np.ndarray:
    """
    Reconstrói adj no espaço ORIGINAL dos ids do grafo (0..n-1 do GraphSample),
    usando seq.order (que mapeia posição -> id original).
    """
    adj_order = reconstruct_adj_from_sequence(seq)  # (n,n) no espaço da ordenação
    p = np.asarray(seq.order, dtype=np.int64)       # p[t] = id original do nó no passo t

    adj_orig = np.zeros_like(adj_order, dtype=np.uint8)
    adj_orig[np.ix_(p, p)] = adj_order
    np.fill_diagonal(adj_orig, 0)
    return adj_orig


def validate_sequence_matches_graph(sample: GraphSample, seq: DecisionSequence) -> None:
    """
    Confirma que a reconstrução bate com a adj original do grafo.
    """
    rec_orig = reconstruct_adj_original_from_sequence(seq)
    target = sample.adj.astype(np.uint8)

    if rec_orig.shape != target.shape:
        raise ValueError("Adj shape mismatch.")

    if not np.array_equal(rec_orig, target):
        diff = np.abs(rec_orig.astype(int) - target.astype(int)).sum()
        raise ValueError(f"Reconstrução não bate com o grafo. diff={diff}")
