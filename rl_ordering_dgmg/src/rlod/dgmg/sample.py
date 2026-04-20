from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import torch

from rlod.dgmg.model import DGMGMinimal
from rlod.sequences.actions import ActionType


@dataclass
class SampleConfig:
    min_nodes: int = 5
    max_nodes: int = 20
    temperature: float = 1.0
    greedy: bool = False
    max_edges_per_node: int = 10  # safety
    force_min_nodes: bool = True  # força ADD_NODE até min_nodes
    force_max_nodes: bool = True  # força STOP_NODE em max_nodes


@dataclass
class _State:
    g: torch.Tensor
    node_embs: List[torch.Tensor]
    pending_edge: bool


def _pick(logits: torch.Tensor, cfg: SampleConfig) -> int:
    if cfg.temperature != 1.0:
        logits = logits / float(cfg.temperature)
    if cfg.greedy:
        return int(torch.argmax(logits).item())
    dist = torch.distributions.Categorical(logits=logits)
    return int(dist.sample().item())


def _apply_teacher_action(model: DGMGMinimal, st: _State, adj: np.ndarray, a: int, arg: int) -> bool:
    """
    Aplica uma ação teacher-forced (sem loss). Retorna True se STOP_NODE.
    """
    dev = st.g.device
    at = ActionType(int(a))

    if at == ActionType.ADD_NODE:
        h0 = model.node_init_proj(model.node_init.to(dev))
        st.node_embs.append(h0)
        st.g = model.graph_gru(h0.unsqueeze(0), st.g.unsqueeze(0)).squeeze(0)
        st.pending_edge = False
        return False

    if at == ActionType.STOP_NODE:
        return True

    cur_node = len(st.node_embs) - 1

    if at == ActionType.ADD_EDGE:
        st.pending_edge = True
        return False

    if at == ActionType.STOP_EDGE:
        st.pending_edge = False
        return False

    if at == ActionType.CHOOSE_DEST:
        if not st.pending_edge:
            return False
        dest = int(arg)
        if cur_node >= 0 and 0 <= dest < cur_node:
            adj[cur_node, dest] = 1
            adj[dest, cur_node] = 1

            cur = st.node_embs[cur_node]
            prev = torch.stack(st.node_embs[:cur_node], dim=0)
            chosen = prev[dest]
            new_cur = model.node_gru(chosen.unsqueeze(0), cur.unsqueeze(0)).squeeze(0)
            st.node_embs[cur_node] = new_cur
            st.g = model.graph_gru(new_cur.unsqueeze(0), st.g.unsqueeze(0)).squeeze(0)

        st.pending_edge = False
        return False

    return False


@torch.no_grad()
def sample_graph(model: DGMGMinimal, cfg: SampleConfig, device: str = "cpu") -> np.ndarray:
    """
    Sampling autoregressivo completo com STOP_NODE aprendido.
    """
    dev = torch.device(device)
    model.eval().to(dev)

    H = model.cfg.hidden_dim
    st = _State(g=model.graph0.to(dev).clone(), node_embs=[], pending_edge=False)

    adj = np.zeros((cfg.max_nodes, cfg.max_nodes), dtype=np.uint8)

    while True:
        n_now = len(st.node_embs)

        # decide ADD_NODE vs STOP_NODE
        if n_now == 0:
            node_choice = 0 
        elif cfg.force_min_nodes and n_now < cfg.min_nodes:
            node_choice = 0
        elif cfg.force_max_nodes and n_now >= cfg.max_nodes:
            node_choice = 1
        else:
            logits_node = model.node_mlp(st.g)
            node_choice = _pick(logits_node, cfg)

        if node_choice == 1:  # STOP_NODE
            break

        # ADD_NODE
        h0 = model.node_init_proj(model.node_init.to(dev))
        st.node_embs.append(h0)
        st.g = model.graph_gru(h0.unsqueeze(0), st.g.unsqueeze(0)).squeeze(0)

        cur_node = len(st.node_embs) - 1
        connected_prev = set()
        edges_done = 0

        while True:
            cur = st.node_embs[cur_node]
            logits_edge = model.edge_mlp(torch.cat([st.g, cur], dim=0))  # (2,)
            edge_choice = _pick(logits_edge, cfg)  # 0 add, 1 stop

            if edge_choice == 1:
                break

            if cur_node == 0:
                break
            if edges_done >= cfg.max_edges_per_node:
                break

            prev = torch.stack(st.node_embs[:cur_node], dim=0)  # (cur_node, H)
            q = model.dest_q(torch.cat([st.g, cur], dim=0))
            k = model.dest_k(prev)
            logits_dest = (k @ q)  # (cur_node,)

            # mask multi-edge
            if connected_prev:
                mask = torch.ones(cur_node, dtype=torch.bool, device=dev)
                for d in connected_prev:
                    mask[d] = False
                logits_dest = logits_dest.masked_fill(~mask, -1e9)

            if torch.all(logits_dest < -1e8):
                break

            dest = _pick(logits_dest, cfg)
            if dest in connected_prev:
                continue
            connected_prev.add(dest)

            adj[cur_node, dest] = 1
            adj[dest, cur_node] = 1

            chosen = prev[dest]
            new_cur = model.node_gru(chosen.unsqueeze(0), cur.unsqueeze(0)).squeeze(0)
            st.node_embs[cur_node] = new_cur
            st.g = model.graph_gru(new_cur.unsqueeze(0), st.g.unsqueeze(0)).squeeze(0)

            edges_done += 1

    n = len(st.node_embs)
    out = adj[:n, :n]
    np.fill_diagonal(out, 0)
    return out


@torch.no_grad()
def sample_graph_from_prefix(
    model: DGMGMinimal,
    prefix_actions: np.ndarray,
    prefix_args: np.ndarray,
    prefix_len: int,
    cfg: SampleConfig,
    device: str = "cpu",
) -> np.ndarray:
    """
    Teacher-force de um prefixo (derivado da ordenação do RL) e depois completa por sampling.
    Isso dá dependência da validade em relação à ordenação.
    """
    dev = torch.device(device)
    model.eval().to(dev)

    st = _State(g=model.graph0.to(dev).clone(), node_embs=[], pending_edge=False)
    adj = np.zeros((cfg.max_nodes, cfg.max_nodes), dtype=np.uint8)

    # aplica prefixo
    stopped = False
    L = min(prefix_len, len(prefix_actions))
    for i in range(L):
        a = int(prefix_actions[i])
        arg = int(prefix_args[i])
        if a == -1:
            break
        stopped = _apply_teacher_action(model, st, adj, a, arg)
        if stopped:
            break

    if stopped:
        n = len(st.node_embs)
        out = adj[:n, :n]
        np.fill_diagonal(out, 0)
        return out

    # completa com sampling
    while True:
        n_now = len(st.node_embs)

        if n_now == 0:
            node_choice = 0
        elif cfg.force_min_nodes and n_now < cfg.min_nodes:
            node_choice = 0
        elif cfg.force_max_nodes and n_now >= cfg.max_nodes:
            node_choice = 1
        else:
            logits_node = model.node_mlp(st.g)
            node_choice = _pick(logits_node, cfg)

        if node_choice == 1:
            break

        h0 = model.node_init_proj(model.node_init.to(dev))
        st.node_embs.append(h0)
        st.g = model.graph_gru(h0.unsqueeze(0), st.g.unsqueeze(0)).squeeze(0)

        cur_node = len(st.node_embs) - 1
        connected_prev = set()
        edges_done = 0

        while True:
            cur = st.node_embs[cur_node]
            logits_edge = model.edge_mlp(torch.cat([st.g, cur], dim=0))
            edge_choice = _pick(logits_edge, cfg)

            if edge_choice == 1:
                break
            if cur_node == 0:
                break
            if edges_done >= cfg.max_edges_per_node:
                break

            prev = torch.stack(st.node_embs[:cur_node], dim=0)
            q = model.dest_q(torch.cat([st.g, cur], dim=0))
            k = model.dest_k(prev)
            logits_dest = (k @ q)

            if connected_prev:
                mask = torch.ones(cur_node, dtype=torch.bool, device=dev)
                for d in connected_prev:
                    mask[d] = False
                logits_dest = logits_dest.masked_fill(~mask, -1e9)

            if torch.all(logits_dest < -1e8):
                break

            dest = _pick(logits_dest, cfg)
            if dest in connected_prev:
                continue
            connected_prev.add(dest)

            adj[cur_node, dest] = 1
            adj[dest, cur_node] = 1

            chosen = prev[dest]
            new_cur = model.node_gru(chosen.unsqueeze(0), cur.unsqueeze(0)).squeeze(0)
            st.node_embs[cur_node] = new_cur
            st.g = model.graph_gru(new_cur.unsqueeze(0), st.g.unsqueeze(0)).squeeze(0)

            edges_done += 1

    n = len(st.node_embs)
    out = adj[:n, :n]
    np.fill_diagonal(out, 0)
    return out


@torch.no_grad()
def sample_many(model: DGMGMinimal, cfg: SampleConfig, num: int, device: str = "cpu") -> List[np.ndarray]:
    return [sample_graph(model, cfg, device=device) for _ in range(num)]
