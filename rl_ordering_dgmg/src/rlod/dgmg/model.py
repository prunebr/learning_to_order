from __future__ import annotations

from dataclasses import dataclass
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from rlod.sequences.actions import ActionType


@dataclass
class DGMGConfig:
    hidden_dim: int = 64
    node_init_dim: int = 64


class DGMGMinimal(nn.Module):
    """
    DGMG minimalista (teacher forcing) com cabeça de STOP_NODE.

    Heads:
      - node_head: decide ADD_NODE vs STOP_NODE
      - edge_head: decide ADD_EDGE vs STOP_EDGE
      - dest_head: escolhe destino (0..cur_node-1)
    """
    def __init__(self, cfg: DGMGConfig):
        super().__init__()
        self.cfg = cfg
        H = cfg.hidden_dim

        # init node + updates
        self.node_init = nn.Parameter(torch.zeros(cfg.node_init_dim))
        self.node_init_proj = nn.Linear(cfg.node_init_dim, H)

        self.node_gru = nn.GRUCell(input_size=H, hidden_size=H)
        self.graph_gru = nn.GRUCell(input_size=H, hidden_size=H)

        # heads
        self.node_mlp = nn.Sequential(
            nn.Linear(H, H),
            nn.ReLU(),
            nn.Linear(H, 2),  # 0=ADD_NODE, 1=STOP_NODE
        )

        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * H, H),
            nn.ReLU(),
            nn.Linear(H, 2),  # 0=ADD_EDGE, 1=STOP_EDGE
        )

        self.dest_q = nn.Linear(2 * H, H)
        self.dest_k = nn.Linear(H, H)

        # initial global state
        self.graph0 = nn.Parameter(torch.zeros(H))

    # ---------- teacher-forcing NLL ----------
    def forward_nll(self, actions: torch.Tensor, args: torch.Tensor) -> torch.Tensor:
        """
        actions: (L,)
        args:    (L,)
        retorna NLL escalar (somatório)
        """
        device = actions.device
        g = self.graph0.to(device)

        node_embs: List[torch.Tensor] = []
        cur_node = -1
        pending_edge = False

        nll = torch.zeros((), device=device)

        for a, arg in zip(actions.tolist(), args.tolist()):
            if a == -1:
                break

            at = ActionType(int(a))

            if at == ActionType.ADD_NODE:
                # node decision BEFORE creating node
                logits_node = self.node_mlp(g)
                nll = nll + F.cross_entropy(logits_node.unsqueeze(0), torch.tensor([0], device=device))

                cur_node += 1
                h0 = self.node_init_proj(self.node_init.to(device))
                node_embs.append(h0)
                g = self.graph_gru(h0.unsqueeze(0), g.unsqueeze(0)).squeeze(0)
                pending_edge = False

            elif at == ActionType.STOP_NODE:
                logits_node = self.node_mlp(g)
                nll = nll + F.cross_entropy(logits_node.unsqueeze(0), torch.tensor([1], device=device))
                break

            elif at == ActionType.ADD_EDGE:
                if cur_node < 0:
                    raise ValueError("ADD_EDGE antes de ADD_NODE")
                cur = node_embs[cur_node]
                logits = self.edge_mlp(torch.cat([g, cur], dim=0))
                nll = nll + F.cross_entropy(logits.unsqueeze(0), torch.tensor([0], device=device))
                pending_edge = True

            elif at == ActionType.STOP_EDGE:
                if cur_node < 0:
                    raise ValueError("STOP_EDGE antes de ADD_NODE")
                cur = node_embs[cur_node]
                logits = self.edge_mlp(torch.cat([g, cur], dim=0))
                nll = nll + F.cross_entropy(logits.unsqueeze(0), torch.tensor([1], device=device))
                pending_edge = False

            elif at == ActionType.CHOOSE_DEST:
                if not pending_edge:
                    raise ValueError("CHOOSE_DEST sem ADD_EDGE anterior.")
                if cur_node <= 0:
                    raise ValueError("CHOOSE_DEST com cur_node <= 0")

                dest = int(arg)
                if dest < 0 or dest >= cur_node:
                    raise ValueError(f"dest inválido {dest} (cur_node={cur_node})")

                cur = node_embs[cur_node]
                prev = torch.stack(node_embs[:cur_node], dim=0)  # (cur_node, H)
                q = self.dest_q(torch.cat([g, cur], dim=0))       # (H,)
                k = self.dest_k(prev)                              # (cur_node, H)
                logits = (k @ q)                                   # (cur_node,)

                nll = nll + F.cross_entropy(logits.unsqueeze(0), torch.tensor([dest], device=device))

                # update (simples) nó atual condicionado ao dest
                chosen = prev[dest]
                new_cur = self.node_gru(chosen.unsqueeze(0), cur.unsqueeze(0)).squeeze(0)
                node_embs[cur_node] = new_cur
                g = self.graph_gru(new_cur.unsqueeze(0), g.unsqueeze(0)).squeeze(0)

                pending_edge = False

            else:
                raise ValueError(f"Ação desconhecida: {a}")

        return nll

    def batch_nll(self, actions_batch: torch.Tensor, args_batch: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        B = actions_batch.size(0)
        nll = torch.zeros((), device=actions_batch.device)
        for i in range(B):
            L = int(lengths[i].item())
            nll = nll + self.forward_nll(actions_batch[i, :L], args_batch[i, :L])
        return nll
