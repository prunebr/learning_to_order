from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import torch

from rlod.graphs.types import GraphSample


@dataclass
class OrderingState:
    adj: torch.Tensor          # (n, n) float32
    mask: torch.Tensor         # (n,) bool  True = disponível
    selected: List[int]        # lista de nós já escolhidos


class NodeOrderingEnv:
    """
    Episódio: construir uma permutação de nós.
    - reset(sample) prepara adj + máscara
    - step(action) marca o nó como escolhido
    """
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.state: Optional[OrderingState] = None

    def reset(self, sample: GraphSample) -> OrderingState:
        adj = torch.tensor(sample.adj, dtype=torch.float32, device=self.device)
        n = sample.n_nodes
        mask = torch.ones(n, dtype=torch.bool, device=self.device)
        self.state = OrderingState(adj=adj, mask=mask, selected=[])
        return self.state

    def step(self, action: int) -> Tuple[OrderingState, bool]:
        assert self.state is not None, "Chame reset() antes de step()."
        n = self.state.mask.shape[0]
        if action < 0 or action >= n:
            raise ValueError(f"Action inválida: {action} para n={n}")
        if not bool(self.state.mask[action]):
            raise ValueError(f"Action inválida: nó {action} já foi escolhido.")

        self.state.mask[action] = False
        self.state.selected.append(int(action))

        done = (len(self.state.selected) == n)
        return self.state, done

    def get_action_mask(self) -> torch.Tensor:
        assert self.state is not None
        return self.state.mask
