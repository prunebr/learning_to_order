from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List
import numpy as np
import torch

from rlod.graphs.types import GraphSample
from rlod.sequences.builder import build_decision_sequence
from rlod.dgmg.model import DGMGMinimal, DGMGConfig
from rlod.dgmg.sample import sample_graph_from_prefix, SampleConfig
from rlod.graphs.validators import is_connected_from_adj, is_tree_from_adj, is_binary_tree_undirected
from rlod.sequences.actions import ActionType


class RewardProvider(ABC):
    @abstractmethod
    def __call__(self, sample: GraphSample, order: List[int]) -> torch.Tensor:
        ...


@dataclass
class DummyReward(RewardProvider):
    value: float = 0.0

    def __call__(self, sample: GraphSample, order: List[int]) -> torch.Tensor:
        return torch.tensor(float(self.value))


class DGMGNLLReward(RewardProvider):
    def __init__(self, ckpt_path: str, device: str = "cpu", nll_scale: float = 1.0):
        self.device = torch.device(device)
        ckpt = torch.load(ckpt_path, map_location=self.device)
        cfg_dict = ckpt.get("cfg", {"hidden_dim": 64, "node_init_dim": 64})
        cfg = DGMGConfig(**cfg_dict)
        self.model = DGMGMinimal(cfg).to(self.device)
        # compatível com checkpoints antigos
        self.model.load_state_dict(ckpt["model_state"], strict=False)
        self.model.eval()
        self.nll_scale = float(nll_scale)

    @torch.no_grad()
    def __call__(self, sample: GraphSample, order: List[int]) -> torch.Tensor:
        seq = build_decision_sequence(sample, order)
        actions = torch.tensor(seq.actions, dtype=torch.long, device=self.device)
        args = torch.tensor(seq.args, dtype=torch.long, device=self.device)
        nll = self.model.forward_nll(actions, args)
        return -self.nll_scale * nll


def _safe_prefix_len(actions: np.ndarray, max_len: int) -> int:
    """
    Evita cortar prefixo no meio de um par ADD_EDGE/CHOOSE_DEST.
    Retorna um len <= max_len em um ponto 'seguro' (pending_edge=False).
    """
    pending = False
    last_ok = 0
    L = min(max_len, len(actions))
    for i in range(L):
        a = int(actions[i])
        if a == -1:
            break
        at = ActionType(a)
        if at == ActionType.ADD_EDGE:
            pending = True
        elif at in (ActionType.CHOOSE_DEST, ActionType.STOP_EDGE, ActionType.ADD_NODE, ActionType.STOP_NODE):
            pending = False
        if not pending:
            last_ok = i + 1
        if at == ActionType.STOP_NODE:
            return i + 1
    return max(1, last_ok)


class DGMGNLLValidityReward(RewardProvider):
    """
    Reward = -NLL(seq) + lambda_valid * validity(prefix-completion)

    validity é estimada gerando K completions do DGMG condicionadas a um prefixo
    da sequência induzida pela ordenação (dependente da ordenação).
    """
    def __init__(
        self,
        ckpt_path: str,
        device: str = "cpu",
        nll_scale: float = 1.0,
        lambda_valid: float = 5.0,
        valid_samples: int = 3,
        prefix_ratio: float = 0.5,
        min_nodes: int = 5,
        max_nodes: int = 20,
        w_connected: float = 0.2,
        w_tree: float = 0.5,
        w_binary: float = 1.0,
        family: str = "binary",
    ):
        self.device = torch.device(device)
        ckpt = torch.load(ckpt_path, map_location=self.device)
        cfg_dict = ckpt.get("cfg", {"hidden_dim": 64, "node_init_dim": 64})
        cfg = DGMGConfig(**cfg_dict)

        self.model = DGMGMinimal(cfg).to(self.device)
        self.model.load_state_dict(ckpt["model_state"], strict=False)
        self.model.eval()

        self.nll_scale = float(nll_scale)
        self.lambda_valid = float(lambda_valid)
        self.valid_samples = int(valid_samples)
        self.prefix_ratio = float(prefix_ratio)

        self.sample_cfg = SampleConfig(min_nodes=min_nodes, max_nodes=max_nodes, greedy=False, temperature=1.0)
        self.wc = float(w_connected)
        self.wt = float(w_tree)
        self.wb = float(w_binary)
        self.wsum = self.wc + self.wt + self.wb
        self.family = family

    @torch.no_grad()
    def __call__(self, sample: GraphSample, order: List[int]) -> torch.Tensor:
        seq = build_decision_sequence(sample, order)

        actions_t = torch.tensor(seq.actions, dtype=torch.long, device=self.device)
        args_t = torch.tensor(seq.args, dtype=torch.long, device=self.device)

        nll = self.model.forward_nll(actions_t, args_t)

        # prefix len seguro
        raw_len = len(seq.actions)
        pref = int(self.prefix_ratio * raw_len)
        pref = max(1, min(pref, raw_len))
        pref = _safe_prefix_len(seq.actions, pref)

        # completions e validade
        scores = []
        for _ in range(self.valid_samples):
            adj = sample_graph_from_prefix(
                self.model,
                prefix_actions=seq.actions,
                prefix_args=seq.args,
                prefix_len=pref,
                cfg=self.sample_cfg,
                device=str(self.device),
            )
            conn = 1.0 if is_connected_from_adj(adj) else 0.0
            if self.family == "ba":
                    n = int(adj.shape[0])
                    valid_size = 1.0 if (self.sample_cfg.min_nodes <= n <= self.sample_cfg.max_nodes) else 0.0
                    score = valid_size * conn
            else:
                tree = 1.0 if is_tree_from_adj(adj) else 0.0
                binary = 1.0 if is_binary_tree_undirected(adj, root=0) else 0.0
                score = (self.wc * conn + self.wt * tree + self.wb * binary) / self.wsum

            scores.append(score)

        valid_bonus = float(np.mean(scores)) if scores else 0.0

        reward = (-self.nll_scale * nll) + (self.lambda_valid * torch.tensor(valid_bonus, device=self.device))
        return reward
