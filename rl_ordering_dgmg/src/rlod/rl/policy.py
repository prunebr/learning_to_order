from __future__ import annotations

from dataclasses import dataclass
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_adj(adj: torch.Tensor) -> torch.Tensor:
    """
    A_norm = D^-1/2 (A + I) D^-1/2
    adj: (n,n) float
    """
    n = adj.size(0)
    a = adj + torch.eye(n, device=adj.device, dtype=adj.dtype)
    deg = a.sum(dim=1)  # (n,)
    deg_inv_sqrt = torch.pow(deg.clamp(min=1.0), -0.5)
    D = torch.diag(deg_inv_sqrt)
    return D @ a @ D


class GraphAttentionBlock(nn.Module):
    def __init__(self, emb_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=emb_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln1 = nn.LayerNorm(emb_dim)
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2),
            nn.ReLU(),
            nn.Linear(emb_dim * 2, emb_dim),
        )
        self.ln2 = nn.LayerNorm(emb_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x: (n, d), adj: (n, n)
        Atenção mascarada para respeitar apenas vizinhos + self-loop.
        """
        n = adj.size(0)
        eye = torch.eye(n, device=adj.device, dtype=adj.dtype)
        allowed = (adj + eye) > 0
        attn_mask = ~allowed  # True = posição bloqueada no MultiheadAttention

        h, _ = self.attn(
            x.unsqueeze(0),
            x.unsqueeze(0),
            x.unsqueeze(0),
            attn_mask=attn_mask,
            need_weights=False,
        )
        h = h.squeeze(0)
        x = self.ln1(x + h)
        x = self.ln2(x + self.ffn(x))
        return x


class GraphEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int = 2,
        emb_dim: int = 64,
        num_layers: int = 5,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, emb_dim)
        self.layers = nn.ModuleList(
            [GraphAttentionBlock(emb_dim=emb_dim, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)]
        )

    def forward(self, adj: torch.Tensor) -> torch.Tensor:
        """
        adj: (n,n) float
        return: node_emb (n, d)
        """
        n = adj.size(0)
        deg = adj.sum(dim=1)  # (n,)
        deg_norm = deg / deg.clamp(min=1.0).max().clamp(min=1.0)
        x = torch.stack([torch.ones(n, device=adj.device), deg_norm], dim=1)  # (n,2)

        # mistura um "GCN smoothing" leve com blocos de atenção
        a_norm = normalize_adj(adj)
        h = self.input_proj(x)
        h = F.relu(a_norm @ h)
        for layer in self.layers:
            h = layer(h, adj)
        return h


class PointerDecoder(nn.Module):
    """
    Decoder tipo 'pointer' com estado recorrente:
      state_{t+1} = GRU(state_t, emb[action_t])
      logits_i = v^T tanh(W_h emb_i + W_s state_t)
    """
    def __init__(self, emb_dim: int = 64, state_dim: int = 64):
        super().__init__()
        self.gru = nn.GRUCell(input_size=emb_dim, hidden_size=state_dim)

        self.W_h = nn.Linear(emb_dim, state_dim, bias=False)
        self.W_s = nn.Linear(state_dim, state_dim, bias=True)
        self.v = nn.Linear(state_dim, 1, bias=False)

        self.state0 = nn.Parameter(torch.zeros(state_dim))

    def init_state(self, device: torch.device) -> torch.Tensor:
        return self.state0.to(device)

    def logits(self, node_emb: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        # node_emb: (n,d), state: (d,)
        n = node_emb.size(0)
        s = state.unsqueeze(0).expand(n, -1)  # (n,d)
        z = torch.tanh(self.W_h(node_emb) + self.W_s(s))
        return self.v(z).squeeze(-1)  # (n,)

    def step_state(self, state: torch.Tensor, chosen_emb: torch.Tensor) -> torch.Tensor:
        return self.gru(chosen_emb.unsqueeze(0), state.unsqueeze(0)).squeeze(0)


@dataclass
class PolicyOutput:
    order: List[int]
    logprob_sum: torch.Tensor
    entropy_sum: torch.Tensor


class GraphOrderingPolicy(nn.Module):
    def __init__(
        self,
        emb_dim: int = 64,
        state_dim: int = 64,
        gnn_layers: int = 5,
        attn_heads: int = 4,
        start_node: str = "degree",
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        self.encoder = GraphEncoder(
            in_dim=2,
            emb_dim=emb_dim,
            num_layers=gnn_layers,
            num_heads=attn_heads,
            dropout=attn_dropout,
        )
        self.decoder = PointerDecoder(emb_dim=emb_dim, state_dim=state_dim)
        self.start_node = start_node

    def _pick_start_node(self, adj: torch.Tensor, mask: torch.Tensor) -> int:
        if self.start_node == "random":
            candidates = torch.where(mask)[0]
            idx = int(torch.randint(0, candidates.numel(), (1,), device=adj.device).item())
            return int(candidates[idx].item())
        if self.start_node != "degree":
            logits = torch.zeros(mask.shape[0], device=adj.device).masked_fill(~mask, -1e9)
            return int(torch.argmax(logits).item())
        deg = adj.sum(dim=1)
        deg = deg.masked_fill(~mask, -1e9)
        return int(torch.argmax(deg).item())

    @torch.no_grad()
    def greedy_order(self, adj: torch.Tensor) -> List[int]:
        self.eval()
        node_emb = self.encoder(adj)
        n = adj.size(0)
        mask = torch.ones(n, dtype=torch.bool, device=adj.device)
        state = self.decoder.init_state(adj.device)

        out: List[int] = []
        for _ in range(n):
            if len(out) == 0:
                a = self._pick_start_node(adj, mask)
            else:
                logits = self.decoder.logits(node_emb, state)
                logits = logits.masked_fill(~mask, -1e9)
                a = int(torch.argmax(logits).item())
            out.append(a)
            mask[a] = False
            state = self.decoder.step_state(state, node_emb[a])
        return out

    def sample_order(self, adj: torch.Tensor) -> PolicyOutput:
        """
        Amostra uma permutação completa.
        Retorna soma de logprobs e entropias (útil no REINFORCE).
        """
        self.train()
        node_emb = self.encoder(adj)
        n = adj.size(0)
        mask = torch.ones(n, dtype=torch.bool, device=adj.device)
        state = self.decoder.init_state(adj.device)

        order: List[int] = []
        logprob_sum = torch.zeros((), device=adj.device)
        entropy_sum = torch.zeros((), device=adj.device)

        for _ in range(n):
            if len(order) == 0:
                ai = self._pick_start_node(adj, mask)
            else:
                logits = self.decoder.logits(node_emb, state)
                logits = logits.masked_fill(~mask, -1e9)

                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample()
                logprob_sum = logprob_sum + dist.log_prob(a)
                entropy_sum = entropy_sum + dist.entropy()
                ai = int(a.item())
            order.append(ai)

            mask[ai] = False
            state = self.decoder.step_state(state, node_emb[ai])

        return PolicyOutput(order=order, logprob_sum=logprob_sum, entropy_sum=entropy_sum)
