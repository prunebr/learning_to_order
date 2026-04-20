from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
import numpy as np
import torch
from torch.utils.data import Dataset

from rlod.sequences.builder import DecisionSequence


@dataclass
class SeqBatch:
    actions: torch.Tensor   # (B, L) int64
    args: torch.Tensor      # (B, L) int64
    n_nodes: torch.Tensor   # (B,) int64
    lengths: torch.Tensor   # (B,) int64


class DGMGSequenceDataset(Dataset):
    """
    Dataset de DecisionSequence serializadas (dicts).
    Retorna tensors (actions, args) por amostra.
    """
    def __init__(self, records: List[Dict[str, Any]]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        d = self.records[idx]
        seq = DecisionSequence.from_dict(d)
        return {
            "actions": torch.tensor(seq.actions, dtype=torch.long),
            "args": torch.tensor(seq.args, dtype=torch.long),
            "n_nodes": torch.tensor(seq.n_nodes, dtype=torch.long),
            "length": torch.tensor(len(seq.actions), dtype=torch.long),
        }


def collate_pad(batch: List[Dict[str, Any]], pad_value_action: int = -1, pad_value_arg: int = -1) -> SeqBatch:
    """
    Padding para batch. Ações padded com -1 (ignoradas na loss).
    """
    B = len(batch)
    lengths = torch.stack([b["length"] for b in batch], dim=0)  # (B,)
    Lmax = int(lengths.max().item())

    actions = torch.full((B, Lmax), pad_value_action, dtype=torch.long)
    args = torch.full((B, Lmax), pad_value_arg, dtype=torch.long)
    n_nodes = torch.stack([b["n_nodes"] for b in batch], dim=0)

    for i, b in enumerate(batch):
        L = int(b["length"].item())
        actions[i, :L] = b["actions"]
        args[i, :L] = b["args"]

    return SeqBatch(actions=actions, args=args, n_nodes=n_nodes, lengths=lengths)
