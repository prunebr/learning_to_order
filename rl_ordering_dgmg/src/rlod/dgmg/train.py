from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any
import time
import torch
from torch.utils.data import DataLoader

from rlod.dgmg.dataset import DGMGSequenceDataset, collate_pad
from rlod.dgmg.model import DGMGMinimal, DGMGConfig


@dataclass
class TrainConfig:
    lr: float = 3e-4
    batch_size: int = 32
    epochs: int = 5
    grad_clip: float = 1.0
    device: str = "cpu"
    log_every: int = 100
    dgmg_lr: float = 5e-4


def train_dgmg(model: DGMGMinimal, train_loader: DataLoader, cfg: TrainConfig) -> Dict[str, Any]:
    device = torch.device(cfg.device)
    model.to(device)
    model.train()

    #opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.dgmg_lr)

    step = 0
    t0 = time.time()
    running = 0.0
    count = 0
    all_losses = []

    for epoch in range(1, cfg.epochs + 1):
        for batch in train_loader:
            step += 1
            actions = batch.actions.to(device)
            args = batch.args.to(device)
            lengths = batch.lengths.to(device)

            nll = model.batch_nll(actions, args, lengths)
            # média por sequência do batch (só para estabilidade numérica)
            loss = nll / actions.size(0)

            opt.zero_grad(set_to_none=True)
            loss.backward()

            if cfg.grad_clip and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            opt.step()

            running += float(loss.item())
            count += 1
            all_losses.append(float(loss.item()))

            if step % cfg.log_every == 0:
                dt = time.time() - t0
                print(f"[ep {epoch}] step {step:6d} loss={running/count:.4f} dt={dt:.1f}s")
                running = 0.0
                count = 0

    mean_loss = float(sum(all_losses) / max(1, len(all_losses)))
    return {
        "steps": step,
        "train_loss_mean": mean_loss,
        "train_loss_last": float(all_losses[-1]) if all_losses else 0.0,
    }
