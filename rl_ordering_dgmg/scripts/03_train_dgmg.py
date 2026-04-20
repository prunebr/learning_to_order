from __future__ import annotations

import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from rlod.utils.io import load_pickle
from rlod.dgmg.dataset import DGMGSequenceDataset, collate_pad
from rlod.dgmg.model import DGMGMinimal, DGMGConfig
from rlod.dgmg.train import train_dgmg, TrainConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train_seq", type=str, required=True)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ckpt_dir", type=str, default="runs/dgmg")
    p.add_argument("--log_every", type=int, default=100)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    records = load_pickle(args.train_seq)
    ds = DGMGSequenceDataset(records)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_pad)

    model = DGMGMinimal(DGMGConfig(hidden_dim=64, node_init_dim=64))
    cfg = TrainConfig(
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=args.device,
        log_every=args.log_every,
    )

    train_dgmg(model, loader, cfg)

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / "dgmg_minimal.pt"
    torch.save({"model_state": model.state_dict(), "cfg": model.cfg.__dict__}, path)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
