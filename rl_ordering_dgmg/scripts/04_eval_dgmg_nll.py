from __future__ import annotations

import argparse
import torch
from torch.utils.data import DataLoader

from rlod.utils.io import load_pickle
from rlod.dgmg.dataset import DGMGSequenceDataset, collate_pad
from rlod.dgmg.model import DGMGMinimal, DGMGConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--seq_path", type=str, required=True)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--batch_size", type=int, default=64)
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    ckpt = torch.load(args.ckpt, map_location=device)
    cfg_dict = ckpt.get("cfg", {"hidden_dim": 64, "node_init_dim": 64})
    cfg = DGMGConfig(**cfg_dict)

    model = DGMGMinimal(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    records = load_pickle(args.seq_path)
    ds = DGMGSequenceDataset(records)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_pad)

    total_nll = 0.0
    total_seqs = 0

    for batch in loader:
        actions = batch.actions.to(device)
        args_t = batch.args.to(device)
        lengths = batch.lengths.to(device)

        nll = model.batch_nll(actions, args_t, lengths)  # soma do batch
        total_nll += float(nll.item())
        total_seqs += int(actions.size(0))

    mean_nll = total_nll / max(1, total_seqs)
    print(f"mean_nll_per_graph = {mean_nll:.6f} over {total_seqs} sequences")


if __name__ == "__main__":
    main()
