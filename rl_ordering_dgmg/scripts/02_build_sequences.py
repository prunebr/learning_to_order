from __future__ import annotations

import argparse
from pathlib import Path
import torch

from rlod.utils.io import load_pickle, save_pickle
from rlod.graphs.types import GraphSample
from rlod.rl.policy import GraphOrderingPolicy
from rlod.sequences.builder import (
    build_decision_sequence,
    validate_sequence_matches_graph,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--in_path", type=str, required=True)
    p.add_argument("--out_path", type=str, required=True)

    p.add_argument("--policy_ckpt", type=str, required=True)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--mode", type=str, default="sample", choices=["sample", "greedy"])

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--validate", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="0 = sem limite")
    return p.parse_args()


def load_policy(ckpt_path: str, device: str, seed: int) -> GraphOrderingPolicy:
    torch.manual_seed(seed)
    ckpt = torch.load(ckpt_path, map_location=device)
    policy_cfg = ckpt.get("policy_cfg", {}) if isinstance(ckpt, dict) else {}
    policy = GraphOrderingPolicy(
        emb_dim=int(policy_cfg.get("emb_dim", 64)),
        state_dim=int(policy_cfg.get("state_dim", 64)),
        gnn_layers=int(policy_cfg.get("gnn_layers", 5)),
        attn_heads=int(policy_cfg.get("attn_heads", 4)),
        start_node=str(policy_cfg.get("start_node", "degree")),
        attn_dropout=float(policy_cfg.get("attn_dropout", 0.0)),
    ).to(device)

    state = ckpt["policy_state"] if isinstance(ckpt, dict) and "policy_state" in ckpt else ckpt
    policy.load_state_dict(state)
    policy.eval()
    return policy


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    raw = load_pickle(args.in_path)  # lista de dicts GraphSample.to_dict()
    samples = [GraphSample.from_dict(d) for d in raw]

    if args.limit and args.limit > 0:
        samples = samples[: args.limit]

    policy = load_policy(args.policy_ckpt, device=str(device), seed=args.seed)

    out_records = []
    for i, s in enumerate(samples):
        adj = torch.tensor(s.adj, dtype=torch.float32, device=device)

        if args.mode == "greedy":
            order = policy.greedy_order(adj)
        else:
            with torch.no_grad():
                order = policy.sample_order(adj).order

        seq = build_decision_sequence(s, order)

        if args.validate:
            validate_sequence_matches_graph(s, seq)

        out_records.append(seq.to_dict())

        if (i + 1) % 200 == 0:
            print(f"Processed {i+1}/{len(samples)}")

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_pickle(out_records, out_path, compress=True)

    print(f"OK: saved {len(out_records)} sequences -> {out_path}")


if __name__ == "__main__":
    main()
