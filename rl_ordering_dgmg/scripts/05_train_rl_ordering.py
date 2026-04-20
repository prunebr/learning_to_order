from __future__ import annotations

import argparse
from pathlib import Path
import time
import torch

from rlod.utils.io import load_pickle, save_pickle
from rlod.graphs.types import GraphSample
from rlod.rl.policy import GraphOrderingPolicy
from rlod.rl.reward import DummyReward
from rlod.rl.algos.reinforce import ReinforceTrainer, ReinforceConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")

    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--print_every", type=int, default=200)
    p.add_argument("--save_every", type=int, default=1000)

    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--entropy_beta", type=float, default=1e-3)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--baseline_ema", type=float, default=0.95)
    p.add_argument("--emb_dim", type=int, default=64)
    p.add_argument("--state_dim", type=int, default=64)
    p.add_argument("--gnn_layers", type=int, default=5)
    p.add_argument("--attn_heads", type=int, default=4)
    p.add_argument("--attn_dropout", type=float, default=0.0)
    p.add_argument("--start_node", type=str, default="degree", choices=["degree", "random"])

    p.add_argument("--ckpt_dir", type=str, default="runs/rl")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    data = load_pickle(args.data_path)  # lista de dicts
    samples = [GraphSample.from_dict(d) for d in data]

    policy = GraphOrderingPolicy(
        emb_dim=args.emb_dim,
        state_dim=args.state_dim,
        gnn_layers=args.gnn_layers,
        attn_heads=args.attn_heads,
        start_node=args.start_node,
        attn_dropout=args.attn_dropout,
    )

    cfg = ReinforceConfig(
        lr=args.lr,
        entropy_beta=args.entropy_beta,
        grad_clip=args.grad_clip,
        baseline_ema=args.baseline_ema,
    )
    trainer = ReinforceTrainer(policy, cfg=cfg, device=args.device)

    # reward stub (depois troca por reward do DGMG)
    reward_fn = DummyReward(value=0.0)

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    ema_reward = 0.0
    alpha = 0.98

    for step in range(1, args.steps + 1):
        s = samples[step % len(samples)]

        adj = torch.tensor(s.adj, dtype=torch.float32, device=trainer.device)
        out = trainer.policy.sample_order(adj)

        r = reward_fn(s, out.order).to(trainer.device)
        stats = trainer.update(r, out.logprob_sum, out.entropy_sum)

        ema_reward = alpha * ema_reward + (1 - alpha) * stats["reward"]

        if step % args.print_every == 0:
            dt = time.time() - t0
            print(
                f"[{step:6d}] loss={stats['loss']:.4f} "
                f"reward={stats['reward']:.4f} emaR={ema_reward:.4f} "
                f"ent={stats['entropy_sum']:.2f} dt={dt:.1f}s"
            )

        if step % args.save_every == 0:
            ckpt = {
                "step": step,
                "policy_state": trainer.policy.state_dict(),
                "cfg": cfg.__dict__,
                "policy_cfg": {
                    "emb_dim": args.emb_dim,
                    "state_dim": args.state_dim,
                    "gnn_layers": args.gnn_layers,
                    "attn_heads": args.attn_heads,
                    "start_node": args.start_node,
                    "attn_dropout": args.attn_dropout,
                },
                "seed": args.seed,
            }
            path = ckpt_dir / f"policy_step{step}.pt"
            torch.save(ckpt, path)
            print(f"Saved: {path}")

    # save final
    final_path = ckpt_dir / "policy_final.pt"
    torch.save(
        {
            "step": args.steps,
            "policy_state": trainer.policy.state_dict(),
            "cfg": cfg.__dict__,
            "policy_cfg": {
                "emb_dim": args.emb_dim,
                "state_dim": args.state_dim,
                "gnn_layers": args.gnn_layers,
                "attn_heads": args.attn_heads,
                "start_node": args.start_node,
                "attn_dropout": args.attn_dropout,
            },
            "seed": args.seed,
        },
        final_path
    )
    print(f"Saved final: {final_path}")


if __name__ == "__main__":
    main()
