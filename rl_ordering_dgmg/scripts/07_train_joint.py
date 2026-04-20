# scripts/07_train_joint.py  (your file)
from __future__ import annotations

import argparse
from rlod.joint.loop import JointConfig, run_joint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_train", type=str, required=True)
    p.add_argument("--raw_val", type=str, required=True)
    p.add_argument("--raw_test", type=str, required=True)

    p.add_argument("--out_dir", type=str, default="runs/joint")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--outer_iters", type=int, default=10)
    p.add_argument("--rl_steps_per_iter", type=int, default=3000)
    p.add_argument("--dgmg_epochs_per_iter", type=int, default=2)
    p.add_argument("--rl_emb_dim", type=int, default=64)
    p.add_argument("--rl_state_dim", type=int, default=64)
    p.add_argument("--rl_gnn_layers", type=int, default=5)
    p.add_argument("--rl_attn_heads", type=int, default=4)
    p.add_argument("--rl_attn_dropout", type=float, default=0.0)
    p.add_argument("--rl_start_node", type=str, default="degree", choices=["degree", "random"])

    p.add_argument("--gen_samples", type=int, default=1000)
    p.add_argument("--n_min", type=int, default=5)
    p.add_argument("--n_max", type=int, default=20)

    p.add_argument("--early_stop_patience", type=int, default=3)
    p.add_argument("--early_stop_min_delta", type=float, default=0.2)

    p.add_argument("--seq_mode", type=str, default="sample", choices=["sample", "greedy"])

    p.add_argument("--dgmg_init_ckpt", type=str, default="")
    p.add_argument("--lambda_valid", type=float, default=5.0)
    p.add_argument("--valid_samples", type=int, default=3)
    p.add_argument("--prefix_ratio", type=float, default=0.5)

    p.add_argument("--eval_every", type=int, default=10)

    # NEW:
    p.add_argument("--family", type=str, default="binary", choices=["binary", "ba"])
    p.add_argument("--dgmg_lr", type=float, default=5e-4)

    return p.parse_args()


def main():
    a = parse_args()
    cfg = JointConfig(
        raw_train=a.raw_train,
        raw_val=a.raw_val,
        raw_test=a.raw_test,
        out_dir=a.out_dir,
        device=a.device,
        seed=a.seed,
        outer_iters=a.outer_iters,
        rl_steps_per_iter=a.rl_steps_per_iter,
        dgmg_epochs_per_iter=a.dgmg_epochs_per_iter,
        rl_emb_dim=a.rl_emb_dim,
        rl_state_dim=a.rl_state_dim,
        rl_gnn_layers=a.rl_gnn_layers,
        rl_attn_heads=a.rl_attn_heads,
        rl_attn_dropout=a.rl_attn_dropout,
        rl_start_node=a.rl_start_node,
        gen_samples=a.gen_samples,
        n_min=a.n_min,
        n_max=a.n_max,
        early_stop_patience=a.early_stop_patience,
        early_stop_min_delta=a.early_stop_min_delta,
        seq_mode=a.seq_mode,
        dgmg_init_ckpt=a.dgmg_init_ckpt,
        lambda_valid=a.lambda_valid,
        valid_samples=a.valid_samples,
        prefix_ratio=a.prefix_ratio,
        eval_every=a.eval_every,

        # NEW:
        family=a.family,
        dgmg_lr=a.dgmg_lr,
    )
    run_joint(cfg)


if __name__ == "__main__":
    main()