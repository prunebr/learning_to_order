from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from rlod.graphs.binary_trees import BinaryTreeSource, BinaryTreeSourceConfig
from rlod.utils.io import save_pickle


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, default="data/raw")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_graphs", type=int, default=4000)

    p.add_argument("--n_min", type=int, default=5)
    p.add_argument("--n_max", type=int, default=20)
    p.add_argument("--mode", type=str, default="any", choices=["any", "full", "perfect"])
    p.add_argument("--relabel", action="store_true")

    p.add_argument("--train_frac", type=float, default=0.8)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--test_frac", type=float, default=0.1)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    fr = args.train_frac + args.val_frac + args.test_frac
    if abs(fr - 1.0) > 1e-9:
        raise ValueError("train_frac + val_frac + test_frac deve somar 1.0")

    cfg = BinaryTreeSourceConfig(
        n_min=args.n_min,
        n_max=args.n_max,
        mode=args.mode,
        relabel_nodes=args.relabel,
    )
    source = BinaryTreeSource(cfg=cfg, seed=args.seed)

    samples = [source.sample().to_dict() for _ in range(args.num_graphs)]

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(args.num_graphs)

    n_train = int(args.train_frac * args.num_graphs)
    n_val = int(args.val_frac * args.num_graphs)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    train = [samples[i] for i in train_idx]
    val = [samples[i] for i in val_idx]
    test = [samples[i] for i in test_idx]

    out_dir = Path(args.out_dir)
    tag = f"binary_trees_{args.mode}_n{args.n_min}-{args.n_max}_relabel{int(args.relabel)}"

    save_pickle(train, out_dir / f"{tag}_train.pkl.gz", compress=True)
    save_pickle(val, out_dir / f"{tag}_val.pkl.gz", compress=True)
    save_pickle(test, out_dir / f"{tag}_test.pkl.gz", compress=True)

    print(f"OK: {tag}")
    print(f" train: {len(train)}  val: {len(val)}  test: {len(test)}")
    print(f" out: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
