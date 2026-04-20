# scripts/01_make_ba_dataset.py
import argparse
import gzip
import os
import pickle
from typing import List

import numpy as np
import networkx as nx


def save_pickle_gz(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def ba_graph_to_adj(n: int, m: int, rng: np.random.Generator, relabel: bool) -> np.ndarray:
    if not (m < n):
        raise ValueError(f"Need m < n, got n={n}, m={m}")

    g_seed = int(rng.integers(0, 2**32 - 1))
    G = nx.barabasi_albert_graph(n=n, m=m, seed=g_seed)

    if relabel:
        perm = rng.permutation(n)
        mapping = {i: int(perm[i]) for i in range(n)}
        G = nx.relabel_nodes(G, mapping, copy=True)

    A = nx.to_numpy_array(G, nodelist=list(range(n)), dtype=np.uint8)
    np.fill_diagonal(A, 0)
    A = ((A + A.T) > 0).astype(np.uint8)
    return A


def make_split(num_graphs: int, n_min: int, n_max: int, m: int, relabel: bool, rng: np.random.Generator) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for _ in range(num_graphs):
        n = int(rng.integers(n_min, n_max + 1))
        out.append(ba_graph_to_adj(n=n, m=m, rng=rng, relabel=relabel))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="data/raw")
    ap.add_argument("--n_min", type=int, default=20)
    ap.add_argument("--n_max", type=int, default=60)
    ap.add_argument("--m", type=int, default=2)

    ap.add_argument("--num_train", type=int, default=6000)
    ap.add_argument("--num_val", type=int, default=750)
    ap.add_argument("--num_test", type=int, default=750)

    ap.add_argument("--relabel", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    name = f"ba_m{args.m}_n{args.n_min}-{args.n_max}_relabel{int(args.relabel)}"
    train_path = os.path.join(args.out_dir, f"{name}_train.pkl.gz")
    val_path = os.path.join(args.out_dir, f"{name}_val.pkl.gz")
    test_path = os.path.join(args.out_dir, f"{name}_test.pkl.gz")

    train = make_split(args.num_train, args.n_min, args.n_max, args.m, args.relabel, rng)
    val = make_split(args.num_val, args.n_min, args.n_max, args.m, args.relabel, rng)
    test = make_split(args.num_test, args.n_min, args.n_max, args.m, args.relabel, rng)

    save_pickle_gz(train_path, train)
    save_pickle_gz(val_path, val)
    save_pickle_gz(test_path, test)

    print(f"OK: {name}")
    print(f" train: {len(train)}  val: {len(val)}  test: {len(test)}")
    print(" saved:", train_path)
    print(" saved:", val_path)
    print(" saved:", test_path)


if __name__ == "__main__":
    main()