from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional
import csv
import time
import statistics
import numpy as np
import torch
from torch.utils.data import DataLoader

from rlod.utils.io import load_pickle, save_pickle
from rlod.graphs.types import GraphSample

from rlod.rl.policy import GraphOrderingPolicy
from rlod.rl.algos.reinforce import ReinforceTrainer, ReinforceConfig
from rlod.rl.reward import DGMGNLLValidityReward

from rlod.sequences.builder import build_decision_sequence
from rlod.dgmg.dataset import DGMGSequenceDataset, collate_pad
from rlod.dgmg.model import DGMGMinimal, DGMGConfig
from rlod.dgmg.train import train_dgmg, TrainConfig

from rlod.dgmg.sample import sample_many, SampleConfig
from rlod.dgmg.eval import evaluate_generated, EvalConfig

from rlod.eval.ba_metrics import evaluate_generated_ba


@dataclass
class JointConfig:
    raw_train: str
    raw_val: str
    raw_test: str

    out_dir: str = "runs/joint"
    seed: int = 42
    device: str = "cpu"

    outer_iters: int = 10

    # RL
    rl_steps_per_iter: int = 3000
    rl_print_every: int = 300
    rl_entropy_beta: float = 1e-3
    rl_lr: float = 3e-4
    rl_emb_dim: int = 64
    rl_state_dim: int = 64
    rl_gnn_layers: int = 5
    rl_attn_heads: int = 4
    rl_attn_dropout: float = 0.0
    rl_start_node: str = "degree"

    # DGMG
    dgmg_epochs_per_iter: int = 2
    dgmg_batch_size: int = 32
    dgmg_lr: float = 5e-4
    family: str = "binary"   # "binary" or "ba"

    # sampling / eval
    gen_samples: int = 1000
    n_min: int = 5
    n_max: int = 20

    # early stopping (por val_nll)
    early_stop_patience: int = 3
    early_stop_min_delta: float = 0.2

    # sequência gerada para treino do DGMG (sample vs greedy)
    seq_mode: str = "sample"  # "sample" ou "greedy"

        # reward validade
    lambda_valid: float = 5.0
    valid_samples: int = 3
    prefix_ratio: float = 0.5

    # init opcional
    dgmg_init_ckpt: str = ""

    eval_every: int = 10




def _build_sequences_from_policy(samples: List[GraphSample], policy: GraphOrderingPolicy, device: str, mode: str) -> List[Dict[str, Any]]:
    dev = torch.device(device)
    out = []
    policy.eval()

    for s in samples:
        adj = torch.tensor(s.adj, dtype=torch.float32, device=dev)
        if mode == "greedy":
            order = policy.greedy_order(adj)
        else:
            with torch.no_grad():
                order = policy.sample_order(adj).order

        seq = build_decision_sequence(s, order)
        out.append(seq.to_dict())
    return out


@torch.no_grad()
def _mean_nll_for_policy(model: DGMGMinimal, samples: List[GraphSample], policy: GraphOrderingPolicy, device: str, mode: str) -> float:
    dev = torch.device(device)
    model.eval()
    policy.eval()

    total = 0.0
    for s in samples:
        adj = torch.tensor(s.adj, dtype=torch.float32, device=dev)
        if mode == "greedy":
            order = policy.greedy_order(adj)
        else:
            order = policy.sample_order(adj).order

        seq = build_decision_sequence(s, order)
        actions = torch.tensor(seq.actions, dtype=torch.long, device=dev)
        args = torch.tensor(seq.args, dtype=torch.long, device=dev)
        nll = model.forward_nll(actions, args)
        total += float(nll.item())

    return total / max(1, len(samples))


def run_joint(cfg: JointConfig) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # carregar raw
    raw_train = [GraphSample.from_dict(d) for d in load_pickle(cfg.raw_train)]
    raw_val = [GraphSample.from_dict(d) for d in load_pickle(cfg.raw_val)]
    raw_test = [GraphSample.from_dict(d) for d in load_pickle(cfg.raw_test)]

    # inicializa modelos
    policy = GraphOrderingPolicy(
        emb_dim=cfg.rl_emb_dim,
        state_dim=cfg.rl_state_dim,
        gnn_layers=cfg.rl_gnn_layers,
        attn_heads=cfg.rl_attn_heads,
        start_node=cfg.rl_start_node,
        attn_dropout=cfg.rl_attn_dropout,
    )
    rl_trainer = ReinforceTrainer(
        policy,
        ReinforceConfig(lr=cfg.rl_lr, entropy_beta=cfg.rl_entropy_beta),
        device=cfg.device,
    )

    dgmg = DGMGMinimal(DGMGConfig(hidden_dim=64, node_init_dim=64)).to(cfg.device)

    # se quiser iniciar com um checkpoint dgmg já treinado, você pode carregar aqui
    # (por enquanto, assume que você já treinou e tem runs/dgmg/dgmg_minimal.pt,
    # mas o loop também pode começar do zero)
    init_path = Path(cfg.dgmg_init_ckpt) if cfg.dgmg_init_ckpt else Path("runs/dgmg/dgmg_minimal.pt")
    if init_path.exists():
        ck = torch.load(init_path, map_location=cfg.device)
        dgmg.load_state_dict(ck["model_state"], strict=False)
        dgmg.eval()


    # log CSV
    csv_path = out_dir / "eval_history.csv"
    fieldnames = [
        "iter",
        "rl_steps_total",
        "dgmg_epochs_total",
        "rl_loss_mean",
        "rl_reward_mean",
        "rl_entropy_mean",
        "rl_adv_mean",
        "rl_logprob_mean",
        "dgmg_train_loss_mean",
        "dgmg_train_loss_last",
        "val_nll_greedy",
        "test_nll_greedy",
        "gen_valid_ratio",
        "gen_tree_ratio",
        "gen_binary_ratio",
        "gen_connected_ratio",
        "gen_valid_size_ratio",
        "gen_avg_nodes",
        "gen_std_nodes",
        "gen_avg_edges",
        "gen_avg_gini_degree",
        "gen_hubiness",
    ]
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()

    best_val = float("inf")
    bad = 0
    rl_steps_total = 0
    dgmg_epochs_total = 0

    for it in range(1, cfg.outer_iters + 1):
        print(f"\n===== JOINT ITER {it}/{cfg.outer_iters} =====")

        # ---- 1) RL update com reward do DGMG atual ----
        reward_fn = DGMGNLLValidityReward(
            ckpt_path=str(_save_dgmg_ckpt(dgmg, out_dir / f"dgmg_iter{it-1}.pt")),
            device=cfg.device,
            lambda_valid=cfg.lambda_valid,
            valid_samples=cfg.valid_samples,
            prefix_ratio=cfg.prefix_ratio,
            min_nodes=cfg.n_min,
            max_nodes=cfg.n_max,
        )

        t0 = time.time()
        rl_loss_values: List[float] = []
        rl_reward_values: List[float] = []
        rl_entropy_values: List[float] = []
        rl_adv_values: List[float] = []
        rl_logprob_values: List[float] = []
        for step in range(1, cfg.rl_steps_per_iter + 1):
            s = raw_train[(rl_steps_total + step) % len(raw_train)]
            adj = torch.tensor(s.adj, dtype=torch.float32, device=rl_trainer.device)

            out = rl_trainer.policy.sample_order(adj)
            r = reward_fn(s, out.order).to(rl_trainer.device)
            stats = rl_trainer.update(r, out.logprob_sum, out.entropy_sum)
            rl_loss_values.append(stats["loss"])
            rl_reward_values.append(stats["reward"])
            rl_entropy_values.append(stats["entropy_sum"])
            rl_adv_values.append(stats["adv"])
            rl_logprob_values.append(stats["logprob_sum"])

            if step % cfg.rl_print_every == 0:
                dt = time.time() - t0
                print(f"[RL] step {step:5d}/{cfg.rl_steps_per_iter} R={stats['reward']:.2f} ent={stats['entropy_sum']:.2f} dt={dt:.1f}s")

        rl_steps_total += cfg.rl_steps_per_iter

        # salva policy
        policy_path = out_dir / f"policy_iter{it}.pt"
        torch.save(
            {
                "step": rl_steps_total,
                "policy_state": rl_trainer.policy.state_dict(),
                "policy_cfg": {
                    "emb_dim": cfg.rl_emb_dim,
                    "state_dim": cfg.rl_state_dim,
                    "gnn_layers": cfg.rl_gnn_layers,
                    "attn_heads": cfg.rl_attn_heads,
                    "start_node": cfg.rl_start_node,
                    "attn_dropout": cfg.rl_attn_dropout,
                },
            },
            policy_path,
        )
        print(f"Saved policy: {policy_path}")

        # ---- 2) Gerar sequências com RL atual ----
        seq_train = _build_sequences_from_policy(raw_train, rl_trainer.policy, cfg.device, cfg.seq_mode)
        seq_val = _build_sequences_from_policy(raw_val, rl_trainer.policy, cfg.device, "greedy")
        seq_test = _build_sequences_from_policy(raw_test, rl_trainer.policy, cfg.device, "greedy")

        seq_dir = out_dir / f"seq_iter{it}"
        seq_dir.mkdir(parents=True, exist_ok=True)
        save_pickle(seq_train, seq_dir / "train.pkl.gz", compress=True)
        save_pickle(seq_val, seq_dir / "val.pkl.gz", compress=True)
        save_pickle(seq_test, seq_dir / "test.pkl.gz", compress=True)
        print(f"Saved sequences: {seq_dir}")

        # ---- 3) (Re)treinar DGMG (fine-tune) ----
        ds = DGMGSequenceDataset(seq_train)
        loader = DataLoader(ds, batch_size=cfg.dgmg_batch_size, shuffle=True, collate_fn=collate_pad)

        train_cfg = TrainConfig(
            lr=cfg.dgmg_lr,
            batch_size=cfg.dgmg_batch_size,
            epochs=cfg.dgmg_epochs_per_iter,
            device=cfg.device,
            log_every=200,
            dgmg_lr=cfg.dgmg_lr,
        )
        dgmg_train_stats = train_dgmg(dgmg, loader, train_cfg)
        dgmg_epochs_total += cfg.dgmg_epochs_per_iter

        dgmg_path = _save_dgmg_ckpt(dgmg, out_dir / f"dgmg_iter{it}.pt")
        print(f"Saved dgmg: {dgmg_path}")

        # ---- 4) Avaliações ----
        val_nll = _mean_nll_for_policy(dgmg, raw_val, rl_trainer.policy, cfg.device, mode="greedy")
        test_nll = _mean_nll_for_policy(dgmg, raw_test, rl_trainer.policy, cfg.device, mode="greedy")

        # geração e métricas (valid_ratio etc.)
        do_gen_eval = (it % cfg.eval_every == 0) or (it == 1) or (it == cfg.outer_iters)

        if do_gen_eval:
            gen_cfg = SampleConfig(min_nodes=cfg.n_min, max_nodes=cfg.n_max, greedy=False, temperature=1.0)
            adjs = sample_many(dgmg, gen_cfg, num=cfg.gen_samples, device=cfg.device)
            
            #metrics = evaluate_generated(adjs, EvalConfig(n_min=cfg.n_min, n_max=cfg.n_max))
            if cfg.family == "ba":
                m = evaluate_generated_ba(adjs, n_min=cfg.n_min, n_max=cfg.n_max)
                metrics = {
                    "valid_ratio": m["valid_size_ratio"] * m["connected_ratio"],
                    "gen_tree_ratio": 0.0,
                    "gen_binary_ratio": 0.0,
                    "tree_ratio": 0.0,
                    "binary_ratio": 0.0,
                    "connected_ratio": m["connected_ratio"],
                    "valid_size_ratio": m["valid_size_ratio"],
                    "avg_nodes": m["avg_nodes"],
                    "std_nodes": m["std_nodes"],
                    "avg_edges": m["avg_edges"],
                    "avg_gini_degree": m["avg_gini_degree"],
                    "hubiness": m["hubiness"],
                }
            else:
                # your existing binary-tree metrics dict
                metrics = evaluate_generated(adjs, EvalConfig(n_min=cfg.n_min, n_max=cfg.n_max))
        else:
            metrics = {}

        row = {
            "iter": it,
            "rl_steps_total": rl_steps_total,
            "dgmg_epochs_total": dgmg_epochs_total,
            "rl_loss_mean": statistics.fmean(rl_loss_values) if rl_loss_values else 0.0,
            "rl_reward_mean": statistics.fmean(rl_reward_values) if rl_reward_values else 0.0,
            "rl_entropy_mean": statistics.fmean(rl_entropy_values) if rl_entropy_values else 0.0,
            "rl_adv_mean": statistics.fmean(rl_adv_values) if rl_adv_values else 0.0,
            "rl_logprob_mean": statistics.fmean(rl_logprob_values) if rl_logprob_values else 0.0,
            "dgmg_train_loss_mean": float(dgmg_train_stats.get("train_loss_mean", 0.0)),
            "dgmg_train_loss_last": float(dgmg_train_stats.get("train_loss_last", 0.0)),
            "val_nll_greedy": val_nll,
            "test_nll_greedy": test_nll,
            "gen_valid_ratio": metrics.get("valid_ratio", 0.0),
            "gen_tree_ratio": metrics.get("tree_ratio", 0.0),
            "gen_binary_ratio": metrics.get("binary_ratio", 0.0),
            "gen_connected_ratio": metrics.get("connected_ratio", 0.0),
            "gen_valid_size_ratio": metrics.get("valid_size_ratio", 0.0),
            "gen_avg_nodes": metrics.get("avg_nodes", 0.0),
            "gen_std_nodes": metrics.get("std_nodes", 0.0),
            "gen_avg_edges": metrics.get("avg_edges", 0.0),
            "gen_avg_gini_degree": metrics.get("avg_gini_degree", 0.0),
            "gen_hubiness": metrics.get("hubiness", 0.0),
        }

        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writerow(row)

        print(f"[EVAL] val_nll={val_nll:.3f} test_nll={test_nll:.3f} "
              f"valid={row['gen_valid_ratio']:.3f} tree={row['gen_tree_ratio']:.3f} bin={row['gen_binary_ratio']:.3f}")

        # ---- 5) Early stopping ----
        if val_nll < best_val - cfg.early_stop_min_delta:
            best_val = val_nll
            bad = 0
            # salva “best”
            torch.save(
                {
                    "step": rl_steps_total,
                    "policy_state": rl_trainer.policy.state_dict(),
                    "policy_cfg": {
                        "emb_dim": cfg.rl_emb_dim,
                        "state_dim": cfg.rl_state_dim,
                        "gnn_layers": cfg.rl_gnn_layers,
                        "attn_heads": cfg.rl_attn_heads,
                        "start_node": cfg.rl_start_node,
                        "attn_dropout": cfg.rl_attn_dropout,
                    },
                },
                out_dir / "policy_best.pt",
            )
            torch.save({"model_state": dgmg.state_dict(), "cfg": dgmg.cfg.__dict__}, out_dir / "dgmg_best.pt")
        else:
            bad += 1
            if bad >= cfg.early_stop_patience:
                print(f"Early stop: val_nll não melhorou por {bad} iterações.")
                break

    print(f"\nFinal: history -> {csv_path.resolve()}")
    print(f"Best val_nll: {best_val:.3f}")


def _save_dgmg_ckpt(model: DGMGMinimal, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "cfg": model.cfg.__dict__}, path)
    return path
