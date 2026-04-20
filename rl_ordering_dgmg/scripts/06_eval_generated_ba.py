# scripts/06_eval_generated_ba.py
import argparse
import csv
import json
from pathlib import Path
import torch

from rlod.dgmg.model import DGMGMinimal, DGMGConfig
from rlod.dgmg.sample import sample_many, SampleConfig
from rlod.eval.ba_metrics import evaluate_generated_ba


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=str)
    ap.add_argument("--device", default="cpu", type=str)
    ap.add_argument("--samples", default=10000, type=int)
    ap.add_argument("--n_min", default=20, type=int)
    ap.add_argument("--n_max", default=60, type=int)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--temperature", default=1.0, type=float)
    ap.add_argument("--metrics_json", default="", type=str)
    ap.add_argument("--metrics_csv", default="", type=str)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location=args.device)
    cfg = DGMGConfig(**ckpt["cfg"])
    model = DGMGMinimal(cfg).to(args.device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()

    gen_cfg = SampleConfig(
        min_nodes=args.n_min,
        max_nodes=args.n_max,
        greedy=args.greedy,
        temperature=args.temperature,
    )
    adjs = sample_many(model, gen_cfg, num=args.samples, device=args.device)
    metrics = evaluate_generated_ba(adjs, n_min=args.n_min, n_max=args.n_max)

    print(metrics)

    if args.metrics_json:
        out_json = Path(args.metrics_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)
        print(f"Saved JSON metrics: {out_json}")

    if args.metrics_csv:
        out_csv = Path(args.metrics_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
            writer.writeheader()
            writer.writerow(metrics)
        print(f"Saved CSV metrics: {out_csv}")


if __name__ == "__main__":
    main()