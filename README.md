# Learning to Order: GNN-Based Deep Reinforcement Learning Policies for Autoregressive Graph Models

A Python framework for learning optimal node orderings for graph generation using reinforcement learning and deep generative models. This project combines a Deep Graph Generative Model (DGMG) with deep reinforcement learning to optimize the order in which graph nodes are added during generation.

The **ordering policy** uses a graph encoder built from stacked **multi-head self-attention** blocks (`torch.nn.MultiheadAttention`) with topology-aware masking, instead of a plain GCN. You can control depth, attention heads, and dropout via the training scripts below.

## Overview

The project implements a complete pipeline for:

1. **Dataset generation**: Create synthetic graph datasets (binary trees, Barabási–Albert networks).
2. **Sequence building**: Convert graphs into ordered node-addition sequences.
3. **Model training**: Train a deep generative model (DGMG) on these sequences.
4. **Reinforcement learning**: Learn node-ordering policies using RL (attention-based policy).
5. **Joint training**: Iteratively improve both the generative model and the ordering policy.

## Attention-based ordering policy

- The policy encoder is implemented in `src/rlod/rl/policy.py` as stacked `GraphAttentionBlock` modules (multi-head attention over nodes with masks for allowed choices).
- Standalone RL scripts (`05_train_rl_ordering.py`, `06_train_rl_with_dgmg_reward.py`) expose **`--gnn_layers`**, **`--attn_heads`**, **`--attn_dropout`**, and **`--start_node`** (`degree` or `random`).
- Joint training (`07_train_joint.py`) exposes the same knobs with the **`rl_`** prefix: **`--rl_gnn_layers`**, **`--rl_attn_heads`**, **`--rl_attn_dropout`**, **`--rl_start_node`**.
- Checkpoints store policy configuration so inference and sequence building stay consistent with training.

## Quick start

### Prerequisites

- **Python 3.10+**
- **PyTorch 2.0+** with CUDA (optional, for GPU acceleration)
- **NetworkX 3.0+**
- **NumPy 1.24+**

### Installation

From the `rl_ordering_dgmg` directory:

```bash
cd /path/to/rl_ordering_dgmg
pip install -e .
```

This installs dependencies and the `rlod` package.

### First run (binary trees)

Example workflow with **attention** defaults on the RL policy (`gnn_layers=5`, `attn_heads=4`):

```bash
# Step 1: Generate a binary tree dataset
python scripts/01_make_trees_dataset.py \
    --out_dir data/raw \
    --seed 0 \
    --num_graphs 4000 \
    --n_min 5 \
    --n_max 20 \
    --mode any \
    --train_frac 0.8 \
    --val_frac 0.1 \
    --test_frac 0.1

# Step 2: Build node ordering sequences (uses a random policy)
python scripts/02_build_sequences.py \
    --graphs_train data/raw/binary_trees_any_n5-20_relabel0/train.pkl \
    --graphs_val data/raw/binary_trees_any_n5-20_relabel0/val.pkl \
    --out_dir data/processed \
    --seed 0 \
    --policy_mode sample

# Step 3: Train the Deep Graph Generative Model (DGMG)
python scripts/03_train_dgmg.py \
    --train_seq data/processed/seq_sample_train.pkl \
    --device cpu \
    --epochs 5 \
    --batch_size 32 \
    --lr 3e-4

# Step 4: Train RL policy with dummy reward (attention policy)
python scripts/05_train_rl_ordering.py \
    --data_path data/raw/binary_trees_any_n5-20_relabel0/train.pkl \
    --seed 0 \
    --device cpu \
    --steps 5000 \
    --lr 3e-4 \
    --gnn_layers 5 \
    --attn_heads 4 \
    --attn_dropout 0.0 \
    --start_node degree

# Alternative: Train RL policy with DGMG-based reward
python scripts/06_train_rl_with_dgmg_reward.py \
    --data_path data/raw/binary_trees_any_n5-20_relabel0/train.pkl \
    --dgmg_ckpt runs/dgmg/dgmg_minimal.pt \
    --device cpu \
    --steps 5000 \
    --lr 3e-4 \
    --gnn_layers 5 \
    --attn_heads 4 \
    --attn_dropout 0.0 \
    --start_node degree
```

## Detailed usage

### Step 1: Generate a graph dataset

#### Option A: Binary trees

```bash
python scripts/01_make_trees_dataset.py [OPTIONS]
```

**Parameters:**

- `--out_dir` (str, default: `data/raw`): Output directory.
- `--seed` (int, default: `0`): Random seed.
- `--num_graphs` (int, default: `4000`): Total number of graphs.
- `--n_min`, `--n_max` (int): Node count range.
- `--mode` (str, default: `any`): `any`, `full`, or `perfect`.
- `--relabel` (flag): Randomly relabel nodes if set.
- `--train_frac`, `--val_frac`, `--test_frac`: Split fractions.

#### Option B: Barabási–Albert

```bash
python scripts/01_make_ba_dataset.py [OPTIONS]
```

Similar layout; preferential-attachment graphs with parameter `--m`, and split sizes via `--num_train`, `--num_val`, `--num_test` (see script help).

### Step 2: Build node ordering sequences

```bash
python scripts/02_build_sequences.py [OPTIONS]
```

**Parameters:**

- `--graphs_train`, `--graphs_val` (str, required): Pickle paths.
- `--out_dir` (str, default: `data/processed`): Output for sequence pickles.
- `--seed` (int): Random seed.
- `--policy_mode` (str): `sample` or `greedy` where supported.

If you load a trained policy checkpoint, the saved `policy_cfg` should match the attention settings used during RL training.

### Step 3: Train DGMG

```bash
python scripts/03_train_dgmg.py [OPTIONS]
```

**Parameters:**

- `--train_seq` (str, required): Training sequences pickle.
- `--device` (str, default: `cpu`): `cpu` or `cuda`.
- `--epochs`, `--batch_size`, `--lr`: Optimization settings.
- `--ckpt_dir` (str, default: `runs/dgmg`): Checkpoint directory.
- `--log_every` (int): Logging interval.

### Step 4: Evaluate DGMG (optional)

```bash
python scripts/04_eval_dgmg_nll.py [OPTIONS]
```

### Step 5: Train RL policy

#### Option A: Dummy reward (baseline)

```bash
python scripts/05_train_rl_ordering.py [OPTIONS]
```

**Core parameters:**

- `--data_path` (str, required): Graph pickle.
- `--device`, `--seed`, `--steps`, `--lr`: Training setup.
- `--entropy_beta`, `--grad_clip`, `--baseline_ema`: REINFORCE stabilizers.
- `--emb_dim`, `--state_dim` (int, default: `64`): Policy widths.
- **`--gnn_layers` (int, default: `5`)**: Number of attention blocks in the encoder.
- **`--attn_heads` (int, default: `4`)**: Multi-head attention heads.
- **`--attn_dropout` (float, default: `0.0`)**: Dropout inside attention.
- **`--start_node` (str, default: `degree`)**: `degree` or `random` for the first node choice.
- `--ckpt_dir` (str, default: `runs/rl`): Output directory.

#### Option B: DGMG-based reward

```bash
python scripts/06_train_rl_with_dgmg_reward.py [OPTIONS]
```

Same policy arguments as above, plus:

- `--dgmg_ckpt` (str, required): Trained DGMG checkpoint.
- `--nll_scale` (float, default: `1.0`): Weight for the NLL-based reward term.

### Step 6: Evaluate generated BA graphs (optional)

```bash
python scripts/06_eval_generated_ba.py [OPTIONS]
```

Example with metrics export:

```bash
python scripts/06_eval_generated_ba.py \
    --ckpt runs/joint_ba_attn_degree/dgmg_best.pt \
    --samples 10000 \
    --n_min 20 --n_max 60 \
    --metrics_json runs/joint_ba_attn_degree/ba_eval_metrics.json \
    --metrics_csv runs/joint_ba_attn_degree/ba_eval_metrics.csv
```

### Step 7: Joint training (recommended for strong results)

```bash
python scripts/07_train_joint.py [OPTIONS]
```

**Core parameters:**

- `--raw_train`, `--raw_val`, `--raw_test` (str, required): Graph pickles.
- `--device`, `--seed`: Runtime.
- `--outer_iters`, `--rl_steps_per_iter`, `--dgmg_epochs_per_iter`: Outer loop.
- **`--rl_emb_dim`, `--rl_state_dim` (int, default: `64`)**: Policy size.
- **`--rl_gnn_layers` (int, default: `5`)**: Attention encoder depth.
- **`--rl_attn_heads` (int, default: `4`)**: Attention heads for the policy.
- **`--rl_attn_dropout` (float, default: `0.0`)**: Attention dropout.
- **`--rl_start_node` (str, default: `degree`)**: `degree` or `random`.
- `--gen_samples`, `--n_min`, `--n_max`: Generation and validity checks.
- `--early_stop_patience`, `--early_stop_min_delta`: Early stopping on validation signal.
- `--seq_mode` (str): `sample` or `greedy`.
- `--dgmg_init_ckpt` (str): Optional warm-start for DGMG.
- `--lambda_valid`, `--valid_samples`, `--prefix_ratio`: Validity reward and sequence prefixes.
- `--family` (str, default: `binary`): `binary` or `ba`.
- `--dgmg_lr` (float): DGMG learning rate in the joint loop.
- `--out_dir`, `--eval_every`: Outputs and evaluation frequency.

**Outputs** (typical):

- `dgmg_iter*.pt`, `policy_iter*.pt`, `dgmg_best.pt`, `policy_best.pt`
- `eval_history.csv`: Metrics per outer iteration (for plotting)
- `summary_for_thesis.csv` (when applicable)

### Barabási–Albert: dataset + joint run with attention

```bash
# Dataset
python scripts/01_make_ba_dataset.py \
    --out_dir data/raw \
    --n_min 20 --n_max 60 --m 2 \
    --num_train 6000 --num_val 750 --num_test 750 \
    --seed 42

# Joint training (attention policy)
python scripts/07_train_joint.py \
    --raw_train data/raw/ba_m2_n20-60_relabel0_train.pkl.gz \
    --raw_val data/raw/ba_m2_n20-60_relabel0_val.pkl.gz \
    --raw_test data/raw/ba_m2_n20-60_relabel0_test.pkl.gz \
    --family ba \
    --n_min 20 --n_max 60 \
    --out_dir runs/joint_ba_attn_degree \
    --outer_iters 10 \
    --rl_steps_per_iter 3000 \
    --dgmg_epochs_per_iter 2 \
    --rl_gnn_layers 5 \
    --rl_attn_heads 4 \
    --rl_attn_dropout 0.0 \
    --rl_start_node degree \
    --seed 42
```

## Visualization (`eval_history`)

Use `notebooks/ba_joint_training_visualizations.ipynb` to plot metrics from `eval_history.csv`. Set the run folder via:

- `RUN_NAME` in the notebook, or
- environment variable `JOINT_RUN_NAME` (e.g. `joint_ba_attn_degree_no_earlystop` for a long 100-iteration run).

Figures are written under `runs/<RUN_NAME>/figures/`.

Useful columns include: `iter`, `val_nll_greedy`, `test_nll_greedy`, `gen_valid_ratio`, `gen_connected_ratio`, `gen_avg_nodes`, `gen_avg_edges`, `gen_avg_gini_degree`, `gen_hubiness`.

## Project structure

```
rl_ordering_dgmg/
├── scripts/
│   ├── 01_make_trees_dataset.py
│   ├── 01_make_ba_dataset.py
│   ├── 02_build_sequences.py
│   ├── 03_train_dgmg.py
│   ├── 04_eval_dgmg_nll.py
│   ├── 05_train_rl_ordering.py          # RL dummy reward (attention policy)
│   ├── 06_train_rl_with_dgmg_reward.py  # RL + DGMG reward (attention policy)
│   ├── 06_eval_generated_ba.py
│   └── 07_train_joint.py                # Joint RL + DGMG (attention policy)
├── notebooks/
│   └── ba_joint_training_visualizations.ipynb
├── data/
│   ├── raw/
│   └── processed/
├── runs/                                 # Checkpoints and logs (examples)
│   ├── dgmg/
│   ├── joint_ba_attn_degree/
│   ├── joint_ba_attn_degree_no_earlystop/
│   └── ...
├── src/rlod/
│   ├── dgmg/
│   ├── graphs/
│   ├── joint/
│   ├── rl/                               # policy.py: attention blocks
│   ├── sequences/
│   └── utils/
├── pyproject.toml
└── README.md
```

## Common examples

### Quick test on small binary trees

```bash
python scripts/01_make_trees_dataset.py --num_graphs 200 --n_min 3 --n_max 8

python scripts/02_build_sequences.py \
    --graphs_train data/raw/binary_trees_any_n3-8_relabel0/train.pkl \
    --graphs_val data/raw/binary_trees_any_n3-8_relabel0/val.pkl \
    --out_dir data/processed

python scripts/03_train_dgmg.py \
    --train_seq data/processed/seq_sample_train.pkl \
    --epochs 2 --batch_size 16

python scripts/05_train_rl_ordering.py \
    --data_path data/raw/binary_trees_any_n3-8_relabel0/train.pkl \
    --steps 1000 \
    --gnn_layers 5 --attn_heads 4 --start_node degree
```

### Full pipeline with DGMG reward

```bash
python scripts/01_make_trees_dataset.py

python scripts/02_build_sequences.py \
    --graphs_train data/raw/binary_trees_any_n5-20_relabel0/train.pkl \
    --graphs_val data/raw/binary_trees_any_n5-20_relabel0/val.pkl

python scripts/03_train_dgmg.py \
    --train_seq data/processed/seq_sample_train.pkl \
    --batch_size 64 --epochs 10

python scripts/06_train_rl_with_dgmg_reward.py \
    --data_path data/raw/binary_trees_any_n5-20_relabel0/train.pkl \
    --dgmg_ckpt runs/dgmg/dgmg_minimal.pt \
    --steps 10000 --nll_scale 2.0 \
    --gnn_layers 5 --attn_heads 4 --start_node degree
```

### Joint training (binary trees)

```bash
python scripts/07_train_joint.py \
    --raw_train data/raw/binary_trees_any_n5-20_relabel0/train.pkl \
    --raw_val data/raw/binary_trees_any_n5-20_relabel0/val.pkl \
    --raw_test data/raw/binary_trees_any_n5-20_relabel0/test.pkl \
    --outer_iters 12 \
    --rl_steps_per_iter 3000 \
    --dgmg_epochs_per_iter 2 \
    --rl_gnn_layers 5 \
    --rl_attn_heads 4 \
    --rl_start_node degree \
    --out_dir runs/my_joint_experiment
```

## Parameter tuning

**Faster experiments:** Fewer graphs, smaller `n_max`, fewer `epochs` / `steps`, smaller `batch_size`.

**Better quality:** More graphs, more DGMG epochs and RL steps, joint training, tune `rl_attn_heads` and `rl_gnn_layers` if you see under- or overfitting.

**Limited memory:** Lower `batch_size`, use CPU, reduce graph counts and `n_max`.

## GPU acceleration

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

python scripts/03_train_dgmg.py --device cuda --train_seq data/processed/seq_sample_train.pkl

python scripts/07_train_joint.py --device cuda [other arguments...]
```

## Troubleshooting

- **`ModuleNotFoundError: rlod`**: Run `pip install -e .` from `rl_ordering_dgmg` and verify with `python -c "import rlod; print(rlod.__file__)"`.
- **Missing files**: Run commands from `rl_ordering_dgmg` (or fix paths); prefer absolute paths if relative paths fail.
- **CUDA OOM**: Reduce `batch_size`, use `--device cpu`, or shrink graphs / batch construction.
- **Policy / sequence mismatch**: Rebuild sequences with a policy checkpoint that matches saved `policy_cfg` (same `gnn_layers`, `attn_heads`, etc.).
