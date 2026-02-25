# Deep Lineage

Predict cell fate from lineage-resolved transcriptomics using deep learning.

## Installation

```bash
uv sync
uv pip install -e . --no-deps
source .venv/bin/activate
```

## Pipeline Overview

Deep Lineage operates on multi-timepoint gene expression trajectories, expression profiles from cells observed at successive timepoints along the same lineage. Each trajectory carries a fate label indicating the terminal cell state.

The pipeline has three stages:

1. **Autoencoder** compresses high-dimensional gene expression into a latent space
2. **LSTM classifier** predicts cell fate from the latent trajectory
3. **LSTM regressor** predicts expression at unobserved timepoints

### Tips

- We use **uv** to manage dependencies and ensure reproducibility.
- The easiest way to understand the repo is to run through the synthetic data flow below end-to-end (you can start with `--genes 200` so the flow finishes quickly).
- The autoencoder training scripts print a reconstruction quality assessment (EXCELLENT / DECENT / POOR). Downstream model performance depends heavily on autoencoder quality. If the assessment is POOR, consider a grid search to optimize parameters.

## Train on Synthetic Data

### 1. Generate data

**Option A: Download pre-generated [LineageSim](https://github.com/garyhlai/lineagesim) dataset**

```bash
uv run python scripts/data/download_and_set_up_data.py --url "https://drive.google.com/file/d/1X-jf73mR6U9CF-lTROi7NNmLy9EPuWkh/view?usp=sharing"
```

**Option B: Generate from scratch using [LineageSim](https://github.com/garyhlai/lineagesim) (Recommended)**

```bash
uv run python scripts/data/generate_dataset.py --new_run --cells 8192 --genes 2000 --beta 0.2 --commit_depth 7 --max_trajectories 8000 --seed 42 --balance_t2
```

Key parameters:

- `--cells`: number of cells in the simulation
- `--genes`: number of genes per cell
- `--beta`: fate signal strength (lower = harder)
- `--commit_depth`: tree depth at which fate commitment occurs
- `--max_trajectories`: cap on extracted trajectories
- `--balance_t2`: balances terminal cell-type distribution

### 2. Split trajectories

```bash
uv run python scripts/data/split_trajectories.py --splits 0.8,0.1,0.1 --seed 42
uv run python scripts/data/verify_split_integrity.py
```

### 3. Train encoder

```bash
uv run python scripts/synthetic_training/train_autoencoder.py --latent_dim 200 --hidden_sizes "1024,512,256" --learning_rate 5e-4 --batch_size 1024
```

### 4. Train models

```bash
# Classifier (one model per timepoint combination)
uv run python scripts/synthetic_training/train_classifier.py --timepoints t0 --name t0_only
uv run python scripts/synthetic_training/train_classifier.py --timepoints t0,t1 --name t0_t1
uv run python scripts/synthetic_training/train_classifier.py --timepoints t0,t1,t2 --name t0_t1_t2

# Regressor (future and intermediate prediction)
uv run python scripts/synthetic_training/train_regressor.py --input_timepoints t0,t1 --target_timepoint t2 --name future
uv run python scripts/synthetic_training/train_regressor.py --input_timepoints t0,t2 --target_timepoint t1 --name intermediate
```

### 5. Evaluate

```bash
# Classifier
uv run python scripts/evaluation/evaluate_classifier.py --model ae_t0_only
uv run python scripts/evaluation/evaluate_classifier.py --model ae_t0_t1
uv run python scripts/evaluation/evaluate_classifier.py --model ae_t0_t1_t2

# Regressor
uv run python scripts/evaluation/evaluate_regressor.py --model ae_future
uv run python scripts/evaluation/evaluate_regressor.py --model ae_intermediate
```

### 6. Collect results

```bash
uv run python scripts/evaluation/collect_results.py
```

---

<details>
<summary><strong>Autoregressive Method (Supplementary)</strong></summary>

Steps 1-2 are the same as the LineageSim experiment above.

```bash
# Train
uv run python scripts/autoregressive/train_expression_classifier.py
uv run python scripts/autoregressive/train_autoregressive_generator.py

# Evaluate
uv run python scripts/autoregressive/evaluate_autoregressive_pipeline.py
```

</details>

## Train on Real Data

### 1. Process data

```bash
uv run python scripts/real_training/generate_reprogramming_dataset.py --mode cumulative
uv run python scripts/real_training/generate_reprogramming_dataset.py --mode single
```

### 2. Validate datasets

```bash
uv run python scripts/real_training/validate_reprogramming_dataset.py
```

### 3. Train autoencoder

```bash
uv run python scripts/real_training/train_autoencoder_reprogramming.py \
    --dataset reprogramming_dataset_cumulative.h5 \
    --output_dir runs/reprog_ae
```

### 4. Train classifier

```bash
uv run python scripts/real_training/train_classifier_reprogramming.py \
    --dataset reprogramming_dataset_cumulative.h5 \
    --encoder runs/reprog_ae/encoder.keras \
    --output_dir runs/reprog_cls
```

### 5. Train regressor

```bash
uv run python scripts/real_training/train_regressor_reprogramming.py \
    --dataset reprogramming_dataset_cumulative.h5 \
    --output_dir runs/reprog_reg
```

### 6. Evaluate

```bash
uv run python scripts/real_training/evaluate_reprogramming.py \
    --ae_dir runs/reprog_ae \
    --cls_dir runs/reprog_cls \
    --reg_dir runs/reprog_reg
```

---

<details>
<summary><strong>Robustness Analysis</strong></summary>

Scripts for noise robustness, cell dropout, and clone misidentification experiments are in `scripts/robustness/`. Run with `--help` for usage.

- `scripts/robustness/analyze_noise_robustness.py`
- `scripts/robustness/simulate_cell_dropout.py`
- `scripts/robustness/simulate_clone_misidentification.py`

</details>

## Citation

If you use this code, please cite the accompanying preprint:

```bibtex
@article{deeplineage2024,
  title   = {Deep Lineage: Single-Cell Lineage Tracing and Fate Inference Using Deep Learning},
  year    = {2024},
  journal = {bioRxiv},
  doi     = {10.1101/2024.04.25.591126}
}
```
