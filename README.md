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

`scripts/real_data/` runs Deep Lineage on mouse hematopoiesis (LARRY; Weinreb et al., 2020) and fibroblast-to-iEP reprogramming (CellTagging; Biddy et al., 2018). Both datasets are downloaded through CoSpar on first use. Lineages (clones, or day-0 barcode families for reprogramming) are split into 10 groups; each rotation tests on one group, validates on the next and trains on the rest.

### 1. Build the datasets

```bash
uv run python -m scripts.real_data.prepare_dataset --dataset hematopoiesis
uv run python -m scripts.real_data.prepare_dataset --dataset reprogramming
```

### 2. Train and evaluate each rotation

```bash
for r in 0 1 2 3 4 5 6 7 8 9; do
  uv run python -m scripts.real_data.run_analyses --dataset hematopoiesis --rotation $r
  uv run python -m scripts.real_data.run_analyses --dataset reprogramming --rotation $r
done
```

Add `--fate_only` to run only the fate-prediction analyses.

### 3. Pool the results

```bash
uv run python -m scripts.real_data.summarize --dataset hematopoiesis
uv run python -m scripts.real_data.summarize --dataset reprogramming
```

Outputs are written to `runs/<dataset>/`.

---

<details>
<summary><strong>Robustness Analysis</strong></summary>

Scripts for noise robustness, cell dropout, and clone misidentification experiments are in `scripts/robustness/`. Run with `--help` for usage.

- `scripts/robustness/analyze_noise_robustness.py`
- `scripts/robustness/simulate_cell_dropout.py`
- `scripts/robustness/simulate_clone_misidentification.py`

</details>

## Citation

If you use this code, please cite:

Sadria, M., Zhang, A., Lai, H.G., and Bader, G.D. (2026). Deep Lineage: Single-cell lineage tracing and fate inference using deep learning. Cell Systems 17, 101759. https://doi.org/10.1016/j.cels.2026.101759

```bibtex
@article{sadria2026deeplineage,
  author  = {Sadria, Mehrshad and Zhang, Allen and Lai, Haizhi Gary and Bader, Gary D.},
  title   = {Deep Lineage: Single-cell lineage tracing and fate inference using deep learning},
  journal = {Cell Systems},
  volume  = {17},
  pages   = {101759},
  year    = {2026},
  doi     = {10.1016/j.cels.2026.101759}
}
```

An earlier version is available as a preprint: https://doi.org/10.1101/2024.04.25.591126

## License

This code is released under the MIT License; see [LICENSE](LICENSE).
