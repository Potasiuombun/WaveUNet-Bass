# WaveUNet-Bass

Waveform-domain loudness enhancement using a residual 1D Wave-U-Net in PyTorch.

## Current Status

- Dataset path in active use: `/mnt/f/fma_small/temp_data`
- Recommended serialized dataset format: rich DataFrame (`.pkl` or `.parquet`)
- Split strategy: grouped by `track_id` (no frame leakage across train/val/test)
- Baseline config: `configs/baseline.yaml`
- Overfit config (batch size 64): `configs/overfit_nmse.yaml`

## Quick Start

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Build serialized dataset (thesis path)

```bash
python scripts/prepare_dataset.py \
  --data-root /mnt/f/fma_small/temp_data \
  --output /mnt/f/fma_small/datasets/waveunet_baseline.pkl \
  --input-suffix _reference_clipped.npy \
  --target-suffix _admm_processed.npy \
  --frame-size 1024 \
  --hop-size 512 \
  --threshold 0.5
```

Memory-safe build options (useful on large corpora):

```bash
python scripts/prepare_dataset.py \
  --data-root /mnt/f/fma_small/temp_data \
  --output /mnt/f/fma_small/datasets/waveunet_baseline_small.pkl \
  --input-suffix _reference_clipped.npy \
  --target-suffix _admm_processed.npy \
  --frame-size 1024 \
  --hop-size 512 \
  --threshold 0.5 \
  --max-groups 2 \
  --max-tracks 120 \
  --max-frames-per-track 128
```

### 3) Inspect dataset

```bash
python scripts/inspect_dataset.py /mnt/f/fma_small/datasets/waveunet_baseline_small.pkl
```

Inspector now reports metadata path consistency per track (coverage + reuse checks).

### 4) Train baseline

```bash
python scripts/train.py \
  --config configs/baseline.yaml \
  --dataset-file /mnt/f/fma_small/datasets/waveunet_baseline.pkl \
  --device cuda
```

### 4b) Stage-1 representation training (small serialized set)

```bash
python scripts/train.py \
  --config configs/baseline_small.yaml \
  --dataset-file /mnt/f/fma_small/datasets/waveunet_baseline_small.pkl \
  --device cuda
```

Artifacts created by stage-1 runs include:

- checkpoints (latest + best)
- CSV/JSON training logs
- config snapshot under `logs/<experiment_name>/config_snapshot.yaml`
- run summary under `logs/<experiment_name>/run_summary.json`
- dataset sidecar summary next to dataset (created if missing):
  `<dataset_name>.summary.json`

### 5) Overfit run (NMSE-focused, batch size 64)

```bash
python scripts/train.py \
  --config configs/overfit_nmse.yaml \
  --dataset-file /mnt/f/fma_small/datasets/waveunet_baseline_small.pkl \
  --device cuda
```

### 6) Stage-2 detectability fine-tuning

```bash
python scripts/finetune_detectability.py \
  --config configs/detectability_finetune.yaml \
  --dataset-file /mnt/f/fma_small/datasets/waveunet_baseline_small.pkl \
  --checkpoint checkpoints/best.pth \
  --device cuda
```

Stage-2 keeps reconstruction + MRSTFT active and adds optional detectability.
Use `finetune.freeze_policy` to choose: `none`, `encoder`,
`early_encoder_only`, or `discriminative_lr` (default).

### 7) Stage-3 variational training (VAE WaveUNet)

```bash
python scripts/train_vae.py \
  --config configs/vae_stage3.yaml \
  --dataset-file /mnt/f/fma_small/datasets/waveunet_baseline_small.pkl \
  --device cuda
```

Stage-3 adds a variational bottleneck and optimizes:

- reconstruction (L1 + NMSE)
- MRSTFT
- KL divergence (`beta_kl`)

Use `training.warm_start_checkpoint` to initialize VAE weights from stage-1.
`configs/vae_stage3.yaml` includes a KL warmup (`beta_kl_start -> beta_kl_end`).

### 8) Regular WaveUNet with self-attention bottleneck

```bash
python scripts/train.py \
  --config configs/baseline_attention_bottleneck.yaml \
  --dataset-file /mnt/f/fma_small/datasets/waveunet_baseline_small.pkl \
  --device cuda
```

## Split and Leakage Guarantee

Dataset splitting is done in `src/data/splits.py` with `split_by_track(...)`:

- Track IDs are shuffled once with a fixed seed.
- Train/val/test receive disjoint sets of track IDs.
- All frames of one track stay in exactly one split.

This prevents overlap leakage where neighboring or overlapping frames from the same track appear in both training and evaluation.

## Data Format (temp_data_outputs)

Expected structure:

```text
/mnt/f/fma_small/temp_data/
  018_output/
    018031_reference_clipped.npy
    018031_admm_processed.npy
    018031_admm_processed.wav          (optional metadata)
    018031_admm_reference_clipped.wav  (optional metadata)
    018031_admm_rescaled.npy           (optional metadata)
  019_output/
    ...
```

Core serialized columns include:

- `input`, `target`
- `track_id`, `frame_index`, `split`
- `source_input_path`, `source_target_path`, `group_id`
- Optional metadata paths: `processed_wav_path`, `reference_wav_path`, `rescaled_npy_path`

## Main Paths

- Configs: `configs/`
- Data scripts: `scripts/prepare_dataset.py`, `scripts/inspect_dataset.py`, `scripts/train.py`
- Core data logic: `src/data/serialized.py`, `src/data/splits.py`
- Logs/checkpoints: `logs/`, `checkpoints/`

## Documentation Map

See `docs/DOCUMENTATION.md` for a concise map of project docs and which files are canonical vs legacy notes.
