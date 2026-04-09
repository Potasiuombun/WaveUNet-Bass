# Documentation Map

This file organises the project documentation and identifies which files to use first.

## Canonical Docs

1. **`README.md`** — primary entrypoint: model families, dataset stats, Stage 1 / Stage 2
   training commands, inference usage, plots and metric tables.

2. **`docs/guides/SERIALIZED_QUICK_REF.md`** — CLI flags and troubleshooting for dataset
   building and inspection.

## Active Configs

| Config | Purpose |
|--------|---------|
| `configs/stage1_fma_curated_cli_norm.yaml` | WaveUNet Stage 1 |
| `configs/stage1_vae_fma_curated_cli_norm.yaml` | VAE WaveUNet Stage 1 |
| `configs/stage1_dsp_residual_fma_curated_cli_norm.yaml` | DSP + residual Stage 1 |
| `configs/stage1_band_controller_fma_curated_cli_norm.yaml` | Band controller Stage 1 |
| `configs/stage1_ddsp_controller_fma_curated_cli_norm.yaml` | DDSP controller Stage 1 |

Stage 2 has no separate config file — all hyper-parameters are passed as CLI flags to
`scripts/stage2_detectability_all_models.py`.

## Active Scripts

| Script | Purpose |
|--------|---------|
| `scripts/train.py` | Stage 1 WaveUNet trainer |
| `scripts/train_vae.py` | Stage 1 VAE WaveUNet trainer |
| `scripts/train_dsp_residual.py` | Stage 1 DSP + residual trainer |
| `scripts/train_band_controller.py` | Stage 1 band controller trainer |
| `scripts/train_ddsp_controller.py` | Stage 1 DDSP controller trainer |
| `scripts/stage2_detectability_all_models.py` | Stage 2 fine-tuning (all 5 models) |
| `scripts/infer_stage2_all_models.py` | Stage 2 inference + NMSE / detectability / PEAQ |
| `scripts/check_contamination.py` | Cross-split leakage verification |
| `scripts/setup_fma_baseline.py` | Build `stage1_fma_curated_cli_norm.pkl` from FMA Small |

## Checkpoints

| Path | Contents |
|------|---------|
| `checkpoints/stage1_*/best.pth` | Stage 1 best checkpoints (5 models) |
| `checkpoints/stage2_detectability_*/best_stage2.pth` | Stage 2 best checkpoints (5 models) |

## Dataset

| Path | Description |
|------|-------------|
| `datasets/stage1_fma_curated_cli_norm.pkl` | Curated FMA Small, 23,680 frames, 44100 Hz |

Split by `track_id` (1200 train / 120 val / 160 test). Zero cross-split leakage
verified by `scripts/check_contamination.py`.

## Outputs and Plots

| Path | Contents |
|------|---------|
| `outputs/stage2_listening/` | Reference track 000890: WAVs + 3 comparison plots |
| `outputs/stage2_listening_je_te/` | External track: WAVs + 3 comparison plots |
| `logs/stage2_detectability_all_models_leaderboard.csv` | Stage 2 val leaderboard |
| `logs/stage2_detectability_all_models_summary.json` | Stage 2 full training summary |

## Key Source Modules

| Module | Purpose |
|--------|---------|
| `src/models/` | All model architectures |
| `src/dsp/` | `FixedBandSplitter`, `FastDSPBaseline`, DDSP control helpers |
| `src/losses/perceptual.py` | `DetectabilityLossWrapper` |
| `src/data/splits.py` | `split_by_track()` — deterministic, seed-fixed split |
| `src/data/serialized.py` | Dataset loading |

## Artifact Archive Layout

- Active roots: `checkpoints/`, `logs/`, `outputs/`
- Archived runs (pre-2026-04-08): `archives/past_until_2026-04-08/`
- Archive pointer: `ACTIVE_ARTIFACTS_ARCHIVE_PATH.txt`

## Historical / Delivery Notes (archival only)

Located in `docs/archive/`. These are milestone snapshots from earlier stages of the
project and may reference outdated paths or commands. Not a source of truth.

## Implementation Guides

Located in `docs/guides/`. Useful when modifying dataset internals or migrating old
workflows: `SERIALIZED_DATASET_IMPL.md`, `SERIALIZED_DATASET_GUIDE.md`, `MIGRATION.md`.
