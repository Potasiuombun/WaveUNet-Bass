# Retraining Audit (2026-04-08)

## Scope
This audit is limited to the active thesis plan:
- S1 Plain non-attention VAE stabilization
- D1 Fast DSP baseline
- D2 DSP + tiny residual model
- B1 Band-split tiny neural controller
- DD1 DDSP-style controller

Policy applied:
- Preserve baseline WaveUNet branch and repo structure.
- Retrain/re-evaluate only where corrected CLI/ADMM references materially affect training/evaluation.
- Do not reactivate attention/detectability stage-2 branches now.

## Keep
- Core baseline architecture and training path:
  - `src/models/waveunet.py`
  - `scripts/train.py`
  - baseline configs (`configs/baseline*.yaml`) used for non-CLI target training.
- Generic utilities not tied to old CLI outputs:
  - data loading/splitting (`src/data/*`)
  - logging/checkpoint helpers (`src/utils/logging.py`, `src/training/checkpointing.py`)
  - generic reconstruction losses/metrics (`src/losses/reconstruction.py`, `src/evaluation/*`).
- S1 stabilization code already present:
  - `scripts/train_vae.py` (cyclical KL, free bits, KL logging, run summary/config snapshot)
  - `configs/vae_plain_cyclical.yaml`.
- D1 fast DSP baseline already present:
  - `src/dsp/fast_baseline.py`
  - `scripts/run_dsp_baseline.py`
  - `configs/dsp_baseline.yaml`.

## Retrain (fresh or warm-start)
- Any branch trained directly against old CLI-derived targets/references (legacy suspect sets under `legacy_cli_validation/*` and prior legacy ADMM-reference artifacts) should be treated as stale.
- For current plan branches:
  - S1 plain VAE: retrain/reconfirm on corrected, clean dataset split for thesis reporting.
  - D2/B1/DD1: new branches; train fresh experiments from run-scoped configs.

## Re-evaluate
- All ADMM-vs-model comparison reports generated from legacy suspect CLI output directories must be re-run on corrected ADMM/reference generation.
- Any detectability/ADMM comparison tables that depended on old `000_output/010_output/018_output` style legacy outputs are stale.
- New corrected comparisons in `WaveUNet-Bass/outputs/admm_compare/*` built from filtered corrected ADMM inputs can be retained.

## Safe warm starts
- Safe warm starts for current plan:
  - S1 plain VAE from baseline or prior plain VAE checkpoint with matching non-attention architecture.
  - D2 tiny residual from random init (recommended default) or optional warm start from generic small conv if shape-compatible.
  - B1 and DD1 controllers: random init recommended (new control heads differ from existing checkpoints).
- Not recommended now:
  - Warm-starting from attention branches for S1.
  - Reusing detectability fine-tune checkpoints for D2/B1/DD1.

## Branch-level decision matrix
- S1 (plain non-attention VAE): **retrain/reconfirm** on corrected references; warm start allowed from plain compatible checkpoint.
- D1 (fast DSP baseline): **no retraining** (deterministic DSP), but **re-evaluate** metrics on corrected comparison sets.
- D2 (DSP + tiny residual): **new training required**.
- B1 (band-split tiny controller): **new training required**.
- DD1 (DDSP-style controller): **new training required**.
- Attention branches: **defer** (out of current plan).
- Detectability stage-2 branches: **defer** unless explicitly requested later.
