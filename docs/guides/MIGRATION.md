# 📚 Migration Guide: Old Scripts → New Baseline

This guide explains how the old monolithic scripts map to the new modular codebase.

---

## 🔄 Overview

| Old File | New Location(s) |
|----------|-----------------|
| `all_in_one_wave_u_net.py` | `src/models/waveunet.py`, `src/data/dataset.py`, `src/training/engine.py` |
| `wave_like_niels.py` | Archived (not in baseline; can extend in phase 2) |
| `wavenet_variational.ipynb` | Archived (VAE expansion for phase 2) |

---

## 📋 Detailed Mapping

### Data Loading

**Old (all_in_one_wave_u_net.py):**
```python
def create_dataset_wav(parent_folder):
    # Hardcoded paths, global variables, inline loops
    for fold in os.listdir(parent_folder):
        in_path = os.path.join(parent_folder, fold, "admm_reference.wav")
        target_path = os.path.join(parent_folder, fold, "admm_processed.wav")
        # MinMaxScaler per window (NOT ideal)
        input_sample = SCALER.fit_transform(input_data[i:i+WINDOW_SIZE])
```

**New:**
```python
from src.data.dataset import PairedAudioDataset, create_dataloaders

# Config-driven, no hardcoding
dataloader = create_dataloaders(
    data_dir="/path/to/data",
    input_pattern="*reference*",
    target_pattern="*processed*",
    threshold=0.0,  # Explicit threshold filtering
    normalization="none"  # No per-window MinMax
)
```

**Key improvements:**
- ✅ No hardcoded paths (config-driven)
- ✅ Configurable pairing logic
- ✅ Track-level train/val/test split (no leakage)
- ✅ Correct threshold filtering (max(abs) >= threshold)
- ✅ Optional per-track normalization (global, not per-window)
- ✅ Type hints and docstrings

---

### Model Definition

**Old (all_in_one_wave_u_net.py):**
```python
class Model(nn.Module):
    def __init__(self, n_layers=8, channels_interval=16):
        super().__init__()
        # ...encoder, middle, decoder in one big class
        
    def forward(self, x):
        # Direct waveform output
        return output
```

**New:**
```python
from src.models.waveunet import ResidualWaveUNet

model = ResidualWaveUNet(
    depth=8,
    base_channels=16,
    kernel_size=9,
    output_activation=None  # Explicit control
)

# Returns: input + delta (residual prediction)
output = model(x)
```

**Key improvements:**
- ✅ Modular blocks (ConvBlock, DownBlock, UpBlock)
- ✅ Residual output (more principled for loudness enhancement)
- ✅ Optional output activation
- ✅ Clean initialization

---

### Loss Functions

**Old (all_in_one_wave_u_net.py):**
```python
criterion = nn.MSELoss()

# In training loop:
loss = criterion(output, target)
loss = loss / (torch.linalg.vector_norm(target, ord=2) + EPSPILON)  # Inline NMSE
```

**New:**
```python
from src.losses.combined import create_combined_loss

criterion = create_combined_loss(
    l1_weight=1.0,
    nmse_weight=1.0,
    mrstft_weight=1.0
)

loss_dict = criterion(output, target)  # Returns breakdown
loss = loss_dict["total"]
```

**Key improvements:**
- ✅ Modular loss composition
- ✅ Multi-resolution STFT loss (perceptually better)
- ✅ Loss breakdown per component
- ✅ Config-driven weights

---

### Training Loop

**Old (all_in_one_wave_u_net.py):**
```python
def train(model, train_loader, val_loader, optimizer, criterion, device, epochs=10):
    for epoch in range(epochs):
        model.train()
        for i, (input, target) in enumerate(train_loader):
            # Training step
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        for i, (input, target) in enumerate(val_loader):
            # Validation step
        
        # Early stopping logic mixed with training
        if early_stopping(val_losses, patience=50):
            break
```

**New:**
```python
from src.training.engine import Trainer

trainer = Trainer(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    scheduler=scheduler,
    gradient_clip=1.0,
    amp=True  # Automatic mixed precision
)

trainer.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=100,
    checkpoint_dir="checkpoints",
    patience=5,
    csv_log_path="logs/metrics.csv"
)
```

**Key improvements:**
- ✅ Clean separation of concerns
- ✅ Built-in checkpointing (latest + best)
- ✅ CSV/JSON logging
- ✅ Proper early stopping
- ✅ Gradient clipping
- ✅ AMP support
- ✅ LR scheduling

---

### Inference

**Old (all_in_one_wave_u_net.py):**
```python
model.load_state_dict(torch.load("models/model_40.pth"))

with torch.no_grad():
    output = model(input)
```

**New:**
```bash
# Command-line:
python scripts/infer_file.py \
  --checkpoint checkpoints/best.pth \
  --input input.wav \
  --output output.wav
```

Or programmatically:
```python
from src.evaluation.inference import infer_file

infer_file(
    model=model,
    input_path="input.wav",
    output_path="output.wav",
    frame_size=1024,
    hop_size=512
)
```

**Key improvements:**
- ✅ Overlap-add reconstruction (not window-by-window)
- ✅ Batch inference for efficiency
- ✅ WAV I/O handled
- ✅ CLI interface

---

### Evaluation

**Old (wavenet_variational.ipynb):**
```python
# Manual metric computation scattered throughout
def compute_similarity(target, predict, weightage=[0.33, 0.33, 0.33]):
    # Complex custom metric NOT in baseline
```

**New:**
```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/best.pth \
  --split test
```

Or:
```python
from src.evaluation.evaluate import evaluate_dataset

metrics = evaluate_dataset(model, dataloader)
# Returns: NMSE, MAE, crest factors, loudness, peak reduction
```

**Key improvements:**
- ✅ Standard audio metrics (no custom similarity scores yet)
- ✅ Per-batch and aggregated metrics
- ✅ CSV export
- ✅ Visualization helpers

---

## 🔧 How to Run the Baseline

### 1. Prepare data directory
```
/path/to/data/
├── track_001/
│   ├── admm_reference.wav
│   └── admm_processed.wav
├── track_002/
│   ├── admm_reference.wav
│   └── admm_processed.wav
```

### 2. Edit config
```bash
editor configs/baseline.yaml
# Update: data.data_dir, model.*, training.*
```

### 3. Train
```bash
python scripts/train.py --config configs/baseline.yaml
```

### 4. Evaluate
```bash
python scripts/evaluate.py --config configs/baseline.yaml --split test
```

### 5. Infer
```bash
python scripts/infer_file.py --checkpoint checkpoints/best.pth --input input.wav --output output.wav
```

---

## ❌ What's NOT in the Baseline

These features from the old scripts are **intentionally excluded** for phase 1:

- ❌ VAE (latent space) — for phase 2
- ❌ Detectability loss (`libdetectability`) — for phase 2
- ❌ GAN discriminator — for phase 2
- ❌ Per-window MinMax normalization — global/per-track only
- ❌ `plotlosses_real_time()` inside training — logging via CSV
- ❌ WaveLike encoder (simpler baseline only)
- ❌ Multi-track / stereo — mono only

These are documented as **TODOs** in phase 2 section (see README.md).

---

## 📂 File Handling Differences

| Old | New |
|-----|-----|
| Inline MinMaxScaler per window | `preprocessing.normalize_audio(mode="peak_per_track")` |
| Global `PARENT_FOLDER` variable | Config file `data.data_dir` |
| Mixed train/val logic | `dataset.split()` with seed control |
| Manual frame loop | `frame_audio()` and `unframe_audio()` (overlap-add) |
| Hardcoded checkpoint paths | `training/checkpointing.py` with automated patterns |
| Print statements only | CSV + JSON logging |

---

## 🎯 Benefits of Migration

✅ **Modularity**: Each component (data, model, loss, training) is independent  
✅ **Testability**: Unit test each module separately  
✅ **Reproducibility**: Config files + seed control  
✅ **Scalability**: Easy to add new losses, schedulers, metrics  
✅ **Type safety**:Type hints catch bugs early  
✅ **Documentation**: Docstrings and clear naming  
✅ **CLI interface**: No more notebook cells to run manually  
✅ **Checkpointing**: Proper save/load without manual state dicts  
✅ **Logging**: Structured output (CSV/JSON) for post-training analysis  

---

## 🚀 Next Steps (Phase 2)

1. Add transformer bottleneck to model
2. Implement detectability loss
3. Add stage-2 fine-tuning workflow
4. Implement VAE variant
5. Add GAN training loop
6. Multi-track and stereo support
7. Advanced visualization (embedding space, attention maps)

Each of these extends the baseline without breaking existing functionality.

---

## 🤔 FAQ

**Q: Can I use the old Jupyter notebooks?**  
A: They work independently but are not part of the baseline. Archive them in `notebooks/archive/`.

**Q: How do I add a new loss function?**  
A: Create a class in `src/losses/`, inherit from `nn.Module`, and add to `CombinedLoss`.

**Q: Can I train on multi-GPU?**  
A: Not yet built-in. Would require `DistributedDataParallel` wrapper (phase 2).

**Q: How do I resume training from checkpoint?**  
A: Modify `train.py` to load checkpoint before calling `trainer.fit()` (TODO in code).

**Q: Why residual output instead of direct output?**  
A: Residuals are easier to learn and have better gradient flow for small changes. Standard in audio enhancement.

---

## 📖 See Also

- [README.md](README.md) — Project overview
- [configs/baseline.yaml](configs/baseline.yaml) — Default config
- `src/data/dataset.py` — Detailed docs on dataset loading
- `src/training/engine.py` — Training loop with type hints
