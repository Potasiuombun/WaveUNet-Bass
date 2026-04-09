#!/usr/bin/env python3
"""Visualize inference results."""

import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np
from pathlib import Path
import librosa

output_dir = Path("inference_outputs")

# Load audio files
original = sf.read(output_dir / "original.wav")[0]
input_norm = sf.read(output_dir / "input_normalized.wav")[0]
output = sf.read(output_dir / "output_reconstructed.wav")[0]
error = sf.read(output_dir / "error.wav")[0]

sr = 44100

print("=" * 70)
print("INFERENCE RESULTS VISUALIZATION")
print("=" * 70)

print("\n📁 Generated files:")
for f in sorted(output_dir.glob("*.wav")):
    size_mb = f.stat().st_size / (1024**2)
    print(f"   ✓ {f.name:30s} ({size_mb:.2f} MB)")

# Create visualization
fig, axes = plt.subplots(4, 2, figsize=(16, 12))

# Time-domain waveforms
time = np.arange(len(original)) / sr

ax = axes[0, 0]
ax.plot(time, original, alpha=0.8, linewidth=0.5)
ax.set_ylabel("Amplitude", fontweight="bold")
ax.set_title("Original Audio (Denormalized)", fontweight="bold")
ax.grid(alpha=0.3)

ax = axes[1, 0]
ax.plot(time, input_norm, alpha=0.8, linewidth=0.5, color="orange")
ax.set_ylabel("Amplitude", fontweight="bold")
ax.set_title("Input (Peak Normalized)", fontweight="bold")
ax.grid(alpha=0.3)

ax = axes[2, 0]
ax.plot(time, output, alpha=0.8, linewidth=0.5, color="green")
ax.set_ylabel("Amplitude", fontweight="bold")
ax.set_title("Model Output (Reconstructed)", fontweight="bold")
ax.grid(alpha=0.3)

ax = axes[3, 0]
ax.plot(time, error, alpha=0.8, linewidth=0.5, color="red")
ax.set_ylabel("Amplitude", fontweight="bold")
ax.set_xlabel("Time (s)", fontweight="bold")
ax.set_title("Reconstruction Error (Output - Input)", fontweight="bold")
ax.grid(alpha=0.3)

# Spectrograms
def plot_spectrogram(ax, y, sr, title):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)
    img = ax.imshow(S_db, aspect="auto", origin="lower", cmap="magma", interpolation="nearest")
    ax.set_ylabel("Mel Frequency", fontweight="bold")
    ax.set_title(title, fontweight="bold")
    return img

plot_spectrogram(axes[0, 1], original, sr, "Original Spectrogram")
plot_spectrogram(axes[1, 1], input_norm, sr, "Input Spectrogram (Normalized)")
plot_spectrogram(axes[2, 1], output, sr, "Output Spectrogram (Reconstructed)")
plot_spectrogram(axes[3, 1], error, sr, "Error Spectrogram")

plt.tight_layout()
plt.savefig(output_dir / "comparison.png", dpi=150, bbox_inches="tight")
print(f"\n✓ Saved visualization: {output_dir / 'comparison.png'}")

# Print statistics
print("\n" + "=" * 70)
print("AUDIO STATISTICS")
print("=" * 70)

print("\nOriginal (Denormalized):")
print(f"   Peak: {np.max(np.abs(original)):.6f}")
print(f"   RMS: {np.sqrt(np.mean(original**2)):.6f}")

print("\nInput (Normalized):")
print(f"   Peak: {np.max(np.abs(input_norm)):.6f}")
print(f"   RMS: {np.sqrt(np.mean(input_norm**2)):.6f}")

print("\nOutput (Reconstructed):")
print(f"   Peak: {np.max(np.abs(output)):.6f}")
print(f"   RMS: {np.sqrt(np.mean(output**2)):.6f}")

print("\nReconstruction Metrics:")
mae = np.mean(np.abs(output - input_norm))
mse = np.mean((output - input_norm) ** 2)
nmse = mse / (np.mean(input_norm ** 2) + 1e-8)
rmse = np.sqrt(mse)
print(f"   MAE:  {mae:.6f}")
print(f"   MSE:  {mse:.6f}")
print(f"   RMSE: {rmse:.6f}")
print(f"   NMSE: {nmse:.6f}")

print("\n" + "=" * 70)
print("✨ All done! Check inference_outputs/ for audio and visualization")
print("=" * 70)
