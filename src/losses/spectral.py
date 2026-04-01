"""Spectral losses using STFT."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class STFTLoss(nn.Module):
    """Single-resolution STFT loss."""
    
    def __init__(
        self,
        fft_size: int = 2048,
        hop_size: int = 512,
        window: str = "hann"
    ):
        """Initialize STFT loss.
        
        Args:
            fft_size: FFT size.
            hop_size: Hop size.
            window: Window function ("hann", "hamming", etc).
        """
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        
        if window == "hann":
            self.window = torch.hann_window(fft_size)
        elif window == "hamming":
            self.window = torch.hamming_window(fft_size)
        else:
            raise ValueError(f"Unknown window: {window}")
    
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute STFT loss.
        
        Args:
            output: Predicted waveform [batch, 1, T].
            target: Target waveform [batch, 1, T].
            
        Returns:
            Scalar loss.
        """
        # Ensure window is on correct device
        window = self.window.to(output.device)
        
        # Compute STFT
        output_stft = torch.stft(
            output.squeeze(1),
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            window=window,
            return_complex=True
        )
        target_stft = torch.stft(
            target.squeeze(1),
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            window=window,
            return_complex=True
        )
        
        # Magnitude spectrogram
        output_mag = torch.abs(output_stft)
        target_mag = torch.abs(target_stft)
        
        # Log magnitude loss
        eps = 1e-6
        loss = F.mse_loss(
            torch.log(output_mag + eps),
            torch.log(target_mag + eps)
        )
        
        return loss


class MultiResolutionSTFTLoss(nn.Module):
    """Multi-resolution STFT loss.
    
    Computes STFT loss at multiple resolutions and averages.
    """
    
    def __init__(
        self,
        fft_sizes: list = [2048, 1024, 512],
        hop_sizes: list = [512, 256, 128],
        window: str = "hann"
    ):
        """Initialize multi-resolution STFT loss.
        
        Args:
            fft_sizes: List of FFT sizes.
            hop_sizes: List of hop sizes.
            window: Window function.
        """
        super().__init__()
        
        if len(fft_sizes) != len(hop_sizes):
            raise ValueError("fft_sizes and hop_sizes must have same length")
        
        self.losses = nn.ModuleList([
            STFTLoss(fft_size, hop_size, window)
            for fft_size, hop_size in zip(fft_sizes, hop_sizes)
        ])
    
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute multi-resolution STFT loss.
        
        Returns:
            Average loss across all resolutions.
        """
        loss = torch.tensor(0.0, device=output.device)
        for stft_loss in self.losses:
            loss = loss + stft_loss(output, target)
        return loss / len(self.losses)


class SpectralConvergence(nn.Module):
    """Spectral Convergence loss (from Mehri et al.)."""
    
    def __init__(self, fft_size: int = 2048, hop_size: int = 512):
        """Initialize spectral convergence loss.
        
        Args:
            fft_size: FFT size.
            hop_size: Hop size.
        """
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.window = torch.hann_window(fft_size)
    
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute spectral convergence.
        
        SC = ||mag(target) - mag(output)||_F / ||mag(target)||_F
        """
        window = self.window.to(output.device)
        
        output_stft = torch.stft(
            output.squeeze(1),
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            window=window,
            return_complex=True
        )
        target_stft = torch.stft(
            target.squeeze(1),
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            window=window,
            return_complex=True
        )
        
        output_mag = torch.abs(output_stft)
        target_mag = torch.abs(target_stft)
        
        numerator = torch.norm(target_mag - output_mag, p='fro')
        denominator = torch.norm(target_mag, p='fro')
        
        return numerator / (denominator + 1e-8)
