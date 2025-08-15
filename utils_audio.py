import torch
import torch.nn.functional as F

def add_white_noise(clean_audio: torch.Tensor, snr_db: float = 10) -> torch.Tensor:
    """
    Añade ruido blanco a un tensor 1D de audio manteniendo un SNR objetivo en dB.
    """
    if clean_audio.dim() != 1:
        raise ValueError("add_white_noise espera un tensor 1D (num_samples)")

    clean = clean_audio.float()
    rms_signal = torch.sqrt(torch.mean(clean ** 2) + 1e-12)

    noise = torch.randn_like(clean)
    rms_noise = torch.sqrt(torch.mean(noise ** 2) + 1e-12)

    desired_rms_noise = rms_signal / (10 ** (snr_db / 20))
    noise = noise * (desired_rms_noise / (rms_noise + 1e-12))

    return clean + noise

def pad_or_trim(signal: torch.Tensor, target_length: int) -> torch.Tensor:
    """
    Recorta o rellena con ceros un tensor 1D hasta "target_length" muestras.
    """
    if signal.dim() != 1:
        raise ValueError("pad_or_trim espera un tensor 1D (num_samples)")
    current = signal.numel()
    if current == target_length:
        return signal
    if current > target_length:
        return signal[:target_length]
    pad = target_length - current
    return F.pad(signal, (0, pad))
