import os
import argparse
import math

import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np

from datasets import LibriSpeechNoisyDataset
from model_unet import DenoiserUNetWrapper


def compute_snr_db(clean: torch.Tensor, estimate: torch.Tensor) -> float:
    eps = 1e-12
    noise = estimate - clean
    rms_sig = torch.sqrt(torch.mean(clean ** 2) + eps)
    rms_nse = torch.sqrt(torch.mean(noise ** 2) + eps)
    return 20.0 * math.log10((rms_sig / (rms_nse + eps)).item() + eps)


def stft_log_mag(wave: torch.Tensor, n_fft: int, hop: int, win: int) -> np.ndarray:
    window = torch.hann_window(win, device=wave.device)
    spec = torch.stft(wave, n_fft=n_fft, hop_length=hop, win_length=win, window=window, center=True, return_complex=True)
    mag = spec.abs().cpu().numpy()
    return np.log1p(mag)


def main():
    parser = argparse.ArgumentParser(description="Evaluación/visualización de modelo U-Net 2D denoiser")
    parser.add_argument("--checkpoint", type=str, default=os.path.join("checkpoints", "unet2d_best.pt"))
    parser.add_argument("--root_dir", type=str, default="LibriSpeech/dev-clean")
    parser.add_argument("--num_samples", type=int, default=2)
    parser.add_argument("--snr_db", type=float, default=5.0)
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        alt = os.path.join("checkpoints", "unet2d_last.pt")
        if os.path.exists(alt):
            print(f"Checkpoint no encontrado: {args.checkpoint}. Usando: {alt}")
            args.checkpoint = alt
        else:
            raise FileNotFoundError(f"No se encontró checkpoint: {args.checkpoint}")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt.get("config", {})
    target_sr = cfg.get("target_sr", 16000)
    n_fft = cfg.get("n_fft", 512)
    hop = cfg.get("hop_length", 128)
    win = cfg.get("win_length", 512)

    # Dataset para obtener (noisy, clean) y también audios originales si se desea
    dataset = LibriSpeechNoisyDataset(
        root_dir=args.root_dir,
        target_sr=target_sr,
        snr_db=args.snr_db,
        fixed_length_sec=5.0,
    )

    model = DenoiserUNetWrapper(n_fft=n_fft, hop_length=hop, win_length=win, base_channels=cfg.get("base_channels", 32))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    out_dir = os.path.join("output", "eval")
    os.makedirs(out_dir, exist_ok=True)
    audio_dir = os.path.join(out_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    # Tomar primeros N ejemplos
    for i in range(min(args.num_samples, len(dataset))):
        noisy, clean = dataset[i]  # [T], [T]

        with torch.no_grad():
            denoised = model(noisy.unsqueeze(0)).squeeze(0)

        # WAVs
        clean_path = os.path.join(audio_dir, f"sample_{i+1}_clean.wav")
        noisy_path = os.path.join(audio_dir, f"sample_{i+1}_noisy.wav")
        denoised_path = os.path.join(audio_dir, f"sample_{i+1}_denoised.wav")
        torchaudio.save(clean_path, clean.unsqueeze(0).to(torch.float32), target_sr)
        torchaudio.save(noisy_path, noisy.unsqueeze(0).to(torch.float32), target_sr)
        torchaudio.save(denoised_path, denoised.unsqueeze(0).to(torch.float32), target_sr)
        print(f"Guardados WAV: {clean_path}, {noisy_path}, {denoised_path}")

        # Métricas SNR
        snr_noisy = compute_snr_db(clean, noisy)
        snr_denoised = compute_snr_db(clean, denoised)
        delta_snr = snr_denoised - snr_noisy

        # Espectrogramas (log magnitude)
        log_noisy = stft_log_mag(noisy, n_fft, hop, win)
        log_clean = stft_log_mag(clean, n_fft, hop, win)
        log_denoised = stft_log_mag(denoised, n_fft, hop, win)

        # Figura: 2 columnas (ondas y espectrogramas), 3 filas (noisy/clean/denoised)
        t = np.arange(clean.numel()) / target_sr
        fig, axes = plt.subplots(3, 2, figsize=(14, 8))

        # Waveforms
        axes[0, 0].plot(t, noisy.numpy())
        axes[0, 0].set_title(f"Noisy | SNR={snr_noisy:.2f} dB")
        axes[1, 0].plot(t, clean.numpy(), color='tab:orange')
        axes[1, 0].set_title("Clean (target)")
        axes[2, 0].plot(t, denoised.numpy(), color='tab:green')
        axes[2, 0].set_title(f"Denoised | ΔSNR={delta_snr:.2f} dB")
        for r in range(3):
            axes[r, 0].set_xlabel("Tiempo (s)")
            axes[r, 0].set_ylabel("Amplitud")

        # Spectrograms (imshow)
        axes[0, 1].imshow(log_noisy, origin='lower', aspect='auto')
        axes[0, 1].set_title("Spec Noisy (log mag)")
        axes[1, 1].imshow(log_clean, origin='lower', aspect='auto')
        axes[1, 1].set_title("Spec Clean (log mag)")
        im2 = axes[2, 1].imshow(log_denoised, origin='lower', aspect='auto')
        axes[2, 1].set_title("Spec Denoised (log mag)")
        for r in range(3):
            axes[r, 1].set_xlabel("Frames")
            axes[r, 1].set_ylabel("Frecuencias")

        fig.colorbar(im2, ax=axes[:, 1].ravel().tolist(), shrink=0.7)
        fig.tight_layout()
        fig_path = os.path.join(out_dir, f"eval_sample_{i+1}.png")
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"Guardada figura: {fig_path}")


if __name__ == "__main__":
    main()


