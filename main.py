from torch.utils.data import DataLoader
from datasets import LibriSpeechNoisyDataset
import torchaudio
import os
import matplotlib.pyplot as plt
import numpy as np
from utils_audio import pad_or_trim, add_white_noise
import torch

if __name__ == "__main__":
    dataset = LibriSpeechNoisyDataset(
        root_dir="LibriSpeech/dev-clean",  # Cambia a tu ruta
        target_sr=16000,
        snr_db=10,
        fixed_length_sec=5.0,
    )

    # Visualización: solo 2 archivos, cada figura con PRE (arriba) y POST-corte (abajo)
    preview_dir = os.path.join("output")
    os.makedirs(preview_dir, exist_ok=True)
    audio_dir = os.path.join(preview_dir, "audio_previews")
    os.makedirs(audio_dir, exist_ok=True)

    sample_paths = dataset.audio_files[:2]
    for i, file_path in enumerate(sample_paths, start=1):
        waveform, sr = torchaudio.load(file_path)

        # Convertir a mono si es necesario
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample si es necesario
        if sr != dataset.target_sr:
            resampler = torchaudio.transforms.Resample(sr, dataset.target_sr)
            waveform = resampler(waveform)
            sr = dataset.target_sr

        # Normalizar con epsilon (igual que en el dataset)
        max_val = waveform.abs().max()
        waveform = waveform / (max_val + 1e-12)
        mono = waveform.squeeze(0)

        # Señal recortada/rellenada a la longitud fija del dataset
        mono_fixed = pad_or_trim(mono, dataset.fixed_length)
        # Señal con ruido al SNR objetivo
        mono_noisy = add_white_noise(mono_fixed, dataset.snr_db)
        # Clamp por seguridad para exportar
        mono_fixed = torch.clamp(mono_fixed, -1.0, 1.0)
        mono_noisy = torch.clamp(mono_noisy, -1.0, 1.0)

        # Tiempos
        t_pre = np.arange(mono.numel()) / sr
        t_post = np.arange(mono_fixed.numel()) / sr

        # Figura con tres subplots: PRE, POST limpio, POST con ruido
        fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=False)
        axes[0].plot(t_pre, mono.numpy())
        axes[0].set_title(f"PRE corte: {os.path.basename(file_path)} | dur: {mono.numel()/sr:.2f}s @ {sr}Hz")
        axes[0].set_ylabel("Amplitud")

        axes[1].plot(t_post, mono_fixed.numpy(), color='tab:orange')
        axes[1].set_title(f"POST corte (limpio): {dataset.fixed_length/sr:.2f}s (longitud fija)")
        axes[1].set_xlabel("Tiempo (s)")
        axes[1].set_ylabel("Amplitud")

        axes[2].plot(t_post, mono_noisy.numpy(), color='tab:green')
        axes[2].set_title(f"POST corte (ruido {dataset.snr_db} dB SNR)")
        axes[2].set_xlabel("Tiempo (s)")
        axes[2].set_ylabel("Amplitud")

        fig.tight_layout()
        out_path = os.path.join(preview_dir, f"preview_{i}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Guardado: {out_path}")

        # Guardar audios WAV (mono, float32)
        clean_wav_path = os.path.join(audio_dir, f"sample_{i}_clean.wav")
        noisy_wav_path = os.path.join(audio_dir, f"sample_{i}_noisy.wav")
        torchaudio.save(clean_wav_path, mono_fixed.unsqueeze(0).to(torch.float32), sr)
        torchaudio.save(noisy_wav_path, mono_noisy.unsqueeze(0).to(torch.float32), sr)
        print(f"Guardado: {clean_wav_path}")
        print(f"Guardado: {noisy_wav_path}")

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Ejemplo de loop
    for noisy_batch, clean_batch in dataloader:
        print("Noisy batch shape:", noisy_batch.shape)
        print("Clean batch shape:", clean_batch.shape)
        print(len(dataset.audio_files))   # Cantidad total de archivos
        break





