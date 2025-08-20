import os
import argparse

import torch
import torchaudio

from model_unet import DenoiserUNetWrapper


def rms(x: torch.Tensor) -> torch.Tensor:
    eps = 1e-12
    return torch.sqrt(torch.mean(x.float() ** 2) + eps)


def stft_mag(wave: torch.Tensor, n_fft: int, hop: int, win: int) -> torch.Tensor:
    window = torch.hann_window(win, device=wave.device, dtype=wave.dtype)
    spec = torch.stft(
        wave, n_fft=n_fft, hop_length=hop, win_length=win,
        window=window, center=True, return_complex=True
    )
    return spec.abs()  # [F, T]


def log_spectral_distance_db(mag_a: torch.Tensor, mag_b: torch.Tensor) -> float:
    # LSD promedio en dB entre dos magnitudes espectrales
    eps = 1e-12
    p1 = mag_a.float() ** 2
    p2 = mag_b.float() ** 2
    l1 = 10.0 * torch.log10(p1 + eps)
    l2 = 10.0 * torch.log10(p2 + eps)
    lsd = torch.sqrt(torch.mean((l1 - l2) ** 2))
    return lsd.item()


def spectral_flatness(mag: torch.Tensor) -> float:
    # Promedio temporal del índice de planitud espectral (0-1). 1 ~ ruido blanco.
    eps = 1e-12
    # mag: [F, T]
    geom = torch.exp(torch.mean(torch.log(mag + eps), dim=0))
    arit = torch.mean(mag + eps, dim=0)
    sfm = (geom / arit).clamp(min=0.0, max=1.0)
    return torch.mean(sfm).item()


def make_output_path(input_path: str, out_path: str | None) -> str:
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        return out_path
    base = os.path.splitext(os.path.basename(input_path))[0]
    out_dir = os.path.join("output", "infer")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{base}_denoised.wav")


def main():
    parser = argparse.ArgumentParser(description="Inferencia sin referencia con métricas no-referenciales (proxy)")
    parser.add_argument("--checkpoint", type=str, default=os.path.join("checkpoints", "unet2d_best.pt"))
    parser.add_argument("--input", type=str, required=True, help="Ruta al audio ruidoso (.wav/.flac)")
    parser.add_argument("--output", type=str, default=None, help="Ruta de salida .wav (por defecto: output/infer/<nombre>_denoised.wav)")
    parser.add_argument("--target_sr", type=int, default=0, help="SR de destino (0 = usar el del checkpoint)")
    parser.add_argument("--no_normalize", action="store_true", help="No normalizar por máximo antes de inferir")
    args = parser.parse_args()

    # Cargar checkpoint y configuración
    if not os.path.exists(args.checkpoint):
        alt = os.path.join("checkpoints", "unet2d_last.pt")
        if os.path.exists(alt):
            print(f"Checkpoint no encontrado: {args.checkpoint}. Usando: {alt}")
            args.checkpoint = alt
        else:
            raise FileNotFoundError(f"No se encontró checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt.get("config", {})
    target_sr = args.target_sr if args.target_sr > 0 else cfg.get("target_sr", 16000)
    n_fft = cfg.get("n_fft", 512)
    hop = cfg.get("hop_length", 128)
    win = cfg.get("win_length", 512)
    base_channels = cfg.get("base_channels", 32)

    # Cargar audio de entrada
    wave, sr = torchaudio.load(args.input)  # [C, T]
    if wave.shape[0] > 1:
        wave = wave.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wave = torchaudio.transforms.Resample(sr, target_sr)(wave)

    # Normalización como en el dataset, guardando escala para restaurar
    orig_max = wave.abs().max().item()
    noisy = wave.squeeze(0)
    if not args.no_normalize:
        noisy = noisy / (orig_max + 1e-12)

    # Modelo
    model = DenoiserUNetWrapper(n_fft=n_fft, hop_length=hop, win_length=win, base_channels=base_channels)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with torch.no_grad():
        den = model(noisy.unsqueeze(0)).squeeze(0)

    # Restaurar escala si se normalizó
    if not args.no_normalize and orig_max > 0:
        den = den * orig_max

    # Guardar denoised
    out_path = make_output_path(args.input, args.output)
    torchaudio.save(out_path, den.unsqueeze(0).to(torch.float32), target_sr)
    print(f"Guardado audio denoised en: {out_path}")

    # Métricas no-referenciales (proxy) entre noisy y denoised
    # 1) Energía y cambios de nivel
    r_noisy = rms(noisy)
    r_deno = rms(den)
    resid = (noisy - den)
    r_resid = rms(resid)

    def to_dbfs(x: torch.Tensor) -> float:
        return 20.0 * torch.log10(x.item() + 1e-12)

    print(f"RMS (dBFS aprox.): noisy={to_dbfs(r_noisy):.2f} dB, denoised={to_dbfs(r_deno):.2f} dB, residual={to_dbfs(r_resid):.2f} dB")
    # 2) Reducción de ruido (proxy): energía del residual relativa al noisy
    nr_db = 20.0 * torch.log10((r_resid / (r_noisy + 1e-12)).item() + 1e-12)
    print(f"Reducción de ruido (proxy): {nr_db:.2f} dB (más negativo = mayor reducción)")

    # 3) Distancias/estadísticas espectrales
    mag_noisy = stft_mag(noisy, n_fft, hop, win)
    mag_deno = stft_mag(den, n_fft, hop, win)
    lsd = log_spectral_distance_db(mag_noisy, mag_deno)
    sfm_noisy = spectral_flatness(mag_noisy)
    sfm_deno = spectral_flatness(mag_deno)
    print(f"LSD(noisy, denoised): {lsd:.3f} dB (distancia espectral log)")
    print(f"Spectral Flatness: noisy={sfm_noisy:.3f}, denoised={sfm_deno:.3f} (menor suele ser más tonal/limpio)")


if __name__ == "__main__":
    main()




