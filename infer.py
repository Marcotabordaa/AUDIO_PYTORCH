import os
import argparse

import torch
import torchaudio

from model_unet import DenoiserUNetWrapper


def make_output_path(input_path: str, out_path: str | None) -> str:
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        return out_path
    base = os.path.splitext(os.path.basename(input_path))[0]
    out_dir = os.path.join("output", "infer")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{base}_denoised.wav")


def load_checkpoint(checkpoint_path: str):
    if not os.path.exists(checkpoint_path):
        alt = os.path.join("checkpoints", "unet2d_last.pt")
        if os.path.exists(alt):
            print(f"Checkpoint no encontrado: {checkpoint_path}. Usando: {alt}")
            checkpoint_path = alt
        else:
            raise FileNotFoundError(f"No se encontró checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = ckpt.get("config", {})
    return ckpt, cfg


def main():
    parser = argparse.ArgumentParser(description="Inferencia de U-Net 2D denoiser sobre audio ruidoso")
    parser.add_argument("--checkpoint", type=str, default=os.path.join("checkpoints", "unet2d_best.pt"))
    parser.add_argument("--input", type=str, required=True, help="Ruta al audio ruidoso (.wav/.flac)")
    parser.add_argument("--output", type=str, default=None, help="Ruta de salida .wav (por defecto: output/infer/<nombre>_denoised.wav)")
    parser.add_argument("--target_sr", type=int, default=0, help="SR de destino (0 = usar el del checkpoint)")
    parser.add_argument("--no_normalize", action="store_true", help="No normalizar por máximo antes de inferir")
    args = parser.parse_args()

    ckpt, cfg = load_checkpoint(args.checkpoint)
    target_sr = args.target_sr if args.target_sr > 0 else cfg.get("target_sr", 48000)
    n_fft = cfg.get("n_fft", 512)
    hop = cfg.get("hop_length", 128)
    win = cfg.get("win_length", 512)
    base_channels = cfg.get("base_channels", 32)

    # Cargar audio de entrada
    wave, sr = torchaudio.load(args.input)  # [C, T]
    if wave.shape[0] > 1:
        wave = wave.mean(dim=0, keepdim=True)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        wave = resampler(wave)

    # Normalización como en el dataset, pero guardando escala para restaurar
    orig_max = wave.abs().max().item()
    if not args.no_normalize:
        wave = wave / (orig_max + 1e-12)

    # Modelo
    model = DenoiserUNetWrapper(n_fft=n_fft, hop_length=hop, win_length=win, base_channels=base_channels)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with torch.no_grad():
        inp = wave.squeeze(0).unsqueeze(0)  # [1, T]
        denoised = model(inp).squeeze(0)    # [T]

    # Restaurar escala si se normalizó
    if not args.no_normalize and orig_max > 0:
        denoised = denoised * orig_max

    out_path = make_output_path(args.input, args.output)
    torchaudio.save(out_path, denoised.unsqueeze(0).to(torch.float32), target_sr)
    print(f"Guardado audio denoised en: {out_path}")


if __name__ == "__main__":
    main()


