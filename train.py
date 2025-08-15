import os
import math
import argparse
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from datasets import LibriSpeechNoisyDataset
from model_unet import DenoiserUNetWrapper


def build_dataloaders(
    root_dir: str,
    target_sr: int,
    fixed_length_sec: float,
    snr_db: float,
    batch_size: int,
    num_workers: int,
    val_ratio: float = 0.1,
) -> Tuple[DataLoader, DataLoader]:
    dataset = LibriSpeechNoisyDataset(
        root_dir=root_dir,
        target_sr=target_sr,
        snr_db=snr_db,
        fixed_length_sec=fixed_length_sec,
    )

    val_len = max(1, int(len(dataset) * val_ratio))
    train_len = len(dataset) - val_len
    train_set, val_set = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=(num_workers > 0),
    )
    return train_loader, val_loader


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    criterion = nn.L1Loss()
    running = 0.0
    num_batches = 0
    pbar = tqdm(loader, desc="Train", leave=False)
    for noisy, clean in pbar:
        noisy = noisy.to(device)
        clean = clean.to(device)

        optimizer.zero_grad(set_to_none=True)
        denoised = model(noisy)
        loss = criterion(denoised, clean)
        loss.backward()
        optimizer.step()

        running += loss.item()
        num_batches += 1
        avg = running / max(1, num_batches)
        pbar.set_postfix({"loss": f"{avg:.5f}"})
    return running / max(1, num_batches)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    criterion = nn.L1Loss()
    running = 0.0
    num_batches = 0
    pbar = tqdm(loader, desc="Val", leave=False)
    for noisy, clean in pbar:
        noisy = noisy.to(device)
        clean = clean.to(device)
        denoised = model(noisy)
        loss = criterion(denoised, clean)
        running += loss.item()
        num_batches += 1
        avg = running / max(1, num_batches)
        pbar.set_postfix({"loss": f"{avg:.5f}"})
    return running / max(1, num_batches)


def main():
    parser = argparse.ArgumentParser(description="Entrenamiento U-Net 2D para denoising")
    parser.add_argument("--root_dir", type=str, default="LibriSpeech/dev-clean")
    parser.add_argument("--target_sr", type=int, default=16000)
    parser.add_argument("--fixed_length_sec", type=float, default=5.0)
    parser.add_argument("--snr_db", type=float, default=10.0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--n_fft", type=int, default=512)
    parser.add_argument("--hop_length", type=int, default=128)
    parser.add_argument("--win_length", type=int, default=512)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    device = torch.device("cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    train_loader, val_loader = build_dataloaders(
        root_dir=args.root_dir,
        target_sr=args.target_sr,
        fixed_length_sec=args.fixed_length_sec,
        snr_db=args.snr_db,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = DenoiserUNetWrapper(
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        base_channels=args.base_channels,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = math.inf
    best_path = os.path.join(args.out_dir, "unet2d_best.pt")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:02d} | train L1: {train_loss:.5f} | val L1: {val_loss:.5f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "config": {
                    "n_fft": args.n_fft,
                    "hop_length": args.hop_length,
                    "win_length": args.win_length,
                    "base_channels": args.base_channels,
                    "target_sr": args.target_sr,
                },
                "val_loss": best_val,
            }, best_path)
            print(f"Guardado mejor modelo en: {best_path}")


if __name__ == "__main__":
    main()


