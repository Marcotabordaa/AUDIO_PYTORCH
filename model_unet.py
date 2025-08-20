import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


def center_crop_like(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Recorta 'source' al tamaño espacial de 'target' centrado.
    Útil cuando por divisiones/impares las dimensiones no coinciden exactamente.
    """
    _, _, h, w = source.shape
    _, _, ht, wt = target.shape
    dh = max((h - ht) // 2, 0)
    dw = max((w - wt) // 2, 0)
    return source[:, :, dh:dh + ht, dw:dw + wt]


class UNet2D(nn.Module):
    def __init__(self, in_channels: int = 1, base_channels: int = 32):
        super().__init__()
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8

        # Encoder
        self.enc1 = ConvBlock2D(in_channels, c1)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock2D(c1, c2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock2D(c2, c3)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock2D(c3, c4)

        # Decoder
        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = ConvBlock2D(c4, c3)

        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock2D(c3, c2)

        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = ConvBlock2D(c2, c1)

        self.out_conv = nn.Conv2d(c1, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))

        # Bottleneck
        xb = self.bottleneck(self.pool3(x3))

        # Decoder con concatenaciones (skip connections)
        u3 = self.up3(xb)
        x3c = center_crop_like(x3, u3)
        d3 = self.dec3(torch.cat([u3, x3c], dim=1))

        u2 = self.up2(d3)
        x2c = center_crop_like(x2, u2)
        d2 = self.dec2(torch.cat([u2, x2c], dim=1))

        u1 = self.up1(d2)
        x1c = center_crop_like(x1, u1)
        d1 = self.dec1(torch.cat([u1, x1c], dim=1))

        logits = self.out_conv(d1)
        mask = torch.sigmoid(logits)
        return mask


class DenoiserUNetWrapper(nn.Module):
    """
    Wrapper que recibe audio ruidoso en el tiempo [B, T],
    aplica STFT -> U-Net 2D para predecir máscara de magnitud -> ISTFT,
    y devuelve audio denoised [B, T].
    """
    def __init__(self, n_fft: int = 512, hop_length: int = 128, win_length: int = 512, base_channels: int = 32):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = torch.hann_window(win_length)
        self.unet = UNet2D(in_channels=1, base_channels=base_channels)

    def stft(self, wave: torch.Tensor) -> torch.Tensor:
        return torch.stft(
            wave,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(device=wave.device, dtype=wave.dtype),
            center=True,
            return_complex=True,
        )

    def istft(self, spec: torch.Tensor, length: int) -> torch.Tensor:
        # Llamada dinámica para evitar falsos positivos de algunos linters/stubs
        istft_fn = getattr(torch, "istft")
        return istft_fn(  # type: ignore
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(device=spec.device, dtype=spec.real.dtype),
            center=True,
            length=length,
        )

    def forward(self, noisy_wave: torch.Tensor) -> torch.Tensor:
        """
        noisy_wave: [B, T]
        return: [B, T]
        """
        spec = self.stft(noisy_wave)               # [B, F, frames]
        mag = spec.abs()
        phase = torch.angle(spec)

        # Log-magnitude para estabilidad numérica
        log_mag = torch.log1p(mag)
        net_in = log_mag.unsqueeze(1)              # [B, 1, F, frames]

        mask = self.unet(net_in)                   # [B, 1, ~F, ~frames]
        # Asegurar tamaño exacto a la entrada del espectrograma
        if mask.shape[-2:] != mag.shape[-2:]:
            mask = F.interpolate(mask, size=mag.shape[-2:], mode="bilinear", align_corners=False)
        mask = mask.clamp(0.0, 1.0)

        mag_est = mask.squeeze(1) * mag            # desenmascarar sobre magnitud real
        spec_est = torch.polar(mag_est, phase)     # reconstrucción compleja

        denoised = self.istft(spec_est, length=noisy_wave.shape[-1])
        return denoised


