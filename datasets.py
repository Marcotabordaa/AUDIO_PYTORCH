import os
import torchaudio
from torch.utils.data import Dataset
from utils_audio import add_white_noise, pad_or_trim

class LibriSpeechNoisyDataset(Dataset):
    """
    Dataset de LibriSpeech dev-clean que devuelve pares (noisy, clean)
    Todos los audios tendrán la misma longitud fija en muestras.
    """
    def __init__(self, root_dir: str, target_sr: int = 16000, snr_db: float = 10, fixed_length_sec: float = 2.0):
        self.root_dir = root_dir
        self.target_sr = target_sr
        self.snr_db = snr_db
        self.fixed_length = int(target_sr * fixed_length_sec)  # longitud fija en muestras

        # Listamos todos los archivos .flac
        self.audio_files = [
            os.path.join(dp, f)
            for dp, _, filenames in os.walk(root_dir)
            for f in filenames if f.endswith(".flac")
        ]
        if len(self.audio_files) == 0:
            raise RuntimeError(f"No se encontraron archivos .flac en {root_dir}")

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        file_path = self.audio_files[idx]
        waveform, sr = torchaudio.load(file_path)

        # Mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample si necesario
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            waveform = resampler(waveform)

        # Normalizar con epsilon para evitar divisiones por cero
        max_val = waveform.abs().max()
        waveform = waveform / (max_val + 1e-12)
        clean = waveform.squeeze(0)

        # ---- RECORTE O PAD ----
        clean = pad_or_trim(clean, self.fixed_length)

        # Añadir ruido
        noisy = add_white_noise(clean, self.snr_db)
        return noisy, clean

