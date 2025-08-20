import torchaudio
import torch

# --- Cargar los audios ---
clean_path = "clean_infer.L.wav"
enhanced_path = "enh16.L.wav"

clean, sr1 = torchaudio.load(clean_path)
enhanced, sr2 = torchaudio.load(enhanced_path)
print("Sample rate clean:", sr1)
print("Sample rate enhanced:", sr2)

# Asegurar mismo sample rate
assert sr1 == sr2, "Los audios deben tener el mismo sample rate"

# Ajustar longitudes (recorte al mínimo)
min_len = min(clean.shape[1], enhanced.shape[1])
clean = clean[:, :min_len]
enhanced = enhanced[:, :min_len]

# --- Métricas ---
def compute_snr(clean, enhanced):
    noise = clean - enhanced
    snr = 10 * torch.log10(torch.sum(clean**2) / (torch.sum(noise**2) + 1e-8))
    return snr.item()

def compute_sdr(clean, enhanced):
    alpha = torch.sum(clean * enhanced) / (torch.sum(clean**2) + 1e-8)
    e_true = alpha * clean
    e_res = enhanced - e_true
    sdr = 10 * torch.log10(torch.sum(e_true**2) / (torch.sum(e_res**2) + 1e-8))
    return sdr.item()

print("SNR:", compute_snr(clean, enhanced))
print("SDR:", compute_sdr(clean, enhanced))
