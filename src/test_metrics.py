import torchaudio
from torchmetrics.audio import (
    PerceptualEvaluationSpeechQuality,
    ShortTimeObjectiveIntelligibility,
)


def compute_metrics(clean_path: str, noisy_path: str, sample_rate: int = 16000):
    print(f"Caricamento file pulito: {clean_path}")
    clean_audio, sr_clean = torchaudio.load(clean_path)

    print(f"Caricamento file rumoroso/predetto: {noisy_path}")
    noisy_audio, sr_noisy = torchaudio.load(noisy_path)

    # 1. Conversione da stereo a mono (se necessario)
    if clean_audio.ndim > 1 and clean_audio.shape[0] > 1:
        clean_audio = clean_audio.mean(dim=0, keepdim=True)
    if noisy_audio.ndim > 1 and noisy_audio.shape[0] > 1:
        noisy_audio = noisy_audio.mean(dim=0, keepdim=True)

    # 2. Resampling a 16kHz (obbligatorio per PESQ Wideband)
    if sr_clean != sample_rate:
        clean_audio = torchaudio.functional.resample(clean_audio, sr_clean, sample_rate)
    if sr_noisy != sample_rate:
        noisy_audio = torchaudio.functional.resample(noisy_audio, sr_noisy, sample_rate)

    # 3. Allineamento della lunghezza (tronca il più lungo)
    min_len = min(clean_audio.shape[-1], noisy_audio.shape[-1])
    clean_audio = clean_audio[..., :min_len]
    noisy_audio = noisy_audio[..., :min_len]

    # Assicuriamoci che i tensori siano in float32 su CPU
    clean_audio = clean_audio.cpu().float()
    noisy_audio = noisy_audio.cpu().float()

    # 4. Inizializzazione delle metriche di torchmetrics
    # mode="wb" richiede 16000 Hz
    pesq_metric = PerceptualEvaluationSpeechQuality(fs=sample_rate, mode="wb")
    stoi_metric = ShortTimeObjectiveIntelligibility(fs=sample_rate, extended=False)

    # 5. Calcolo
    pesq_score = pesq_metric(noisy_audio, clean_audio)
    stoi_score = stoi_metric(noisy_audio, clean_audio)

    print("\n--- Risultati ---")
    print(f"PESQ: {pesq_score.item():.4f}")
    print(f"STOI: {stoi_score.item():.4f}")


if __name__ == "__main__":
    # Inserisci qui i percorsi ai tuoi file .wav di test
    CLEAN_FILE = "data/audio/validation/clean/84_121123_1.wav"
    NOISY_FILE = "data/audio/validation/noisy/84_121123_1.wav"

    compute_metrics(CLEAN_FILE, NOISY_FILE)
