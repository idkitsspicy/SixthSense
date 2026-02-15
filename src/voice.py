import os
import sounddevice as sd
import soundfile as sf

import numpy as np
import torch
import torchaudio
from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor


# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO_PATH = os.path.join(BASE_DIR, "audio", "live.wav")

# ---------------- MODEL LOAD (load once) ----------------
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-large-ls960-ft")
model = HubertForSequenceClassification.from_pretrained(
    "xbgoose/hubert-speech-emotion-recognition-russian-dusha-finetuned"
)
model.eval()

num2emotion = {0: "neutral", 1: "angry", 2: "positive", 3: "sad", 4: "other"}


# ---------------- LABEL MAPPING (6 emotions) ----------------
def map_to_targets_6(label: str) -> str:
    """
    Maps voice model labels -> 6 required classes.
    """
    label = label.lower()

    if label == "positive":
        return "Happy"
    if label == "sad":
        return "Sad"
    if label == "angry":
        return "Angry"
    if label == "neutral":
        return "Neutral"
    return "Confused"  # 'other' -> Confused


# ---------------- RECORD MIC ----------------
def record_audio(out_file=AUDIO_PATH, seconds=5, sr=16000):
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    print("🎙️ Recording... Speak now!")
    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    sf.write(out_file, audio, sr)
    print("✅ Saved recording:", out_file)
    return out_file


# ---------------- LOAD WAV (NO TORCHCODEC) ----------------
def load_wav(filepath: str, target_sr=16000):
    audio, sr = sf.read(filepath)  # numpy

    waveform = torch.tensor(audio).float()
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    else:
        waveform = waveform.T

    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        sr = target_sr

    return waveform, sr


# ---------------- ENERGY (for Excited) ----------------
def compute_energy(waveform: torch.Tensor) -> float:
    """
    waveform: [1, T]
    Returns RMS energy.
    """
    x = waveform.squeeze(0).cpu().numpy()
    rms = float(np.sqrt(np.mean(x ** 2) + 1e-9))
    return rms


# ---------------- PREDICT VOICE ----------------
def predict_voice(filepath: str):
    waveform, sr = load_wav(filepath, target_sr=16000)

    inputs = feature_extractor(
        waveform,
        sampling_rate=sr,
        return_tensors="pt",
        padding=True,
        max_length=16000 * 10,
        truncation=True,
    )

    with torch.no_grad():
        logits = model(inputs["input_values"][0]).logits  # [1, 5]

    probs = torch.softmax(logits, dim=-1)[0]
    pred_idx = int(torch.argmax(probs))
    voice_conf = float(probs[pred_idx])

    raw_emotion = num2emotion[pred_idx]
    voice_label = map_to_targets_6(raw_emotion)

    # Excited detection: if positive + high energy
    energy = compute_energy(waveform)
    if voice_label == "Happy" and energy > 0.03:
             voice_label = "Excited"


    return voice_label, voice_conf, raw_emotion, energy


if __name__ == "__main__":
    record_audio(AUDIO_PATH, seconds=5, sr=16000)
    voice_label, voice_conf, raw, energy = predict_voice(AUDIO_PATH)

    print("\n🎧 Voice Emotion Results")
    print("Raw model output :", raw)
    print("Mapped emotion   :", voice_label)
    print("Confidence       :", round(voice_conf, 3))
    print("Energy (RMS)     :", round(energy, 4))
