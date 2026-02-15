import os, json, time
import pandas as pd
from collections import Counter
from datetime import datetime
import sys
import os

# Force UTF-8 output so Windows terminal doesn't crash


from voice import record_audio, predict_voice

# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
LOG_FILE = os.path.join(BASE_DIR, "emotions_log.csv")
AUDIO_PATH = os.path.join(BASE_DIR, "audio", "live.wav")
MEMORY_FILE = os.path.join(BASE_DIR, "memory", "user_001.json")

os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
os.environ["PYTHONIOENCODING"] = "utf-8"

# ---------------- FACE HELPERS ----------------
def map_face_label(label):
    label = str(label).lower()
    if label == "happy": return "Happy"
    if label == "sad": return "Sad"
    if label == "angry": return "Angry"
    return "Neutral"  # neutral + anything else -> Neutral


def get_face_emotion_during_window(start_ts, end_ts):
    """
    Reads emotions_log.csv and returns face emotion computed from SAME window.
    Strategy: majority label in window + average confidence for that label.
    """
    if not os.path.exists(LOG_FILE):
        raise FileNotFoundError(f"Face log not found: {LOG_FILE}. Run webcam main.py first.")

    # ✅ if file corrupted anywhere, this prevents crash
    df = pd.read_csv(LOG_FILE, engine="python", on_bad_lines="skip")

    if df.empty:
        raise ValueError("Face log is empty. Wait until webcam writes at least 1 row.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    window = df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)]

    if window.empty:
        # fallback to last row
        last = df.iloc[-1]
        face_label = map_face_label(str(last["emotion"]))
        face_conf = float(last["confidence"])
        return face_label, face_conf, "fallback_last"

    window["emotion_mapped"] = window["emotion"].astype(str).apply(map_face_label)

    # majority emotion
    majority = window["emotion_mapped"].value_counts().idxmax()

    # avg confidence for majority emotion
    face_conf = float(window[window["emotion_mapped"] == majority]["confidence"].mean())

    return majority, face_conf, "window"


# ---------------- MASKING ----------------
def masking_detection(face_label, face_conf, voice_label, voice_conf, thresh=0.55):
    face_conf = float(face_conf)
    voice_conf = float(voice_conf)
    return bool((face_label != voice_label) and (face_conf >= thresh) and (voice_conf >= thresh))


# ---------------- FUSION ----------------
def fuse(face_label, face_conf, voice_label, voice_conf):
    """
    Simple fusion: pick modality with higher confidence.
    """
    return face_label if float(face_conf) >= float(voice_conf) else voice_label


# ---------------- MEMORY ----------------
def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return []
    try:
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return []

def save_memory(mem):
    with open(MEMORY_FILE, "w") as f:
        json.dump(mem, f, indent=2)

def baseline_emotion(mem, last_n=10):
    if not mem:
        return None
    finals = [x["final_emotion"] for x in mem[-last_n:]]
    return Counter(finals).most_common(1)[0][0]


# ---------------- CONTEXT-AWARE PERSONALISATION ----------------
def recent_stats(mem, last_n=10):
    """
    Stats from last N sessions: trend + masking frequency
    """
    if not mem:
        return {}

    recent = mem[-last_n:]
    finals = [x["final_emotion"] for x in recent]
    masking_count = sum(1 for x in recent if x.get("masking") is True)

    top_emotion = Counter(finals).most_common(1)[0][0]

    return {
        "n": len(recent),
        "top_emotion": top_emotion,
        "masking_count": masking_count,
        "emotion_counts": dict(Counter(finals))
    }


def get_today_stats(mem):
    """
    Today's emotion distribution + total sessions today
    """
    if not mem:
        return {"today_sessions": 0}

    today = datetime.now().date()
    today_entries = []

    for x in mem:
        try:
            dt = datetime.strptime(x["timestamp"], "%Y-%m-%d %H:%M:%S")
            if dt.date() == today:
                today_entries.append(x)
        except:
            pass

    if not today_entries:
        return {"today_sessions": 0}

    finals = [x["final_emotion"] for x in today_entries]
    masking_count = sum(1 for x in today_entries if x.get("masking") is True)

    return {
        "today_sessions": len(today_entries),
        "today_top": Counter(finals).most_common(1)[0][0],
        "today_emotions": dict(Counter(finals)),
        "today_masking": masking_count
    }


def deviation_alert(final_emotion, baseline):
    if baseline is None:
        return None
    if final_emotion != baseline:
        return f"Deviation: current mood ({final_emotion}) differs from baseline ({baseline})."
    return None


# ---------------- AGENT RESPONSE ----------------
def agent_response(final_emotion, masking, baseline=None, mem=None):
    """
    Emotion-aware + context-aware response.
    """
    # base empathetic response
    if masking:
        base_msg = (
            "Masking suspected (face ≠ voice). "
            "Sometimes people try to stay composed even when they aren’t okay. "
            "Want to share what’s going on?"
        )
    else:
        if final_emotion == "Sad":
            base_msg = "You seem sad. I’m here for you. Want to talk about it?"
        elif final_emotion == "Angry":
            base_msg = "You seem angry/frustrated. Want to vent?"
        elif final_emotion == "Happy":
            base_msg = "You seem happy.That’s nice!"
        else:
            base_msg = "You seem calm/neutral."

    # ----- context personalization -----
    personal = []

    if baseline and baseline != final_emotion:
        personal.append(f"{deviation_alert(final_emotion, baseline)}")

    if mem:
        stats = recent_stats(mem, last_n=10)
        today = get_today_stats(mem)

        # trend
        if stats.get("n", 0) >= 5:
            personal.append(
                f"Trend: In the last {stats['n']} sessions, your most frequent emotion was **{stats['top_emotion']}**."
            )

        # repeated masking
        if stats.get("masking_count", 0) >= 2:
            personal.append(
                f"Pattern: masking detected **{stats['masking_count']} times** recently."
            )

        # today summary
        if today.get("today_sessions", 0) >= 3:
            personal.append(
                f"Today summary: {today['today_sessions']} runs, dominant emotion = **{today['today_top']}**."
            )

    if personal:
        return base_msg + "\n\n" + "\n".join(personal)

    return base_msg


# ---------------- MAIN ----------------
if __name__ == "__main__":
    print("\nMake sure webcam face agent is running in another terminal:")
    print("   python main.py\n")

    # --- SAME TIME WINDOW ---
    start_ts = pd.Timestamp.now()

    print("Recording voice for 5 seconds... (look into camera & speak)")
    record_audio(AUDIO_PATH, seconds=5, sr=16000)

    end_ts = pd.Timestamp.now()

    # voice prediction
    voice_label, voice_conf, raw, voice_energy = predict_voice(AUDIO_PATH)

    # face prediction from same window
    face_label, face_conf, mode = get_face_emotion_during_window(start_ts, end_ts)

    # masking + fusion
    masking = masking_detection(face_label, face_conf, voice_label, voice_conf)
    final_emotion = fuse(face_label, face_conf, voice_label, voice_conf)

    # memory + baseline
    mem = load_memory()
    base = baseline_emotion(mem)

    entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "face": str(face_label),
        "face_conf": float(face_conf),
        "voice": str(voice_label),
        "voice_conf": float(voice_conf),
        "final_emotion": str(final_emotion),
        "masking": bool(masking),
        "window_mode": str(mode),
        "voice_raw": str(raw)
    }
    mem.append(entry)
    save_memory(mem)

    print("\n---------------- RESULTS ----------------")
    print("Face :", face_label, "| conf:", round(face_conf, 3), "| mode:", mode)
    print("Voice:", voice_label, "| conf:", round(voice_conf, 3), "| raw:", raw)
    print("FINAL:", final_emotion)
    print("MASKING:", masking)
    print("BASELINE:", base)
    print("-----------------------------------------")

    response = agent_response(final_emotion, masking, baseline=base, mem=mem)
    print("\nAgent Response:\n", response)

    print("\nMemory saved ->", MEMORY_FILE)
