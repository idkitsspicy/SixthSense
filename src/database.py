import pandas as pd
from datetime import datetime
import os

DB_FILE = "emotions_log.csv"

def log_emotion(emotion, confidence):
    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "emotion": emotion,
        "confidence": round(confidence, 3)
        
    }

    df = pd.DataFrame([record])

    if not os.path.exists(DB_FILE):
        df.to_csv(DB_FILE, index=False)
    else:
        df.to_csv(DB_FILE, mode="a", header=False, index=False)