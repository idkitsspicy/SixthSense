import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/aparn/Downloads/rampage/emotions_log.csv", engine="python", on_bad_lines="skip")

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.dropna(subset=["timestamp"])

# emotion distribution
plt.figure(figsize=(8,4))
df["emotion"].value_counts().plot(kind="bar")
plt.title("Face Emotion Distribution")
plt.xlabel("Emotion")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# confidence over time (last 200 frames)
last = df.tail(200)
plt.figure(figsize=(10,3))
plt.plot(last["timestamp"], last["confidence"])
plt.title("Face Confidence Over Time (last 200 frames)")
plt.xlabel("Time")
plt.ylabel("Confidence")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
