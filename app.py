import streamlit as st
import subprocess
import sys

st.set_page_config(page_title="Emotion AI Control Panel", layout="centered")

st.title("🧠 Multimodal Emotion AI System")
st.write("Control the Face Emotion Agent and Fusion Agent")

# Store face agent process
if "face_process" not in st.session_state:
    st.session_state.face_process = None


# ================= FACE AGENT =================
st.subheader("👁 Face Emotion Agent (Webcam)")

col1, col2 = st.columns(2)

with col1:
    if st.button("▶ Start Face Agent"):
        if st.session_state.face_process is None:
            try:
                st.session_state.face_process = subprocess.Popen(
                    [sys.executable, "main.py"]
                )
                st.success("Face Agent started. Webcam window opened.")
            except Exception as e:
                st.error(f"Error starting Face Agent: {e}")
        else:
            st.warning("Face Agent already running.")

with col2:
    if st.button("⛔ Stop Face Agent"):
        if st.session_state.face_process:
            st.session_state.face_process.terminate()
            st.session_state.face_process = None
            st.success("Face Agent stopped.")
        else:
            st.warning("Face Agent is not running.")


# ================= FUSION AGENT =================
st.subheader("🎙 Fusion Agent (Voice + Face + Memory)")

st.write("⚠ Make sure Face Agent is running before fusion.")

if st.button("🚀 Run Fusion Agent"):
    st.info("Fusion Agent running... Check TERMINAL for voice recording and logs.")

    try:
        # Runs normally in terminal (no output capture)
        subprocess.Popen([sys.executable, "fusion_agent.py"])
        st.success("Fusion Agent started successfully.")
    except Exception as e:
        st.error(f"Fusion Agent failed: {e}")


# ================= FOOTER =================
st.markdown("---")
st.caption("Press 'q' in webcam window to close camera manually.")