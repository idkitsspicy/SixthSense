from .face_detector import detect_face
from .emotion_recognizer import recognize_emotion
from .database import log_emotion

def process_frame(frame):
    face, bbox = detect_face(frame)

    if face is None:
        return None, None

    emotions, dominant, confidence = recognize_emotion(face)
    log_emotion(dominant, confidence)

    return {
        "emotions": emotions,
        "dominant_emotion": dominant,
        "confidence": confidence
    }, bbox