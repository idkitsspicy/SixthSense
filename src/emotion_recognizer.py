from deepface import DeepFace

def recognize_emotion(face_img):
    result = DeepFace.analyze(
        face_img,
        actions=['emotion'],
        enforce_detection=False
    )

    emotions = result[0]["emotion"]
    dominant = result[0]["dominant_emotion"]
    confidence = max(emotions.values())

    return emotions, dominant, confidence