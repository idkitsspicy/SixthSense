import cv2
from src.camera import start_camera
from src.pipeline import process_frame

#to start camera and handle error
try:
    cap = start_camera()
except RuntimeError as e:
    print(e)
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    output, bbox = process_frame(frame)  #display the box on the camera

    if output and bbox:
        x, y, w, h = bbox
        label = f"{output['dominant_emotion']} ({output['confidence']:.2f})"

        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    else:
        cv2.putText(frame, "No face detected",
                    (30,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,0,255), 2)

    cv2.imshow("Facial Emotion Recognition", frame) #shows on

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()