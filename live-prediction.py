import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("me-vs-others-model.keras")
class_labels = ["ME", "OTHERS"]

threshold = 0.15
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def preprocess_face(face_crop):
    face_resized = cv2.resize(face_crop, (224, 224))
    face_normalized = face_resized / 255.0
    return np.expand_dims(face_normalized, axis=0)

def predict_label(image, face_coordinates):
    x, y, width, height = face_coordinates
    x, y = max(0, x), max(0, y)
    face_crop = image[y:y + height, x:x + width]

    face_input = preprocess_face(face_crop)
    predictions = model.predict(face_input, verbose=0)
    prediction_score = predictions[0][0]
    label = "me" if prediction_score < threshold else "others"
    return label, prediction_score

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    for face_coordinates in faces:
        label, score = predict_label(frame, face_coordinates)

        x, y, width, height = face_coordinates
        color = (0, 255, 0) if label == "me" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
        cv2.putText(frame, f"{label} ({score:.4f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Live Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quit signal received. Exiting.")
        break

cap.release()
cv2.destroyAllWindows()
