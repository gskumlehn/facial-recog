import cv2
import numpy as np
import os
from mtcnn.mtcnn import MTCNN
from tensorflow.keras.models import load_model

model = load_model("me-vs-others-model.keras")
detector = MTCNN()
class_labels = ["ME", "OTHERS"]

def recognize_and_label_faces(image_path, output_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return

    faces = detector.detect_faces(image)
    print(image_path)
    for face in faces:
        x, y, width, height = face["box"]
        x, y = max(0, x), max(0, y)
        face_crop = image[y:y+height, x:x+width]

        face_resized = cv2.resize(face_crop, (224, 224))
        face_normalized = face_resized / 255.0
        face_expanded = np.expand_dims(face_normalized, axis=0)

        predictions = model.predict(face_expanded)
        print(predictions)
        label = "ME" if predictions[0] < 0.01 else "OTHERS"

        color = (0, 255, 0) if label == "ME" else (0, 0, 255)
        cv2.rectangle(image, (x, y), (x + width, y + height), color, 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imwrite(output_path, image)

for filename in os.listdir("test"):
    recognize_and_label_faces(f"test/{filename}", f"results/{filename}")