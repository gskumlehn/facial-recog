import cv2
import os
from mtcnn.mtcnn import MTCNN

detector = MTCNN()

def process_data(source_folder  , output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(source_folder):
        image_path = os.path.join(source_folder, filename)
        image = cv2.imread(image_path)

        if image is None:
            continue

        faces = detector.detect_faces(image)

        for i, face in enumerate(faces):
            x, y, w, h = face["box"]
            face_image = image[y:y + h, x:x + w]

            face_resized = cv2.resize(face_image, (224, 224))
            output_image_path = os.path.join(output_folder, f"{filename}_{i}.jpg")
            cv2.imwrite(output_image_path, face_resized)

    print("Face extraction complete.")

process_data("my-pictures", "data/me")
process_data("OIDv6/train/person", "data/others")