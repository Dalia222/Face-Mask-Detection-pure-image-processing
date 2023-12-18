import os
import cv2

# Assuming all your images are in the same folder
folder_path = './testing_images'

def detect_mouth(face_roi):
    mouth_cascade = cv2.CascadeClassifier('./mouth.xml')
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

    mouths = mouth_cascade.detectMultiScale(gray, scaleFactor=3, minNeighbors=11, minSize=(10, 10))

    print(f"Number of mouths detected: {len(mouths)}")

    return len(mouths) > 0


def detect_edges(image, threshold1, threshold2):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, threshold1, threshold2)
    return edges

def detect_and_label_faces(image_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Unable to read the image at {image_path}")
        return

    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_roi = image[y:y+h, x:x+w]

        # Uncomment the following lines if you have the corresponding XML files
        # mask_edges = detect_mask_edges(face_roi)
        # face_edges = detect_edges(face_roi, 30, 150)

        # Detect mouth
        has_mouth = detect_mouth(face_roi)

        color = (0, 255, 0)
        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)

        # Uncomment the following lines if you have the corresponding XML files
        # cv2.imshow("Mask Edges", mask_edges)
        # cv2.imshow("Face Edges", face_edges)

        if has_mouth:
            label = "No Mask"
            color = (0, 0, 255)
        else:
            label = "Mask"
            color = (0, 255, 0)

        cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)

    cv2.imshow("Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(('.jpg', '.png', '.jpeg')):
        image_path = os.path.join(folder_path, filename)
        detect_and_label_faces(image_path)
