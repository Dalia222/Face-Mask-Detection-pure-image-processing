import cv2
import numpy as np

def detect_mouth(face_roi):
    mouth_cascade = cv2.CascadeClassifier('./mouth.xml')
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

    mouths = mouth_cascade.detectMultiScale(gray, scaleFactor=3, minNeighbors=11, minSize=(10, 10))

    print(f"Number of mouths detected: {len(mouths)}")

    return len(mouths) > 0

def detect_and_label_faces():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    mouth_cascade = cv2.CascadeClassifier('./mouth.xml')

    # Open the camera (use 0 for the default camera)
    cap = cv2.VideoCapture(0)

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        if not ret:
            print("Error: Unable to capture frame")
            break

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]

            # # Detect mask edges
            # mask_edges = detect_mask_edges(face_roi)

            # # Detect face edges
            # face_edges = detect_edges(face_roi, 30, 150)

            # Detect mouth
            has_mouth = detect_mouth(face_roi)

            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

            # cv2.imshow("Mask Edges", mask_edges)
            # cv2.imshow("Face Edges", face_edges)

            if has_mouth:
                label = "No Mask"
                color = (0, 0, 255)
            else:
                label = "Mask"

            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # Display the frame
        cv2.imshow("Live Camera", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Call the function to start live camera processing
detect_and_label_faces()
