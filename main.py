import cv2
import dlib

# Function to draw the rectangle and get/put the face count


def draw_rectangle_and_label(frame, face, face_count):
    x, y, x1, y1 = face.left(), face.top(), face.right(), face.bottom()
    cv2.rectangle(frame, (x, y), (x1, y1), (240, 113, 202), 2)
    label = f'Face #{face_count}'
    cv2.putText(frame, label, (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


cap = cv2.VideoCapture(0)

# Get the coordinates of the face
detector = dlib.get_frontal_face_detector()

while True:
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the face
    faces = detector(gray)

    face_count = 0

    for face in faces:
        face_count += 1
        draw_rectangle_and_label(frame, face, face_count)

    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
