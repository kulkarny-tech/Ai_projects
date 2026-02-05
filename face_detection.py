import cv2

# Load Haar Cascade file
cascade_path = "C:\\Users\\kulka\\Downloads\\haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

# Open webcam
cam = cv2.VideoCapture(0)   # change to 1 if 0 doesn't work

if not cam.isOpened():
    print("Camera not accessible")
    exit()

while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed to capture image")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4)

    # Draw rectangle around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 225, 0), 2)

    # Show output
    cv2.imshow("Face Detection", frame)

    # Press ESC to exit
    if cv2.waitKey(10) == 27:
        break

cam.release()
cv2.destroyAllWindows()
