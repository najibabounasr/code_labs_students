"""
WORKSHOP 3 - STEP 1: Face Detection
====================================
Goal: Detect faces in real-time using OpenCV's Haar Cascade

What you'll learn:
- What is face detection (finding faces, not identifying them)
- How to use pre-trained Haar Cascade classifier
- How to draw bounding boxes around detected faces
- Difference between detection and recognition

Total lines: ~25 (excluding comments)
"""

import cv2

# STEP 1: Load the pre-trained face detection model
# Haar Cascade is a classic machine learning approach for face detection
# This model was trained on thousands of face images
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# STEP 2: Open webcam
cap = cv2.VideoCapture(0)

print("Face Detection Started!")
print("This detects WHERE faces are, but doesn't recognize WHO they are")
print("Press 'q' to quit")

# STEP 3: Main loop
while True:
    # Read a frame from webcam
    success, frame = cap.read()

    if not success:
        print("Failed to grab frame")
        break

    # Flip frame horizontally (mirror effect)
    frame = cv2.flip(frame, 1)

    # STEP 4: Convert to grayscale
    # Haar Cascade works better with grayscale images (simpler, faster)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # STEP 5: Detect faces
    # detectMultiScale parameters:
    # - scaleFactor=1.1: How much the image size is reduced at each scale (smaller = more accurate but slower)
    # - minNeighbors=5: How many neighbors each rectangle should have to be kept (higher = fewer false positives)
    # - minSize=(30, 30): Minimum face size to detect
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # STEP 6: Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        # Draw rectangle (green color, thickness 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Add label
        cv2.putText(
            frame,
            'Face',
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # STEP 7: Display face count
    cv2.putText(
        frame,
        f'Faces detected: {len(faces)}',
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # STEP 8: Show the frame
    cv2.imshow('Step 1: Face Detection', frame)

    # Check if user pressed 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# STEP 9: Clean up
cap.release()
cv2.destroyAllWindows()

print("Face detection stopped!")
