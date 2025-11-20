"""
WORKSHOP 2 - STEP 1: Basic Hand Detection
==========================================
Goal: Detect a hand and draw landmarks on it in real-time

What you'll learn:
- How to open your webcam with OpenCV
- How to use MediaPipe to detect hands
- How to draw hand landmarks (21 points on your hand)

Total lines: ~30 (excluding comments)
"""



import cv2
import mediapipe as mp



# STEP 1: Initialize MediaPipe Hands
# MediaPipe is Google's library for hand tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils



# Configure hand detection
# max_num_hands=1 means we only track one hand (simpler for beginners)
# min_detection_confidence=0.7 means 70% sure it's a hand before tracking
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)



# STEP 2: Open webcam
# 0 means use the first camera on your computer
cap = cv2.VideoCapture(0)



print("Hand Detection Started!")
print("Press 'q' to quit")



# STEP 3: Main loop - runs continuously
while True:
    # Read a frame from the webcam
    success, frame = cap.read()

    if not success:
        print("Failed to grab frame")
        break

    # Flip the frame horizontally (makes it mirror-like, more natural)
    frame = cv2.flip(frame, 1)

    # Convert BGR (OpenCV format) to RGB (MediaPipe format)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # STEP 4: Detect hands in the frame
    results = hands.process(rgb_frame)

    # STEP 5: Draw hand landmarks if a hand is detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw all 21 hand landmarks and connections between them
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    # STEP 6: Display the frame
    cv2.imshow('Step 1: Hand Detection', frame)

    # STEP 7: Check if user pressed 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

                    

# STEP 8: Clean up - release camera and close windows
cap.release()
cv2.destroyAllWindows()
hands.close()



print("Hand detection stopped!")
