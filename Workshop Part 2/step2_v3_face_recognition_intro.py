"""
WORKSHOP 3 - STEP 2: Face Detection with MediaPipe (v3 - WORKING)
==================================================================
Goal: Detect faces using MediaPipe 0.10.x+ API

What you'll learn:
- Face detection with MediaPipe Tasks API (latest version)
- How to draw bounding boxes and landmarks
- Extract face detection confidence scores

This version uses MediaPipe 0.10.x+ Tasks API that actually works!

Total lines: ~70 (excluding comments)
"""

import cv2
import mediapipe as mp
import numpy as np

# STEP 1: Initialize MediaPipe Face Detector using Tasks API
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# STEP 2: Open webcam
cap = cv2.VideoCapture(0)

print("Face Detection with MediaPipe Tasks API - v3 (WORKING!)")
print("=" * 50)
print("This uses the MediaPipe 0.10.x+ Tasks API")
print("Press 'q' to quit")
print("=" * 50)

# STEP 3: Configure face detector options
# Download the model first if not present
import os
model_path = os.path.join(os.path.dirname(__file__), 'detector.tflite')

options = FaceDetectorOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    min_detection_confidence=0.7
)

# STEP 4: Create face detector with context manager
with FaceDetector.create_from_options(options) as detector:

    frame_count = 0

    # STEP 5: Main loop
    while True:
        success, frame = cap.read()

        if not success:
            print("Failed to grab frame")
            break

        # Flip frame horizontally
        frame = cv2.flip(frame, 1)

        # Get frame dimensions
        frame_height, frame_width, _ = frame.shape

        # Convert to RGB and create MediaPipe Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # STEP 6: Detect faces
        # Use timestamp in milliseconds for video mode
        detection_result = detector.detect_for_video(mp_image, frame_count)
        frame_count += 1

        # STEP 7: Draw detection results
        face_count = 0
        if detection_result.detections:
            for detection in detection_result.detections:
                face_count += 1

                # Get bounding box
                bbox = detection.bounding_box
                x = int(bbox.origin_x)
                y = int(bbox.origin_y)
                w = int(bbox.width)
                h = int(bbox.height)

                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Get confidence score (if available)
                if detection.categories:
                    confidence = detection.categories[0].score
                    label = f'Face {face_count}: {confidence:.2f}'
                else:
                    label = f'Face {face_count}'

                # Draw label above the box
                cv2.putText(
                    frame,
                    label,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                # Draw keypoints if available (eyes, nose, mouth, etc)
                if detection.keypoints:
                    for keypoint in detection.keypoints:
                        kp_x = int(keypoint.x * frame_width)
                        kp_y = int(keypoint.y * frame_height)
                        cv2.circle(frame, (kp_x, kp_y), 4, (255, 0, 255), -1)

        # STEP 8: Display face count
        cv2.putText(
            frame,
            f'Faces detected: {face_count}',
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # STEP 9: Display info panel
        cv2.rectangle(frame, (10, 50), (frame_width - 10, 150), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 50), (frame_width - 10, 150), (0, 255, 0), 2)

        info_text = [
            "MediaPipe Tasks API (v3)",
            "Green boxes = detected faces",
            "Pink dots = facial keypoints"
        ]

        for i, text in enumerate(info_text):
            cv2.putText(
                frame,
                text,
                (20, 75 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

        # STEP 10: Show the frame
        cv2.imshow('Step 2 v3: MediaPipe Face Detection', frame)

        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# STEP 11: Clean up
cap.release()
cv2.destroyAllWindows()

print("\nDemo stopped!")
print("\nWhat's different from v2?")
print("  - Uses MediaPipe Tasks API (0.10.x+)")
print("  - Proper video mode with frame timestamps")
print("  - Draws facial keypoints for visualization")
print("  - Works with latest MediaPipe version!")
