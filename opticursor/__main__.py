# from facemesh.facemesh import FaceMeshApp

# class Main:
#     def __init__(self):
#         self.face_mesh_app = FaceMeshApp()

#     def start(self, withDrawing = True):
#         print("Starting Face Mesh Application...")
#         self.face_mesh_app.run(withDrawing)

# if __name__ == "__main__":
#     main_app = Main()
#     main_app.start(False)

import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize Mediapipe for face and eye detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Set up webcam
cap = cv2.VideoCapture(0)

# Initialize face detector
with mp_face_detection.FaceDetection(min_detection_confidence=0.2) as face_detection:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to RGB for mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)
        
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                
                # Draw bounding box around face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Track eyes based on detected face
                eye_center_x = int(x + w / 2)
                eye_center_y = int(y + h / 2)
                
                # Get screen dimensions
                screen_width, screen_height = pyautogui.size()

                # Map eye position to screen coordinates
                mouse_x = np.interp(eye_center_x, [0, iw], [0, screen_width])
                mouse_y = np.interp(eye_center_y, [0, ih], [0, screen_height])

                # Move mouse cursor
                pyautogui.moveTo(mouse_x, mouse_y)

        # Display the video feed with face and eye tracking
        cv2.imshow("Eye Tracking Cursor", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release webcam and close all windows
cap.release()
cv2.destroyAllWindows()
