# import cv2
# import mediapipe as mp
# import numpy as np
# import tensorflow as tf
# import pyautogui
# from sklearn.preprocessing import StandardScaler
# import joblib  # To load the saved scaler

# pyautogui.FAILSAFE = False

# class CursorController:
#     def __init__(self, model_path, scaler_path, unique_landmark_indices):
#         self.model = tf.keras.models.load_model(model_path)
#         self.scaler = joblib.load(scaler_path)  # Load the saved scaler
#         self.face_mesh = mp.solutions.face_mesh.FaceMesh(
#             static_image_mode=False,
#             max_num_faces=1,
#             refine_landmarks=True,
#             min_detection_confidence=0.5,
#             min_tracking_confidence=0.5
#         )
#         self.unique_landmark_indices = unique_landmark_indices

#     def preprocess_landmarks(self, face_landmarks):
#         # Extract the selected landmarks (x, y, z coordinates) from the face landmarks
#         landmarks = []
#         for idx in self.unique_landmark_indices:
#             x = face_landmarks.landmark[idx].x
#             y = face_landmarks.landmark[idx].y
#             z = face_landmarks.landmark[idx].z
#             landmarks.extend([x, y, z])  # Append x, y, z for each landmark

#         landmarks = np.array(landmarks).reshape(1, -1)  # Reshape to match model input
#         return self.scaler.transform(landmarks)  # Scale the landmarks using the saved scaler

#     def predict_cursor_position(self, landmarks_scaled):
#         # Predict the mouse position using the trained model
#         predicted = self.model.predict(landmarks_scaled)
#         print(f"Predicted mouse position: {predicted}")
#         return predicted

#     def move_cursor(self, predicted_mouse_pos):
#         # Get the screen size to move the cursor correctly
#         screen_width, screen_height = pyautogui.size()
        
#         # Ensure mouse_x and mouse_y are in the range [0, 1] and move the cursor
#         mouse_x, mouse_y = predicted_mouse_pos[0]  # Extract predicted mouse position
        
#         # Clip values to ensure they are within [0, 1]
#         mouse_x = max(0, min(mouse_x, 1))
#         mouse_y = max(0, min(mouse_y, 1))
        
#         # Convert to pixel coordinates on the screen
#         pyautogui.moveTo(int(mouse_x * screen_width), int(mouse_y * screen_height))

#     def run(self):
#         cap = cv2.VideoCapture(0)
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = self.face_mesh.process(frame_rgb)

#             if results.multi_face_landmarks:
#                 for face_landmarks in results.multi_face_landmarks:
#                     landmarks_scaled = self.preprocess_landmarks(face_landmarks)
#                     predicted_mouse_pos = self.predict_cursor_position(landmarks_scaled)
#                     self.move_cursor(predicted_mouse_pos)

#             # Show the frame
#             cv2.imshow("Real-Time Cursor Control", frame)
#             if cv2.waitKey(5) & 0xFF == 27:  # ESC to exit
#                 break

#         cap.release()
#         cv2.destroyAllWindows()

# # Usage
# if __name__ == "__main__":
#     # Define unique landmark indices (e.g., face oval, eyes, etc.)
#     mp_face_mesh = mp.solutions.face_mesh
#     face_oval_indices = list(set([i for connection in mp_face_mesh.FACEMESH_FACE_OVAL for i in connection]))
#     left_eye_indices = list(set([i for connection in mp_face_mesh.FACEMESH_LEFT_EYE for i in connection]))
#     right_eye_indices = list(set([i for connection in mp_face_mesh.FACEMESH_RIGHT_EYE for i in connection]))
#     irises_indices = list(set([i for connection in mp_face_mesh.FACEMESH_IRISES for i in connection]))
#     unique_landmark_indices = sorted(set(face_oval_indices + left_eye_indices + right_eye_indices + irises_indices))

#     # Initialize the controller with the model and scaler paths
#     controller = CursorController(
#         "saved_model/cursor_control_model.keras",
#         "saved_model/scaler.pkl",
#         unique_landmark_indices
#     )
    
#     # Start controlling the cursor
#     controller.run()

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pyautogui
from sklearn.preprocessing import StandardScaler
import joblib  # To load the saved scaler

pyautogui.FAILSAFE = False

class CursorController:
    def __init__(self, model_path, scaler_path, unique_landmark_indices, sequence_length=10):
        # Load model and scaler
        self.model = tf.keras.models.load_model(model_path)
        self.scaler = joblib.load(scaler_path)  # Load the saved scaler
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.unique_landmark_indices = unique_landmark_indices
        self.sequence_length = sequence_length  # Number of frames to consider for temporal processing
        self.landmarks_sequence = []  # To store the last few frames of landmarks

    def preprocess_landmarks(self, face_landmarks):
        # Extract the selected landmarks (x, y, z coordinates) from the face landmarks
        landmarks = []
        for idx in self.unique_landmark_indices:
            x = face_landmarks.landmark[idx].x
            y = face_landmarks.landmark[idx].y
            z = face_landmarks.landmark[idx].z
            landmarks.extend([x, y, z])  # Append x, y, z for each landmark

        landmarks = np.array(landmarks).reshape(1, -1)  # Reshape to match model input
        return self.scaler.transform(landmarks)  # Scale the landmarks using the saved scaler

    def update_landmarks_sequence(self, landmarks_scaled):
        # Maintain a sequence of recent frames to handle temporal data
        if len(self.landmarks_sequence) >= self.sequence_length:
            self.landmarks_sequence.pop(0)  # Remove the oldest frame
        self.landmarks_sequence.append(landmarks_scaled)

        # Ensure the sequence is in the correct shape for LSTM (samples, timesteps, features)
        if len(self.landmarks_sequence) == self.sequence_length:
            sequence_input = np.array(self.landmarks_sequence).reshape(1, self.sequence_length, -1)  # 1 batch, sequence_length timesteps, feature size
            return sequence_input
        else:
            return None  # Not enough frames yet

    def predict_cursor_position(self, sequence_input):
        # Predict the mouse position using the trained model
        if sequence_input is not None:
            predicted = self.model.predict(sequence_input)
            print(f"Predicted mouse position: {predicted}")
            return predicted
        return None

    def move_cursor(self, predicted_mouse_pos):
        # Get the screen size to move the cursor correctly
        screen_width, screen_height = pyautogui.size()
        
        # Ensure mouse_x and mouse_y are in the range [0, 1] and move the cursor
        mouse_x, mouse_y = predicted_mouse_pos[0]  # Extract predicted mouse position
        
        # Clip values to ensure they are within [0, 1]
        mouse_x = max(0, min(mouse_x, 1))
        mouse_y = max(0, min(mouse_y, 1))
        
        # Convert to pixel coordinates on the screen
        pyautogui.moveTo(int(mouse_x * screen_width), int(mouse_y * screen_height))

    def run(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame_rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    landmarks_scaled = self.preprocess_landmarks(face_landmarks)
                    sequence_input = self.update_landmarks_sequence(landmarks_scaled)
                    
                    if sequence_input is not None:
                        predicted_mouse_pos = self.predict_cursor_position(sequence_input)
                        self.move_cursor(predicted_mouse_pos)

            # Show the frame
            cv2.imshow("Real-Time Cursor Control", frame)
            if cv2.waitKey(5) & 0xFF == 27:  # ESC to exit
                break

        cap.release()
        cv2.destroyAllWindows()

# Usage
if __name__ == "__main__":
    # Define unique landmark indices (e.g., face oval, eyes, etc.)
    mp_face_mesh = mp.solutions.face_mesh
    face_oval_indices = list(set([i for connection in mp_face_mesh.FACEMESH_FACE_OVAL for i in connection]))
    left_eye_indices = list(set([i for connection in mp_face_mesh.FACEMESH_LEFT_EYE for i in connection]))
    right_eye_indices = list(set([i for connection in mp_face_mesh.FACEMESH_RIGHT_EYE for i in connection]))
    irises_indices = list(set([i for connection in mp_face_mesh.FACEMESH_IRISES for i in connection]))
    unique_landmark_indices = sorted(set(face_oval_indices + left_eye_indices + right_eye_indices + irises_indices))

    # Initialize the controller with the model and scaler paths
    controller = CursorController(
        "saved_model/cursor_control_model.keras",
        "saved_model/scaler.pkl",
        unique_landmark_indices,
        sequence_length=10  # Number of frames for temporal input
    )
    
    # Start controlling the cursor
    controller.run()
