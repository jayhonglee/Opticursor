import cv2
import mediapipe as mp
import csv
import numpy as np
import os

class TrainingDataGenerator:
    def __init__(self, output_file="training_data.csv"):
        self.output_file = output_file
        self.data = []
        self.mouse_x, self.mouse_y = 0, 0
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_oval_indices = list(set([i for connection in self.mp_face_mesh.FACEMESH_FACE_OVAL for i in connection]))
        self.left_eye_indices = list(set([i for connection in self.mp_face_mesh.FACEMESH_LEFT_EYE for i in connection]))
        self.right_eye_indices = list(set([i for connection in self.mp_face_mesh.FACEMESH_RIGHT_EYE for i in connection]))
        self.irises_indices = list(set([i for connection in self.mp_face_mesh.FACEMESH_IRISES for i in connection]))
        self.unique_landmark_indices = sorted(set(self.face_oval_indices + self.left_eye_indices + self.right_eye_indices + self.irises_indices))

        self.fieldnames = (
            ['mouse_x', 'mouse_y'] +
            [f'landmark_{i}_x' for i in self.unique_landmark_indices] +
            [f'landmark_{i}_y' for i in self.unique_landmark_indices] +
            [f'landmark_{i}_z' for i in self.unique_landmark_indices]
        )

        # Setup OpenCV window
        cv2.namedWindow("Fullscreen", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Fullscreen", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:  # Update mouse position
            self.mouse_x, self.mouse_y = x, y

    def collect_data(self):
        cap = cv2.VideoCapture(0)
        cv2.setMouseCallback("Fullscreen", self.mouse_callback)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame_rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    landmarks = [coord for idx in self.unique_landmark_indices for coord in (
                        face_landmarks.landmark[idx].x,
                        face_landmarks.landmark[idx].y,
                        face_landmarks.landmark[idx].z
                    )]
                    self.data.append([self.mouse_x, self.mouse_y] + landmarks)

            cv2.imshow("Fullscreen", frame)
            if cv2.waitKey(5) & 0xFF == 27:  # ESC key to exit
                break

        cap.release()
        cv2.destroyAllWindows()
        self.save_data()

    def save_data(self):
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        with open(self.output_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(self.fieldnames)
            writer.writerows(self.data)
        print(f"Data saved to {self.output_file}")

if __name__ == "__main__":
    collector = TrainingDataGenerator("data/training_data.csv")
    collector.collect_data()
