import cv2
import mediapipe as mp

class FaceMeshApp:
    def __init__(self, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face_mesh = mp.solutions.face_mesh

    def process_frame(self, frame, withDrawing):
        # Convert frame to RGB
        frame.flags.writeable = False
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        # Convert frame back to BGR
        frame.flags.writeable = True
        frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                if withDrawing: self.draw_landmarks(frame, face_landmarks)
                self.print_eye_landmarks(face_landmarks) # TODO: Remove later

        return frame

    def draw_landmarks(self, frame, face_landmarks):
        self.mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks,
            connections=self.mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
        )
        self.mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks,
            connections=self.mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
        )
        self.mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks,
            connections=self.mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_iris_connections_style()
        )

    # TODO: Remove later ?
    def print_eye_landmarks(self, face_landmarks):
        left_eye_indices = self.mp_face_mesh.FACEMESH_LEFT_EYE
        right_eye_indices = self.mp_face_mesh.FACEMESH_RIGHT_EYE

        print("Left Eye Landmarks:")
        for idx in left_eye_indices:
            print(f"Landmark {idx}: {face_landmarks.landmark[idx[0]]}, {face_landmarks.landmark[idx[1]]}")
        
        print("\nRight Eye Landmarks:")
        for idx in right_eye_indices:
            print(f"Landmark {idx}: {face_landmarks.landmark[idx[0]]}, {face_landmarks.landmark[idx[1]]}")

    def run(self, withDrawing):
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            frame = self.process_frame(frame, withDrawing)

            # Flip the frame horizontally for a selfie-view display
            cv2.imshow('MediaPipe Face Mesh', cv2.flip(frame, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break
        
        cap.release()
        cv2.destroyAllWindows()

