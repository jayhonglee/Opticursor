from facemesh.facemesh import FaceMeshApp

class Main:
    def __init__(self):
        self.face_mesh_app = FaceMeshApp()

    def start(self):
        print("Starting Face Mesh Application...")
        self.face_mesh_app.run()

if __name__ == "__main__":
    main_app = Main()
    main_app.start()