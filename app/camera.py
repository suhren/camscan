import cv2

from camscan import scanner


FRAME_WIDTH = 3260  # 1536  # 3264
FRAME_HEIGHT = 2444  # 1152  # 2448

TARGET_FPS = 30


class Camera:
    def __init__(
        self,
        index: int = 0,
        resolution: tuple = (FRAME_WIDTH, FRAME_HEIGHT),
    ):
        self.capture = None
        self.index = index
        self.resolution = resolution
        self.initialize()

    def initialize(self):
        self.capture = cv2.VideoCapture(self.index + cv2.CAP_DSHOW)

        if not self.capture.isOpened():
            print("Cannot open camera")

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self.capture.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    def set_index(self, index: int):
        self.index = index
        self.initialize()

    def set_resolution(self, resolution: tuple[int]):
        self.resolution = resolution
        self.initialize()

    def show_settings(self):
        """
        Show the settings dialog of the camera.
        """
        self.capture.set(cv2.CAP_PROP_SETTINGS, 1)

    def take_image(self) -> cv2.Mat:
        is_frame_read_correctly, img_capture = self.capture.read()

        if not is_frame_read_correctly:
            print("Can't receive frame (stream end?). Exiting ...")
            return None

        scan_result = scanner.main(img_capture)

        return (
            img_capture,
            scan_result.warped,
            scan_result.contour,
        )

    def get_available_camera_indices(self) -> list[int]:
        found_camera_indices = []

        for index in range(10):
            dummy_capture = cv2.VideoCapture(index + cv2.CAP_DSHOW)
            if dummy_capture.isOpened():
                found_camera_indices.append(index)
            dummy_capture.release()

        self.initialize()

        return found_camera_indices
