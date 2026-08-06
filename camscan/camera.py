"""
This module provides an abstracted Camera class wrapping OpenCV video capture.
"""

import platform

import cv2

from camscan.logging import logger

# Depending on the platform, there might be the need to change the API backend
# preference. For Windows specifically we use DirectShow. Read more here:
# https://docs.opencv.org/3.4/d8/dfe/classcv_1_1VideoCapture.html
SYSTEM = platform.system()
if SYSTEM == "Windows":
    API_PREFERENCE = cv2.CAP_DSHOW
else:
    API_PREFERENCE = cv2.CAP_ANY


class CameraError(Exception):
    """
    Error raised when there is a problem with the camera.
    """


class Camera:
    """
    A class representing a video camera. It mainly wraps and abstracts an OpenCV
    VideoCapture object, and makes tasks like changing resolution and input
    devices a bit easier.
    :param index: A device index of the desired camera
    :param resolution: A tuple of the resolution on the form (width, height)
    :param target_fps: The target framerate of the camera in frames per second
    """

    def __init__(
        self,
        index: int = 0,
        resolution: tuple[int, int] = (800, 600),
        target_fps: int = 30,
    ):
        self.index = index
        self.resolution = resolution
        self.target_fps = target_fps
        self._video_capture: cv2.VideoCapture | None = None
        self.initialize()

    def initialize(self) -> None:
        """
        Initialize the camera by opening a video capture feed using settings
        like resolution and framerate specified in this instance.
        """
        self._video_capture = cv2.VideoCapture(
            index=self.index,
            apiPreference=API_PREFERENCE,
        )
        self._video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self._video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self._video_capture.set(cv2.CAP_PROP_FPS, self.target_fps)

        if not self._video_capture.isOpened():
            logger.error("Cannot open camera")

    def set_index(self, index: int) -> None:
        """
        Set the OpenCV device indexof the camera.
        :param index: A device index of the desired camera
        """
        self.index = index
        self.initialize()

    def set_resolution(self, resolution: tuple[int, int]) -> None:
        """
        Set the capture resolution of the camera.
        :param resolution: A tuple of the resolution on the form (width, height)
        """
        self.resolution = resolution
        self.initialize()

    def show_settings(self) -> None:
        """
        Bring up the settings of the camera as a dialog window.
        """
        if self._video_capture is None:
            raise CameraError("The VideoCapture object is not inititalized")

        self._video_capture.set(cv2.CAP_PROP_SETTINGS, 1)

    def capture(self) -> cv2.typing.MatLike | None:
        """
        Capture an image from the video stream and extract documents from it.
        :raises CameraError: If the VideoCapture is not initialized
        :return: An OpenCV image of the captured frame
        """

        if self._video_capture is None:
            raise CameraError("The VideoCapture object is not inititalized")

        is_frame_read_correctly, img_capture = self._video_capture.read()

        if not is_frame_read_correctly:
            return None

        return img_capture

    def get_available_device_indices(self) -> list[int]:
        """
        Identify cameras available to OpenCV by naively attempting to initiate
        video capture on a range of device indices and saving the ones that
        successfully open.
        :return: A list of device indices where the video capture was successful
        """
        found_camera_indices = []
        for index in range(10):
            dummy_capture = cv2.VideoCapture(
                index=index,
                apiPreference=API_PREFERENCE,
            )
            if dummy_capture.isOpened():
                found_camera_indices.append(index)
            dummy_capture.release()

        # Ensure the camera is still properly initiated after opening captures
        self.initialize()

        return found_camera_indices
