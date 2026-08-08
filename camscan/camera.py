"""
This module provides an abstracted Camera class wrapping OpenCV video capture.
"""

import platform
import re

import cv2
from cv2_enumerate_cameras import enumerate_cameras
from cv2_enumerate_cameras.camera_info import CameraInfo

from camscan.logging import logger

# Depending on the platform, there might be the need to change the API backend
# preference. For Windows specifically we use DirectShow. Read more here:
# https://docs.opencv.org/3.4/d8/dfe/classcv_1_1VideoCapture.html
SYSTEM = platform.system()
if SYSTEM == "Windows":
    API_PREFERENCE = cv2.CAP_DSHOW
else:
    API_PREFERENCE = cv2.CAP_ANY


RE_RES_STR = re.compile(r"^(\d+)x(\d+)$")


class CameraError(Exception):
    """
    Error raised when there is a problem with the camera.
    """


class Camera:
    """
    A class representing a video camera. It mainly wraps and abstracts an OpenCV
    VideoCapture object, and makes tasks like changing resolution and input
    devices a bit easier.
    :param info: A cv2_enumerate_cameras.camera_info.CameraInfo object
    :param resolution: A tuple of the resolution on the form (width, height)
    :param target_fps: The target framerate of the camera in frames per second
    """

    def __init__(
        self,
        info: CameraInfo,
        resolution: tuple[int, int] = (800, 600),
        target_fps: int = 30,
    ):
        self.info = info
        self.resolution = resolution
        self.target_fps = target_fps
        self._video_capture: cv2.VideoCapture | None = None

    @property
    def index(self) -> int:
        return self.info.index

    @property
    def name(self) -> str:
        return self.info.name

    @property
    def display_name(self) -> str:
        return f"{self.info.name} ({self.info.index})"

    @property
    def info_string(self) -> str:
        return f"{self.info} ({self.resolution[0]}x{self.resolution[1]})"

    def initialize(self, attempts: int = 3) -> None:
        """
        Initialize the camera by opening a video capture feed using settings
        like resolution and framerate specified in this instance.
        """
        for _ in range(attempts):
            self._video_capture = cv2.VideoCapture(
                index=self.index,
                apiPreference=API_PREFERENCE,
            )
            if self._video_capture.isOpened():
                logger.debug(f"VideoCapture initialized for '{self.display_name}'")
                self.set_target_fps(self.target_fps)
                self.set_resolution(self.resolution)
                return

        logger.error(
            f"Could not initialize VideoCapture object for '{self.display_name}'"
            f" after {attempts} attempts"
        )

    @classmethod
    def resolution_string_to_tuple(cls, string: str) -> tuple:
        if matches := RE_RES_STR.findall(string):
            return (int(matches[0][0]), int(matches[0][1]))
        raise ValueError(f"Resolution {string} does not match {RE_RES_STR.pattern}")

    def set_target_fps(self, value: int) -> None:
        """
        Set the target FPS of the camera.
        :param value: The FPS value
        """
        self.target_fps = value
        if self._video_capture is not None:
            self._video_capture.set(cv2.CAP_PROP_FPS, self.target_fps)

    def get_resoltion_string(self) -> str:
        return f"{self.resolution[0]}x{self.resolution[1]}"

    def set_resolution(self, value: tuple[int, int] | str) -> None:
        """
        Set the capture resolution of the camera.
        :raises TypeError: If an invalid resolution value is passed.
        :param value: A tuple of the resolution on the form (width, height) or a string on the form 'WxH'
        """

        if isinstance(value, str):
            self.resolution = Camera.resolution_string_to_tuple(value)
        elif isinstance(value, tuple):
            self.resolution = value
        else:
            raise TypeError(f"Unknown resolution value type: {value} ({type(value)})")

        logger.debug(f"Setting camera resolution to {value}")

        if self._video_capture is not None:
            self._video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self._video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

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
        :return: An OpenCV image of the captured frame
        """

        if self._video_capture is None:
            self.initialize()

        if self._video_capture is None:
            return None

        is_frame_read_correctly, img_capture = self._video_capture.read()

        if not is_frame_read_correctly:
            return None

        return img_capture


class CameraManager:
    def __init__(self) -> None:
        self._cameras: dict[str, Camera] = {}
        self.update_available_cameras()

    def get_camera_names(self) -> list[str]:
        return list(self._cameras.keys())

    def get_camera_by_name(self, name: str) -> Camera:
        return self._cameras[name]

    def update_available_cameras(self) -> None:
        """
        Update available cameras.
        """
        # NOTE: We need to use API_PREFERENCE set correctly on Windows to get the
        # correct camera indices to be used by opencv. These indices may seem
        # strange, since opencv defaults to using the high digits of index to
        # represent the backend. For example, 701 indicates the second camera on the
        # DSHOW backend (700).
        # See https://github.com/lukehugh/cv2_enumerate_cameras/blob/v1.3.0/README.md

        self.cameras: dict[str, Camera] = {}

        for info in enumerate_cameras(apiPreference=API_PREFERENCE):
            camera = Camera(info=info)
            self._cameras[camera.display_name] = camera
