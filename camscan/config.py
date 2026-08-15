"""
Application configuration.
"""

from camscan import (
    __app_display_name__,
    __version__,
    postprocessing,
)
from camscan.model.model import BaseModel
from camscan.model.models.find_contours import FindContours
from camscan.model.models.grabcut_contours import GrabCutConotours
from camscan.model.models.grabcut_hough import GrabCutHough
from camscan.model.models.simple_hough import SimpleHough

MODELS: dict[str, BaseModel] = {
    "SimpleHough": SimpleHough(),
    "GrabCutHough": GrabCutHough(),
    "FindContours": FindContours(),
    "GrabCutConotours": GrabCutConotours(),
}

DEFAULT_MODEL_OPTION = "GrabCutConotours"

# Define the window title
WINDOW_TITLE = f"{__app_display_name__} {__version__}"

# Define the initial application window size
WINDOW_WIDTH = 1536
WINDOW_HEIGHT = 864

# Define the desired frames per second in the camera feed
CAMERA_FPS = 30
CAMERA_WAIT_MS = int(1000 / CAMERA_FPS)

# Define constants related to the styling of widgets in the GUI
LEFT_MENU_PAD_X = 20
LEFT_MENU_PAD_Y = 5
RIGHT_MENU_PAD_X = 10
RIGHT_MENU_PAD_Y = 5
LEFT_MENU_PACK_KWARGS = {"padx": LEFT_MENU_PAD_X, "pady": LEFT_MENU_PAD_Y}
RIGHT_MENU_PACK_KWARGS = {"padx": RIGHT_MENU_PAD_X, "pady": RIGHT_MENU_PAD_Y}

# Keybind used to capture images with the cameras
CAPTURE_KEYBIND = "<space>"

# Specify supported file formats when exporting images as separate files.
# See the OpenCV documentation for more information on the supported file types:
# https://docs.opencv.org/3.4/d4/da8/group__imgcodecs.html
EXPORT_SEPARATE_FILE_TYPES = [
    "png",
    "bmp",
    "dib",
    "jpeg",
    "jpg",
    "jpe",
    "jp2",
    "webp",
    "pbm",
    "pgm",
    "ppm",
    "pxm",
    "pnm",
    "sr",
    "ras",
    "tiff",
    "tif",
    "exr",
    "hdr",
    "pic",
]

# Specify supported file formats when exporting images as a single merged file
EXPORT_MERGED_FILE_TYPES = [
    "pdf",
]

# Specify the supported postprocessing functions for the captured images
POSTPROCESSING_OPTIONS = {
    "None": postprocessing.dummy,
    "Sharpen": postprocessing.sharpen,
    "Grayscale": postprocessing.grayscale,
    "Black and White": postprocessing.black_and_white,
}

DEFAULT_POSTPROCESSING_OPTION = "None"

# Define the list of pre-defined camera resolutions. In addition to these, the
# user can also enter custom resolutions manually.
RESOLUTIONS = [
    "3264x2448",
    "3264x1836",
    "2592x1944",
    "2048x1536",
    "1920x1080",
    "1600x1200",
    "1280x720",
    "1024x768",
    "800x600",
    "640x480",
]

# Collection of tooltip strings shown for various widgets
TOOLTIPS = {
    # Left panel
    "camera_configuration": (
        "Open camera configuration for selecting camera and resolution"
    ),
    "camera_driver_settings": (
        "Open camera driver settings dialog (determined by the selected camera)"
    ),
    "postprocessing": "Set the postprocessing effect applied to the captured images",
    "system_appearance": "Set the user interface appearance of the application",
    "system_ui_scaling": "Set the user interface scale of the application",
    "free_capture_mode": (
        "Ignore the document detection algorithm and capture the entire image"
    ),
    "two_page_mode": "Split the captured image into equal left and right parts",
    "capture": (
        f"Capture an image and save to the captures pane (key bind {CAPTURE_KEYBIND})"
    ),
    "export_separate": "Export captures as separate files in a directory",
    "export_merged": "Export captures as a single merged file",
    # Right panel
    "select_all": "Select or deselect all captures",
    "delete": "Delete the selected captures",
    # Camera Configuration Window
    "camera_name": (
        "Select a camera by choosing its name. Update this list with available"
        " devices using the camera identification button."
    ),
    "identify_cameras": (
        "Identify available cameras on the system and populate the camera list"
    ),
    "camera_resolution": "Set the camera resolution from a preset list of resolutions",
    "custom_camera_resolution": (
        "Set a custom camera resolution using a string on the form <width>x<height>"
    ),
}
