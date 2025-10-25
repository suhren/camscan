import cv2
import PIL
import customtkinter as ctk
from camscan.utils import resize_with_aspect_ratio


def opencv_to_pil_image(
    image: cv2.Mat,
    width: int = None,
    height: int = None,
) -> PIL.Image:
    """
    Given an OpenCV image, convert to to a PIL image. The function also supports
    resizing the image while keeping its original aspect ratio.
    :param image: The input OpenCV image
    :param width: Optional width to scale the image to
    :param width: Optional height to scale the image to
    :return: The image converted to a PIL image
    """
    return PIL.Image.fromarray(
        resize_with_aspect_ratio(
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            width=width,
            height=height,
        )
    )


def opencv_to_ctk_image(
    image: cv2.Mat,
    width: int = None,
    height: int = None,
) -> ctk.CTkImage:
    """
    Given an OpenCV image, convert to to a CTkImage. The function also supports
    resizing the image while keeping its original aspect ratio.
    :param image: The input OpenCV image
    :param width: Optional width to scale the image to
    :param width: Optional height to scale the image to
    :return: The image converted to a CTkImage
    """
    pil_image = opencv_to_pil_image(image=image, width=width, height=height)
    return ctk.CTkImage(
        pil_image,
        size=(pil_image.width, pil_image.height),
    )
