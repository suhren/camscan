"""
This module provides utility postprocessing functions for images.
"""

import cv2


def dummy(image: cv2.Mat) -> cv2.Mat:
    """
    Apply no processing whatsoever and simply return the image again.
    :param image: The input image
    :return: The original image with no modification
    """
    return image


def sharpen(image: cv2.Mat) -> cv2.Mat:
    """
    Apply a sharpening effect to the input image.
    :param image: The input image
    :return: The image with the effect applied
    """
    blurred = cv2.GaussianBlur(
        src=image,
        ksize=(0, 0),
        sigmaX=3,
    )
    sharpened = cv2.addWeighted(
        src1=image,
        alpha=1.5,
        src2=blurred,
        beta=-0.5,
        gamma=0,
    )
    return sharpened


def grayscale(image: cv2.Mat) -> cv2.Mat:
    """
    Convert the input image to grayscale.
    :param image: The input image
    :return: The image with the effect applied
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def black_and_white(image: cv2.Mat) -> cv2.Mat:
    """
    Convert the image to black and white (it looks like a pencil sketch).
    This is done by converting it to grayscale, applying a sharpening effect,
    and then an adaptive threshold.
    :param image: The input image
    :return: The image with the effect applied
    """
    gray = grayscale(image=image)
    sharpened = sharpen(image=gray)
    thresholded = cv2.adaptiveThreshold(
        src=sharpened,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=21,
        C=15,
    )
    return thresholded
