"""
This module defines the algorithm for detecting and extracting documents from
images and its related parameters and helper functions.
"""

import math

import cv2
import numpy as np

from camscan import types, utils
from camscan.model import hough_utils
from camscan.model.model import BaseModel, ModelResult


class SimpleHough(BaseModel):
    RESCALED_HEIGHT = 500.0

    BLUR_KSIZE = 13
    MORPH_KSIZE = 13
    CANNY_THRESHOLD1 = 0
    CANNY_THRESHOLD2 = 84
    HOUGH_RHO = 2
    HOUGH_THETA = np.pi / 180
    HOUGH_THRESHOLDS = (100, 150, 200)
    HOUGH_MAX_LINES = 16
    MIN_INTERSECTION_ANGLE = 60 * math.pi / 180
    MIN_CONTOUR_AREA_RATIO = 0.20
    MIN_CONTOUR_CORNER_DISTANCE = 50

    def run(self, img: types.Image) -> ModelResult:
        """
        Detect and extract a document found in the input image.
        :param img: The input image to detect and extract the document
        :return: A ScanResult object containing the extracted document (if found)
        """

        result = ModelResult(
            img=img,
            debug_images={
                "img": img,
                "img_scale": None,
                "img_scale_gray": None,
                "img_scale_gray_blur": None,
                "img_scale_gray_blur_dilated": None,
                "img_edge": None,
                "img_hough_preview": None,
                "img_hough_best_contour": None,
                "best_mask": None,
                "warped": None,
            },
            contour=None,
            warped=None,
        )

        # When detecting documents, we can assume that they make up a large portion
        # of the image and have a somewhat clear and defined rectangular shape. To
        # detect this overall shape, small details are therefore best avoided. The
        # image can then be resized to a lower resolution, with the added benefit of
        # also speeding up the algorithm as it is faster to process.
        img_scale = utils.resize_with_aspect_ratio(
            image=img,
            height=self.RESCALED_HEIGHT,
        )

        result.debug_images["img_scale"] = img_scale

        # The result is converted back to original scale later, so save this ratio
        original_scale = img.shape[0] / img_scale.shape[0]

        # The algorithm works by detecting edges in the image. For this application
        # color is not (usually) interesting, so convert the image to grayscale.
        img_scale_gray = cv2.cvtColor(
            src=img_scale,
            code=cv2.COLOR_BGR2GRAY,
        )

        result.debug_images["img_scale_gray"] = img_scale_gray

        # We can remove unnecessary details from the image by applying a Gaussian
        # blur, which will suppress high frequency noise (like the actual text on
        # the pages of a document) while leaving larger details (like the actual
        # contour of the document).
        img_scale_gray_blur = cv2.GaussianBlur(
            src=img_scale_gray,
            ksize=(self.BLUR_KSIZE, self.BLUR_KSIZE),
            sigmaX=0,
        )

        result.debug_images["img_scale_gray_blur"] = img_scale_gray_blur

        # We can then apply a morphological 'Close' transformation to the image.
        # Closing is a 'Dilation' followed by 'Erosion'. Dilation will grow bright
        # areas of the picture, which will fill in potential small holes. Erosion
        # will then shrink these areas back down. It is useful in closing small
        # holes inside the foreground objects, or small black points on the object.
        # See https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
        img_scale_gray_blur_dilated = cv2.morphologyEx(
            src=img_scale_gray_blur,
            op=cv2.MORPH_CLOSE,
            kernel=cv2.getStructuringElement(
                shape=cv2.MORPH_RECT,
                ksize=(self.MORPH_KSIZE, self.MORPH_KSIZE),
            ),
        )

        result.debug_images["img_scale_gray_blur_dilated"] = img_scale_gray_blur_dilated

        # We can then apply the Canny edge detection algorithm to the image
        img_edge = cv2.Canny(
            image=img_scale_gray_blur_dilated,
            threshold1=self.CANNY_THRESHOLD1,
            threshold2=self.CANNY_THRESHOLD2,
        )

        result.debug_images["img_edge"] = img_edge

        # Apply the Hough Line transform to the edged image
        # The following code will try an increasing range of threshold values to
        # attempt to minimize the amount of returned lines
        lines = None
        for threshold in self.HOUGH_THRESHOLDS:
            lines = cv2.HoughLines(
                image=img_edge,
                rho=self.HOUGH_RHO,
                theta=self.HOUGH_THETA,
                threshold=threshold,
            )
            if lines is not None and len(lines) <= self.HOUGH_MAX_LINES:
                break

        # Return if no lines were found in the Hough Transform
        if lines is None:
            return result

        img_hough_preview = img_scale.copy()

        # HoughLines produces an array with shape (num_lines, 1, 2) which we are
        # reshaping to (num_lines, 2)
        lines = lines.reshape((lines.shape[0], lines.shape[2]))
        img_hough_preview = hough_utils.draw_hough_lines(
            image=img_hough_preview, lines=lines
        )

        result.debug_images["img_hough_preview"] = img_hough_preview

        # Run the contour finding algorithm to get a list of contours
        contours = hough_utils.find_contours(
            lines=lines,
            max_x=img_edge.shape[1],
            max_y=img_edge.shape[0],
        )

        # Find the best contour by scoring them and filtering out invalid ones
        best_contour, best_mask = hough_utils.find_best_contour(
            contours=contours,
            image_edged=img_edge,
        )

        result.debug_images["best_mask"] = best_mask

        # Return if no best contour could be found
        if best_contour is None:
            return result

        # Scale the best contour back to the original input image scale
        best_contour = (best_contour * original_scale).astype(np.int32)

        img_hough_best_contour = utils.draw_contour(
            image=img,
            contour=best_contour,
        )

        result.debug_images["img_hough_best_contour"] = img_hough_best_contour

        # Extract the area within the contour to a separate image
        warped, best_contour = hough_utils.extract_contour(
            image=img, contour=best_contour
        )

        result.debug_images["warped"] = warped
        result.contour = best_contour
        result.warped = warped

        return result
