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


class GrabCutHough(BaseModel):
    RESCALED_HEIGHT = 256.0

    MARGIN = 0.05
    BLUR_KSIZE = 13
    MORPH_KSIZE = 13
    CANNY_THRESHOLD1 = 0
    CANNY_THRESHOLD2 = 1
    HOUGH_RHO = 1
    HOUGH_THETA = math.pi / 60
    HOUGH_THRESHOLD = 100
    HOUGH_MAX_LINES = 16
    MIN_INTERSECTION_ANGLE = 60 * math.pi / 180
    MIN_CONTOUR_AREA_RATIO = 0.10
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
                "mask": None,
                "mask2": None,
                "img_grabcut": None,
                "blur": None,
                "dilated": None,
                "edge": None,
                "img_hough_preview": None,
                "best_mask": None,
                "img_hough_best_contour": None,
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

        # The result is converted back to original scale later, so save this ratio
        original_scale = img.shape[0] / img_scale.shape[0]

        result.debug_images["img_scale"] = img_scale

        # Apply a margin to the image to avoid the edges
        height, width = img_scale.shape[:2]
        margin = self.MARGIN
        x = int(width * margin)
        y = int(height * margin)
        rect = (x, y, width - 2 * x, height - 2 * y)

        # Create mask and models
        mask = np.zeros(img_scale.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        result.debug_images["mask"] = mask

        # Apply GrabCut
        cv2.grabCut(img_scale, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

        # Create binary mask
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")

        result.debug_images["mask2"] = mask2

        # Convert to RGBA and apply mask to alpha channel
        img_grabcut = cv2.cvtColor(img_scale, cv2.COLOR_BGR2BGRA)
        img_grabcut[:, :, 3] = mask2 * 255  # 0 for background, 255 for foreground

        result.debug_images["img_grabcut"] = img_grabcut

        blur = cv2.GaussianBlur(
            src=mask2,
            ksize=(self.BLUR_KSIZE, self.BLUR_KSIZE),
            sigmaX=0,
        )

        result.debug_images["blur"] = blur

        # We can then apply a morphological 'Close' transformation to the image.
        # Closing is a 'Dilation' followed by 'Erosion'. Dilation will grow bright
        # areas of the picture, which will fill in potential small holes. Erosion
        # will then shrink these areas back down. It is useful in closing small
        # holes inside the foreground objects, or small black points on the object.
        # See https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
        dilated = cv2.morphologyEx(
            src=blur,
            op=cv2.MORPH_CLOSE,
            kernel=cv2.getStructuringElement(
                shape=cv2.MORPH_RECT,
                ksize=(self.MORPH_KSIZE, self.MORPH_KSIZE),
            ),
        )

        result.debug_images["dilated"] = blur

        edge = cv2.Canny(
            image=dilated,
            threshold1=self.CANNY_THRESHOLD1,
            threshold2=self.CANNY_THRESHOLD2,
        )

        result.debug_images["edge"] = edge

        lines = cv2.HoughLines(
            image=edge,
            rho=2,
            theta=math.pi / 180,
            threshold=100,
            srn=0,
            stn=0,
            min_theta=0,
            max_theta=np.pi,
        )

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
            max_x=edge.shape[1],
            max_y=edge.shape[0],
            min_intersection_angle=self.MIN_INTERSECTION_ANGLE,
            min_corner_distance=self.MIN_CONTOUR_CORNER_DISTANCE,
        )

        # Find the best contour by scoring them and filtering out invalid ones
        best_contour, best_mask = hough_utils.find_best_contour(
            contours=contours,
            image_edged=edge,
            min_contour_area_ratio=self.MIN_CONTOUR_AREA_RATIO,
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
