"""
This module defines the algorithm for detecting and extracting documents from
images and its related parameters and helper functions.
"""

import cv2
import numpy as np

from camscan import types, utils
from camscan.model import hough_utils
from camscan.model.model import BaseModel, ModelResult


class FindContours(BaseModel):
    RESCALED_HEIGHT = 256
    BLUR_KSIZE = 13
    MORPH_KSIZE = 7
    CANNY_THRESHOLD1 = 0
    CANNY_THRESHOLD2 = 84

    def run(self, img: types.Image) -> ModelResult:
        """
        Detect and extract a document found in the input image.
        :param img: The input image to detect and extract the document
        :return: A ScanResult object containing the extracted document (if found)
        """

        result = ModelResult(
            img=img,
            debug_images={
                "input": img,
                "scale": None,
                "grayscale": None,
                "blur": None,
                "threshold": None,
                "morphology": None,
                "canny": None,
                "contours": None,
                "polygon": None,
                "warped": None,
            },
            contour=None,
            warped=None,
        )

        img_scale = utils.resize_with_aspect_ratio(
            image=img,
            height=256,
        )
        original_scale = img.shape[0] / img_scale.shape[0]

        result.debug_images["scale"] = img_scale

        img_grayscale = cv2.cvtColor(
            src=img_scale,
            code=cv2.COLOR_BGR2GRAY,
        )

        result.debug_images["grayscale"] = img_grayscale

        img_blur = cv2.GaussianBlur(
            src=img_grayscale,
            ksize=(self.BLUR_KSIZE, self.BLUR_KSIZE),
            sigmaX=0,
        )

        result.debug_images["blur"] = img_blur

        img_threshold = cv2.threshold(
            src=img_blur,
            thresh=0,
            maxval=255,
            type=cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )[1]

        result.debug_images["threshold"] = img_threshold

        kernel = np.ones((self.MORPH_KSIZE, self.MORPH_KSIZE), np.uint8)
        img_morphology = cv2.morphologyEx(img_threshold, cv2.MORPH_CLOSE, kernel)
        img_morphology = cv2.morphologyEx(img_morphology, cv2.MORPH_OPEN, kernel)

        result.debug_images["morphology"] = img_morphology

        img_canny = cv2.Canny(
            image=img_morphology,
            threshold1=self.CANNY_THRESHOLD1,
            threshold2=self.CANNY_THRESHOLD2,
        )

        result.debug_images["canny"] = img_canny

        contours, _hierarchy = cv2.findContours(
            image=img_canny,
            mode=cv2.RETR_EXTERNAL,
            method=cv2.CHAIN_APPROX_SIMPLE,
        )

        area_thresh = 0.0

        largest_contour = contours[0]
        largest_contour_idx = 0

        for i, c in enumerate(contours):
            area = cv2.contourArea(c)
            if area > area_thresh:
                area_thresh = area
                largest_contour = c
                largest_contour_idx = i

        img_contours = img_scale.copy()

        for i, contour in enumerate(contours):
            if i == largest_contour_idx:
                cv2.drawContours(img_contours, [contour], 0, (0, 255, 0), 2)
            else:
                cv2.drawContours(img_contours, [contour], 0, (255, 0, 0), 1)

        result.debug_images["contours"] = img_contours

        # get perimeter and approximate a polygon
        largest_contour_perimeter = cv2.arcLength(largest_contour, True)
        largest_contour_corners = cv2.approxPolyDP(
            curve=largest_contour,
            epsilon=0.04 * largest_contour_perimeter,
            closed=True,
        )

        # draw polygon on input image from detected corners
        img_polygon = img_scale.copy()
        cv2.polylines(
            img=img_polygon,
            pts=[largest_contour_corners],
            isClosed=True,
            color=(0, 0, 255),
            thickness=1,
            lineType=cv2.LINE_AA,
        )
        # Alternate: cv2.drawContours(page,[corners],0,(0,0,255),1)

        result.debug_images["polygon"] = img_polygon

        # Scale the best contour back to the original input image scale
        contour = (largest_contour_corners * original_scale).astype(np.int32)

        if contour.shape[0] != 4:
            return result

        # At this point, the contour has shape (4, 1, 2) and we want it as (4, 2)
        contour = np.reshape(contour, (4, 2))

        # Extract the area within the contour to a separate image
        warped, contour = hough_utils.extract_contour(image=img, contour=contour)

        result.debug_images["warped"] = warped
        result.contour = contour
        result.warped = warped

        return result
