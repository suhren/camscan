"""
This module defines the algorithm for detecting and extracting documents from
images and its related parameters and helper functions.
"""

import cv2
import numpy as np

from camscan import types, utils
from camscan.model import hough_utils
from camscan.model.model import BaseModel, ModelResult


class GrabCutConotours(BaseModel):
    RESCALED_HEIGHT = 128
    MORPH_KSIZE = 7
    MARGIN = 0.05
    GRABCUT_ITER_COUNT = 1

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
                "mask": None,
                "grabcut": None,
                "contours": None,
                "polygon": None,
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

        result.debug_images["scale"] = img_scale

        # Apply a margin to the image to avoid the edges
        height, width = img_scale.shape[:2]
        margin = self.MARGIN
        x = int(width * margin)
        y = int(height * margin)
        rect = (x, y, width - 2 * x, height - 2 * y)

        # Create mask and models
        img_mask1 = np.zeros(img_scale.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        # Apply GrabCut
        cv2.grabCut(
            img=img_scale,
            mask=img_mask1,
            rect=rect,
            bgdModel=bgdModel,
            fgdModel=fgdModel,
            iterCount=self.GRABCUT_ITER_COUNT,
            mode=cv2.GC_INIT_WITH_RECT,
        )

        # Create binary mask
        img_mask2 = np.where((img_mask1 == 2) | (img_mask1 == 0), 0, 1).astype("uint8")

        result.debug_images["mask"] = img_mask2 * 255

        # Convert to RGBA and apply mask to alpha channel
        img_grabcut = cv2.cvtColor(img_scale, cv2.COLOR_BGR2BGRA)
        img_grabcut[:, :, 3] = img_mask2 * 255  # 0 for background, 255 for foreground

        result.debug_images["grabcut"] = img_grabcut

        contours, _hierarchy = cv2.findContours(
            image=img_mask2,
            mode=cv2.RETR_EXTERNAL,
            method=cv2.CHAIN_APPROX_SIMPLE,
        )

        if len(contours) == 0:
            return result

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
