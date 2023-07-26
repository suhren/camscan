"""
This module defines the algorithm for detecting and extracting documents from
images and its related parameters and helper functions.
"""

import math
import itertools
from dataclasses import dataclass
import collections

import cv2
import numpy as np

import utils


RESCALED_HEIGHT = 500.0

BLUR_KSIZE = 13
MORPH_KSIZE = 13
CANNY_THRESHOLD1 = 0
CANNY_THRESHOLD2 = 84
HOUGH_RHO = 2
HOUGH_THETA = np.pi / 180
HOUGH_THRESHOLDS = [100, 150, 200]
HOUGH_MAX_LINES = 16
MIN_INTERSECTION_ANGLE = 60 * math.pi / 180
MIN_CONTOUR_AREA_RATIO = 0.20
MIN_CONTOUR_CORNER_DISTANCE = 50


def euclidean_distance(p1: tuple[int, int], p2: tuple[int, int]) -> float:
    """
    Calculate the euclidean distance between two points.
    :param p1: The first point on the form (x, y)
    :param p2: The second point on the form (x, y)
    :return: The euclidean distance between two points
    """
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def draw_hough_lines(
    image: cv2.Mat,
    lines: np.ndarray,
    color: tuple = (0, 255, 0),
    thickness: int = 2,
) -> cv2.Mat:
    """
    Given an input image, draw lines expressed in Hesse normal form produced by
    the Hough Transform onto it.
    :param image: The input image to draw lines onto
    :param lines: An array on the form [[rho1, theta1], [rho2, theta2], ...]
    :param color: The color of the drawn line
    :param thickness: The thickness of the drawn line
    :return: The image with the lines drawn
    """
    out = image.copy()
    for rho, theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (round(x0 + 10000 * (-b)), round(y0 + 10000 * a))
        pt2 = (round(x0 - 10000 * (-b)), round(y0 - 10000 * a))
        cv2.line(img=out, pt1=pt1, pt2=pt2, color=color, thickness=thickness)

    return out


def intersection(
    rho1: float, theta1: float, rho2: float, theta2: float
) -> tuple[int, int]:
    """
    Finds the intersection of two lines expressed in Hesse normal form produced
    by the Hough Transform. See https://en.wikipedia.org/wiki/Hough_transform
    and https://stackoverflow.com/a/383527/5087436 about how this intersection
    algorithm was derived.
    :param rho1: The rho parameter of the first line
    :param theta1: The theta parameter of the first line
    :param rho2: The rho parameter of the second line
    :param theta2: The theta parameter of the second line
    :return: The intersection (x, y) with the closest integer pixel coordinates
    """
    A = [[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]]
    b = [[rho1], [rho2]]
    x, y = np.linalg.solve(A, b).flatten()
    return round(x), round(y)


def find_cycles(graph: list[set[int]], length: int) -> list[list[int]]:
    """
    Find cycles in a graph of a given length.

    Note that we are only interested in the *unique* cycles found in the graph.
    For example, the cycle [0, 1, 2] in a fully connected graph of three points
    could be found in many duplicate ways, like [0, 2, 1], [2, 1, 0], etc.

    For a cycle of length N there are N different starting points, and two
    possible directions of travel, leading to 2N different ways of expressing a
    single cycle. For this reason, the algorithm will deduplicate these in a
    deterministic way such that each cycle always starts with the two lowest
    node indices in the cycle. In the above example, it would be the [0, 1, 2].

    :param graph: The graph represented as a list of sets on the form
        [{neighbor_11, neighbor_12, ...}, {neighbor_21, neighbor_22, ...}, ...]
    :param length: The length of the cycles to find
    :return: A list of cycles where each cycle is a list of node indices
    """

    def _find_cycles_recursive(
        graph: list[set[int]],
        length: int,
        visited: list[int],
        cycles: list,
    ):
        successors = graph[visited[-1]]
        if len(visited) == length:
            if visited[0] in successors:
                cycles.append(visited)
                return
        elif len(visited) < length:
            for v in successors:
                if v in visited:
                    continue
                _find_cycles_recursive(graph, length, visited + [v], cycles)

    all_found_cycles = []
    for i in range(len(graph)):
        cycles = []
        _find_cycles_recursive(
            graph,
            length=length,
            visited=[i],
            cycles=cycles,
        )
        all_found_cycles += cycles

    deduplicated_cycles = []

    for cycle in all_found_cycles:
        # Use the convention of having the lowest index of the cycle first
        # Do this by rotating the list to keep the cycle representation
        # Rotating left with the index, the lowest value should now be leftmost
        forward_cycle = collections.deque(cycle)
        reverse_cycle = collections.deque(reversed(cycle))
        forward_cycle.rotate(-forward_cycle.index(min(forward_cycle)))
        reverse_cycle.rotate(-reverse_cycle.index(min(reverse_cycle)))
        # Sorting ensures that we take the cycle with lowest two start indices
        cycle = sorted([tuple(forward_cycle), tuple(reverse_cycle)])[0]
        if cycle not in deduplicated_cycles:
            deduplicated_cycles.append(cycle)

    # Not needed, but also sort the list of cycles on indices for determinism
    return list(map(list, sorted(deduplicated_cycles)))


def find_intersections(
    lines: np.ndarray,
    max_x: int,
    max_y: int,
    min_angle: float = MIN_INTERSECTION_ANGLE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Given lines expressed in Hesse normal form produced by the Hough Transform,
    find the intersection coordinates of these lines under some conditions:
    - Line intersections where the angle of incidence is larger than 'min_angle'
    - Line intersections where 0 <= x <= 'max_x' and 0 <= y <= 'max_y'
    :param lines: An array on the form [[rho1, theta1], [rho2, theta2], ...]
    :param max_x: Maximum allowed intersection x coordinate value
    :param max_y: Maximum allowed intersection y coordinate value
    :param min_angle: Minimum angle of incidence of two intersecting lines
    :return:
        A tuple with an array of intersection coordinates, and another array
        containing the indices of the lines that make up that intersection.
    """
    intersections = []
    line_pairs = []

    for i, j in itertools.combinations(range(len(lines)), r=2):
        rho1, theta1 = lines[i]
        rho2, theta2 = lines[j]

        # For two intersecting lines there are two supplementary angles
        # If either of these is too small, we don't consider the line
        intersect_angle_1 = abs(theta1 - theta2)
        intersect_angle_2 = math.pi - intersect_angle_1
        if min(intersect_angle_1, intersect_angle_2) < min_angle:
            continue

        x, y = intersection(rho1=rho1, theta1=theta1, rho2=rho2, theta2=theta2)

        if 0 <= x <= max_x and 0 <= y <= max_y:
            intersections.append([x, y])
            line_pairs.append([i, j])

    return np.array(intersections), np.array(line_pairs)


def find_contours(
    lines: np.ndarray,
    max_x: int,
    max_y: int,
    min_corner_distance: float = MIN_CONTOUR_CORNER_DISTANCE,
) -> list[list[tuple[int, int]]]:
    """
    Given lines expressed in Hesse normal form produced by the Hough Transform,
    find the corners of four-sided polygons made up of their intersections.
    :param lines: An array on the form [[rho1, theta1], [rho2, theta2], ...]
    :param max_x: Maximum allowed x coordinate value
    :param max_y: Maximum allowed y coordinate value
    :param min_corner_distance: Minimum required distance of two corner points
    :return: An array where each entry is a list of four corner points
    """
    # Build a graph where the nodes are the intersection points of the lines,
    # and the edges represent the lines between the corners.
    intersections, line_pairs = find_intersections(
        lines=lines,
        max_x=max_x,
        max_y=max_y,
    )

    graph = []

    # Consider each intersection points as a node in the graph
    for i, p1 in enumerate(intersections):
        # For each such point, store a set of possible neighbors
        neighbors = set()
        # Iterate over all candidate neighbor intersection points
        for j, p2 in enumerate(intersections):
            # If the candidate neighbor is not on the same line, ignore it
            if not set(line_pairs[i]).intersection(line_pairs[j]):
                continue
            # If the candidate neighbor is too close, ignore it
            if euclidean_distance(p1, p2) < min_corner_distance:
                continue
            # Otherwise, add it as a valid neighbor
            neighbors.add(j)

        graph.append(neighbors)

    # Find all four-sided polygons by finding cycles in the graph with length 4
    cycles = find_cycles(graph=graph, length=4)

    # Use node indices of the cycles to get the coordinates for each polygon
    return [intersections[index_list] for index_list in cycles]


def order_contour(contour: np.ndarray) -> np.ndarray:
    """
    Sort a contour of four corners such that they will be ordered as
    [top left, top right, bottom right, bottom left].
    :param contour: An array of shape (4, 2) with the corners of the contour
    :return: The contour with the corners sorted
    """
    # Identify the top left point, with the smallest sum of x and y coordinates
    i0 = np.argmin(np.sum(contour, axis=1))
    x0, y0 = contour[i0]

    # For the remaining corners, get their relative angle to the top left corner
    candidates = []
    for i in filter(lambda i: i != i0, range(4)):
        x, y = contour[i]
        candidates.append((math.atan2(y - y0, x - x0), i))

    # Sorting on the angles gives the remaining corners in the order TR, BR, BL
    idxs = [i for _, i in sorted(candidates)]

    return np.array(
        [
            contour[i0],
            contour[idxs[0]],
            contour[idxs[1]],
            contour[idxs[2]],
        ]
    )


def find_best_contour(
    contours: list[list[tuple[int, int]]],
    image_edged: cv2.Mat = None,
    image: cv2.Mat = None,
):
    """
    Given a list of contours, score them according to some metric and filter out
    invalid ones. Depending on on the image that is supplied, a different metric
    will be used for the scoring.
    :param contours: An list of contours, where each contour is a list of points
    :param image_edged:
        The edge detected image. If supplied, the scoring of the contours will
        be based on their overlap of the edges.
    :param image:
        The original or processed image. If supplied, the scoring of the
        contours will be based on the standard deviation of pixels within the
        area of the contours.
    :return: A tuple of the best contour, and an image showing its scoring mask
    """
    best_score = 0
    best_contour = None
    best_mask = None

    count = 0

    if image is not None:
        shape = image.shape
    elif image_edged is not None:
        shape = image_edged.shape
    else:
        raise ValueError("At least one image must be supplied")

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < shape[0] * shape[1] * MIN_CONTOUR_AREA_RATIO:
            continue

        count += 1

        if image_edged is not None:
            mask = cv2.drawContours(
                image=np.zeros(shape=image_edged.shape, dtype=np.uint8),
                contours=[contour],
                contourIdx=-1,
                color=255,
                thickness=16,
            )

            mask = image_edged & mask
            score = mask.sum()
        else:
            mask = cv2.drawContours(
                image=np.zeros(shape=image.shape, dtype=np.uint8),
                contours=[contour],
                contourIdx=-1,
                color=255,
                thickness=-1,
            )
            mean, std = cv2.meanStdDev(
                src=image,
                mask=mask,
            )
            mean = mean[0][0]
            std = std[0][0]
            score = area * 1 / (1 + std)

        if score > best_score:
            best_score = score
            best_contour = contour
            best_mask = mask

    return best_contour, best_mask


def extract_contour(
    image: cv2.Mat,
    contour: list[tuple[int, int]],
) -> tuple[cv2.Mat, list[tuple[int, int]]]:
    """
    Given an image and a contour, extract the image contained withing the
    contour and apply a perspective transform on it to make it rectangular.
    :param image: The image to extract the contour from
    :param contour: A list of points defining the corners of the contour
    :return: The extracted perspective-warped image and the sorted contour
    """
    # Ensure that the corners of the contour are ordered as
    # (top left, top right, bottom right, bottom left)
    ordered_contour = order_contour(contour=contour)
    src = ordered_contour.astype(np.float32)
    (tl, tr, br, bl) = src
    # Find the longest width and height on the sides of the contour
    w = max(euclidean_distance(br, bl), euclidean_distance(tr, tl))
    h = max(euclidean_distance(tr, br), euclidean_distance(tl, bl))
    # Apply the perspective transform to warp the contour image to a rectangle
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    warped = cv2.warpPerspective(
        src=image,
        M=cv2.getPerspectiveTransform(src=src, dst=dst.astype(np.float32)),
        dsize=(int(w), int(h)),
    )
    return warped, ordered_contour


@dataclass
class ScanResult:
    """
    Helper class that contains the result of the document scanner output.
    :param debug_images: Dictionary of intermediate images of the algorithm
    :param contour: Array with the contour corners of the detected document
    :param warped: The extracted document inside the detected contour
    """

    debug_images: dict[str, cv2.Mat]
    contour: np.ndarray
    warped: cv2.Mat


def main(img: cv2.Mat) -> ScanResult:
    """
    Detect and extract a document found in the input image.
    :param img: The input image to detect and extract the document
    :return: A ScanResult object containing the extracted document (if found)
    """

    # When detecting documents, we can assume that they make up a large portion
    # of the image and have a somewhat clear and defined rectangular shape. To
    # detect this overall shape, small details are therefore best avoided. The
    # image can then be resized to a lower resolution, with the added benefit of
    # also speeding up the algorithm as it is faster to process.
    img_scale = utils.resize_with_aspect_ratio(
        image=img,
        height=RESCALED_HEIGHT,
    )

    # The result is converted back to original scale later, so save this ratio
    original_scale = img.shape[0] / img_scale.shape[0]

    # The algorithm works by detecting edges in the image. For this application
    # color is not (usually) interesting, so convert the image to grayscale.
    img_scale_gray = cv2.cvtColor(
        src=img_scale,
        code=cv2.COLOR_BGR2GRAY,
    )

    # We can remove unnecessary details from the image by applying a Gaussian
    # blur, which will suppress high frequency noise (like the actual text on
    # the pages of a document) while leaving larger details (like the actual
    # contour of the document).
    img_scale_gray_blur = cv2.GaussianBlur(
        src=img_scale_gray,
        ksize=(BLUR_KSIZE, BLUR_KSIZE),
        sigmaX=0,
    )

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
            ksize=(MORPH_KSIZE, MORPH_KSIZE),
        ),
    )

    # We can then apply the Canny edge detection algorithm to the image
    img_edge = cv2.Canny(
        image=img_scale_gray_blur_dilated,
        threshold1=CANNY_THRESHOLD1,
        threshold2=CANNY_THRESHOLD2,
    )

    # Apply the Hough Line transform to the edged image
    # The following code will try an increasing range of threshold values to
    # attempt to minimize the amount of returned lines
    lines = None
    for threshold in HOUGH_THRESHOLDS:
        lines = cv2.HoughLines(
            image=img_edge,
            rho=HOUGH_RHO,
            theta=HOUGH_THETA,
            threshold=threshold,
        )
        if lines is not None and len(lines) <= HOUGH_MAX_LINES:
            break

    # Return if no lines were found in the Hough Transform
    if lines is None:
        return ScanResult(
            debug_images=dict(
                img=img,
                img_scale=img_scale,
                img_scale_gray=img_scale_gray,
                img_scale_gray_blur=img_scale_gray_blur,
                img_scale_gray_blur_dilated=img_scale_gray_blur_dilated,
                img_edge=img_edge,
                img_hough_preview=None,
                img_hough_best_contour=None,
                best_mask=None,
                warped=None,
            ),
            contour=None,
            warped=None,
        )

    img_hough_preview = img_scale.copy()

    # HoughLines produces an array with shape (num_lines, 1, 2) which we are
    # reshaping to (num_lines, 2)
    lines = lines.reshape((lines.shape[0], lines.shape[2]))
    img_hough_preview = draw_hough_lines(image=img_hough_preview, lines=lines)

    # Run the contour finding algorithm to get a list of contours
    contours = find_contours(
        lines=lines,
        max_x=img_edge.shape[1],
        max_y=img_edge.shape[0],
    )

    # Find the best contour by scoring them and filtering out invalid ones
    best_contour, best_mask = find_best_contour(
        contours=contours,
        image_edged=img_edge,
    )

    # Return if no best contour could be found
    if best_contour is None:
        return ScanResult(
            debug_images=dict(
                img=img,
                img_scale=img_scale,
                img_scale_gray=img_scale_gray,
                img_scale_gray_blur=img_scale_gray_blur,
                img_scale_gray_blur_dilated=img_scale_gray_blur_dilated,
                img_edge=img_edge,
                img_hough_preview=img_hough_preview,
                img_hough_best_contour=None,
                best_mask=None,
                warped=None,
            ),
            contour=None,
            warped=None,
        )

    # Scale the best contour back to the original input image scale
    best_contour = (best_contour * original_scale).astype(np.int32)

    img_hough_best_contour = utils.draw_contour(
        image=img,
        contour=best_contour,
    )

    # Extract the area within the contour to a separate image
    warped, best_contour = extract_contour(image=img, contour=best_contour)

    return ScanResult(
        debug_images=dict(
            img=img,
            img_scale=img_scale,
            img_scale_gray=img_scale_gray,
            img_scale_gray_blur=img_scale_gray_blur,
            img_scale_gray_blur_dilated=img_scale_gray_blur_dilated,
            img_edge=img_edge,
            img_hough_preview=img_hough_preview,
            img_hough_best_contour=img_hough_best_contour,
            best_mask=best_mask,
            warped=warped,
        ),
        contour=best_contour,
        warped=warped,
    )


if __name__ == "__main__":
    main()
