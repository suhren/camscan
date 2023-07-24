import math
import itertools
import typing as t
from dataclasses import dataclass

import cv2
import numpy as np

import image_utils


RESCALED_HEIGHT = 500.0

MORPH_KSIZE = 13
CANNY_THRESHOLD1 = 0
CANNY_THRESHOLD2 = 84

HOUGH = 25


def draw_hough_lines(
    image: cv2.Mat,
    lines: list,
    color: tuple = (0, 255, 0),
    thickness: int = 2,
):
    out = image.copy()
    for line in lines:
        rho, theta = line
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 10000 * (-b))
        y1 = int(y0 + 10000 * (a))
        x2 = int(x0 - 10000 * (-b))
        y2 = int(y0 - 10000 * (a))
        cv2.line(
            img=out,
            pt1=(x1, y1),
            pt2=(x2, y2),
            color=color,
            thickness=thickness,
        )

    return out


def intersection(rho_1, theta_1, rho_2, theta_2):
    """Finds the intersection of two lines given in Hesse normal form.

    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    A = np.array(
        [[np.cos(theta_1), np.sin(theta_1)], [np.cos(theta_2), np.sin(theta_2)]]
    )
    b = np.array([[rho_1], [rho_2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [x0, y0]


def find_intersections(hough_lines: list, max_x: int, max_y: int):
    combinations = list(itertools.combinations(hough_lines, r=2))

    corners = []

    for (rho_1, theta_1), (rho_2, theta_2) in combinations:
        theta_min = min(theta_1, theta_2)
        theta_max = max(theta_1, theta_2)

        d = min(theta_max - theta_min, math.pi - theta_max + theta_min)

        if d < 60 * math.pi / 180:
            continue

        x, y = intersection(rho_1, theta_1, rho_2, theta_2)

        if 0 <= x <= max_x and 0 <= y <= max_y:
            corners.append((x, y))

    return np.array(corners)


def find_cycles_recursive(
    graph: list[set[int]],
    cycle_length: int,
    visited: list[int],
):
    successors = graph[visited[-1]]
    if len(visited) == cycle_length:
        if visited[0] in successors:
            yield visited
    elif len(visited) < cycle_length:
        for v in successors:
            if v in visited:
                continue
            yield from find_cycles_recursive(graph, cycle_length, visited + [v])


def find_cycles(graph: list[set[int]], cycle_length: int):
    for i in range(len(graph)):
        yield from find_cycles_recursive(
            graph,
            cycle_length=cycle_length,
            visited=[i],
        )


def find_contours(hough_lines: list, max_x: int, max_y: int):
    num_lines = len(hough_lines)
    combinations = itertools.combinations(range(num_lines), r=2)
    corners = []

    crossing_lines = []

    for i, j in combinations:
        rho_1, theta_1 = hough_lines[i]
        rho_2, theta_2 = hough_lines[j]
        theta_min = min(theta_1, theta_2)
        theta_max = max(theta_1, theta_2)

        d = min(theta_max - theta_min, math.pi - theta_max + theta_min)

        if d < 60 * math.pi / 180:
            continue

        x, y = intersection(rho_1, theta_1, rho_2, theta_2)

        if 0 <= x <= max_x and 0 <= y <= max_y:
            corners.append((x, y))
            crossing_lines.append(set([i, j]))

    corners = np.array(corners)

    graph = []

    for i, (xi, yi) in enumerate(corners):
        neighbors = set()
        for j, (xj, yj) in enumerate(corners):
            if len(crossing_lines[i].intersection(crossing_lines[j])) == 0:
                continue
            if math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2) < 50:
                continue

            neighbors.add(j)
        graph.append(neighbors)

    # Perform depth-first search of the Graph of corners with a max depth of 4
    contour_indices = list(find_cycles(graph=graph, cycle_length=4))

    unique_contour_indices = {}

    for index_list in contour_indices:
        index_list_id = str(set(index_list))
        if index_list_id not in unique_contour_indices:
            unique_contour_indices[index_list_id] = index_list

    contour_corners = [
        corners[index_list] for index_list in unique_contour_indices.values()
    ]
    return contour_corners


def order_contour(contour):
    result = np.zeros((4, 2), dtype="float32")
    s = contour.sum(axis=1)

    result[0] = contour[np.argmin(s)]
    result[2] = contour[np.argmax(s)]

    diff = np.diff(contour, axis=1)
    result[1] = contour[np.argmin(diff)]
    result[3] = contour[np.argmax(diff)]

    return result


@dataclass
class ScanResult:
    debug_images: dict[str, cv2.Mat]
    contour: np.ndarray
    warped: cv2.Mat


def main(img: cv2.Mat):
    img_scale = image_utils.resize_with_aspect_ratio(
        image=img,
        height=RESCALED_HEIGHT,
    )

    original_scale = img.shape[0] / img_scale.shape[0]

    img_scale_gray = cv2.cvtColor(
        src=img_scale,
        code=cv2.COLOR_BGR2GRAY,
    )

    img_scale_gray_blur = cv2.GaussianBlur(
        src=img_scale_gray,
        ksize=(13, 13),
        sigmaX=0,
    )

    img_scale_gray_blur_dilated = cv2.morphologyEx(
        src=img_scale_gray_blur,
        op=cv2.MORPH_CLOSE,
        kernel=cv2.getStructuringElement(
            shape=cv2.MORPH_RECT,
            ksize=(MORPH_KSIZE, MORPH_KSIZE),
        ),
    )

    img_edge = cv2.Canny(
        image=img_scale_gray_blur_dilated,
        threshold1=CANNY_THRESHOLD1,
        threshold2=CANNY_THRESHOLD2,
    )

    hough_lines = None

    for threshold in [100, 150, 200]:
        hough_lines = cv2.HoughLines(
            image=img_edge,
            rho=2,
            theta=np.pi / 180,
            threshold=threshold,
        )
        if hough_lines is not None and len(hough_lines) <= 16:
            break

    if hough_lines is None:
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

    # Convert shape from (num_lines, 1, 2) to (num_lines, 2)
    hough_lines = hough_lines.squeeze()
    img_hough_preview = draw_hough_lines(img_hough_preview, hough_lines)

    contours = find_contours(
        hough_lines=hough_lines,
        max_x=img_edge.shape[1],
        max_y=img_edge.shape[0],
    )

    MIN_AREA_RATIO = 0.25

    best_score = 0
    best_contour = None
    best_mask = None

    count = 0

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < img_scale.shape[0] * img_scale.shape[1] * MIN_AREA_RATIO:
            continue

        count += 1

        scoring = "edges"

        if scoring == "edges":
            mask = cv2.drawContours(
                image=np.zeros(shape=img_edge.shape, dtype=np.uint8),
                contours=[contour],
                contourIdx=-1,
                color=255,
                thickness=16,
            )

            mask = img_edge & mask
            score = mask.sum()
        elif scoring == "std":
            mask = cv2.drawContours(
                image=np.zeros(shape=img_scale_gray_blur_dilated.shape, dtype=np.uint8),
                contours=[contour],
                contourIdx=-1,
                color=255,
                thickness=-1,
            )
            mean, std = cv2.meanStdDev(
                src=img_scale_gray_blur_dilated,
                mask=mask,
            )
            mean = mean[0][0]
            std = std[0][0]
            score = area * 1 / (1 + std)
        else:
            raise Exception(f"Unknown scoring: {scoring}")

        if score > best_score:
            best_score = score
            best_contour = contour
            best_mask = mask

    img_hough_best_contour = img.copy()

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

    best_contour = order_contour(contour=best_contour)
    best_contour_original_scale = (best_contour * original_scale).astype(np.int32)

    img_hough_best_contour = image_utils.draw_contour(
        image=img_hough_best_contour,
        contour=best_contour_original_scale,
    )

    src = best_contour_original_scale.astype(np.float32)

    (tl, tr, br, bl) = src

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array(
        [
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1],
        ],
        dtype=np.float32,
    )

    warped = cv2.warpPerspective(
        src=img,
        M=cv2.getPerspectiveTransform(
            src=src,
            dst=dst,
        ),
        dsize=(maxWidth, maxHeight),
    )

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
        contour=best_contour_original_scale,
        warped=warped,
    )


if __name__ == "__main__":
    main()
