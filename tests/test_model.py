"""
This module tests the functionality of the model code.
"""

import math

import cv2
import numpy as np
import pytest

from camscan.model.models.simple_hough import SimpleHough

model = SimpleHough()


@pytest.mark.parametrize(
    "image_file, expected_contour",
    [
        [
            "tests/images/IMG_1842.jpg",
            [[47, 84], [938, 84], [950, 681], [47, 685]],
        ],
        [
            "tests/images/IMG_1843.jpg",
            [[306, 101], [725, 310], [453, 869], [31, 667]],
        ],
        [
            "tests/images/IMG_1844.jpg",
            [[363, 154], [712, 339], [405, 845], [26, 576]],
        ],
        [
            "tests/images/IMG_1845.jpg",
            [[370, 266], [697, 356], [567, 837], [175, 678]],
        ],
        [
            "tests/images/IMG_1846.jpg",
            [[285, 164], [842, 269], [806, 681], [136, 515]],
        ],
    ],
)
def test_scanner(image_file: str, expected_contour: np.ndarray) -> None:
    """
    Test the algorithm's ability to accurately detect the contour corners of
    a few test images.
    """
    image = cv2.imread(image_file)
    assert image is not None
    scan_result = model.run(img=image)
    actual_contour = scan_result.contour
    assert actual_contour is not None, "No contour produced"
    assert actual_contour.shape == (4, 2), "Wrong shape contour"

    # The expected and actual contours should be ordered as (TL, TR, BR, BL)
    names = ("TL", "TR", "BR", "BL")
    max_distance = 30
    failed = []

    # For each pair of expected and actual corners, check the distance
    for p1, p2, name in zip(expected_contour, actual_contour, names):
        d = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
        message = f"Corner {name}: Expected: {p1}, Actual: {p2}, Distance: {d}"
        print(message)
        # If the corner distance is too big, it is a failed corner
        if d > max_distance:
            failed.append(f"{message}: Distance > {max_distance}")

    # Assert that no corners failed, or print them if they did
    assert not failed, "Bad corners:\n" + "\n".join(failed)
