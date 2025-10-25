"""
This module tests the functionality of the scanner code.
"""

import cv2
import math
import pytest
import numpy as np

from camscan import scanner


@pytest.mark.parametrize(
    "graph, length, expected_cycles",
    [
        [
            # Graph with 5 nodes and 6 edges
            [[1, 3], [0, 2, 4], [1, 3], [2, 4], [1, 3]],
            4,
            [
                [0, 1, 2, 3],
                [0, 1, 4, 3],
                [1, 2, 3, 4],
            ],
        ],
        [
            # Fully connected graph with 4 nodes and 8 edges
            # Here, all cycles will traverse the same nodes in different ways
            [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]],
            4,
            [
                [0, 1, 3, 2],
                [0, 2, 1, 3],
                [0, 1, 2, 3],
            ],
        ],
    ],
)
def test_find_cycles(
    graph: list[set[int]],
    length: int,
    expected_cycles: list[list[int]],
):
    """
    Test the functionality of the cycle finder on some test graphs.
    """
    actual_cycles = scanner.find_cycles(graph=graph, length=length)
    actual_cycles = list(sorted(map(tuple, actual_cycles)))
    expected_cycles = list(sorted(map(tuple, expected_cycles)))
    assert actual_cycles == expected_cycles


@pytest.mark.parametrize(
    "contour, expected_contour",
    [
        [
            # The input contour is on the form (TL, BL, BR, TR)
            # The ordered contour should be as (TL, TR, BR, BL)
            np.array([[0, 0], [0, 1], [1, 1], [1, 0]]),
            np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),
        ],
        [
            # The input contour is on the form (TL, BL, BR, TR)
            # The ordered contour should be as (TL, TR, BR, BL)
            np.array([[0, 1], [1, 2], [2, 1], [1, 0]]),
            np.array([[0, 1], [1, 0], [2, 1], [1, 2]]),
        ],
        [
            # This is a 45 degree rhombus on in the region (0, 0) to (2, 2)
            # It is not obvious which of the 'left' or 'top' corners to consider
            # the 'top left'. By convention, the algorithm should pick the first
            # such occurrence in the input array.
            # The input contour is on the form (TL, BL, BR, TR)
            # The ordered contour should be as (TL, TR, BR, BL)
            np.array([[0, 1], [1, 2], [2, 1], [1, 0]]),
            np.array([[0, 1], [1, 0], [2, 1], [1, 2]]),
        ],
    ],
)
def test_order_contour(contour: np.ndarray, expected_contour: np.ndarray):
    """
    Test the function to order a contour of four corners so that they will be
    in the order [Top Left, Top Right, Bottom Right, Bottom Left].
    """
    actual_contour = scanner.order_contour(contour=contour)
    np.testing.assert_array_equal(actual=actual_contour, desired=expected_contour)


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
def test_scanner(image_file: str, expected_contour: list[tuple[int, int]]):
    """
    Test the algorithm's ability to accurately detect the contour corners of
    a few test images.
    """
    image = cv2.imread(image_file)
    scan_result = scanner.main(img=image)
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
