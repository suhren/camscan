"""
This module tests the functionality of the hough utility code.
"""


import numpy as np
import pytest

from camscan.model import hough_utils


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
) -> None:
    """
    Test the functionality of the cycle finder on some test graphs.
    """
    actual_cycles = hough_utils.find_cycles(graph=graph, length=length)
    sorted_actual_cycles = sorted([tuple(x) for x in actual_cycles])
    sorted_expected_cycles = sorted([tuple(x) for x in expected_cycles])
    assert sorted_actual_cycles == sorted_expected_cycles


@pytest.mark.parametrize(
    "contour, expected_contour",
    [
        [
            # Square in the region (0, 0) to (1, 1):
            # The input contour is on the form (TL, BL, BR, TR)
            # The ordered contour should be as (TL, TR, BR, BL)
            np.array([[0, 0], [0, 1], [1, 1], [1, 0]]),
            np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),
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
def test_order_contour(contour: np.ndarray, expected_contour: np.ndarray) -> None:
    """
    Test the function to order a contour of four corners so that they will be
    in the order [Top Left, Top Right, Bottom Right, Bottom Left].
    """
    actual_contour = hough_utils.order_contour(contour=contour)
    np.testing.assert_array_equal(actual=actual_contour, desired=expected_contour)
