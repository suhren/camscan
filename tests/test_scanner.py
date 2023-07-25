"""
This module tests the functionality of the scanner code.
"""

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
    actual_contour = scanner.order_contour(contour=contour)
    np.testing.assert_array_equal(x=actual_contour, y=expected_contour)
