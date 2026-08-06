import numpy as np

Point = tuple[int, int] | list[int] | np.ndarray[tuple[int]]
Contour = np.ndarray[tuple[int, int]]
Line = tuple[float, float]

GrayScaleImage = np.ndarray[tuple[int, int]]
ColorImage = np.ndarray[tuple[int, int, int]]
Image = GrayScaleImage | ColorImage

Color = tuple[int, int, int]
