from abc import ABC, abstractmethod
from dataclasses import dataclass

from camscan import types


@dataclass
class ModelResult:
    """
    Helper class that contains the result of the model output.
    :param img: The input image
    :param debug_images: List of  of intermediate images of the algorithm
    :param contour: Array with the contour corners of the detected document
    :param warped: The extracted document inside the detected contour
    """

    img: types.Image
    debug_images: dict[str, types.Image | None]
    contour: types.Contour | None
    warped: types.Image | None


class BaseModel(ABC):
    @abstractmethod
    def run(self, img: types.Image) -> ModelResult:
        pass
