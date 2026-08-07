import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass

from camscan import types


@dataclass
class ModelParameter:
    name: str
    value: t.Any

    def __post_init__(self) -> None:
        self.default_value = self.value

    def set(self, value: t.Any) -> None:
        self.value = value


@dataclass
class StringParameter(ModelParameter):
    pass


@dataclass
class IntParameter(ModelParameter):
    min_value: int | None = None
    max_value: int | None = None


@dataclass
class FloatParameter(ModelParameter):
    min_value: float | None = None
    max_value: float | None = None


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
    error_message: str | None = None


class BaseModel(ABC):
    @abstractmethod
    def __init__(self, parameters: list[ModelParameter]):
        self.parameters = parameters
        self.__parameters_lookup = {p.name: p for p in parameters}

    def run(self, img: types.Image) -> ModelResult:
        """
        Detect and extract a document found in the input image.
        :param img: The input image to detect and extract the document
        :return: A ScanResult object containing the extracted document (if found)
        """
        try:
            return self._run(img=img)
        except Exception as e:  # noqa: BLE001
            return ModelResult(
                img=img,
                debug_images={},
                contour=None,
                warped=None,
                error_message=str(e),
            )

    @abstractmethod
    def _run(self, img: types.Image) -> ModelResult:
        pass

    def param(self, name: str) -> ModelParameter:
        return self.__parameters_lookup[name]
