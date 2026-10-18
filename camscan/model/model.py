import time
import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

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
    :param debug_images: List of intermediate images of the algorithm
    :param contour: Array with the contour corners of the detected document
    :param warped: The extracted document inside the detected contour
    :param step_timings: Optional dict of step names to execution times (seconds)
    """

    img: types.Image
    debug_images: dict[str, types.Image | None] = field(default_factory=dict)
    contour: types.Contour | None = None
    warped: types.Image | None = None
    error_message: str | None = None
    step_timings: dict[str, float] = field(default_factory=dict)
    _last_step_time: float = field(default=0.0, repr=False)

    def __post_init__(self) -> None:
        """Initialize the last step time to now when the object is created."""
        self._last_step_time = time.perf_counter()

    def add_step(self, name: str, image: types.Image | None = None) -> None:
        """
        Add a debug image and automatically record timing for this step.
        :param name: Name of the step (used as key in debug_images and step_timings)
        :param image: The intermediate image to store (optional)
        """
        now = time.perf_counter()

        self.step_timings[name] = now - self._last_step_time
        self._last_step_time = now

        # Store image if provided
        if image is not None:
            self.debug_images[name] = image

    @property
    def total_time(self) -> float:
        """Total processing time across all steps."""
        return sum(self.step_timings.values())

    def get_step_proportion(self, step_name: str) -> float:
        """
        Get the proportion of total time spent on a specific step.

        :param step_name: Name of the step
        :return: Proportion of total time (0.0 to 1.0)
        """
        if self.total_time == 0:
            return 0.0
        return self.step_timings.get(step_name, 0.0) / self.total_time

    def get_timing_summary(self) -> dict[str, dict[str, float]]:
        """
        Get a summary of timing information for all steps.

        :return: Dict mapping step names to {'time': seconds, 'proportion': 0.0-1.0}
        """
        total = self.total_time
        return {
            name: {
                "time": time_val,
                "proportion": time_val / total if total > 0 else 0.0,
            }
            for name, time_val in self.step_timings.items()
        }


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
