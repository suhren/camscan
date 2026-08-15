import functools
import typing as t
from tkinter import messagebox as tk_messagebox

import customtkinter as ctk

from camscan.camera import Camera, CameraError
from camscan.widgets.input import InputInt


class CameraSettings(ctk.CTkToplevel):
    """
    Settings for a camera.
    See https://docs.opencv.org/3.4.20/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704d
    """

    def __init__(
        self,
        master: t.Any,
        camera: Camera,
    ) -> None:
        super().__init__(master=master)
        self.resizable(width=False, height=False)
        self.title("Camera Settings")

        # Make sure this window is on top of the main window
        # We could simply just set topmost to True and leave it at that, but
        # that will prevent the Tooltips from working properly. We can instead
        # set it to topmost temporarily, use grab_set to set focus, and then
        # set topmost back to False. This brings the window to the front.
        # From the documentation it seems that using .lift(aboveThis=self) would
        # work, but I was not able to make that work.
        self.attributes("-topmost", True)
        self.grab_set()
        self.attributes("-topmost", False)

        props = camera.get_capture_properties()

        inputs: dict[str, InputInt] = {}

        def _on_value(value: float, index: int, name: str) -> None:
            try:
                camera.set_property_value(index=index, value=value)
            except CameraError as e:
                inputs[name].set_value(inputs[name]._previous_value)
                tk_messagebox.showerror(title="Error", message=str(e))

        for prop in props:
            inputs[prop.name] = InputInt(
                master=self,
                label=prop.name,
                value=int(prop.value),
                default_value=int(prop.value),
                on_value=functools.partial(_on_value, index=prop.index, name=prop.name),
            )
